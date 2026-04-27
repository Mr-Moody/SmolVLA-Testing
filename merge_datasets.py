"""
Merge multiple LeRobotDataset v3 directories into a single combined dataset.

lerobot's LeRobotMultiDataset is not yet implemented in the version used here,
so this script merges at the file level:
  - Data parquets (one per chunk, multi-episode) are concatenated with updated
    episode_index, global frame index, and task_index
  - Episodes parquets (rich per-episode metadata with video file refs) are
    concatenated with all index/offset columns updated
  - Video files are hard-linked with new sequential file indices
  - meta/tasks.parquet, info.json, and stats.json are regenerated

Usage:
    uv run --project ../lerobot python merge_datasets.py \\
        /scratch0/$USER/lerobot_datasets/001 \\
        /scratch0/$USER/lerobot_datasets/002 \\
        /scratch0/$USER/lerobot_datasets/003 \\
        --output /scratch0/$USER/lerobot_datasets/merged \\
        --force
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

UNKNOWN_TASK_TEMPLATE = "unknown task ({dataset}:{index})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_info(dataset_root: Path) -> dict:
    path = dataset_root / "meta" / "info.json"
    with path.open() as f:
        return json.load(f)


def get_video_keys(info: dict) -> list[str]:
    return [
        k for k, v in info.get("features", {}).items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    ]


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src → dst (instant on same filesystem), fall back to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_tasks(dataset_root: Path) -> dict[int, str]:
    """Return {local_task_index: task_string} from meta/tasks.parquet."""
    path = dataset_root / "meta" / "tasks.parquet"
    if not path.exists():
        return {}
    t = pq.read_table(path)
    rows = t.to_pydict()
    return {rows["task_index"][i]: rows["task"][i] for i in range(len(t))}


def read_all_parquets(directory: Path) -> pa.Table | None:
    """Read and concatenate all *.parquet files found under directory."""
    if not directory.exists():
        return None
    tables: list[pa.Table] = []
    for chunk_dir in sorted(directory.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for pfile in sorted(chunk_dir.iterdir()):
            if pfile.suffix == ".parquet":
                tables.append(pq.read_table(pfile))
    if not tables:
        return None
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def count_video_files(source_root: Path, video_key: str) -> int:
    """Count total .mp4 files for this video key across all chunk directories."""
    vid_root = source_root / "videos" / video_key
    if not vid_root.exists():
        return 0
    count = 0
    for chunk_dir in vid_root.iterdir():
        if chunk_dir.is_dir():
            count += sum(1 for f in chunk_dir.iterdir() if f.suffix == ".mp4")
    return count


def collect_video_files_by_index(source_root: Path, video_key: str) -> dict[int, Path]:
    """Collect {file_index: file_path} for one video key within a source dataset.

    We key by file index because episodes parquet references videos via
    `videos/<key>/file_index`.
    """
    vid_root = source_root / "videos" / video_key
    if not vid_root.exists():
        return {}

    files_by_index: dict[int, Path] = {}
    for chunk_dir in sorted(vid_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for vfile in sorted(chunk_dir.iterdir()):
            if vfile.suffix != ".mp4":
                continue
            old_fi = int(vfile.stem.replace("file-", ""))
            if old_fi in files_by_index and files_by_index[old_fi] != vfile:
                raise ValueError(
                    f"Duplicate video file index {old_fi} for key '{video_key}' in {source_root}. "
                    "Cannot merge safely because episodes metadata only stores file_index."
                )
            files_by_index[old_fi] = vfile
    return files_by_index


def build_video_file_index_remap(source_root: Path, video_key: str, global_offset: int) -> tuple[dict[int, int], dict[int, Path]]:
    """Return remap {old_file_index: new_file_index} and source files mapping.

    Remaps sparse/non-contiguous local indices onto a contiguous global index
    space to avoid collisions when merging multiple datasets.
    """
    files_by_index = collect_video_files_by_index(source_root, video_key)
    remap: dict[int, int] = {}
    for rank, old_fi in enumerate(sorted(files_by_index.keys())):
        remap[old_fi] = global_offset + rank
    return remap, files_by_index


def update_int64_col(table: pa.Table, col_name: str, offset: int) -> pa.Table:
    """Add offset to every value in an int64 column."""
    idx = table.schema.get_field_index(col_name)
    if idx < 0:
        return table
    new_col = pa.array(
        [v + offset for v in table.column(col_name).to_pylist()],
        type=pa.int64(),
    )
    return table.set_column(idx, col_name, new_col)


def remap_int64_col(table: pa.Table, col_name: str, remap: dict[int, int], strict: bool = False) -> pa.Table:
    """Remap values in an int64 column via a lookup dict (identity for missing keys)."""
    if not remap:
        return table
    idx = table.schema.get_field_index(col_name)
    if idx < 0:
        return table
    values = table.column(col_name).to_pylist()
    if strict:
        missing = sorted({v for v in values if v not in remap})
        if missing:
            raise ValueError(
                f"Column '{col_name}' references missing file_index value(s): {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )

    new_col = pa.array([remap.get(v, v) for v in values], type=pa.int64())
    return table.set_column(idx, col_name, new_col)


def replace_null_int64_with_value(table: pa.Table, col_name: str, value: int) -> pa.Table:
    """Replace nulls in an int64 column with a provided fallback value."""
    idx = table.schema.get_field_index(col_name)
    if idx < 0:
        return table
    values = table.column(col_name).to_pylist()
    if all(v is not None for v in values):
        return table
    new_col = pa.array([value if v is None else v for v in values], type=pa.int64())
    return table.set_column(idx, col_name, new_col)


def set_int64_col_constant(table: pa.Table, col_name: str, value: int) -> pa.Table:
    """Set every value in an int64 column to a constant."""
    idx = table.schema.get_field_index(col_name)
    if idx < 0:
        return table
    n = len(table)
    new_col = pa.array([value] * n, type=pa.int64())
    return table.set_column(idx, col_name, new_col)


# ---------------------------------------------------------------------------
# Stats merge
# ---------------------------------------------------------------------------

def merge_stats(stats_list: list[dict], frame_counts: list[int]) -> dict:
    """Weighted average of per-feature mean/std/quantiles; min/max are element-wise.

    Handles arbitrarily nested list structures (e.g. per-channel image stats)
    via numpy, so the caller doesn't need to know the shape.
    """
    if not stats_list:
        return {}
    total_frames = float(sum(frame_counts))
    merged: dict = {}
    all_keys: set[str] = set()
    for s in stats_list:
        all_keys.update(s.keys())

    for key in all_keys:
        entries = [s[key] for s in stats_list if key in s]
        if not entries:
            continue
        sample = entries[0]
        if not isinstance(sample, dict):
            merged[key] = sample
            continue

        feature_merged: dict = {}
        stat_names: set[str] = set()
        for e in entries:
            stat_names.update(e.keys())

        weights = np.array(
            [frame_counts[i] for i, s in enumerate(stats_list) if key in s],
            dtype=float,
        )

        for stat in stat_names:
            arrays = [np.array(e[stat]) for e in entries if stat in e]
            if not arrays:
                continue

            if stat in ("mean", "std") or stat.startswith("q"):
                # Weighted average (approximation for quantiles)
                result = sum(a * w for a, w in zip(arrays, weights)) / total_frames
            elif stat == "min":
                result = np.minimum.reduce(arrays)
            elif stat == "max":
                result = np.maximum.reduce(arrays)
            elif stat == "count":
                result = np.add.reduce(arrays)
            else:
                result = arrays[0]

            feature_merged[stat] = result.tolist() if isinstance(result, np.ndarray) else result

        merged[key] = feature_merged

    return merged


# ---------------------------------------------------------------------------
# Main merge logic
# ---------------------------------------------------------------------------

def merge(source_roots: list[Path], output_root: Path, force: bool) -> Path:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"{output_root} already exists. Pass --force to overwrite.")
        LOGGER.info("Removing existing output directory: %s", output_root)
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)
    (output_root / "meta").mkdir()

    # -------------------------------------------------------------------------
    # Validate + load metadata
    # -------------------------------------------------------------------------
    infos = [load_info(r) for r in source_roots]
    fps = infos[0]["fps"]
    features = infos[0]["features"]
    robot_type = infos[0].get("robot_type", "unknown")
    chunks_size = infos[0].get("chunks_size", 1000)

    for root, info in zip(source_roots, infos):
        if info.get("codebase_version") != "v3.0":
            raise ValueError(
                f"{root}: expected codebase_version v3.0, got {info.get('codebase_version')}"
            )
        if info["fps"] != fps:
            raise ValueError(f"{root}: fps mismatch ({info['fps']} vs {fps})")

    LOGGER.info("Merging %d datasets:", len(source_roots))
    for r, info in zip(source_roots, infos):
        LOGGER.info("  %s — %d episodes, %d frames", r.name, info["total_episodes"], info["total_frames"])

    video_keys = get_video_keys(infos[0])
    LOGGER.info("Video keys: %s", video_keys)

    # -------------------------------------------------------------------------
    # Task registry (preserve per-episode task diversity across sources)
    # -------------------------------------------------------------------------
    global_task_to_idx: dict[str, int] = {}
    local_task_remaps: list[dict[int, int]] = []
    for source_root in source_roots:
        local_tasks = read_tasks(source_root)
        remap: dict[int, int] = {}
        for local_idx, task_text in sorted(local_tasks.items()):
            if task_text not in global_task_to_idx:
                global_task_to_idx[task_text] = len(global_task_to_idx)
            remap[local_idx] = global_task_to_idx[task_text]
        local_task_remaps.append(remap)
    LOGGER.info("Global task registry initialized with %d task(s)", len(global_task_to_idx))

    # -------------------------------------------------------------------------
    # Process each source
    # -------------------------------------------------------------------------
    all_data_tables: list[pa.Table] = []
    all_episodes_tables: list[pa.Table] = []
    video_file_offsets: dict[str, int] = {k: 0 for k in video_keys}
    stats_list: list[dict] = []
    frame_counts: list[int] = []

    global_ep_offset = 0
    global_frame_offset = 0

    for source_root, info, task_remap in zip(source_roots, infos, local_task_remaps):
        src_eps = info["total_episodes"]
        src_frames = info["total_frames"]
        LOGGER.info("Processing %s (%d episodes, %d frames)...", source_root.name, src_eps, src_frames)

        # --- Data parquet (one multi-episode file per chunk) ---
        data_table = read_all_parquets(source_root / "data")
        if data_table is not None:
            task_idx_col = data_table.schema.get_field_index("task_index")
            if task_idx_col >= 0:
                task_values = data_table.column("task_index").to_pylist()
                local_task_indices = sorted({int(v) for v in task_values if v is not None})

                # Some datasets may reference task indices that are absent from
                # tasks.parquet; keep them distinct instead of collapsing.
                for local_idx in local_task_indices:
                    if local_idx not in task_remap:
                        synthetic_task = UNKNOWN_TASK_TEMPLATE.format(
                            dataset=source_root.name,
                            index=local_idx,
                        )
                        if synthetic_task not in global_task_to_idx:
                            global_task_to_idx[synthetic_task] = len(global_task_to_idx)
                        task_remap[local_idx] = global_task_to_idx[synthetic_task]

                if any(v is None for v in task_values):
                    null_task = UNKNOWN_TASK_TEMPLATE.format(
                        dataset=source_root.name,
                        index="null",
                    )
                    if null_task not in global_task_to_idx:
                        global_task_to_idx[null_task] = len(global_task_to_idx)
                    data_table = replace_null_int64_with_value(data_table, "task_index", -1)
                    task_remap[-1] = global_task_to_idx[null_task]

            data_table = update_int64_col(data_table, "episode_index", global_ep_offset)
            data_table = update_int64_col(data_table, "index", global_frame_offset)
            data_table = remap_int64_col(data_table, "task_index", task_remap, strict=True)
            all_data_tables.append(data_table)

        # Build robust video file-index remaps for this source dataset.
        # This prevents collisions when local file indices are sparse.
        video_index_remaps: dict[str, dict[int, int]] = {}
        video_source_files: dict[str, dict[int, Path]] = {}
        for vkey in video_keys:
            remap, files_by_index = build_video_file_index_remap(
                source_root, vkey, video_file_offsets[vkey]
            )
            video_index_remaps[vkey] = remap
            video_source_files[vkey] = files_by_index

        # --- Episodes parquet (rich per-episode metadata) ---
        episodes_table = read_all_parquets(source_root / "meta" / "episodes")
        if episodes_table is not None:
            episodes_table = update_int64_col(episodes_table, "episode_index", global_ep_offset)
            episodes_table = update_int64_col(episodes_table, "dataset_from_index", global_frame_offset)
            episodes_table = update_int64_col(episodes_table, "dataset_to_index", global_frame_offset)
            for vkey in video_keys:
                fi_col = f"videos/{vkey}/file_index"
                episodes_table = remap_int64_col(
                    episodes_table,
                    fi_col,
                    video_index_remaps[vkey],
                    strict=True,
                )
            all_episodes_tables.append(episodes_table)

        # --- Video files: hardlink with new sequential file indices ---
        for vkey in video_keys:
            for old_fi, vfile in sorted(video_source_files[vkey].items()):
                new_fi = video_index_remaps[vkey][old_fi]
                dst = output_root / "videos" / vkey / "chunk-000" / f"file-{new_fi:03d}.mp4"
                link_or_copy(vfile, dst)

        # --- Update running offsets ---
        for vkey in video_keys:
            video_file_offsets[vkey] += len(video_index_remaps[vkey])

        stats_path = source_root / "meta" / "stats.json"
        if stats_path.exists():
            with stats_path.open() as f:
                stats_list.append(json.load(f))
            frame_counts.append(src_frames)

        global_ep_offset += src_eps
        global_frame_offset += src_frames
        LOGGER.info("  Done.")

    # -------------------------------------------------------------------------
    # Write merged data parquet
    # -------------------------------------------------------------------------
    if all_data_tables:
        merged_data = pa.concat_tables(all_data_tables)
        dst_data = output_root / "data" / "chunk-000"
        dst_data.mkdir(parents=True, exist_ok=True)
        pq.write_table(merged_data, dst_data / "file-000.parquet")
        LOGGER.info("Data parquet: %d rows written.", len(merged_data))

    # -------------------------------------------------------------------------
    # Write merged episodes parquet
    # -------------------------------------------------------------------------
    if all_episodes_tables:
        merged_episodes = pa.concat_tables(all_episodes_tables)
        dst_eps = output_root / "meta" / "episodes" / "chunk-000"
        dst_eps.mkdir(parents=True, exist_ok=True)
        pq.write_table(merged_episodes, dst_eps / "file-000.parquet")
        LOGGER.info("Episodes parquet: %d rows written.", len(merged_episodes))

    # -------------------------------------------------------------------------
    # Write merged tasks.parquet
    # -------------------------------------------------------------------------
    if global_task_to_idx:
        task_indices = list(range(len(global_task_to_idx)))
        task_strings = [task for task, _ in sorted(global_task_to_idx.items(), key=lambda x: x[1])]
        tasks_table = pa.table({
            "task_index": pa.array(task_indices, type=pa.int64()),
            "task": pa.array(task_strings, type=pa.string()),
        })
        pq.write_table(tasks_table, output_root / "meta" / "tasks.parquet")

    # -------------------------------------------------------------------------
    # Write info.json
    # -------------------------------------------------------------------------
    total_episodes = global_ep_offset
    total_frames = global_frame_offset
    total_video_files = sum(video_file_offsets.values())
    total_chunks = max(1, (total_episodes + chunks_size - 1) // chunks_size)

    merged_info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(global_task_to_idx),
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": infos[0].get("data_path"),
        "video_path": infos[0].get("video_path"),
        "features": features,
    }
    with (output_root / "meta" / "info.json").open("w") as f:
        json.dump(merged_info, f, indent=2)

    if stats_list:
        merged_stats = merge_stats(stats_list, frame_counts)
        with (output_root / "meta" / "stats.json").open("w") as f:
            json.dump(merged_stats, f, indent=2)

    LOGGER.info(
        "Merge complete: %d episodes, %d frames → %s",
        total_episodes, total_frames, output_root,
    )
    return output_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobotDataset v3 directories into one."
    )
    parser.add_argument("sources", nargs="+", type=Path, help="Source dataset directories.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists.")
    args = parser.parse_args()

    for src in args.sources:
        if not (src / "meta" / "info.json").exists():
            raise FileNotFoundError(f"Not a valid LeRobotDataset v3: {src}")

    merge(args.sources, args.output, force=args.force)


if __name__ == "__main__":
    main()
