"""
Merge multiple LeRobotDataset v3 directories into a single combined dataset.

lerobot's LeRobotMultiDataset is not yet implemented in the version used here,
so this script merges at the file level:
  - Video files are hard-linked (same filesystem = instant, no re-encoding)
  - Parquet files are copied with episode_index updated to the global index
  - meta/info.json, episodes.jsonl, tasks.jsonl are regenerated

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

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

CHUNKS_SIZE = 1000  # episodes per chunk — must match lerobot default


def load_info(dataset_root: Path) -> dict:
    path = dataset_root / "meta" / "info.json"
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def episode_chunk(episode_index: int) -> int:
    return episode_index // CHUNKS_SIZE


def ep_parquet_path(root: Path, episode_index: int) -> Path:
    chunk = episode_chunk(episode_index)
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def ep_video_path(root: Path, video_key: str, episode_index: int) -> Path:
    chunk = episode_chunk(episode_index)
    return root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link src → dst, falling back to copy if cross-device."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def merge_stats(stats_list: list[dict], frame_counts: list[int]) -> dict:
    """Weighted average of per-feature mean/std; min/max are element-wise."""
    if not stats_list:
        return {}
    merged: dict = {}
    total_frames = sum(frame_counts)

    all_keys = set()
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
        stat_names = set()
        for e in entries:
            stat_names.update(e.keys())

        for stat in stat_names:
            values = [e[stat] for e in entries if stat in e]
            if not values:
                continue

            # scalar or list
            if isinstance(values[0], (int, float)):
                if stat == "mean":
                    weights = [frame_counts[i] for i, s in enumerate(stats_list) if key in s]
                    feature_merged[stat] = sum(v * w for v, w in zip(values, weights)) / total_frames
                elif stat == "std":
                    weights = [frame_counts[i] for i, s in enumerate(stats_list) if key in s]
                    feature_merged[stat] = (
                        sum(v * w for v, w in zip(values, weights)) / total_frames
                    )
                elif stat == "min":
                    feature_merged[stat] = min(values)
                elif stat == "max":
                    feature_merged[stat] = max(values)
                else:
                    feature_merged[stat] = values[0]
            elif isinstance(values[0], list):
                length = len(values[0])
                if stat in ("mean", "std"):
                    weights = [frame_counts[i] for i, s in enumerate(stats_list) if key in s]
                    feature_merged[stat] = [
                        sum(v[j] * w for v, w in zip(values, weights)) / total_frames
                        for j in range(length)
                    ]
                elif stat == "min":
                    feature_merged[stat] = [min(v[j] for v in values) for j in range(length)]
                elif stat == "max":
                    feature_merged[stat] = [max(v[j] for v in values) for j in range(length)]
                else:
                    feature_merged[stat] = values[0]
            else:
                feature_merged[stat] = values[0]

        merged[key] = feature_merged

    return merged


def merge(source_roots: list[Path], output_root: Path, force: bool) -> Path:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"{output_root} already exists. Pass --force to overwrite.")
        LOGGER.info("Removing existing output directory: %s", output_root)
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)
    (output_root / "meta").mkdir()

    # -------------------------------------------------------------------------
    # Validate and load metadata from all sources
    # -------------------------------------------------------------------------
    infos = [load_info(r) for r in source_roots]

    fps = infos[0]["fps"]
    features = infos[0]["features"]
    robot_type = infos[0].get("robot_type", "unknown")

    for i, (root, info) in enumerate(zip(source_roots, infos)):
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"{root}: expected codebase_version v3.0, got {info.get('codebase_version')}")
        if info["fps"] != fps:
            raise ValueError(f"{root}: fps mismatch ({info['fps']} vs {fps})")

    LOGGER.info("Merging %d datasets:", len(source_roots))
    for r, info in zip(source_roots, infos):
        LOGGER.info("  %s — %d episodes, %d frames", r.name, info["total_episodes"], info["total_frames"])

    # -------------------------------------------------------------------------
    # Merge tasks — build a global task → task_index mapping
    # -------------------------------------------------------------------------
    global_task_to_index: dict[str, int] = {}
    for root in source_roots:
        for row in load_jsonl(root / "meta" / "tasks.jsonl"):
            task = row["task"]
            if task not in global_task_to_index:
                global_task_to_index[task] = len(global_task_to_index)

    write_jsonl(
        output_root / "meta" / "tasks.jsonl",
        [{"task_index": idx, "task": task} for task, idx in sorted(global_task_to_index.items(), key=lambda x: x[1])],
    )
    LOGGER.info("Global task registry: %d task(s)", len(global_task_to_index))

    # -------------------------------------------------------------------------
    # Process each source dataset — copy parquet + link videos
    # -------------------------------------------------------------------------
    global_episode_index = 0
    merged_episodes: list[dict] = []
    total_frames = 0
    stats_list: list[dict] = []
    frame_counts: list[int] = []

    # Detect video keys from first source that has videos
    video_keys: list[str] = []
    for root in source_roots:
        videos_root = root / "videos"
        if videos_root.exists():
            for chunk_dir in sorted(videos_root.iterdir()):
                for vkey_dir in sorted(chunk_dir.iterdir()):
                    if vkey_dir.is_dir():
                        vk = vkey_dir.name
                        if vk not in video_keys:
                            video_keys.append(vk)

    LOGGER.info("Video keys: %s", video_keys)

    for source_root, info in zip(source_roots, infos):
        src_episodes = load_jsonl(source_root / "meta" / "episodes.jsonl")
        src_n = info["total_episodes"]
        src_frames = info["total_frames"]

        # Build local task_index → global task_index remap
        src_tasks = {row["task_index"]: row["task"] for row in load_jsonl(source_root / "meta" / "tasks.jsonl")}
        task_remap = {local_idx: global_task_to_index[task] for local_idx, task in src_tasks.items()}

        LOGGER.info("Processing %s (%d episodes)...", source_root.name, src_n)

        # Load stats if present
        stats_path = source_root / "meta" / "stats.json"
        if stats_path.exists():
            with stats_path.open() as f:
                stats_list.append(json.load(f))
            frame_counts.append(src_frames)

        for local_ep_idx in range(src_n):
            global_ep_idx = global_episode_index

            # --- parquet ---
            src_parquet = ep_parquet_path(source_root, local_ep_idx)
            dst_parquet = ep_parquet_path(output_root, global_ep_idx)
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)

            table = pq.read_table(src_parquet)

            # Update episode_index column
            new_ep_col = pa.array([global_ep_idx] * len(table), type=pa.int64())
            table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", new_ep_col)

            # Remap task_index if column exists
            if "task_index" in table.schema.names:
                old_task_col = table.column("task_index").to_pylist()
                new_task_col = pa.array([task_remap.get(t, t) for t in old_task_col], type=pa.int64())
                table = table.set_column(table.schema.get_field_index("task_index"), "task_index", new_task_col)

            pq.write_table(table, dst_parquet)
            ep_frames = len(table)
            total_frames += ep_frames

            # --- videos ---
            for vkey in video_keys:
                src_video = ep_video_path(source_root, vkey, local_ep_idx)
                dst_video = ep_video_path(output_root, vkey, global_ep_idx)
                if src_video.exists():
                    link_or_copy(src_video, dst_video)

            # --- episode metadata ---
            src_ep_meta = next((e for e in src_episodes if e.get("episode_index") == local_ep_idx), {})
            tasks_for_ep = src_ep_meta.get("tasks", [])
            # remap task strings through global mapping (tasks list stores strings, not indices)
            merged_episodes.append({
                "episode_index": global_ep_idx,
                "tasks": tasks_for_ep,
                "length": ep_frames,
            })

            global_episode_index += 1

        LOGGER.info("  Done — %d episodes copied.", src_n)

    # -------------------------------------------------------------------------
    # Write merged metadata
    # -------------------------------------------------------------------------
    write_jsonl(output_root / "meta" / "episodes.jsonl", merged_episodes)

    total_episodes = global_episode_index
    total_chunks = max(1, (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE)
    total_video_files = total_episodes * len(video_keys)

    merged_info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(global_task_to_index),
        "total_videos": total_video_files,
        "total_chunks": total_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple LeRobotDataset v3 directories into one.")
    parser.add_argument("sources", nargs="+", type=Path, help="Source dataset directories.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for merged dataset.")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it exists.")
    args = parser.parse_args()

    for src in args.sources:
        if not (src / "meta" / "info.json").exists():
            raise FileNotFoundError(f"Not a valid LeRobotDataset v3: {src}")

    merge(args.sources, args.output, force=args.force)


if __name__ == "__main__":
    main()
