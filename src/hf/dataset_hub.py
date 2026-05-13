"""
dataset_hub.py — Download, validate, and prepare HuggingFace datasets for training.

Provides a lightweight wrapper around huggingface_hub.snapshot_download
to fetch LeRobotDataset v3 datasets, validate they are properly cleaned/
converted/annotated, and optionally merge multiple datasets before training.

Usage (CLI):
    # Download + validate, print local path
    python src/hf/dataset_hub.py prepare NexusDwin/msd-connector-200-209 \\
        --cache-root .hf_cache

    # Multiple datasets → merged
    python src/hf/dataset_hub.py prepare \\
        NexusDwin/msd-connector-200 NexusDwin/msd-connector-201 \\
        --cache-root .hf_cache \\
        --merge-output lerobot_datasets/merged_msd

    # Validate a local dataset
    python src/hf/dataset_hub.py validate /path/to/dataset

Usage (Python):
    from src.hf.dataset_hub import prepare_datasets, validate_dataset
    path = prepare_datasets(["NexusDwin/msd-connector-200-209"])
    result = validate_dataset(Path("/path/to/dataset"))
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger(__name__)

REQUIRED_CODEBASE_VERSION = "v3.0"
DEFAULT_CACHE_SUBDIR = ".hf_cache"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class DatasetValidationError(Exception):
    def __init__(self, repo_id: str, result: ValidationResult):
        self.repo_id = repo_id
        self.result = result
        errors = "\n  ".join(result.errors)
        super().__init__(f"Dataset {repo_id} failed validation:\n  {errors}")


def validate_dataset(root: Path) -> ValidationResult:
    """Validate that a local dataset is properly cleaned, converted, and annotated.

    Checks:
        1. meta/info.json exists and is valid JSON
        2. codebase_version == "v3.0" (converted)
        3. total_episodes > 0
        4. total_frames > 0
        5. total_tasks > 0 (annotated)
        6. meta/tasks.parquet exists
        7. data/ directory has parquet files
        8. videos/ directory has mp4 files
        9. meta/stats.json exists (warning only)
    """
    result = ValidationResult()
    root = Path(root)

    # 1. meta/info.json
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        result.add_error("meta/info.json not found — dataset not converted")
        return result

    try:
        info = json.loads(info_path.read_text())
    except json.JSONDecodeError as e:
        result.add_error(f"meta/info.json is invalid JSON: {e}")
        return result

    # 2. codebase version
    version = info.get("codebase_version")
    if version != REQUIRED_CODEBASE_VERSION:
        result.add_error(
            f"codebase_version={version!r}, expected {REQUIRED_CODEBASE_VERSION!r}"
        )

    # 3-4. episodes and frames
    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)
    if total_episodes <= 0:
        result.add_error(f"total_episodes={total_episodes} — no episodes")
    if total_frames <= 0:
        result.add_error(f"total_frames={total_frames} — no frames")

    # 5. tasks (annotation check)
    total_tasks = info.get("total_tasks", 0)
    if total_tasks <= 0:
        result.add_error(f"total_tasks={total_tasks} — not annotated")

    # 6. tasks.parquet
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        result.add_error("meta/tasks.parquet not found — not annotated")

    # 7. data directory
    data_dir = root / "data"
    if not data_dir.exists():
        result.add_error("data/ directory not found — not converted")
    else:
        parquets = list(data_dir.rglob("*.parquet"))
        if not parquets:
            result.add_error("data/ contains no parquet files")

    # 8. videos directory
    videos_dir = root / "videos"
    if not videos_dir.exists():
        result.add_error("videos/ directory not found — not converted")
    else:
        mp4s = list(videos_dir.rglob("*.mp4"))
        if not mp4s:
            result.add_error("videos/ contains no mp4 files")

    # 9. stats.json (warning only)
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        result.add_warning("meta/stats.json not found — stats will be computed on first train")

    if result.valid:
        LOGGER.info(
            "Validated %s: %d episodes, %d frames, %d tasks",
            root.name, total_episodes, total_frames, total_tasks,
        )

    return result


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dataset(
    repo_id: str,
    cache_root: Path | None = None,
    force: bool = False,
    token: str | None = None,
) -> Path:
    """Download a dataset from HuggingFace Hub to a local cache directory.

    Args:
        repo_id: HF dataset repo ID (e.g. "NexusDwin/msd-connector-200-209").
        cache_root: Local directory for cached downloads. Defaults to
            cwd / .hf_cache.
        force: Re-download even if cached.
        token: HF auth token. If None, uses cached token.

    Returns:
        Path to the downloaded dataset directory.
    """
    from huggingface_hub import snapshot_download

    if cache_root is None:
        cache_root = Path.cwd() / DEFAULT_CACHE_SUBDIR
    cache_root = Path(cache_root)

    local_path = cache_root / repo_id

    # Cache hit: skip if info.json already present
    if not force and (local_path / "meta" / "info.json").exists():
        LOGGER.info("Cache hit: %s → %s", repo_id, local_path)
        return local_path

    if force and local_path.exists():
        LOGGER.info("Force re-download: removing %s", local_path)
        shutil.rmtree(local_path)

    LOGGER.info("Downloading %s → %s", repo_id, local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_path),
        token=token,
    )

    LOGGER.info("Download complete: %s", local_path)
    return local_path


# ---------------------------------------------------------------------------
# Prepare (orchestrator)
# ---------------------------------------------------------------------------

def prepare_datasets(
    repo_ids: list[str],
    cache_root: Path | None = None,
    merge_output: Path | None = None,
    force: bool = False,
    token: str | None = None,
) -> Path:
    """Download, validate, and optionally merge HF datasets for training.

    Args:
        repo_ids: One or more HF dataset repo IDs.
        cache_root: Local cache directory for downloads.
        merge_output: Output path for merged dataset (required if len > 1).
        force: Re-download even if cached.
        token: HF auth token.

    Returns:
        Path to the final dataset directory (single or merged).
    """
    local_paths: list[Path] = []

    for repo_id in repo_ids:
        local_path = download_dataset(repo_id, cache_root, force=force, token=token)
        result = validate_dataset(local_path)

        for w in result.warnings:
            LOGGER.warning("[%s] %s", repo_id, w)

        if not result.valid:
            raise DatasetValidationError(repo_id, result)

        local_paths.append(local_path)

    # Single dataset — no merge needed
    if len(local_paths) == 1:
        return local_paths[0]

    # Multiple datasets — merge
    if merge_output is None:
        if cache_root is None:
            cache_root = Path.cwd() / DEFAULT_CACHE_SUBDIR
        names = "_".join(r.split("/")[-1] for r in repo_ids)
        merge_output = Path(cache_root) / f"merged_{names}"

    LOGGER.info("Merging %d datasets → %s", len(local_paths), merge_output)

    # Import merge from existing src/merge_datasets.py
    src_dir = Path(__file__).resolve().parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from merge_datasets import merge

    merge(source_roots=local_paths, output_root=merge_output, force=True)

    # Validate merged output
    merged_result = validate_dataset(merge_output)
    if not merged_result.valid:
        raise DatasetValidationError("merged", merged_result)

    return merge_output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_prepare(args: argparse.Namespace) -> None:
    cache_root = Path(args.cache_root) if args.cache_root else None
    merge_output = Path(args.merge_output) if args.merge_output else None

    path = prepare_datasets(
        repo_ids=args.repo_ids,
        cache_root=cache_root,
        merge_output=merge_output,
        force=args.force,
        token=args.token,
    )
    # Print final path to stdout for shell capture
    print(str(path))


def _cmd_validate(args: argparse.Namespace) -> None:
    root = Path(args.dataset_path)
    result = validate_dataset(root)

    if result.warnings:
        for w in result.warnings:
            LOGGER.warning("%s", w)

    if result.valid:
        LOGGER.info("Dataset is valid: %s", root)
    else:
        for e in result.errors:
            LOGGER.error("%s", e)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, validate, and prepare HuggingFace datasets for training."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- prepare --
    p_prep = sub.add_parser(
        "prepare",
        help="Download, validate, and optionally merge HF datasets.",
    )
    p_prep.add_argument("repo_ids", nargs="+", help="HF dataset repo IDs.")
    p_prep.add_argument(
        "--cache-root", type=str, default=None,
        help="Local cache directory (default: .hf_cache).",
    )
    p_prep.add_argument(
        "--merge-output", type=str, default=None,
        help="Output path for merged dataset (auto-generated if omitted).",
    )
    p_prep.add_argument("--force", action="store_true", help="Re-download even if cached.")
    p_prep.add_argument("--token", type=str, default=None, help="HF auth token.")
    p_prep.set_defaults(func=_cmd_prepare)

    # -- validate --
    p_val = sub.add_parser(
        "validate",
        help="Validate a local dataset is training-ready.",
    )
    p_val.add_argument("dataset_path", help="Path to local dataset directory.")
    p_val.set_defaults(func=_cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
