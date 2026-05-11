"""Write per-episode task prompts into a cleaned dataset's annotations.jsonl.

Usage:
    uv --project ../lerobot run python src/scripts/generate_annotations.py --data-name 200
    uv --project ../lerobot run python src/scripts/generate_annotations.py --data-name 200 --dataset-root /path/to/cleaned_datasets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.create_labels import generate_prompts
from src.dataset_utils import read_jsonl, save_annotation

_DEFAULT_CLEANED_ROOT = Path(__file__).resolve().parents[2] / "cleaned_datasets"


def count_episodes(dataset_dir: Path) -> int:
    events_path = dataset_dir / "episode_events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"episode_events.jsonl not found in {dataset_dir}")
    rows = read_jsonl(events_path)
    return sum(1 for r in rows if r.get("event") == "episode_start")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and write task prompts into a cleaned dataset.")
    parser.add_argument("--data-name", required=True, help="Dataset folder name under cleaned_datasets/")
    parser.add_argument(
        "--dataset-root",
        default=str(_DEFAULT_CLEANED_ROOT),
        help=f"Root directory containing cleaned datasets (default: {_DEFAULT_CLEANED_ROOT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotations.jsonl if present",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_root) / args.data_name
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    annotations_path = dataset_dir / "annotations.jsonl"
    if annotations_path.exists() and not args.overwrite:
        print(
            f"ERROR: {annotations_path} already exists. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        sys.exit(1)

    num_episodes = count_episodes(dataset_dir)
    if num_episodes == 0:
        print("ERROR: No episodes found in episode_events.jsonl.", file=sys.stderr)
        sys.exit(1)

    prompts = generate_prompts(num_episodes)

    for episode_index, task in enumerate(prompts):
        save_annotation(dataset_dir, episode_index, task)

    print(f"Wrote {num_episodes} prompts to {annotations_path}")
    print("\nExamples:")
    for i, p in enumerate(prompts[:5]):
        print(f"  [{i}] {p}")


if __name__ == "__main__":
    main()
