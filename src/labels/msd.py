"""
create_labels_msd.py — MSD connector insertion task label generator.

Writes annotations.jsonl to a cleaned dataset directory.
15 canonical prompts are assigned cyclically by episode index.

Usage:
    python src/labels/msd.py <cleaned_ds_dir> [<cleaned_ds_dir2> ...]

    e.g.
    python src/labels/msd.py cleaned_datasets/200 cleaned_datasets/201 ...

The episode count is inferred from episode_events.jsonl in each directory.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.labels import write_annotations

CANONICAL_PROMPTS = [
    "Insert the orange MSD connector inside the mating socket.",
    "Put the orange connector fully into the socket.",
    "Push the MSD connector fully into the connector housing.",
    "Seat the connector plug into the opening of the socket.",
    "Drive the plug inside the mating socket.",
    "Connect the orange connector fully into the connector housing.",
    "Insert the orange plug into the socket.",
    "Plug the connector plug securely into the housing.",
    "Seat the MSD plug into the socket.",
    "Push the orange MSD connector inside the mating socket.",
    "Attach the MSD connector into the connector housing.",
    "Insert the MSD connector inside the socket.",
    "Put the orange MSD connector into the housing.",
    "Connect the connector plug inside the housing.",
    "Push the MSD plug fully into the connector slot.",
]


def generate_prompts(num_episodes: int, seed: int = 42) -> list:
    """Return *num_episodes* task prompts, cycling through CANONICAL_PROMPTS."""
    return [CANONICAL_PROMPTS[i % len(CANONICAL_PROMPTS)] for i in range(num_episodes)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MSD insertion annotations.jsonl for cleaned datasets."
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        type=Path,
        help="Paths to cleaned dataset directories (each must contain episode_events.jsonl).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Ignored (retained for CLI compatibility).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing annotations.jsonl.",
    )
    args = parser.parse_args()

    total = 0
    for ds_dir in args.dataset_dirs:
        total += write_annotations(ds_dir, CANONICAL_PROMPTS, force=args.force)

    print(f"\nDone. Total episodes annotated: {total}")


if __name__ == "__main__":
    main()
