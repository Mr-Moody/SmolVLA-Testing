"""
create_labels_soup.py — Soup can pick-and-place task label generator.

Writes annotations.jsonl to a cleaned dataset directory.
5 canonical prompts are assigned cyclically by episode index.

Usage:
    python src/labels/soup.py <cleaned_ds_dir> [<cleaned_ds_dir2> ...]

    e.g.
    python src/labels/soup.py cleaned_datasets/100 cleaned_datasets/101 ...

The episode count is inferred from episode_events.jsonl in each directory.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.labels import write_annotations

CANONICAL_PROMPTS = [
    "Pick up the soup can and place it in the green tray.",
    "Place the soup can inside the tray.",
    "Put the red and white can in the green cardboard tray.",
    "Grab the soup can and set it inside the green tray.",
    "Move the soup can into the green tray.",
]


def generate_prompts(num_episodes: int) -> list:
    """Return *num_episodes* task prompts, cycling through CANONICAL_PROMPTS."""
    return [CANONICAL_PROMPTS[i % len(CANONICAL_PROMPTS)] for i in range(num_episodes)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate soup pick-and-place annotations.jsonl for cleaned datasets."
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        type=Path,
        help="Paths to cleaned dataset directories (each must contain episode_events.jsonl).",
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
