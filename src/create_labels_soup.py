"""
create_labels_soup.py — Soup can pick-and-place task label generator.

Writes annotations.jsonl to a cleaned dataset directory.
5 canonical prompts are assigned cyclically by episode index.

Usage:
    python src/create_labels_soup.py <cleaned_ds_dir> [<cleaned_ds_dir2> ...]

    e.g.
    python src/create_labels_soup.py cleaned_datasets/100 cleaned_datasets/101 ...

The episode count is inferred from episode_events.jsonl in each directory.
"""

import argparse
import json
from pathlib import Path

CANONICAL_PROMPTS = [
    "Pick up the soup can and place it in the green tray.",
    "Place the soup can inside the tray.",
    "Put the red and white can in the green cardboard tray.",
    "Grab the soup can and set it inside the green tray.",
    "Move the soup can into the green tray.",
]


def generate_prompts(num_episodes: int) -> list[str]:
    """Return *num_episodes* task prompts, cycling through CANONICAL_PROMPTS."""
    return [CANONICAL_PROMPTS[i % len(CANONICAL_PROMPTS)] for i in range(num_episodes)]


def _count_episodes(ds_dir: Path) -> int:
    """Count kept episodes from episode_events.jsonl in a cleaned dataset dir."""
    events_path = ds_dir / "episode_events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"episode_events.jsonl not found in {ds_dir}")
    with events_path.open() as f:
        lines = [l.strip() for l in f if l.strip()]
    count = len(lines)
    if count == 0:
        raise ValueError(f"No episodes found in {events_path}")
    return count


def write_annotations(ds_dir: Path, force: bool = False) -> int:
    """Generate and write annotations.jsonl. Returns number of episodes written."""
    ds_dir = Path(ds_dir)
    out_path = ds_dir / "annotations.jsonl"

    if out_path.exists() and not force:
        print(f"  [{ds_dir.name}] annotations.jsonl already exists — skipping (use --force to overwrite).")
        return 0

    n = _count_episodes(ds_dir)
    prompts = generate_prompts(n)

    with out_path.open("w") as f:
        for i, prompt in enumerate(prompts):
            f.write(json.dumps({"episode_index": i, "task": prompt}) + "\n")

    print(f"  [{ds_dir.name}] wrote {n} annotations → {out_path}")
    print(f"    sample: {prompts[0]}")
    if n > 1:
        print(f"           {prompts[1]}")
    return n


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
        total += write_annotations(ds_dir, force=args.force)

    print(f"\nDone. Total episodes annotated: {total}")


if __name__ == "__main__":
    main()
