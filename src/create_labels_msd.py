"""
create_labels_msd.py — MSD connector insertion task label generator.

Writes annotations.jsonl to a cleaned dataset directory.
15 canonical prompts are assigned cyclically by episode index.

Usage:
    python src/create_labels_msd.py <cleaned_ds_dir> [<cleaned_ds_dir2> ...]

    e.g.
    python src/create_labels_msd.py cleaned_datasets/200 cleaned_datasets/201 ...

The episode count is inferred from episode_events.jsonl in each directory.
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# 15 canonical prompts — assigned cyclically (episode_index % 15)
# ---------------------------------------------------------------------------

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


def generate_prompts(num_episodes: int, seed: int = 42) -> list[str]:
    """Return *num_episodes* task prompts, cycling through CANONICAL_PROMPTS."""
    return [CANONICAL_PROMPTS[i % len(CANONICAL_PROMPTS)] for i in range(num_episodes)]


def _count_episodes(ds_dir: Path) -> int:
    """Count kept episodes from episode_events.jsonl in a cleaned dataset dir."""
    events_path = ds_dir / "episode_events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"episode_events.jsonl not found in {ds_dir}")
    with events_path.open() as f:
        lines = [l.strip() for l in f if l.strip()]
    # Cleaned episode_events has one line per kept episode (all are episode_start).
    # Count lines as episode count.
    count = len(lines)
    if count == 0:
        raise ValueError(f"No episodes found in {events_path}")
    return count


def write_annotations(ds_dir: Path, seed: int = 42, force: bool = False) -> int:
    """Generate and write annotations.jsonl. Returns number of episodes written."""
    ds_dir = Path(ds_dir)
    out_path = ds_dir / "annotations.jsonl"

    if out_path.exists() and not force:
        print(f"  [{ds_dir.name}] annotations.jsonl already exists — skipping (use --force to overwrite).")
        return 0

    n = _count_episodes(ds_dir)
    prompts = generate_prompts(n, seed=seed)

    with out_path.open("w") as f:
        for i, prompt in enumerate(prompts):
            f.write(json.dumps({"episode_index": i, "task": prompt}) + "\n")

    print(f"  [{ds_dir.name}] wrote {n} annotations → {out_path}")
    print(f"    sample: {prompts[0]}")
    if n > 1:
        print(f"           {prompts[1]}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MSD insertion annotations.jsonl for cleaned datasets.")
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
        total += write_annotations(ds_dir, seed=args.seed, force=args.force)

    print(f"\nDone. Total episodes annotated: {total}")


if __name__ == "__main__":
    main()
