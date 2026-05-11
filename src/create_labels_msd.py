"""
create_labels_msd.py — MSD connector insertion task label generator.

Writes annotations.jsonl to a cleaned dataset directory.
One unique task description per episode, seeded for reproducibility.

Usage:
    python src/create_labels_msd.py <cleaned_ds_dir> [<cleaned_ds_dir2> ...]

    e.g.
    python src/create_labels_msd.py cleaned_datasets/200 cleaned_datasets/201 ...

The episode count is inferred from episode_events.jsonl in each directory.
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabulary — MSD plug / connector insertion task
# ---------------------------------------------------------------------------

_VERBS = [
    "Insert",
    "Plug",
    "Connect",
    "Push",
    "Seat",
    "Mate",
    "Attach",
    "Drive",
]

_OBJECTS = [
    "the MSD connector",
    "the orange plug",
    "the orange connector",
    "the connector plug",
    "the plug",
    "the MSD plug",
    "the orange MSD connector",
]

_ACTIONS = [
    "into",
    "inside",
    "into the opening of",
    "fully into",
    "securely into",
]

_TARGETS = [
    "the socket",
    "the housing",
    "the connector slot",
    "the blue housing",
    "the slot",
    "the receptacle",
    "the port",
    "the blue socket",
    "the connector housing",
    "the mating socket",
]


def _all_prompts() -> list[str]:
    return [
        f"{v} {o} {a} {t}."
        for v in _VERBS
        for o in _OBJECTS
        for a in _ACTIONS
        for t in _TARGETS
    ]


def generate_prompts(num_episodes: int, seed: int = 42) -> list[str]:
    """Return *num_episodes* unique, shuffled task prompt strings."""
    pool = _all_prompts()
    if num_episodes > len(pool):
        raise ValueError(
            f"Requested {num_episodes} prompts but only {len(pool)} unique "
            "combinations exist. Add more synonym entries to create_labels_msd.py."
        )
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:num_episodes]


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
        help="RNG seed for prompt shuffling (default: 42).",
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
