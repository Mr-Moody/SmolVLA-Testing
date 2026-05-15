"""Shared helpers for task prompt generators."""
import json
from pathlib import Path


def count_episodes(ds_dir: Path) -> int:
    events_path = Path(ds_dir) / "episode_events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"episode_events.jsonl not found in {ds_dir}")
    lines = [l.strip() for l in events_path.open() if l.strip()]
    if not lines:
        raise ValueError(f"No episodes found in {events_path}")
    return len(lines)


def write_annotations(ds_dir: Path, prompts: list, force: bool = False) -> int:
    """Generate and write annotations.jsonl. Returns number of episodes written."""
    ds_dir = Path(ds_dir)
    out_path = ds_dir / "annotations.jsonl"
    if out_path.exists() and not force:
        print(f"  [{ds_dir.name}] annotations.jsonl already exists — skipping (use --force to overwrite).")
        return 0
    n = count_episodes(ds_dir)
    task_prompts = [prompts[i % len(prompts)] for i in range(n)]
    with out_path.open("w") as f:
        for i, p in enumerate(task_prompts):
            f.write(json.dumps({"episode_index": i, "task": p}) + "\n")
    print(f"  [{ds_dir.name}] wrote {n} annotations → {out_path}")
    if n > 0:
        print(f"    sample: {task_prompts[0]}")
    if n > 1:
        print(f"           {task_prompts[1]}")
    return n
