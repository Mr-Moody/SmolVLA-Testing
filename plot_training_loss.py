#!/usr/bin/env python3
"""Plot SmolVLA training loss from a launcher log and checkpoint directory.

The script is designed to work after a run finishes or from a rescued copy of
the run directory. It scans the training log for step/loss pairs, then plots a
loss curve and optionally marks checkpoint steps discovered under the checkpoint
directory.

Examples:
    python plot_training_loss.py \
        --log-file /scratch0/eredhead/smolvla_outputs/train_100_101_102_103.log \
        --checkpoint-dir /scratch0/eredhead/smolvla_outputs/train_100_101_102_103_smolvla/checkpoints \
        --output /scratch0/eredhead/smolvla_outputs/train_100_101_102_103_loss.png

    python plot_training_loss.py --dataset-tag train_100_101_102_103
"""

from __future__ import annotations

import argparse
import getpass
import re
from pathlib import Path


LOSS_LINE_RE = re.compile(r"step:(?P<step>[0-9]+)(?P<step_suffix>[KkMm]?)\s+.*?loss:(?P<loss>[0-9]*\.?[0-9]+)")
STEP_DIR_RE = re.compile(r"(?P<step>[0-9]{3,})")


def parse_step_value(step_text: str, suffix: str) -> int:
    step = int(step_text)
    suffix = suffix.lower()
    if suffix == "k":
        return step * 1000
    if suffix == "m":
        return step * 1_000_000
    return step


def load_loss_points(log_file: Path) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LOSS_LINE_RE.search(line)
        if not match:
            continue
        step = parse_step_value(match.group("step"), match.group("step_suffix"))
        loss = float(match.group("loss"))
        points.append((step, loss))
    return points


def discover_checkpoint_steps(checkpoint_dir: Path | None) -> list[int]:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return []

    steps: set[int] = set()
    for path in checkpoint_dir.iterdir():
        if not path.is_dir():
            continue
        match = STEP_DIR_RE.search(path.name)
        if match:
            steps.add(int(match.group("step")))
    return sorted(steps)


def default_log_file(dataset_tag: str) -> Path:
    return Path(f"/scratch0/{getpass.getuser()}/smolvla_outputs/{dataset_tag}.log")


def default_checkpoint_dir(dataset_tag: str) -> Path:
    return Path(f"/scratch0/{getpass.getuser()}/smolvla_outputs/{dataset_tag}_smolvla/checkpoints")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot SmolVLA training loss from a log file.")
    parser.add_argument("--log-file", type=Path, default=None, help="Training log to parse.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory to annotate on the plot.",
    )
    parser.add_argument(
        "--dataset-tag",
        type=str,
        default=None,
        help="Run tag used to infer default log and checkpoint paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <log stem>_loss.png.",
    )
    parser.add_argument(
        "--show-checkpoints",
        action="store_true",
        help="Draw checkpoint markers when a checkpoint directory is available.",
    )
    return parser


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local Python env
        raise SystemExit(
            "matplotlib is required for plotting. Install it in the active Python environment and retry."
        ) from exc

    parser = build_parser()
    args = parser.parse_args()

    if args.log_file is None and args.dataset_tag is None:
        parser.error("Provide either --log-file or --dataset-tag.")

    log_file = args.log_file
    checkpoint_dir = args.checkpoint_dir

    if args.dataset_tag is not None:
        if log_file is None:
            log_file = default_log_file(args.dataset_tag)
        if checkpoint_dir is None:
            checkpoint_dir = default_checkpoint_dir(args.dataset_tag)

    assert log_file is not None

    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    loss_points = load_loss_points(log_file)
    if not loss_points:
        raise ValueError(f"No loss lines found in {log_file}")

    steps = [step for step, _ in loss_points]
    losses = [loss for _, loss in loss_points]
    checkpoint_steps = discover_checkpoint_steps(checkpoint_dir) if args.show_checkpoints else []

    if args.output is not None:
        output_path = args.output
    else:
        output_path = log_file.with_name(f"{log_file.stem}_loss.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(steps, losses, color="#0f766e", linewidth=1.8, label="loss")
    ax.scatter(steps, losses, color="#0f766e", s=12, alpha=0.35)

    for checkpoint_step in checkpoint_steps:
        ax.axvline(checkpoint_step, color="#dc2626", linestyle="--", alpha=0.18, linewidth=1)

    ax.set_title("SmolVLA training loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)

    print(f"Saved loss plot to {output_path}")
    print(f"Parsed {len(loss_points)} loss points from {log_file}")
    if checkpoint_steps:
        print(f"Annotated {len(checkpoint_steps)} checkpoint steps from {checkpoint_dir}")


if __name__ == "__main__":
    main()