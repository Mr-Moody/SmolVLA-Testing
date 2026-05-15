"""Interactive CLI for building gold-standard phase annotations.

Usage:
    # Annotate a new episode
    python src/scripts/build_gold_set.py \\
        --dataset-root lerobot_datasets/my_dataset \\
        --episode-id 0 \\
        --task pick_place

    # Review / edit an existing annotation
    python src/scripts/build_gold_set.py \\
        --dataset-root lerobot_datasets/my_dataset \\
        --episode-id 0 --task pick_place --review
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import typer
from rich.console import Console
from rich.prompt import FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from src.common.phases import Phase, PHASE_NAMES
from src.annotation.schema import EpisodeAnnotation, PhaseSegment
from src.annotation.validator import Validator

app = typer.Typer(help="Build gold-standard phase annotations interactively.")
console = Console()

GOLD_DIR = Path("data/gold")


def _save_frames(video_path: Path, episode_id: int, n_frames: int = 8):
    """Save uniformly spaced thumbnail frames to /tmp."""
    try:
        import cv2
    except ImportError:
        console.print("[yellow]opencv-python not installed — skipping frame thumbnails.[/yellow]")
        return []

    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_dir = Path(f"/tmp/gold_frames/{episode_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        t = idx / fps
        p = out_dir / f"frame_{idx:05d}_t{t:.2f}s.jpg"
        cv2.imwrite(str(p), frame)
        paths.append((t, str(p)))
    cap.release()
    return paths


def _find_video(dataset_root: Path, episode_id: int) -> Path:
    """Locate the first video file for the given episode index."""
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"No videos/ dir in {dataset_root}")
    # Match episode_XXX_*.mp4 or any *.mp4 with the right index
    candidates = sorted(videos_dir.glob(f"episode_{episode_id:03d}*.mp4"))
    if not candidates:
        candidates = sorted(videos_dir.glob("*.mp4"))
    if not candidates:
        raise FileNotFoundError(f"No video found for episode {episode_id} in {videos_dir}")
    return candidates[0]


def _get_episode_duration(video_path: Path) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return float(total / fps)
    except Exception:
        return float(Prompt.ask("Enter episode duration in seconds"))


def _annotate_interactively(episode_id: int, task: str, duration: float) -> EpisodeAnnotation:
    """Prompt user to enter phase segments."""
    console.print(f"\n[bold]Annotating episode {episode_id} ({task}), duration={duration:.2f}s[/bold]")
    console.print("[dim]Phase IDs:[/dim]")
    for ph in Phase:
        console.print(f"  {int(ph)} → {PHASE_NAMES[ph]}")
    console.print()

    segments = []
    t = 0.0
    while t < duration - 0.01:
        console.print(f"[cyan]Current time: {t:.3f}s[/cyan]")
        phase_id = IntPrompt.ask("  Phase ID (0–4)", default=segments[-1].phase_id if segments else 0)
        end_t = FloatPrompt.ask(f"  End time for this segment (> {t:.3f})", default=round(duration, 3))
        evidence = Prompt.ask("  Evidence (one sentence)")
        confidence = FloatPrompt.ask("  Confidence (0–1)", default=0.9)

        try:
            seg = PhaseSegment(
                phase_id=phase_id,
                phase_name=PHASE_NAMES[Phase(phase_id)],
                start_t=round(t, 4),
                end_t=round(end_t, 4),
                confidence=confidence,
                evidence=evidence,
            )
        except Exception as exc:
            console.print(f"[red]Invalid segment: {exc}. Try again.[/red]")
            continue

        segments.append(seg)
        t = end_t

    # Fix last segment end_t to match duration exactly
    if segments:
        last = segments[-1]
        segments[-1] = PhaseSegment(
            phase_id=last.phase_id,
            phase_name=last.phase_name,
            start_t=last.start_t,
            end_t=round(duration, 4),
            confidence=last.confidence,
            evidence=last.evidence,
        )

    overall_conf = FloatPrompt.ask("Overall confidence (0–1)", default=0.9)
    notes = Prompt.ask("Notes (optional)", default="")

    return EpisodeAnnotation(
        episode_id=str(episode_id),
        task=task,
        segments=segments,
        overall_confidence=overall_conf,
        notes=notes or None,
    )


def _print_annotation_table(ann: EpisodeAnnotation):
    table = Table(title="Annotation Summary")
    table.add_column("Phase")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Conf")
    table.add_column("Evidence")
    for seg in ann.segments:
        table.add_row(
            seg.phase_name,
            f"{seg.start_t:.3f}s",
            f"{seg.end_t:.3f}s",
            f"{seg.confidence:.2f}",
            seg.evidence[:50],
        )
    console.print(table)


@app.command()
def main(
    dataset_root: Path = typer.Option(..., help="Path to lerobot_datasets/<name>/"),
    episode_id: int = typer.Option(..., help="Episode index to annotate"),
    task: str = typer.Option(..., help="pick_place or msd_plug"),
    review: bool = typer.Option(False, help="Re-open and edit an existing annotation"),
):
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / f"{episode_id}.json"

    # Find video + duration
    try:
        video_path = _find_video(dataset_root, episode_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    duration = _get_episode_duration(video_path)

    # Save thumbnails
    frames = _save_frames(video_path, episode_id)
    if frames:
        console.print(f"\n[bold]Frame thumbnails saved to /tmp/gold_frames/{episode_id}/[/bold]")
        for t, p in frames:
            console.print(f"  t={t:.2f}s  {p}")
        console.print()

    # Review mode: load existing annotation
    if review and out_path.exists():
        existing = EpisodeAnnotation.model_validate(json.loads(out_path.read_text()))
        _print_annotation_table(existing)
        if not typer.confirm("Edit this annotation?"):
            raise typer.Exit(0)

    ann = _annotate_interactively(episode_id, task, duration)

    # Validate before saving
    validator = Validator()
    result = validator.validate(ann, episode_duration=duration)
    if result.issues:
        console.print("\n[yellow]Validation issues:[/yellow]")
        for issue in result.issues:
            color = "red" if issue.severity == "error" else "yellow"
            console.print(f"  [{color}]{issue.severity.upper()}[/{color}] [{issue.rule}] {issue.message}")

    if not result.passed:
        if not typer.confirm("\nAnnotation has errors. Save anyway?"):
            console.print("[red]Annotation discarded.[/red]")
            raise typer.Exit(1)

    out_path.write_text(ann.model_dump_json(indent=2))
    console.print(f"\n[green]Annotation saved to {out_path}[/green]")
    _print_annotation_table(ann)


if __name__ == "__main__":
    app()
