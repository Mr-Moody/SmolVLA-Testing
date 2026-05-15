"""Batch annotation driver — annotates a full LeRobot dataset with Qwen3-VL.

Usage:
    python src/scripts/annotate_dataset.py \\
        --dataset-root lerobot_datasets/my_dataset \\
        --task pick_place \\
        --output-dir outputs/annotations
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import typer
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()
app = typer.Typer(help="Batch-annotate a LeRobot dataset with Qwen3-VL phase labels.")


def _load_episode_meta(dataset_root: Path) -> list:
    """Return a list of episode metadata dicts."""
    meta_file = dataset_root / "meta" / "info.json"
    if not meta_file.exists():
        # Try to find episode count from video files
        videos = sorted((dataset_root / "videos").glob("episode_*.mp4")) if (dataset_root / "videos").exists() else []
        return [{"episode_index": i, "video_path": v} for i, v in enumerate(videos)]
    meta = json.loads(meta_file.read_text())
    n_episodes = meta.get("total_episodes", 0)
    return [{"episode_index": i} for i in range(n_episodes)]


def _get_episode_video(dataset_root: Path, episode_idx: int) -> Optional[Path]:
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        return None
    candidates = sorted(videos_dir.glob(f"episode_{episode_idx:03d}*.mp4"))
    return candidates[0] if candidates else None


def _get_episode_duration(video_path: Path) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return float(total / fps)
    except Exception:
        return 10.0  # fallback


def _load_episode_arrays(dataset_root: Path, episode_idx: int) -> dict:
    """Load numpy arrays from episode parquet if available."""
    import numpy as np
    chunk_file = dataset_root / "chunks" / f"episode_{episode_idx:06d}.parquet"
    if not chunk_file.exists():
        # Try alternative naming
        chunk_file = dataset_root / "chunks" / f"episode_{episode_idx:03d}.parquet"
    if not chunk_file.exists():
        return {}
    try:
        df = pd.read_parquet(chunk_file)
        ep = {}
        # Map common column names
        for ts_col in ["timestamp", "timestamps", "t"]:
            if ts_col in df.columns:
                ep["timestamps"] = df[ts_col].to_numpy()
                break
        for g_col in ["gripper_state", "observation.state"]:
            if g_col in df.columns:
                arr = df[g_col].to_numpy()
                # If multi-dim, take last dim as gripper
                if arr.ndim > 1:
                    arr = arr[:, -1]
                ep["gripper_state"] = arr
                break
        return ep
    except Exception:
        return {}


def _dense_phase_ids(segments, n_frames: int, fps: float, control_rate: float) -> list:
    """Convert segment boundaries to per-frame phase IDs."""
    from src.common.phases import Phase
    import numpy as np
    timestamps = np.arange(n_frames) / fps
    ids = []
    seg_iter = iter(segments)
    seg = next(seg_iter, None)
    next_seg = next(seg_iter, None)
    for t in timestamps:
        # Advance segment pointer
        while next_seg is not None and t >= next_seg.start_t:
            seg = next_seg
            next_seg = next(seg_iter, None)
        ids.append(seg.phase_id if seg else 0)
    return ids


@app.command()
def main(
    dataset_root: Path = typer.Option(..., help="Path to lerobot_datasets/<name>/"),
    task: str = typer.Option(..., help="pick_place or msd_plug"),
    output_dir: Path = typer.Option(Path("outputs/annotations"), help="Output directory"),
    batch_size: int = typer.Option(4, help="Episodes per vLLM batch"),
    max_retries: int = typer.Option(2, help="Retries on validation failure"),
):
    from src.annotation.serve_qwen import QwenAnnotator
    from src.annotation.sampler import KeyframeSampler
    from src.annotation.prompt import PromptBuilder
    from src.annotation.schema import parse_qwen_output
    from src.annotation.validator import Validator

    output_dir.mkdir(parents=True, exist_ok=True)
    failed_file = Path("outputs/failed_episodes.txt")
    qwen_log = Path("outputs/qwen_log.jsonl")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    episodes = _load_episode_meta(dataset_root)
    console.print(f"Found {len(episodes)} episode(s) in {dataset_root}")

    annotator = QwenAnnotator()
    sampler = KeyframeSampler()
    prompt_builder = PromptBuilder(task)
    validator = Validator()

    failed_ids = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_bar = progress.add_task("Annotating", total=len(episodes))

        for ep_meta in episodes:
            ep_idx = ep_meta["episode_index"]
            out_file = output_dir / f"episode_{ep_idx:06d}.parquet"

            if out_file.exists():
                progress.advance(task_bar)
                continue

            video_path = _get_episode_video(dataset_root, ep_idx)
            if video_path is None:
                console.print(f"[yellow]No video for episode {ep_idx} — skipping.[/yellow]")
                progress.advance(task_bar)
                continue

            duration = _get_episode_duration(video_path)
            episode_arrays = _load_episode_arrays(dataset_root, ep_idx)

            # Keyframe sampling
            keyframes = sampler.sample(episode_arrays) if episode_arrays else []

            # Build prompt
            user_msg = prompt_builder.build_user_message(str(ep_idx), duration)
            system_msg = prompt_builder.build_system_message()
            combined_prompt = system_msg + "\n\n---\n\n" + user_msg

            annotation = None
            for attempt in range(max_retries + 1):
                try:
                    raw = annotator.annotate_episode(str(video_path), combined_prompt)

                    # Log Qwen call
                    with open(qwen_log, "a") as f:
                        f.write(json.dumps({
                            "episode_id": ep_idx, "attempt": attempt, "output": raw[:500]
                        }) + "\n")

                    parsed = parse_qwen_output(raw, duration)
                    result = validator.validate(parsed, episode=episode_arrays, episode_duration=duration)

                    if result.passed:
                        annotation = parsed
                        break
                    else:
                        console.print(
                            f"[yellow]Episode {ep_idx} attempt {attempt}: "
                            f"{len(result.errors())} error(s)[/yellow]"
                        )
                except Exception as exc:
                    console.print(f"[red]Episode {ep_idx} attempt {attempt}: {exc}[/red]")

            if annotation is None:
                console.print(f"[red]Episode {ep_idx} failed after {max_retries+1} attempts.[/red]")
                failed_ids.append(ep_idx)
                progress.advance(task_bar)
                continue

            # Dense phase IDs at control rate
            try:
                import cv2
                import numpy as np
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception:
                fps = 30.0
                n_frames = max(1, int(duration * fps))

            phase_ids = _dense_phase_ids(annotation.segments, n_frames, fps, fps)
            from src.common.phases import Phase, PHASE_NAMES
            subtask_labels = [
                f"{task}.{PHASE_NAMES[Phase(pid)]}" for pid in phase_ids
            ]

            df = pd.DataFrame({
                "frame_idx": list(range(n_frames)),
                "phase": phase_ids,
                "subtask": subtask_labels,
            })
            df.to_parquet(out_file, index=False)
            console.print(f"[green]Episode {ep_idx} annotated → {out_file}[/green]")
            progress.advance(task_bar)

    if failed_ids:
        with open(failed_file, "a") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")
        console.print(f"[red]{len(failed_ids)} failed episodes written to {failed_file}[/red]")

    console.print("[bold green]Annotation run complete.[/bold green]")


if __name__ == "__main__":
    app()
