"""Annotate a cleaned_dataset (raw recording format) with phase + subtask labels.

Works directly on the ``cleaned_datasets/<name>/`` directory produced by
``main.py clean``.  Does NOT require the LeRobot v3 conversion step.

Annotation strategy (in priority order):
  1. Qwen3-VL (requires vLLM + GPU) — used when ``--use-qwen`` is set and vLLM is installed
  2. Runtime FSM — deterministic, proprioception-only; always available

Output (``cleaned_datasets/<output_name>/``):
  - Copies all original files unchanged
  - Adds ``annotations/<episode_NNN>.json`` per episode (EpisodeAnnotation JSON)
  - Adds ``annotations_summary.json`` with aggregate stats

Usage:
    # FSM annotation (no GPU needed):
    python src/scripts/annotate_cleaned_dataset.py \\
        --dataset-dir cleaned_datasets/102 \\
        --output-name 102_qwen_tagged \\
        --task pick_place

    # Qwen annotation (requires vLLM + GPU):
    python src/scripts/annotate_cleaned_dataset.py \\
        --dataset-dir cleaned_datasets/102 \\
        --output-name 102_qwen_tagged \\
        --task pick_place \\
        --use-qwen
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.common.phases import Phase, PHASE_NAMES
from src.annotation.schema import EpisodeAnnotation, PhaseSegment
from src.annotation.validator import Validator

console = Console()
app = typer.Typer(help="Annotate a cleaned_dataset with phase + subtask labels.")

_GRIPPER_CLOSE_WIDTH_M = 0.035   # below this → gripper considered closed
_MIN_EPISODE_DURATION_S = 1.0    # discard very short episodes


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_episode_boundaries(dataset_dir: Path) -> list[dict]:
    """Return list of {idx, start_ns, end_ns, duration_s, teleop_state}."""
    events = [
        json.loads(l)
        for l in (dataset_dir / "episode_events.jsonl").read_text().splitlines()
        if l.strip()
    ]
    pairs = []
    ep_idx = 0
    pending_start = None
    for ev in events:
        if ev["event"] == "episode_start":
            pending_start = ev
        elif ev["event"] == "episode_end" and pending_start is not None:
            dur_s = (ev["receive_host_time_ns"] - pending_start["receive_host_time_ns"]) / 1e9
            if dur_s >= _MIN_EPISODE_DURATION_S:
                pairs.append({
                    "idx": ep_idx,
                    "start_ns": pending_start["robot_timestamp_ns"],
                    "end_ns": ev["robot_timestamp_ns"],
                    "start_host_ns": pending_start["receive_host_time_ns"],
                    "end_host_ns": ev["receive_host_time_ns"],
                    "duration_s": dur_s,
                    "teleop_state": ev.get("teleop_state", "UNKNOWN"),
                })
                ep_idx += 1
            pending_start = None
    return pairs


def _load_robot_for_episode(robot_lines: list[str], ep: dict) -> dict:
    """Extract robot.jsonl rows that fall within this episode's timestamp range."""
    rows = []
    for line in robot_lines:
        if not line.strip():
            continue
        r = json.loads(line)
        ts = r.get("timestamp_ns", 0)
        if ep["start_ns"] <= ts <= ep["end_ns"]:
            rows.append(r)
    if not rows:
        return {}

    timestamps = np.array([r["timestamp_ns"] for r in rows], dtype=float)
    t0 = timestamps[0]
    timestamps_s = (timestamps - t0) / 1e9

    tcp_pos = np.array([r["robot_state"]["tcp_position_xyz"] for r in rows])
    tcp_ori = np.array([r["robot_state"]["tcp_orientation_xyzw"] for r in rows])
    tcp_pose = np.concatenate([tcp_pos, tcp_ori], axis=1)   # (N, 7)

    gripper_width = np.array([r["robot_state"]["gripper_width"] for r in rows])
    gripper_state_str = [r["robot_state"]["gripper_state"] for r in rows]

    # Compute TCP velocity via finite differences
    dt = np.diff(timestamps_s, prepend=timestamps_s[0])
    dt = np.where(dt < 1e-6, 1e-6, dt)
    tcp_vel_xyz = np.gradient(tcp_pos, axis=0) / dt[:, None]
    tcp_vel = np.concatenate([tcp_vel_xyz, np.zeros_like(tcp_vel_xyz)], axis=1)  # (N, 6)

    # Executed actions
    actions = np.array([
        r["executed_action"]["cartesian_delta_translation"]
        + r["executed_action"]["cartesian_delta_rotation"]
        + [r["executed_action"]["gripper_command"]]
        for r in rows
    ])  # (N, 7)

    return {
        "timestamps": timestamps_s,
        "tcp_pose": tcp_pose,
        "tcp_velocity": tcp_vel,
        "gripper_width": gripper_width,
        "gripper_state_str": gripper_state_str,
        "gripper_state": (gripper_width < _GRIPPER_CLOSE_WIDTH_M).astype(float),
        "wrench": np.zeros((len(rows), 6)),  # no F/T sensor in this dataset
        "actions": actions,
        "n_frames": len(rows),
    }


def _extract_episode_video_clip(
    video_path: Path,
    frames_jsonl: Path,
    ep_start_host_ns: int,
    ep_end_host_ns: int,
    out_path: Path,
) -> Optional[Path]:
    """Trim the continuous camera video to the episode time window."""
    try:
        import cv2
    except ImportError:
        return None

    frame_meta = [
        json.loads(l)
        for l in frames_jsonl.read_text().splitlines()
        if l.strip()
    ]

    # Find frame indices within episode window
    ep_frames = [
        fm for fm in frame_meta
        if ep_start_host_ns <= fm["host_timestamp_ns"] <= ep_end_host_ns
    ]
    if not ep_frames:
        return None

    start_frame = ep_frames[0]["rgb_video_frame"]
    end_frame = ep_frames[-1]["rgb_video_frame"]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return out_path


# ── Heuristic annotation (gripper + velocity + height) ───────────────────────
#
# The RuntimeFSM requires target_pose to compute xy-distance (free→fine_align)
# and F/T wrench data for contact detection.  Neither is available in raw
# cleaned_datasets.  This heuristic annotator uses only the signals present:
#
#   • gripper_state_str transitions (OPEN→CLOSE, CLOSE→OPEN)
#   • TCP z-height (table clearance)
#   • TCP speed (approach = fast, alignment = slow)
#   • gripper_command in executed_action (motor intent signal)
#
# Phase assignment rules for pick-and-place:
#   FREE_MOTION       : before robot slows + descends for pre-grasp
#   FINE_ALIGN        : low speed + low z before gripper closes
#   CONTACT_ESTABLISH : gripper_command rising toward close; state transition
#   CONSTRAINED_MOTION: gripper CLOSED and robot moving (lift/transport)
#   VERIFY_RELEASE    : gripper opens + retract motion after placement

def _find_gripper_transitions(gripper_state_str: list, timestamps: np.ndarray) -> list[tuple]:
    """Return list of (t, from_state, to_state) for each gripper state change."""
    transitions = []
    for i in range(1, len(gripper_state_str)):
        if gripper_state_str[i] != gripper_state_str[i - 1]:
            transitions.append((float(timestamps[i]), gripper_state_str[i - 1], gripper_state_str[i]))
    return transitions


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple box-car smoothing."""
    if len(arr) <= window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _heuristic_annotate_episode(robot_data: dict, task: str, ep_duration: float) -> EpisodeAnnotation:
    """Segment episode into phases using gripper events + velocity + height."""
    timestamps = robot_data["timestamps"]
    tcp_pose = robot_data["tcp_pose"]       # (N, 7)
    tcp_vel = robot_data["tcp_velocity"]    # (N, 6)
    gripper_state_str = robot_data["gripper_state_str"]
    actions = robot_data["actions"]         # (N, 7)  last col = gripper_cmd
    n = robot_data["n_frames"]

    t_start = float(timestamps[0])
    t_end = float(timestamps[-1])

    tcp_z = tcp_pose[:, 2]
    speed = np.linalg.norm(tcp_vel[:, :3], axis=1)
    speed_sm = _smooth(speed, window=15)

    # Gripper command (last column of actions)
    gripper_cmd = actions[:, -1] if actions.shape[1] >= 7 else np.zeros(n)
    gripper_cmd_sm = _smooth(gripper_cmd, window=15)

    # Gripper state transitions
    g_trans = _find_gripper_transitions(gripper_state_str, timestamps)
    close_times = [t for t, frm, to in g_trans if frm == "OPEN" and to == "CLOSE"]
    open_times  = [t for t, frm, to in g_trans if frm == "CLOSE" and to == "OPEN"]

    segments = []

    if not close_times:
        # No gripper activity detected — single FREE_MOTION segment
        segments = [PhaseSegment(
            phase_id=int(Phase.FREE_MOTION),
            phase_name=PHASE_NAMES[Phase.FREE_MOTION],
            start_t=round(t_start, 4), end_t=round(t_end, 4),
            confidence=0.5,
            evidence="No gripper transitions detected; entire episode labelled free_motion.",
        )]
        return EpisodeAnnotation(
            episode_id="0", task=task, segments=segments,
            overall_confidence=0.5,
            notes="Heuristic: no gripper activity.",
        )

    # Find the first OPEN→CLOSE / CLOSE→OPEN pair where open comes AFTER close.
    # If the episode starts with the gripper already CLOSED, skip any leading
    # CLOSE→OPEN transitions until we find a proper pick-and-place cycle.
    t_close, t_open = None, None
    for tc in close_times:
        for to in open_times:
            if to > tc + 0.1:   # open must be at least 100ms after close
                t_close, t_open = tc, to
                break
        if t_close is not None:
            break

    if t_close is None:
        # No valid pick-and-place cycle found
        segments = [PhaseSegment(
            phase_id=int(Phase.FREE_MOTION),
            phase_name=PHASE_NAMES[Phase.FREE_MOTION],
            start_t=round(t_start, 4), end_t=round(t_end, 4),
            confidence=0.5,
            evidence="No valid pick-and-place cycle detected (gripper state unusual).",
        )]
        return EpisodeAnnotation(
            episode_id="0", task=task, segments=segments,
            overall_confidence=0.5,
            notes="Heuristic: no valid close→open cycle found.",
        )

    # Estimate control rate (Hz) from timestamps
    dt_median = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1/30
    hz = max(1.0, 1.0 / dt_median)

    # ── FINE_ALIGN window: look back from close event for slow+low region ─────
    # Walk backward from t_close to find where speed & z first rise above threshold.
    # If the robot is already fast/high exactly at t_close (backward walk fires at i=0),
    # we skip FINE_ALIGN (no observable pre-grasp slow-down) rather than collapsing
    # the boundary onto t_close.
    fine_align_window = min(2.0, t_close * 0.4)
    low_speed_thresh  = float(np.percentile(speed_sm, 20)) + 0.005
    low_z_thresh      = float(np.percentile(tcp_z, 30))

    idx_close = int(np.searchsorted(timestamps, t_close))
    idx_fa_start = max(0, idx_close - int(fine_align_window * hz))
    fine_align_start = None   # None = no clear fine-align phase observable

    for i in range(idx_close - 1, idx_fa_start, -1):   # start at idx_close-1, not idx_close
        if speed_sm[i] > low_speed_thresh * 2.5 or tcp_z[i] > low_z_thresh * 1.5:
            # This frame is still "approaching" — fine-align begins just after this
            fa = float(timestamps[min(i + 1, len(timestamps) - 1)])
            # Only accept fine_align if it gives at least 200 ms before t_close
            if t_close - fa >= 0.2:
                fine_align_start = fa
            break

    # ── CONTACT_ESTABLISH: gripper command ramps up to close ─────────────────
    # Default: 300ms window before the close state event.
    contact_start = t_close - 0.3
    idx_close_search = max(0, idx_close - int(1.5 * hz))
    for i in range(idx_close - 1, idx_close_search, -1):
        if gripper_cmd_sm[i] < 0.3:
            contact_start = float(timestamps[min(i + 1, len(timestamps) - 1)])
            break
    # Keep at least 200ms before t_close and after fine_align (or t_start)
    fa_end = fine_align_start if fine_align_start is not None else t_start
    contact_start = max(fa_end + 0.1, min(contact_start, t_close - 0.1))

    # ── CONSTRAINED_MOTION: gripper CLOSED, robot lifting/transporting ────────
    # Must come strictly after contact and before verify; spaced at least 150 ms.
    constrained_start = max(t_close + 0.15, contact_start + 0.15)

    # ── VERIFY_RELEASE: gripper opens ─────────────────────────────────────────
    verify_start = t_open

    # ── Assemble ordered boundary list ───────────────────────────────────────
    # LEGAL TRANSITION PATH: free_motion → fine_align → contact_establish
    #                         → constrained_motion → verify_release
    # FINE_ALIGN is required between FREE_MOTION and CONTACT_ESTABLISH.
    # If the backward walk found a clear fine-align onset, use it; otherwise
    # insert a synthetic 0.5s window right before contact_establish.
    MIN_SEG_S = 0.15
    SYNTH_FA_WINDOW_S = 0.5    # fallback fine_align window when not observable

    if fine_align_start is None:
        fine_align_start = max(t_start + MIN_SEG_S, contact_start - SYNTH_FA_WINDOW_S)

    # Ensure proper ordering with at least MIN_SEG_S gaps
    fine_align_start = min(fine_align_start, contact_start - MIN_SEG_S)
    fine_align_start = max(fine_align_start, t_start + MIN_SEG_S)
    contact_start    = max(contact_start, fine_align_start + MIN_SEG_S)
    constrained_start = max(constrained_start, contact_start + MIN_SEG_S)

    raw_boundaries: list[tuple[float, Phase]] = [(t_start, Phase.FREE_MOTION)]
    raw_boundaries.append((fine_align_start, Phase.FINE_ALIGN))

    if verify_start - contact_start >= MIN_SEG_S + MIN_SEG_S:
        raw_boundaries.append((contact_start, Phase.CONTACT_ESTABLISH))
        if verify_start - constrained_start >= MIN_SEG_S:
            raw_boundaries.append((constrained_start, Phase.CONSTRAINED_MOTION))
    elif verify_start - contact_start >= MIN_SEG_S:
        raw_boundaries.append((contact_start, Phase.CONTACT_ESTABLISH))

    raw_boundaries.append((verify_start, Phase.VERIFY_RELEASE))

    # Sort and keep only strictly-increasing timestamps
    raw_boundaries.sort(key=lambda x: x[0])
    merged: list[tuple[float, Phase]] = [raw_boundaries[0]]
    for t, ph in raw_boundaries[1:]:
        if t > merged[-1][0] + MIN_SEG_S / 2:   # > half minimum segment
            merged.append((t, ph))

    for i, (t_seg_start, ph) in enumerate(merged):
        t_seg_end = merged[i + 1][0] if i + 1 < len(merged) else t_end
        if t_seg_end <= t_seg_start:
            continue
        confidence = 0.9 if ph in (Phase.CONTACT_ESTABLISH, Phase.CONSTRAINED_MOTION) else 0.75
        segments.append(PhaseSegment(
            phase_id=int(ph),
            phase_name=PHASE_NAMES[ph],
            start_t=round(t_seg_start, 4),
            end_t=round(t_seg_end, 4),
            confidence=confidence,
            evidence=(
                f"gripper_close@{t_close:.2f}s gripper_open@{t_open:.2f}s"
                if ph in (Phase.CONTACT_ESTABLISH, Phase.CONSTRAINED_MOTION, Phase.VERIFY_RELEASE)
                else f"heuristic speed/z analysis (speed_thresh={low_speed_thresh:.4f})"
            ),
        ))

    return EpisodeAnnotation(
        episode_id="0", task=task,
        segments=segments,
        overall_confidence=0.75,
        notes=(
            f"Heuristic annotator: gripper_close@{t_close:.2f}s, "
            f"gripper_open@{t_open:.2f}s. "
            "No F/T sensor; Qwen VLM annotation available with --use-qwen + vLLM + GPU."
        ),
    )


# ── Qwen annotation ───────────────────────────────────────────────────────────

def _qwen_annotate_episode(
    video_clip: Optional[Path],
    task: str,
    ep_idx: int,
    ep_duration: float,
) -> Optional[EpisodeAnnotation]:
    """Attempt Qwen annotation; return None if vLLM unavailable or video missing."""
    if video_clip is None or not video_clip.exists():
        return None
    try:
        from src.annotation.serve_qwen import QwenAnnotator
        from src.annotation.prompt import PromptBuilder
        from src.annotation.schema import parse_qwen_output

        builder = PromptBuilder(task)
        system_msg = builder.build_system_message()
        user_msg = builder.build_user_message(str(ep_idx), ep_duration)
        prompt = system_msg + "\n\n---\n\n" + user_msg

        annotator = QwenAnnotator()
        raw = annotator.annotate_episode(str(video_clip), prompt)
        ann = parse_qwen_output(raw, ep_duration)
        ann = ann.model_copy(update={"episode_id": str(ep_idx), "task": task})
        return ann
    except Exception as exc:
        console.print(f"[yellow]Qwen unavailable for episode {ep_idx}: {exc}[/yellow]")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    dataset_dir: Path = typer.Option(..., help="cleaned_datasets/<name>/ directory"),
    output_name: str = typer.Option(..., help="Output directory name under cleaned_datasets/"),
    task: str = typer.Option("pick_place", help="Task name: pick_place or msd_plug"),
    use_qwen: bool = typer.Option(False, help="Attempt Qwen3-VL annotation (requires vLLM + GPU)"),
    primary_camera: str = typer.Option("ee_zed_m_left", help="Camera used for Qwen video clips"),
    output_root: Path = typer.Option(Path("cleaned_datasets"), help="Root for output"),
):
    if not dataset_dir.exists():
        console.print(f"[red]Dataset directory not found: {dataset_dir}[/red]")
        raise typer.Exit(1)

    output_dir = output_root / output_name
    ann_dir = output_dir / "annotations"

    # Copy dataset to output directory
    if output_dir.exists():
        console.print(f"[yellow]Output dir {output_dir} exists — removing and recreating.[/yellow]")
        shutil.rmtree(output_dir)
    console.print(f"Copying {dataset_dir} → {output_dir} ...")
    shutil.copytree(dataset_dir, output_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Load episode boundaries
    episodes = _load_episode_boundaries(output_dir)
    console.print(f"Found [bold]{len(episodes)}[/bold] episodes (≥{_MIN_EPISODE_DURATION_S}s each).")

    # Preload robot.jsonl lines (shared across all episodes)
    console.print("Loading robot.jsonl ...")
    robot_lines = (output_dir / "robot.jsonl").read_text().splitlines()
    console.print(f"  {len(robot_lines)} rows loaded.")

    # Camera video + frames for Qwen clip extraction
    cam_video = output_dir / "cameras" / primary_camera / "rgb.mp4"
    cam_frames_jsonl = output_dir / "cameras" / primary_camera / "frames.jsonl"

    validator = Validator()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task("Annotating episodes", total=len(episodes))

        for ep in episodes:
            ep_idx = ep["idx"]
            ep_dur = ep["duration_s"]
            progress.update(bar, description=f"Episode {ep_idx:3d}/{len(episodes)-1}  ({ep_dur:.1f}s)")

            # Load per-episode proprioceptive arrays
            robot_data = _load_robot_for_episode(robot_lines, ep)
            if not robot_data:
                console.print(f"[yellow]Episode {ep_idx}: no robot data — skipping.[/yellow]")
                progress.advance(bar)
                continue

            annotation = None
            method = "fsm"

            # Attempt Qwen if requested
            if use_qwen and cam_video.exists() and cam_frames_jsonl.exists():
                clip_path = ann_dir / "clips" / f"episode_{ep_idx:03d}.mp4"
                _extract_episode_video_clip(
                    cam_video, cam_frames_jsonl,
                    ep["start_host_ns"], ep["end_host_ns"],
                    clip_path,
                )
                qwen_ann = _qwen_annotate_episode(clip_path, task, ep_idx, ep_dur)
                if qwen_ann is not None:
                    annotation = qwen_ann
                    method = "qwen"

            # Heuristic fallback (gripper + velocity + height)
            if annotation is None:
                annotation = _heuristic_annotate_episode(robot_data, task, ep_dur)
                method = "heuristic"

            # Update episode_id in annotation
            annotation = EpisodeAnnotation.model_validate({
                **annotation.model_dump(),
                "episode_id": str(ep_idx),
                "task": task,
            })

            # Validate
            val_result = validator.validate(annotation, episode_duration=ep_dur)
            if not val_result.passed:
                errs = val_result.errors()
                console.print(f"[yellow]Episode {ep_idx}: {len(errs)} validation error(s)[/yellow]")
                for iss in errs:
                    console.print(f"  [red]{iss.rule}[/red]: {iss.message}")

            # Save annotation JSON
            out_path = ann_dir / f"episode_{ep_idx:03d}.json"
            ann_payload = annotation.model_dump()
            ann_payload["_meta"] = {
                "method": method,
                "duration_s": ep_dur,
                "n_robot_frames": robot_data["n_frames"],
                "validation_passed": val_result.passed,
                "validation_warnings": len(val_result.warnings()),
            }
            out_path.write_text(json.dumps(ann_payload, indent=2))

            results.append({
                "episode_idx": ep_idx,
                "duration_s": ep_dur,
                "method": method,
                "n_segments": len(annotation.segments),
                "phases": [s.phase_name for s in annotation.segments],
                "validation_passed": val_result.passed,
            })
            progress.advance(bar)

    # Write summary
    phase_freq: dict[str, int] = {}
    for r in results:
        for ph in r["phases"]:
            phase_freq[ph] = phase_freq.get(ph, 0) + 1

    summary = {
        "dataset": str(dataset_dir),
        "output": str(output_dir),
        "task": task,
        "n_episodes": len(results),
        "n_qwen": sum(1 for r in results if r["method"] == "qwen"),
        "n_heuristic": sum(1 for r in results if r["method"] == "heuristic"),
        "n_validation_passed": sum(1 for r in results if r["validation_passed"]),
        "phase_frequency": phase_freq,
        "episodes": results,
    }
    (output_dir / "annotations_summary.json").write_text(json.dumps(summary, indent=2))

    # Print summary table
    table = Table(title=f"Annotation Summary — {output_name}")
    table.add_column("Episode", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Method")
    table.add_column("Segments", justify="right")
    table.add_column("Phase sequence")
    table.add_column("Valid?", justify="center")
    for r in results:
        table.add_row(
            str(r["episode_idx"]),
            f"{r['duration_s']:.1f}s",
            r["method"],
            str(r["n_segments"]),
            " → ".join(p.replace("_", " ") for p in r["phases"]),
            "✓" if r["validation_passed"] else "✗",
        )
    console.print(table)

    # Phase frequency table
    freq_table = Table(title="Phase Frequency Across All Episodes")
    freq_table.add_column("Phase")
    freq_table.add_column("Episodes containing phase", justify="right")
    for ph_name in [PHASE_NAMES[p] for p in Phase]:
        cnt = phase_freq.get(ph_name, 0)
        freq_table.add_row(ph_name, str(cnt))
    console.print(freq_table)

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"{len(results)} episodes annotated "
        f"({summary['n_qwen']} Qwen, {summary['n_heuristic']} heuristic) → {output_dir}"
    )


if __name__ == "__main__":
    app()
