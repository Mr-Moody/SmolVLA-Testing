"""
Episode video labeler — Flask server with a browser-based UI.

Usage:
  python labeler.py [--cleaned-root cleaned_datasets] [--raw-root raw_datasets] [--port 5000]

Opens http://localhost:<port> automatically. Select a dataset from the
dropdown, browse episodes, watch both camera feeds, write a task prompt,
and click Submit. Episode metadata and labels are read/written under
cleaned_datasets/<name>/annotations.jsonl, while browser video conversion
artifacts remain under raw_datasets/<name>/cameras/<camera>/.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from bisect import bisect_left
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, render_template, request, send_file

from src.data.utils import (
    find_episode_boundaries,
    infer_timestamp_ns,
    load_annotations,
    read_jsonl,
    save_annotation,
)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

CLEANED_DATASETS_ROOT = Path("cleaned_datasets")
RAW_DATASETS_ROOT = Path("raw_datasets")

_PROJECT_ROOT = Path(__file__).parent.parent

app = Flask(
    __name__,
    template_folder=str(_PROJECT_ROOT / "frontend" / "templates"),
    static_folder=str(_PROJECT_ROOT / "frontend"),
    static_url_path="/static",
)

_TRANSCODE_LOCKS: dict[Path, threading.Lock] = {}
_TRANSCODE_LOCKS_GUARD = threading.Lock()
_VIDEO_STATUS: dict[tuple[str, str], dict[str, Any]] = {}
_VIDEO_STATUS_GUARD = threading.Lock()
_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
_TRANSCODE_SETTINGS_VERSION = "h264_seek_gop15_v1"
_EPISODE_MARKER_EVENTS = {
  "episode_start",
  "episode_end",
  "episode_stop",
  "episode_finish",
  "episode_finished",
}


def _get_transcode_lock(video_path: Path) -> threading.Lock:
  with _TRANSCODE_LOCKS_GUARD:
    lock = _TRANSCODE_LOCKS.get(video_path)
    if lock is None:
      lock = threading.Lock()
      _TRANSCODE_LOCKS[video_path] = lock
    return lock


def _video_key(dataset: str, camera: str) -> tuple[str, str]:
  return dataset, camera


def _set_video_status(dataset: str, camera: str, **fields: Any) -> dict[str, Any]:
  with _VIDEO_STATUS_GUARD:
    key = _video_key(dataset, camera)
    current = dict(_VIDEO_STATUS.get(key, {}))
    current.update(fields)
    _VIDEO_STATUS[key] = current
    return dict(current)


def _get_video_status(dataset: str, camera: str) -> dict[str, Any] | None:
  with _VIDEO_STATUS_GUARD:
    key = _video_key(dataset, camera)
    status = _VIDEO_STATUS.get(key)
    return dict(status) if status is not None else None


def _video_paths(dataset: str, camera: str) -> tuple[Path, Path, Path]:
  cam_dir = RAW_DATASETS_ROOT / dataset / "cameras" / camera
  return cam_dir / "rgb.mp4", cam_dir / "rgb_h264.mp4", cam_dir / "rgb_browser.mp4"


def _probe_duration_seconds(path: Path) -> float | None:
  ffprobe_bin = shutil.which("ffprobe")
  if ffprobe_bin is None:
    return None
  cmd = [
    ffprobe_bin,
    "-v",
    "error",
    "-show_entries",
    "format=duration",
    "-of",
    "default=noprint_wrappers=1:nokey=1",
    str(path),
  ]
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    return None
  try:
    value = float(result.stdout.strip())
  except ValueError:
    return None
  return value if value > 0 else None


def _transcode_meta_path(web_path: Path) -> Path:
  return web_path.with_suffix(web_path.suffix + ".json")


def _needs_transcode(source_path: Path, web_path: Path) -> bool:
  if not source_path.exists():
    return False
  if not web_path.exists():
    return True
  if web_path.stat().st_size == 0:
    return True
  meta_path = _transcode_meta_path(web_path)
  if not meta_path.exists():
    return True
  try:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return True
  if meta.get("settings_version") != _TRANSCODE_SETTINGS_VERSION:
    return True
  return web_path.stat().st_mtime < source_path.stat().st_mtime


def _transcode_for_web(
  source_path: Path,
  output_path: Path,
  progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[bool, str | None]:
  """Transcode source video to browser-friendly H.264 MP4."""
  ffmpeg_bin = shutil.which("ffmpeg")
  if ffmpeg_bin is None:
    app.logger.warning("ffmpeg not found; cannot auto-convert %s", source_path)
    return False, "ffmpeg not found"

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temp_path = output_path.with_name(output_path.stem + ".tmp.mp4")
  duration_s = _probe_duration_seconds(source_path)

  cmd = [
    ffmpeg_bin,
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-i",
    str(source_path),
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "20",
    "-pix_fmt",
    "yuv420p",
    "-g",
    "15",
    "-keyint_min",
    "15",
    "-sc_threshold",
    "0",
    "-movflags",
    "+faststart",
    "-an",
    "-progress",
    "pipe:1",
    "-nostats",
    str(temp_path),
  ]

  proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
  )

  if progress_cb is not None:
    progress_cb(0.0, "Starting conversion...")

  if proc.stdout is not None:
    for line in proc.stdout:
      clean = line.strip()
      if clean.startswith("out_time_ms=") and duration_s:
        try:
          out_ms = int(clean.split("=", 1)[1])
          out_s = max(0.0, out_ms / 1_000_000.0)
          pct = min(99.0, (out_s / duration_s) * 100.0)
          if progress_cb is not None:
            progress_cb(pct, f"Converting... {pct:.0f}%")
        except ValueError:
          pass
      elif clean.startswith("out_time=") and duration_s:
        m = _FFMPEG_TIME_RE.search(clean)
        if m:
          h = int(m.group(1))
          mnt = int(m.group(2))
          sec = float(m.group(3))
          out_s = h * 3600 + mnt * 60 + sec
          pct = min(99.0, (out_s / duration_s) * 100.0)
          if progress_cb is not None:
            progress_cb(pct, f"Converting... {pct:.0f}%")

  stderr_text = ""
  if proc.stderr is not None:
    stderr_text = proc.stderr.read().strip()
  proc.wait()

  if proc.returncode != 0:
    app.logger.warning(
      "Auto-conversion failed for %s (exit=%s): %s",
      source_path,
      proc.returncode,
      stderr_text,
    )
    try:
      if temp_path.exists():
        temp_path.unlink()
    except OSError:
      pass
    return False, stderr_text or "ffmpeg conversion failed"

  temp_path.replace(output_path)
  _transcode_meta_path(output_path).write_text(json.dumps({
    "settings_version": _TRANSCODE_SETTINGS_VERSION,
  }) + "\n", encoding="utf-8")
  if progress_cb is not None:
    progress_cb(100.0, "Conversion complete")
  app.logger.info("Created browser video: %s", output_path)
  return True, None


def _build_video_status(dataset: str, camera: str) -> dict[str, Any]:
  source_path, web_path, alt_web_path = _video_paths(dataset, camera)
  status = _get_video_status(dataset, camera)
  if status is not None and status.get("state") in {"converting", "error"}:
    return status

  if alt_web_path.exists():
    return _set_video_status(dataset, camera, state="ready", progress=100.0, message="Browser video ready")
  if web_path.exists() and not _needs_transcode(source_path, web_path):
    return _set_video_status(dataset, camera, state="ready", progress=100.0, message="Browser video ready")
  if not source_path.exists():
    return _set_video_status(dataset, camera, state="missing", progress=0.0, message="rgb.mp4 not found")
  if shutil.which("ffmpeg") is None:
    return _set_video_status(dataset, camera, state="source_only", progress=0.0, message="ffmpeg not installed")
  return _set_video_status(dataset, camera, state="pending", progress=0.0, message="Waiting to convert")


def _run_transcode_task(dataset: str, camera: str, source_path: Path, web_path: Path) -> None:
  lock = _get_transcode_lock(web_path)
  with lock:
    if not _needs_transcode(source_path, web_path):
      _set_video_status(dataset, camera, state="ready", progress=100.0, message="Browser video ready")
      return

    def on_progress(percent: float, message: str) -> None:
      _set_video_status(dataset, camera, state="converting", progress=round(percent, 1), message=message)

    ok, error_msg = _transcode_for_web(source_path, web_path, progress_cb=on_progress)
    if ok:
      _set_video_status(dataset, camera, state="ready", progress=100.0, message="Browser video ready")
    else:
      _set_video_status(dataset, camera, state="error", progress=0.0, message=error_msg or "Conversion failed")


def _prepare_video(dataset: str, camera: str) -> dict[str, Any]:
  source_path, web_path, alt_web_path = _video_paths(dataset, camera)
  status = _build_video_status(dataset, camera)

  if status.get("state") in {"ready", "converting", "error", "missing", "source_only"}:
    return status

  if alt_web_path.exists():
    return _set_video_status(dataset, camera, state="ready", progress=100.0, message="Browser video ready")
  if not source_path.exists():
    return _set_video_status(dataset, camera, state="missing", progress=0.0, message="rgb.mp4 not found")

  _set_video_status(dataset, camera, state="converting", progress=0.0, message="Starting conversion...")
  worker = threading.Thread(
    target=_run_transcode_task,
    args=(dataset, camera, source_path, web_path),
    daemon=True,
  )
  worker.start()
  return _get_video_status(dataset, camera) or {"state": "converting", "progress": 0.0, "message": "Starting conversion..."}


def _resolve_video_path(dataset: str, camera: str) -> Path | None:
  """Return best playable video path without blocking request thread."""
  source_path, web_path, alt_web_path = _video_paths(dataset, camera)

  if alt_web_path.exists():
    return alt_web_path
  if web_path.exists() and not _needs_transcode(source_path, web_path):
    return web_path
  if source_path.exists():
    return source_path
  return None


def _normalize_episode_event_name(row: dict[str, Any]) -> str | None:
  for key in ("event", "episode_event"):
    value = row.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip().lower()
  return None


def _episode_marker_row_indices(episode_rows: list[dict[str, Any]]) -> list[int]:
  indices: list[int] = []
  for row_idx, row in enumerate(episode_rows):
    name = _normalize_episode_event_name(row)
    if name not in _EPISODE_MARKER_EVENTS:
      continue
    if row.get("robot_timestamp_ns") is None:
      continue
    indices.append(row_idx)
  return indices


def _write_annotations(dataset_dir: Path, annotations: dict[int, str]) -> None:
  path = dataset_dir / "annotations.jsonl"
  with path.open("w", encoding="utf-8") as handle:
    for episode_index in sorted(annotations):
      handle.write(json.dumps({
        "episode_index": episode_index,
        "task": annotations[episode_index],
      }) + "\n")


def load_subtasks(dataset_dir: Path) -> dict[int, list[dict]]:
  """Load subtask phases from subtasks.jsonl → {episode_index: [phase_dicts]}."""
  path = dataset_dir / "subtasks.jsonl"
  if not path.exists():
    return {}
  result: dict[int, list[dict]] = {}
  for row in read_jsonl(path):
    idx = row.get("episode_index")
    if idx is None:
      continue
    subtasks = row.get("subtasks", [])
    if isinstance(subtasks, list):
      result[int(idx)] = subtasks
  return result


def _write_subtasks(dataset_dir: Path, subtasks_by_ep: dict[int, list[dict]]) -> None:
  path = dataset_dir / "subtasks.jsonl"
  with path.open("w", encoding="utf-8") as handle:
    for episode_index in sorted(subtasks_by_ep):
      handle.write(json.dumps({
        "episode_index": episode_index,
        "subtasks": subtasks_by_ep[episode_index],
      }) + "\n")


def load_trims(dataset_dir: Path) -> dict[int, dict[str, float]]:
  """Load trim points from trims.jsonl → {episode_index: {trim_start_s, trim_end_s}}."""
  path = dataset_dir / "trims.jsonl"
  if not path.exists():
    return {}
  result: dict[int, dict[str, float]] = {}
  for row in read_jsonl(path):
    idx = row.get("episode_index")
    if idx is None:
      continue
    result[int(idx)] = {
      "trim_start_s": float(row.get("trim_start_s", 0)),
      "trim_end_s": float(row.get("trim_end_s", 0)),
    }
  return result


def _write_trims(dataset_dir: Path, trims: dict[int, dict[str, float]]) -> None:
  path = dataset_dir / "trims.jsonl"
  with path.open("w", encoding="utf-8") as handle:
    for episode_index in sorted(trims):
      handle.write(json.dumps({
        "episode_index": episode_index,
        "trim_start_s": trims[episode_index]["trim_start_s"],
        "trim_end_s": trims[episode_index]["trim_end_s"],
      }) + "\n")


def _delete_episode_marker(dataset_dir: Path, ep_idx: int) -> None:
  robot_rows = read_jsonl(dataset_dir / "robot.jsonl")
  episode_rows = read_jsonl(dataset_dir / "episode_events.jsonl")
  boundaries = find_episode_boundaries(robot_rows, episode_rows)

  if ep_idx < 0 or ep_idx >= len(boundaries):
    raise ValueError("episode index out of range")

  marker_indices = _episode_marker_row_indices(episode_rows)
  if not marker_indices:
    raise ValueError("no episode markers found in cleaned dataset")

  marker_pos: int
  if len(marker_indices) == len(boundaries):
    marker_pos = ep_idx
  elif len(marker_indices) == len(boundaries) - 1:
    # First boundary can be implicit at robot row 0, so episode 0 maps to
    # the first explicit marker in episode_events.jsonl.
    marker_pos = 0 if ep_idx == 0 else ep_idx - 1
  else:
    start_robot_idx, _ = boundaries[ep_idx]
    target_ts = infer_timestamp_ns(robot_rows[start_robot_idx])
    marker_pos = min(
      range(len(marker_indices)),
      key=lambda pos: abs(int(episode_rows[marker_indices[pos]].get("robot_timestamp_ns", target_ts)) - target_ts),
    )

  del episode_rows[marker_indices[marker_pos]]

  events_path = dataset_dir / "episode_events.jsonl"
  with events_path.open("w", encoding="utf-8") as handle:
    for row in episode_rows:
      handle.write(json.dumps(row) + "\n")

  annotations = load_annotations(dataset_dir)
  if annotations:
    reindexed: dict[int, str] = {}
    for old_idx, task in annotations.items():
      if old_idx == ep_idx:
        continue
      new_idx = old_idx - 1 if old_idx > ep_idx else old_idx
      reindexed[new_idx] = task
    _write_annotations(dataset_dir, reindexed)

  trims = load_trims(dataset_dir)
  if trims:
    reindexed_trims: dict[int, dict[str, float]] = {}
    for old_idx, trim_data in trims.items():
      if old_idx == ep_idx:
        continue
      new_idx = old_idx - 1 if old_idx > ep_idx else old_idx
      reindexed_trims[new_idx] = trim_data
    _write_trims(dataset_dir, reindexed_trims)

  subtasks_data = load_subtasks(dataset_dir)
  if subtasks_data:
    reindexed_subtasks: dict[int, list[dict]] = {}
    for old_idx, ep_subtasks in subtasks_data.items():
      if old_idx == ep_idx:
        continue
      new_idx = old_idx - 1 if old_idx > ep_idx else old_idx
      reindexed_subtasks[new_idx] = ep_subtasks
    _write_subtasks(dataset_dir, reindexed_subtasks)


def _discover_datasets() -> list[dict[str, Any]]:
  """Return sorted list of {name, episode_count, labeled_count} dicts."""
  result = []
  if not CLEANED_DATASETS_ROOT.exists():
    return result
  for d in sorted(CLEANED_DATASETS_ROOT.iterdir()):
    if not d.is_dir():
      continue
    ep_file = d / "episode_events.jsonl"
    robot_file = d / "robot.jsonl"
    if not ep_file.exists() or not robot_file.exists():
      continue
    try:
      robot_rows = read_jsonl(robot_file)
      ep_rows = read_jsonl(ep_file)
      boundaries = find_episode_boundaries(robot_rows, ep_rows)
      annotations = load_annotations(d)
      result.append({
        "name": d.name,
        "episode_count": len(boundaries),
        "labeled_count": len(annotations),
      })
    except Exception:
      continue
  return result


def _episode_list(dataset_name: str) -> list[dict[str, Any]]:
    """Build episode metadata for one dataset: boundaries → camera timestamps."""
    dataset_dir = CLEANED_DATASETS_ROOT / dataset_name
    robot_rows = read_jsonl(dataset_dir / "robot.jsonl")
    ep_rows = read_jsonl(dataset_dir / "episode_events.jsonl")
    boundaries = find_episode_boundaries(robot_rows, ep_rows)
    annotations = load_annotations(dataset_dir)
    trims = load_trims(dataset_dir)
    subtasks_by_ep = load_subtasks(dataset_dir)

    # Load camera frames for timestamp→video-frame mapping
    cameras_root = dataset_dir / "cameras"
    camera_data: dict[str, tuple[list[int], list[int], int]] = {}
    # camera_data[name] = (sorted host timestamps, video_frame_indices, fps)
    for cam_dir in sorted(p for p in cameras_root.iterdir() if p.is_dir()):
        frames_path = cam_dir / "frames.jsonl"
        if not frames_path.exists():
            continue
        rows = read_jsonl(frames_path)
        ts_list: list[int] = []
        vf_list: list[int] = []
        for row in rows:
            ts_list.append(infer_timestamp_ns(row))
            vf_list.append(int(row.get("rgb_video_frame", row.get("frame_index", 0))))
        # Infer FPS from median inter-frame delta
        fps = 30
        if len(ts_list) > 1:
            import statistics
            deltas = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1) if ts_list[i + 1] > ts_list[i]]
            if deltas:
                fps = max(1, round(1_000_000_000.0 / statistics.median(deltas)))
        camera_data[cam_dir.name] = (ts_list, vf_list, fps)

    def ts_to_video_s(cam_name: str, ts_ns: int) -> float:
        ts_list, vf_list, fps = camera_data[cam_name]
        if not ts_list:
            return 0.0
        idx = bisect_left(ts_list, ts_ns)
        idx = max(0, min(idx, len(ts_list) - 1))
        return vf_list[idx] / fps

    episodes: list[dict[str, Any]] = []
    for ep_idx, (start_robot_idx, end_robot_idx) in enumerate(boundaries):
        start_ts = infer_timestamp_ns(robot_rows[start_robot_idx])
        end_ts = infer_timestamp_ns(robot_rows[max(start_robot_idx, end_robot_idx - 1)])

        cam_info: dict[str, dict[str, float]] = {}
        for cam_name in camera_data:
            _, _, fps = camera_data[cam_name]
            start_s = ts_to_video_s(cam_name, start_ts)
            end_s = ts_to_video_s(cam_name, end_ts)
            # Ensure at least a tiny non-zero duration
            if end_s <= start_s:
                end_s = start_s + 0.1
            cam_info[cam_name] = {"start_s": round(start_s, 4), "end_s": round(end_s, 4), "fps": fps}

        trim = trims.get(ep_idx)
        episodes.append({
            "index": ep_idx,
            "cameras": cam_info,
            "annotation": annotations.get(ep_idx),
            "trim_start_s": trim["trim_start_s"] if trim else None,
            "trim_end_s": trim["trim_end_s"] if trim else None,
            "subtasks": subtasks_by_ep.get(ep_idx, []),
        })

    return episodes


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
  return render_template("labeler.html")


@app.route("/api/datasets")
def api_datasets():
    return jsonify(_discover_datasets())


@app.route("/api/datasets/<name>/episodes")
def api_episodes(name: str):
    dataset_dir = CLEANED_DATASETS_ROOT / name
    if not dataset_dir.exists():
        return jsonify({"error": "dataset not found"}), 404
    try:
        return jsonify(_episode_list(name))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<name>/episodes/<int:ep_idx>/annotate", methods=["POST"])
def api_annotate(name: str, ep_idx: int):
    dataset_dir = CLEANED_DATASETS_ROOT / name
    if not dataset_dir.exists():
        return jsonify({"error": "dataset not found"}), 404
    body = request.get_json(silent=True) or {}
    task = body.get("task", "").strip()
    if not task:
        return jsonify({"error": "task is required"}), 400
    try:
        save_annotation(dataset_dir, ep_idx, task)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<name>/episodes/<int:ep_idx>/trim", methods=["POST"])
def api_trim(name: str, ep_idx: int):
    dataset_dir = CLEANED_DATASETS_ROOT / name
    if not dataset_dir.exists():
        return jsonify({"error": "dataset not found"}), 404
    body = request.get_json(silent=True) or {}
    trim_start_s = body.get("trim_start_s")
    trim_end_s = body.get("trim_end_s")
    if trim_start_s is None or trim_end_s is None:
        return jsonify({"error": "trim_start_s and trim_end_s required"}), 400
    try:
        episodes = _episode_list(name)
        if ep_idx < 0 or ep_idx >= len(episodes):
            return jsonify({"error": "episode index out of range"}), 400
        episode = episodes[ep_idx]
        primary_camera = next(iter(episode["cameras"].values()), None)
        if primary_camera is None:
            return jsonify({"error": "episode has no camera timing metadata"}), 400

        trim_start = float(trim_start_s)
        trim_end = float(trim_end_s)
        episode_start = float(primary_camera["start_s"])
        episode_end = float(primary_camera["end_s"])
        if trim_end <= trim_start:
            return jsonify({"error": "trim_end_s must be greater than trim_start_s"}), 400
        if trim_start < episode_start or trim_end > episode_end:
            return jsonify({"error": "trim range must stay within the episode video bounds"}), 400

        trims = load_trims(dataset_dir)
        trims[ep_idx] = {"trim_start_s": round(trim_start, 4), "trim_end_s": round(trim_end, 4)}
        _write_trims(dataset_dir, trims)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<name>/episodes/<int:ep_idx>/subtasks", methods=["POST"])
def api_subtasks(name: str, ep_idx: int):
    dataset_dir = CLEANED_DATASETS_ROOT / name
    if not dataset_dir.exists():
        return jsonify({"error": "dataset not found"}), 404
    body = request.get_json(silent=True) or {}
    subtasks = body.get("subtasks")
    if not isinstance(subtasks, list):
        return jsonify({"error": "subtasks must be a list"}), 400
    try:
        all_subtasks = load_subtasks(dataset_dir)
        all_subtasks[ep_idx] = subtasks
        _write_subtasks(dataset_dir, all_subtasks)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<name>/episodes/<int:ep_idx>", methods=["DELETE"])
def api_delete_episode(name: str, ep_idx: int):
    dataset_dir = CLEANED_DATASETS_ROOT / name
    if not dataset_dir.exists():
        return jsonify({"error": "dataset not found"}), 404
    try:
        _delete_episode_marker(dataset_dir, ep_idx)
        return jsonify({"ok": True})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/video/<dataset>/<camera>/prepare", methods=["POST"])
def api_video_prepare(dataset: str, camera: str):
    status = _prepare_video(dataset, camera)
    return jsonify(status)


@app.route("/api/video/<dataset>/<camera>/status")
def api_video_status(dataset: str, camera: str):
    status = _build_video_status(dataset, camera)
    return jsonify(status)


@app.route("/video/<dataset>/<camera>")
def serve_video(dataset: str, camera: str):
    """Serve an MP4 with proper HTTP range support for browser playback."""
    video_path = _resolve_video_path(dataset, camera)
    if video_path is None:
        return Response("not found", status=404)

    # Let Werkzeug handle RFC-compliant Range and conditional requests.
    response = send_file(video_path.resolve(), mimetype="video/mp4", conditional=True, max_age=3600)
    response.cache_control.public = True
    return response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Episode video labeler.")
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path("cleaned_datasets"),
        help="Root directory containing cleaned datasets for episode metadata and annotations (default: cleaned_datasets).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("raw_datasets"),
        help="Root directory containing raw videos and browser-converted video files (default: raw_datasets).",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        help="Deprecated alias for --cleaned-root.",
    )
    parser.add_argument("--port", type=int, default=3001)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser.")
    return parser.parse_args()


def main() -> None:
    global CLEANED_DATASETS_ROOT, RAW_DATASETS_ROOT
    args = parse_args()
    CLEANED_DATASETS_ROOT = args.datasets_root if args.datasets_root else args.cleaned_root
    RAW_DATASETS_ROOT = args.raw_root

    if not CLEANED_DATASETS_ROOT.exists():
        print(f"Error: cleaned datasets root not found: {CLEANED_DATASETS_ROOT}", file=sys.stderr)
        sys.exit(1)

    if not RAW_DATASETS_ROOT.exists():
        print(f"Error: raw datasets root not found: {RAW_DATASETS_ROOT}", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"Labeler running at {url}")
    print(f"Episodes + annotations root: {CLEANED_DATASETS_ROOT.resolve()}")
    print(f"Video source + converted output root: {RAW_DATASETS_ROOT.resolve()}")
    print("Press Ctrl+C to quit.")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
