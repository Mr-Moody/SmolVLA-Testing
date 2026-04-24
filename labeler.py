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

# ---------------------------------------------------------------------------
# Add project root to path so dataset_utils can be imported directly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from dataset_utils import (
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

app = Flask(__name__)

_TRANSCODE_LOCKS: dict[Path, threading.Lock] = {}
_TRANSCODE_LOCKS_GUARD = threading.Lock()
_VIDEO_STATUS: dict[tuple[str, str], dict[str, Any]] = {}
_VIDEO_STATUS_GUARD = threading.Lock()
_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")


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


def _needs_transcode(source_path: Path, web_path: Path) -> bool:
  if not source_path.exists():
    return False
  if not web_path.exists():
    return True
  if web_path.stat().st_size == 0:
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
            start_s = ts_to_video_s(cam_name, start_ts)
            end_s = ts_to_video_s(cam_name, end_ts)
            # Ensure at least a tiny non-zero duration
            if end_s <= start_s:
                end_s = start_s + 0.1
            cam_info[cam_name] = {"start_s": round(start_s, 4), "end_s": round(end_s, 4)}

        episodes.append({
            "index": ep_idx,
            "cameras": cam_info,
            "annotation": annotations.get(ep_idx),
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
    return send_file(video_path, mimetype="video/mp4", conditional=True)


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
    parser.add_argument("--port", type=int, default=5000)
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
