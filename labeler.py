"""
Episode video labeler — Flask server with a browser-based UI.

Usage:
    python labeler.py [--datasets-root raw_datasets] [--port 5000]

Opens http://localhost:<port> automatically. Select a dataset from the
dropdown, browse episodes, watch both camera feeds, write a task prompt,
and click Submit. Labels are saved to raw_datasets/<name>/annotations.jsonl
and are automatically picked up by data_converter.py on the next run.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from bisect import bisect_left
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

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
# HTML page (single-file SPA)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Episode Labeler</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }

  /* ── Header bar ── */
  #header {
    background: #1a1a1a;
    border-bottom: 1px solid #333;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
    flex-wrap: wrap;
  }
  #header label { font-size: 13px; color: #aaa; }
  select, input[type=number], input[type=text], textarea {
    background: #2a2a2a; color: #e0e0e0; border: 1px solid #444;
    border-radius: 4px; padding: 4px 8px; font-size: 13px;
  }
  select:focus, input:focus, textarea:focus { outline: none; border-color: #4a9eff; }
  button {
    background: #2a2a2a; color: #e0e0e0; border: 1px solid #444;
    border-radius: 4px; padding: 5px 12px; font-size: 13px; cursor: pointer;
  }
  button:hover { background: #3a3a3a; border-color: #555; }
  button:disabled { opacity: 0.4; cursor: default; }
  button.primary { background: #1a6fd4; border-color: #2a7fe4; }
  button.primary:hover { background: #2a7fe4; }

  #ep-info { margin-left: auto; font-size: 13px; color: #aaa; white-space: nowrap; }
  #progress-badge {
    font-size: 12px; padding: 2px 8px; border-radius: 10px;
    background: #1a3d1a; color: #6ddc6d; border: 1px solid #2a5a2a;
  }

  /* ── Video grid ── */
  #video-grid {
    display: flex;
    flex: 1;
    min-height: 0;
    gap: 4px;
    padding: 4px;
    background: #0a0a0a;
  }
  .cam-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .cam-label {
    font-size: 11px; color: #888; padding: 2px 6px;
    background: #1a1a1a; border-radius: 4px 4px 0 0; text-align: center;
  }
  video {
    width: 100%; height: 100%; object-fit: contain;
    background: #000; display: block;
    flex: 1; min-height: 0;
  }

  /* ── Controls bar ── */
  #controls {
    background: #1a1a1a;
    border-top: 1px solid #333;
    padding: 8px 16px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    flex-shrink: 0;
  }
  #seek-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  #seekbar {
    flex: 1;
    height: 4px;
    accent-color: #4a9eff;
    cursor: pointer;
  }
  #time-display { font-size: 12px; color: #aaa; min-width: 90px; text-align: right; font-variant-numeric: tabular-nums; }
  #btn-row { display: flex; gap: 8px; align-items: center; }

  /* ── Annotation area ── */
  #annotation-area {
    background: #141414;
    border-top: 1px solid #333;
    padding: 10px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex-shrink: 0;
  }
  #annotation-area label { font-size: 12px; color: #aaa; }
  #task-input {
    width: 100%; resize: vertical; min-height: 52px;
    font-size: 14px; padding: 6px 10px; line-height: 1.4;
  }
  #submit-row { display: flex; align-items: center; gap: 10px; }
  #status-msg { font-size: 12px; transition: opacity 0.4s; }
  #status-msg.ok  { color: #6ddc6d; }
  #status-msg.err { color: #f06060; }
  #ep-annotation-status { font-size: 12px; color: #888; }

  /* ── Loading overlay ── */
  #loading {
    position: fixed; inset: 0; background: rgba(0,0,0,0.75);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: #aaa; z-index: 100;
  }
  #loading.hidden { display: none; }
</style>
</head>
<body>

<div id="loading">Loading datasets…</div>

<!-- ── Header ── -->
<div id="header">
  <label for="dataset-select">Dataset</label>
  <select id="dataset-select"></select>

  <span id="progress-badge">0 / 0 labeled</span>

  <button id="btn-prev" disabled>&#9664; Prev</button>

  <label for="jump-input">Jump to</label>
  <input id="jump-input" type="number" min="0" step="1" style="width:70px" placeholder="0">
  <button id="btn-go">Go</button>

  <button id="btn-next" disabled>Next &#9654;</button>

  <span id="ep-info">Episode — / —</span>
</div>

<!-- ── Video grid ── -->
<div id="video-grid">
  <div class="cam-panel" id="panel-cam0">
    <div class="cam-label" id="label-cam0">camera 0</div>
    <video id="video0" preload="auto" playsinline></video>
  </div>
  <div class="cam-panel" id="panel-cam1">
    <div class="cam-label" id="label-cam1">camera 1</div>
    <video id="video1" preload="auto" playsinline></video>
  </div>
</div>

<!-- ── Transport controls ── -->
<div id="controls">
  <div id="seek-row">
    <input id="seekbar" type="range" min="0" max="1000" value="0" step="1">
    <span id="time-display">0:00 / 0:00</span>
  </div>
  <div id="btn-row">
    <button id="btn-playpause">&#9654; Play</button>
    <button id="btn-skip-back">&#8249;&#8249; 5s</button>
    <button id="btn-skip-fwd">5s &#8250;&#8250;</button>
  </div>
</div>

<!-- ── Annotation area ── -->
<div id="annotation-area">
  <label for="task-input">Task prompt <span id="ep-annotation-status"></span></label>
  <textarea id="task-input" placeholder="Describe the action in this episode…"></textarea>
  <div id="submit-row">
    <button id="btn-submit" class="primary">Submit Label</button>
    <span id="status-msg"></span>
  </div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let datasets = [];          // [{name, episode_count, labeled_count}]
let episodes = [];          // [{index, cameras:{name:{start_s, end_s}}, annotation}]
let currentDataset = null;
let currentEpIdx = 0;       // index into episodes[]
let isPlaying = false;
let seekSuppressed = false; // prevent seekbar feedback loop

const video0 = document.getElementById('video0');
const video1 = document.getElementById('video1');
const videos = [video0, video1];

// ── Helpers ────────────────────────────────────────────────────────────────
function fmt(sec) {
  if (!isFinite(sec)) return '0:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function epDuration() {
  if (!episodes.length) return 0;
  const ep = episodes[currentEpIdx];
  const cams = Object.values(ep.cameras);
  if (!cams.length) return 0;
  return cams[0].end_s - cams[0].start_s;
}

function primaryStart() {
  if (!episodes.length) return 0;
  const ep = episodes[currentEpIdx];
  const cams = Object.values(ep.cameras);
  return cams.length ? cams[0].start_s : 0;
}

function primaryEnd() {
  if (!episodes.length) return 0;
  const ep = episodes[currentEpIdx];
  const cams = Object.values(ep.cameras);
  return cams.length ? cams[0].end_s : 0;
}

// Returns the current playback position within the episode (0 … duration)
function relativeTime() {
  return Math.max(0, video0.currentTime - primaryStart());
}

// ── Dataset loading ─────────────────────────────────────────────────────────
async function loadDatasets() {
  const res = await fetch('/api/datasets');
  datasets = await res.json();
  const sel = document.getElementById('dataset-select');
  sel.innerHTML = datasets.map(d =>
    `<option value="${d.name}">${d.name} (${d.episode_count} episodes)</option>`
  ).join('');
  document.getElementById('loading').classList.add('hidden');
  if (datasets.length) selectDataset(datasets[0].name);
}

async function selectDataset(name) {
  currentDataset = name;
  document.getElementById('loading').classList.remove('hidden');
  const res = await fetch(`/api/datasets/${name}/episodes`);
  episodes = await res.json();
  document.getElementById('loading').classList.add('hidden');
  currentEpIdx = 0;
  loadEpisode(0);
  updateProgressBadge();
}

// ── Episode loading ─────────────────────────────────────────────────────────
function loadEpisode(idx) {
  if (!episodes.length) return;
  idx = Math.max(0, Math.min(idx, episodes.length - 1));
  currentEpIdx = idx;

  const ep = episodes[idx];
  const camNames = Object.keys(ep.cameras);

  // Update camera labels and video sources
  camNames.forEach((cam, i) => {
    const labelEl = document.getElementById(`label-cam${i}`);
    const videoEl = document.getElementById(`video${i}`);
    const panelEl = document.getElementById(`panel-cam${i}`);
    if (labelEl) labelEl.textContent = cam;
    if (videoEl && panelEl) {
      panelEl.style.display = 'flex';
      const src = `/video/${currentDataset}/${cam}`;
      if (videoEl.dataset.src !== src) {
        videoEl.dataset.src = src;
        videoEl.src = src;
        videoEl.load();
      }
    }
  });

  // Hide unused camera panels
  for (let i = camNames.length; i < 2; i++) {
    const panelEl = document.getElementById(`panel-cam${i}`);
    if (panelEl) panelEl.style.display = 'none';
  }

  // Seek to episode start once enough data is available
  videos.forEach((v, i) => {
    const cam = camNames[i];
    if (!cam) return;
    const info = ep.cameras[cam];
    const seekTo = () => {
      v.currentTime = info.start_s;
    };
    if (v.readyState >= 1) {
      seekTo();
    } else {
      v.addEventListener('loadedmetadata', seekTo, { once: true });
    }
  });

  // Update UI
  const epInfo = document.getElementById('ep-info');
  epInfo.textContent = `Episode ${idx + 1} / ${episodes.length}`;
  document.getElementById('btn-prev').disabled = idx === 0;
  document.getElementById('btn-next').disabled = idx === episodes.length - 1;
  document.getElementById('jump-input').value = idx;

  // Load existing annotation
  const taskInput = document.getElementById('task-input');
  const annoStatus = document.getElementById('ep-annotation-status');
  const statusMsg = document.getElementById('status-msg');
  statusMsg.textContent = '';
  if (ep.annotation) {
    taskInput.value = ep.annotation;
    annoStatus.textContent = '(labeled)';
    annoStatus.style.color = '#6ddc6d';
  } else {
    taskInput.value = '';
    annoStatus.textContent = '(unlabeled)';
    annoStatus.style.color = '#888';
  }

  // Reset playback state
  pauseAll();
  updateSeekbar();
}

// ── Playback controls ──────────────────────────────────────────────────────
function playAll() {
  videos.forEach(v => { if (v.src) v.play().catch(() => {}); });
  document.getElementById('btn-playpause').textContent = '❚❚ Pause';
  isPlaying = true;
}

function pauseAll() {
  videos.forEach(v => v.pause());
  document.getElementById('btn-playpause').textContent = '▶ Play';
  isPlaying = false;
}

function togglePlay() {
  if (isPlaying) pauseAll(); else playAll();
}

function skipSeconds(delta) {
  if (!episodes.length) return;
  const ep = episodes[currentEpIdx];
  const camNames = Object.keys(ep.cameras);
  const primaryCam = camNames[0];
  if (!primaryCam) return;
  const info = ep.cameras[primaryCam];
  const newTime = Math.max(info.start_s, Math.min(video0.currentTime + delta, info.end_s));
  const offset = newTime - video0.currentTime;
  videos.forEach((v, i) => {
    const cam = camNames[i];
    if (!cam) return;
    v.currentTime = Math.max(ep.cameras[cam].start_s, Math.min(v.currentTime + offset, ep.cameras[cam].end_s));
  });
}

function seekToFraction(frac) {
  if (!episodes.length) return;
  const ep = episodes[currentEpIdx];
  const camNames = Object.keys(ep.cameras);
  camNames.forEach((cam, i) => {
    const v = videos[i];
    if (!v) return;
    const info = ep.cameras[cam];
    v.currentTime = info.start_s + frac * (info.end_s - info.start_s);
  });
}

// ── Seekbar sync ──────────────────────────────────────────────────────────
function updateSeekbar() {
  if (seekSuppressed) return;
  const dur = epDuration();
  const rel = relativeTime();
  const seekbar = document.getElementById('seekbar');
  seekbar.value = dur > 0 ? Math.round((rel / dur) * 1000) : 0;
  document.getElementById('time-display').textContent = `${fmt(rel)} / ${fmt(dur)}`;
}

function updateProgressBadge() {
  const ds = datasets.find(d => d.name === currentDataset);
  if (!ds) return;
  const labeled = episodes.filter(e => e.annotation).length;
  document.getElementById('progress-badge').textContent = `${labeled} / ${episodes.length} labeled`;
}

// ── Annotation submit ─────────────────────────────────────────────────────
async function submitAnnotation() {
  if (!currentDataset || !episodes.length) return;
  const task = document.getElementById('task-input').value.trim();
  if (!task) {
    showStatus('Please enter a task prompt.', false);
    return;
  }
  const ep = episodes[currentEpIdx];
  const res = await fetch(`/api/datasets/${currentDataset}/episodes/${ep.index}/annotate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task }),
  });
  if (res.ok) {
    ep.annotation = task;
    const annoStatus = document.getElementById('ep-annotation-status');
    annoStatus.textContent = '(labeled)';
    annoStatus.style.color = '#6ddc6d';
    updateProgressBadge();
    showStatus('Saved!', true);
  } else {
    const body = await res.json().catch(() => ({}));
    showStatus('Error: ' + (body.error || res.statusText), false);
  }
}

function showStatus(msg, ok) {
  const el = document.getElementById('status-msg');
  el.textContent = msg;
  el.className = ok ? 'ok' : 'err';
  el.style.opacity = '1';
  setTimeout(() => { el.style.opacity = '0'; }, 3000);
}

// ── Event wiring ──────────────────────────────────────────────────────────
document.getElementById('dataset-select').addEventListener('change', e => {
  selectDataset(e.target.value);
});
document.getElementById('btn-prev').addEventListener('click', () => loadEpisode(currentEpIdx - 1));
document.getElementById('btn-next').addEventListener('click', () => loadEpisode(currentEpIdx + 1));
document.getElementById('btn-go').addEventListener('click', () => {
  const v = parseInt(document.getElementById('jump-input').value, 10);
  if (!isNaN(v)) loadEpisode(v);
});
document.getElementById('jump-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-go').click();
});
document.getElementById('btn-playpause').addEventListener('click', togglePlay);
document.getElementById('btn-skip-back').addEventListener('click', () => skipSeconds(-5));
document.getElementById('btn-skip-fwd').addEventListener('click', () => skipSeconds(5));
document.getElementById('btn-submit').addEventListener('click', submitAnnotation);

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.code === 'ArrowLeft') skipSeconds(-5);
  if (e.code === 'ArrowRight') skipSeconds(5);
  if (e.code === 'ArrowUp') loadEpisode(currentEpIdx - 1);
  if (e.code === 'ArrowDown') loadEpisode(currentEpIdx + 1);
});

// Seekbar interaction
const seekbar = document.getElementById('seekbar');
seekbar.addEventListener('mousedown', () => { seekSuppressed = true; });
seekbar.addEventListener('input', () => {
  seekToFraction(seekbar.value / 1000);
  const dur = epDuration();
  document.getElementById('time-display').textContent = `${fmt((seekbar.value / 1000) * dur)} / ${fmt(dur)}`;
});
seekbar.addEventListener('mouseup', () => { seekSuppressed = false; });
seekbar.addEventListener('touchend', () => { seekSuppressed = false; });

// video0 drives the seekbar and enforces episode bounds
video0.addEventListener('timeupdate', () => {
  const ep = episodes[currentEpIdx];
  if (!ep) return;
  const camNames = Object.keys(ep.cameras);
  const primaryCam = camNames[0];
  if (!primaryCam) return;
  const end = ep.cameras[primaryCam].end_s;
  if (video0.currentTime >= end - 0.05) {
    pauseAll();
    video0.currentTime = end;
  }
  updateSeekbar();
});

// ── Boot ──────────────────────────────────────────────────────────────────
loadDatasets();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

DATASETS_ROOT = Path("raw_datasets")

app = Flask(__name__)


def _discover_datasets() -> list[dict[str, Any]]:
    """Return sorted list of {name, episode_count, labeled_count} dicts."""
    result = []
    if not DATASETS_ROOT.exists():
        return result
    for d in sorted(DATASETS_ROOT.iterdir()):
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
    dataset_dir = DATASETS_ROOT / dataset_name
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
    return HTML_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/datasets")
def api_datasets():
    return jsonify(_discover_datasets())


@app.route("/api/datasets/<name>/episodes")
def api_episodes(name: str):
    dataset_dir = DATASETS_ROOT / name
    if not dataset_dir.exists():
        return jsonify({"error": "dataset not found"}), 404
    try:
        return jsonify(_episode_list(name))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<name>/episodes/<int:ep_idx>/annotate", methods=["POST"])
def api_annotate(name: str, ep_idx: int):
    dataset_dir = DATASETS_ROOT / name
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


@app.route("/video/<dataset>/<camera>")
def serve_video(dataset: str, camera: str):
    """Stream an MP4 with Range support so the browser can seek."""
    video_path = DATASETS_ROOT / dataset / "cameras" / camera / "rgb.mp4"
    if not video_path.exists():
        return Response("not found", status=404)

    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        # Parse "bytes=start-end"
        byte_range = range_header.replace("bytes=", "")
        parts = byte_range.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        def generate():
            with open(video_path, "rb") as fh:
                fh.seek(start)
                remaining = length
                chunk = 1 << 20  # 1 MB chunks
                while remaining > 0:
                    data = fh.read(min(chunk, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": "video/mp4",
        }
        return Response(generate(), status=206, headers=headers)

    # Full file (no Range header)
    def generate_full():
        with open(video_path, "rb") as fh:
            while True:
                data = fh.read(1 << 20)
                if not data:
                    break
                yield data

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
    }
    return Response(generate_full(), status=200, headers=headers)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Episode video labeler.")
    parser.add_argument("--datasets-root", type=Path, default=Path("raw_datasets"),
                        help="Root directory containing raw_datasets (default: raw_datasets).")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser.")
    return parser.parse_args()


def main() -> None:
    global DATASETS_ROOT
    args = parse_args()
    DATASETS_ROOT = args.datasets_root

    if not DATASETS_ROOT.exists():
        print(f"Error: datasets root not found: {DATASETS_ROOT}", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"Labeler running at {url}")
    print(f"Serving datasets from: {DATASETS_ROOT.resolve()}")
    print("Press Ctrl+C to quit.")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
