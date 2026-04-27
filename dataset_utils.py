"""Shared data structures and I/O utilities for data_cleaner.py and data_converter.py."""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CAMERA_TOLERANCE_NS = 150_000_000
DEFAULT_TEXT_TOLERANCE_NS = 2_000_000_000


@dataclass(frozen=True)
class TextRecord:
    text: str
    timestamp_ns: int
    raw: dict[str, Any]


@dataclass(frozen=True)
class CameraFrame:
    camera_name: str
    frame_index: int
    timestamp_ns: int
    video_path: Path
    video_frame_index: int
    width: int
    height: int
    raw: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def first_existing_path(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def infer_timestamp_ns(row: dict[str, Any]) -> int:
    for key in (
        "host_timestamp_ns",
        "timestamp_ns",
        "robot_timestamp_ns",
        "receive_host_time_ns",
        "created_unix_time_ns",
    ):
        value = row.get(key)
        if value is not None:
            return int(value)
    raise KeyError(f"Could not find a timestamp field in row: {row}")


def infer_text_value(row: dict[str, Any]) -> str:
    for key in ("text", "instruction", "task", "transcript", "utterance", "caption"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Could not find a text field in row: {row}")


def closest_index(sorted_timestamps: list[int], target: int) -> int | None:
    if not sorted_timestamps:
        return None
    insert_at = bisect_left(sorted_timestamps, target)
    candidates: list[tuple[int, int]] = []
    if insert_at < len(sorted_timestamps):
        candidates.append((abs(sorted_timestamps[insert_at] - target), insert_at))
    if insert_at > 0:
        prev_idx = insert_at - 1
        candidates.append((abs(sorted_timestamps[prev_idx] - target), prev_idx))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def load_text_rows(dataset_dir: Path) -> list[TextRecord]:
    text_path = first_existing_path((
        dataset_dir / "text.jsonl",
        dataset_dir / "language.jsonl",
        dataset_dir / "text" / "text.jsonl",
    ))
    if text_path is None:
        return []
    records = [
        TextRecord(text=infer_text_value(row), timestamp_ns=infer_timestamp_ns(row), raw=row)
        for row in read_jsonl(text_path)
    ]
    records.sort(key=lambda r: r.timestamp_ns)
    return records


def load_camera_frames(dataset_dir: Path) -> dict[str, list[CameraFrame]]:
    cameras_root = dataset_dir / "cameras"
    if not cameras_root.exists():
        raise FileNotFoundError(f"Missing cameras directory: {cameras_root}")
    camera_frames: dict[str, list[CameraFrame]] = {}
    for camera_dir in sorted(p for p in cameras_root.iterdir() if p.is_dir()):
        frames_path = camera_dir / "frames.jsonl"
        if not frames_path.exists():
            continue
        frames = []
        for row in read_jsonl(frames_path):
            video_name = row.get("rgb_video", "rgb.mp4")
            frames.append(CameraFrame(
                camera_name=row.get("camera", camera_dir.name),
                frame_index=int(row["frame_index"]),
                timestamp_ns=infer_timestamp_ns(row),
                video_path=camera_dir / video_name,
                video_frame_index=int(row.get("rgb_video_frame", row["frame_index"])),
                width=int(row["width"]),
                height=int(row["height"]),
                raw=row,
            ))
        frames.sort(key=lambda f: f.timestamp_ns)
        if frames:
            camera_frames[camera_dir.name] = frames
    if not camera_frames:
        raise ValueError(f"No camera frames found under {cameras_root}")
    return camera_frames


def _normalize_event_name(row: dict[str, Any]) -> str | None:
    for key in ("event", "episode_event"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def load_annotations(dataset_dir: Path) -> dict[int, str]:
    """Load episode-index → task string from annotations.jsonl."""
    path = dataset_dir / "annotations.jsonl"
    if not path.exists():
        return {}
    result: dict[int, str] = {}
    for row in read_jsonl(path):
        if "episode_index" in row and "task" in row:
            result[int(row["episode_index"])] = row["task"]
    return result


def load_trims(dataset_dir: Path) -> dict[int, dict[str, float]]:
    """Load episode-index → trim bounds from trims.jsonl."""
    path = dataset_dir / "trims.jsonl"
    if not path.exists():
        return {}
    result: dict[int, dict[str, float]] = {}
    for row in read_jsonl(path):
        if "episode_index" not in row:
            continue
        result[int(row["episode_index"])] = {
            "trim_start_s": float(row.get("trim_start_s", 0.0)),
            "trim_end_s": float(row.get("trim_end_s", 0.0)),
        }
    return result


def load_subtasks(dataset_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Load subtask phases from subtasks.jsonl → {episode_index: [phase_dicts]}."""
    path = dataset_dir / "subtasks.jsonl"
    if not path.exists():
        return {}
    result: dict[int, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        idx = row.get("episode_index")
        if idx is None:
            continue
        phases = row.get("subtasks", [])
        if isinstance(phases, list):
            result[int(idx)] = phases
    return result


def save_annotation(dataset_dir: Path, episode_index: int, task: str) -> None:
    """Write/update annotation for one episode in annotations.jsonl (last write wins per episode)."""
    import datetime
    path = dataset_dir / "annotations.jsonl"
    existing: dict[int, Any] = {}
    if path.exists():
        for row in read_jsonl(path):
            if "episode_index" in row:
                existing[int(row["episode_index"])] = row
    existing[episode_index] = {
        "episode_index": episode_index,
        "task": task,
        "annotated_at": datetime.datetime.utcnow().isoformat(),
    }
    write_jsonl(path, list(existing.values()))


def find_episode_boundaries(
    robot_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) slices into robot_rows for each episode."""
    robot_times = [infer_timestamp_ns(row) for row in robot_rows]
    if not robot_times:
        return []
    split_indices: set[int] = {0, len(robot_times)}
    for row in episode_rows:
        event_name = _normalize_event_name(row)
        if event_name not in {"episode_start", "episode_end", "episode_stop", "episode_finish", "episode_finished"}:
            continue
        if row.get("robot_timestamp_ns") is None:
            continue
        split_indices.add(bisect_left(robot_times, int(row["robot_timestamp_ns"])))
    sorted_splits = sorted(split_indices)
    boundaries = [(s, e) for s, e in zip(sorted_splits[:-1], sorted_splits[1:]) if e > s]
    return boundaries or [(0, len(robot_times))]
