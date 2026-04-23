"""
Convert a raw recorded session into a native LeRobotDataset v3 export.

Expected raw layout
===================

datasets/<dataset_name>/
    session_metadata.json
    robot.jsonl
    episode_events.jsonl
    text.jsonl                      # optional
    language.jsonl                  # optional
    cameras/
        <camera_name>/
            frames.jsonl
            rgb.mp4

The converter is built around the recording format already present in this
repository. It aligns robot timesteps with the nearest camera frame, derives
SmolVLA-friendly keys such as `observation.state` and
`observation.images.top`, and writes a real LeRobotDataset v3 dataset using
`LeRobotDataset.create(...)`.

Run this script from the `lerobot` virtual environment so `lerobot`, `torch`,
and the dataset extras are available.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_DATASETS_ROOT = Path("raw_datasets")
DEFAULT_OUTPUT_ROOT = Path("lerobot_datasets")
DEFAULT_TEXT_TOLERANCE_NS = 2_000_000_000
DEFAULT_CAMERA_TOLERANCE_NS = 150_000_000
DEFAULT_REPO_OWNER = "local"
DEFAULT_VCODEC = "h264"
DEFAULT_JOINT_MOTION_THRESHOLD = 5e-4
DEFAULT_GRIPPER_MOTION_THRESHOLD = 2e-4
DEFAULT_ACTION_TRANSLATION_THRESHOLD = 5e-6
DEFAULT_ACTION_ROTATION_THRESHOLD = 5e-5


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


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


@dataclass
class VideoReaderState:
    capture: cv2.VideoCapture
    last_frame_index: int | None = None
    last_frame_rgb: np.ndarray | None = None


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


class SmolVLADatasetConverter:
    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        repo_id: str,
        primary_camera: str | None = None,
        camera_tolerance_ns: int = DEFAULT_CAMERA_TOLERANCE_NS,
        text_tolerance_ns: int = DEFAULT_TEXT_TOLERANCE_NS,
        force_overwrite: bool = False,
        vcodec: str = DEFAULT_VCODEC,
        max_episodes: int | None = None,
        max_steps_per_episode: int | None = None,
        joint_motion_threshold: float = DEFAULT_JOINT_MOTION_THRESHOLD,
        gripper_motion_threshold: float = DEFAULT_GRIPPER_MOTION_THRESHOLD,
        action_translation_threshold: float = DEFAULT_ACTION_TRANSLATION_THRESHOLD,
        action_rotation_threshold: float = DEFAULT_ACTION_ROTATION_THRESHOLD,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.repo_id = repo_id
        self.primary_camera = primary_camera
        self.camera_tolerance_ns = camera_tolerance_ns
        self.text_tolerance_ns = text_tolerance_ns
        self.force_overwrite = force_overwrite
        self.vcodec = vcodec
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.joint_motion_threshold = float(joint_motion_threshold)
        self.gripper_motion_threshold = float(gripper_motion_threshold)
        self.action_translation_threshold = float(action_translation_threshold)
        self.action_rotation_threshold = float(action_rotation_threshold)

        self.session_metadata = read_json(self.dataset_dir / "session_metadata.json")
        self.robot_rows = read_jsonl(self.dataset_dir / "robot.jsonl")
        self.episode_rows = read_jsonl(self.dataset_dir / "episode_events.jsonl")
        self.text_rows = self._load_text_rows()
        self.camera_frames = self._load_camera_frames()

    def _load_text_rows(self) -> list[TextRecord]:
        text_path = first_existing_path(
            (
                self.dataset_dir / "text.jsonl",
                self.dataset_dir / "language.jsonl",
                self.dataset_dir / "text" / "text.jsonl",
            )
        )
        if text_path is None:
            return []

        text_records = [
            TextRecord(
                text=infer_text_value(row),
                timestamp_ns=infer_timestamp_ns(row),
                raw=row,
            )
            for row in read_jsonl(text_path)
        ]
        text_records.sort(key=lambda item: item.timestamp_ns)
        return text_records

    def _load_camera_frames(self) -> dict[str, list[CameraFrame]]:
        cameras_root = self.dataset_dir / "cameras"
        if not cameras_root.exists():
            raise FileNotFoundError(f"Missing cameras directory: {cameras_root}")

        camera_frames: dict[str, list[CameraFrame]] = {}
        for camera_dir in sorted(path for path in cameras_root.iterdir() if path.is_dir()):
            frames_path = camera_dir / "frames.jsonl"
            if not frames_path.exists():
                continue

            frames = []
            for row in read_jsonl(frames_path):
                video_name = row.get("rgb_video", "rgb.mp4")
                frames.append(
                    CameraFrame(
                        camera_name=row.get("camera", camera_dir.name),
                        frame_index=int(row["frame_index"]),
                        timestamp_ns=infer_timestamp_ns(row),
                        video_path=camera_dir / video_name,
                        video_frame_index=int(row.get("rgb_video_frame", row["frame_index"])),
                        width=int(row["width"]),
                        height=int(row["height"]),
                        raw=row,
                    )
                )

            frames.sort(key=lambda item: item.timestamp_ns)
            if frames:
                camera_frames[camera_dir.name] = frames

        if not camera_frames:
            raise ValueError(f"No camera frames were found under {cameras_root}")

        return camera_frames

    @staticmethod
    def sync_text_to_frames(
        frames: list[CameraFrame],
        text_records: list[TextRecord],
        max_delta_ns: int = DEFAULT_TEXT_TOLERANCE_NS,
    ) -> list[dict[str, Any]]:
        if not frames:
            return []

        text_timestamps = [record.timestamp_ns for record in text_records]
        aligned: list[dict[str, Any]] = []

        for frame in frames:
            match_index = closest_index(text_timestamps, frame.timestamp_ns)
            text: str | None = None
            delta_ns: int | None = None

            if match_index is not None:
                record = text_records[match_index]
                candidate_delta_ns = abs(record.timestamp_ns - frame.timestamp_ns)
                if candidate_delta_ns <= max_delta_ns:
                    text = record.text
                    delta_ns = candidate_delta_ns

            aligned.append(
                {
                    "camera_name": frame.camera_name,
                    "frame_index": frame.frame_index,
                    "frame_timestamp_ns": frame.timestamp_ns,
                    "text": text,
                    "text_time_delta_ns": delta_ns,
                }
            )

        return aligned

    @staticmethod
    def _normalize_event_name(row: dict[str, Any]) -> str | None:
        for key in ("event", "episode_event"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
        return None

    def _episode_boundaries(self) -> list[tuple[int, int]]:
        robot_times = [infer_timestamp_ns(row) for row in self.robot_rows]
        if not robot_times:
            return []

        split_indices: set[int] = {0, len(robot_times)}
        for row in self.episode_rows:
            event_name = self._normalize_event_name(row)
            if event_name not in {"episode_start", "episode_end", "episode_stop", "episode_finish", "episode_finished"}:
                continue
            if row.get("robot_timestamp_ns") is None:
                continue
            split_indices.add(bisect_left(robot_times, int(row["robot_timestamp_ns"])))

        sorted_splits = sorted(split_indices)
        boundaries: list[tuple[int, int]] = []
        for start_idx, end_idx in zip(sorted_splits[:-1], sorted_splits[1:]):
            if end_idx > start_idx:
                boundaries.append((start_idx, end_idx))

        return boundaries or [(0, len(robot_times))]

    @staticmethod
    def _extract_joint_positions(row: dict[str, Any]) -> np.ndarray:
        values = [float(value) for value in row.get("robot_state", {}).get("q", [])]
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _extract_gripper_width(row: dict[str, Any]) -> float:
        return float(row.get("robot_state", {}).get("gripper_width", 0.0))

    @staticmethod
    def _extract_action_norms(row: dict[str, Any]) -> tuple[float, float]:
        executed_action = row.get("executed_action", {})
        translation = np.asarray(executed_action.get("cartesian_delta_translation", []), dtype=np.float64)
        rotation = np.asarray(executed_action.get("cartesian_delta_rotation", []), dtype=np.float64)
        translation_norm = float(np.linalg.norm(translation)) if translation.size else 0.0
        rotation_norm = float(np.linalg.norm(rotation)) if rotation.size else 0.0
        return translation_norm, rotation_norm

    def _is_moving_step(self, current: dict[str, Any], previous: dict[str, Any] | None) -> bool:
        translation_norm, rotation_norm = self._extract_action_norms(current)
        if translation_norm > self.action_translation_threshold:
            return True
        if rotation_norm > self.action_rotation_threshold:
            return True
        if previous is None:
            return False

        current_q = self._extract_joint_positions(current)
        previous_q = self._extract_joint_positions(previous)
        if current_q.size and previous_q.size and current_q.shape == previous_q.shape:
            joint_delta = float(np.max(np.abs(current_q - previous_q)))
            if joint_delta > self.joint_motion_threshold:
                return True

        gripper_delta = abs(self._extract_gripper_width(current) - self._extract_gripper_width(previous))
        if gripper_delta > self.gripper_motion_threshold:
            return True

        return False

    def _trim_to_motion(self, robot_slice: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
        if not robot_slice:
            return [], 0, 0

        moving_indices = [
            index
            for index in range(len(robot_slice))
            if self._is_moving_step(robot_slice[index], robot_slice[index - 1] if index > 0 else None)
        ]
        if not moving_indices:
            return [], len(robot_slice), 0

        first_moving = moving_indices[0]
        last_moving = moving_indices[-1]
        trimmed = robot_slice[first_moving : last_moving + 1]
        removed_leading = first_moving
        removed_trailing = len(robot_slice) - 1 - last_moving
        return trimmed, removed_leading, removed_trailing

    def _state_vector(self, row: dict[str, Any]) -> np.ndarray:
        robot_state = row["robot_state"]
        joint_positions = [float(value) for value in robot_state.get("q", [])]
        gripper_width = float(robot_state.get("gripper_width", 0.0))
        state = joint_positions + [gripper_width]
        if not state:
            raise KeyError("robot_state.q was missing; cannot build observation.state")
        return np.asarray(state, dtype=np.float32)

    def _action_vector(self, row: dict[str, Any]) -> np.ndarray:
        executed_action = row["executed_action"]
        translation = [float(value) for value in executed_action.get("cartesian_delta_translation", [])]
        rotation = [float(value) for value in executed_action.get("cartesian_delta_rotation", [])]
        gripper = [float(executed_action.get("gripper_command", 0.0))]
        action = translation + rotation + gripper
        if not action:
            raise KeyError("executed_action was missing; cannot build action")
        return np.asarray(action, dtype=np.float32)

    def _camera_feature_name(self, camera_name: str) -> str:
        if self.primary_camera is not None:
            return "observation.images.top" if camera_name == self.primary_camera else f"observation.images.{camera_name}"

        default_camera = sorted(self.camera_frames)[0]
        return "observation.images.top" if camera_name == default_camera else f"observation.images.{camera_name}"

    def _match_camera_frame(self, timestamp_ns: int, frames: list[CameraFrame]) -> CameraFrame | None:
        frame_timestamps = [frame.timestamp_ns for frame in frames]
        match_index = closest_index(frame_timestamps, timestamp_ns)
        if match_index is None:
            return None
        matched = frames[match_index]
        if abs(matched.timestamp_ns - timestamp_ns) > self.camera_tolerance_ns:
            return None
        return matched

    def _dominant_episode_text(self, synced_texts: list[str], fallback: str) -> str:
        if not synced_texts:
            return fallback
        return Counter(synced_texts).most_common(1)[0][0]

    def build_episodes(self) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []

        episode_boundaries = self._episode_boundaries()
        LOGGER.info("Built %d event-based segment(s) from episode events.", len(episode_boundaries))
        if self.max_episodes is not None:
            episode_boundaries = episode_boundaries[: self.max_episodes]

        skipped_static = 0
        for episode_index, (start_idx, end_idx) in enumerate(episode_boundaries):
            robot_slice = self.robot_rows[start_idx:end_idx]
            if not robot_slice:
                continue

            trimmed_slice, removed_leading, removed_trailing = self._trim_to_motion(robot_slice)
            if not trimmed_slice:
                skipped_static += 1
                LOGGER.info(
                    "Skipping segment %d (%d steps): no arm motion detected.",
                    episode_index,
                    len(robot_slice),
                )
                continue
            robot_slice = trimmed_slice

            if self.max_steps_per_episode is not None:
                robot_slice = robot_slice[: self.max_steps_per_episode]

            if removed_leading or removed_trailing:
                LOGGER.info(
                    "Segment %d trimmed: removed %d leading and %d trailing static step(s); kept %d step(s).",
                    episode_index,
                    removed_leading,
                    removed_trailing,
                    len(robot_slice),
                )

            steps: list[dict[str, Any]] = []
            matched_frames_by_camera: dict[str, list[CameraFrame]] = {name: [] for name in self.camera_frames}

            for step_index, robot_row in enumerate(robot_slice):
                timestamp_ns = infer_timestamp_ns(robot_row)
                camera_matches: dict[str, CameraFrame] = {}

                for camera_name, frames in self.camera_frames.items():
                    matched_frame = self._match_camera_frame(timestamp_ns, frames)
                    if matched_frame is None:
                        raise ValueError(
                            f"No camera frame within {self.camera_tolerance_ns / 1e6:.1f} ms "
                            f"for episode {episode_index}, step {step_index}, camera '{camera_name}'."
                        )
                    camera_matches[camera_name] = matched_frame
                    matched_frames_by_camera[camera_name].append(matched_frame)

                steps.append(
                    {
                        "step_index": step_index,
                        "timestamp_ns": timestamp_ns,
                        "observation.state": self._state_vector(robot_row),
                        "action": self._action_vector(robot_row),
                        "camera_matches": camera_matches,
                    }
                )

            synced_text_values: list[str] = []
            text_alignment: dict[str, list[dict[str, Any]]] = {}
            for camera_name, matched_frames in matched_frames_by_camera.items():
                alignment = self.sync_text_to_frames(matched_frames, self.text_rows, self.text_tolerance_ns)
                text_alignment[camera_name] = alignment
                synced_text_values.extend(item["text"] for item in alignment if item["text"])

            output_episode_index = len(episodes)
            task = self._dominant_episode_text(
                synced_text_values,
                fallback=f"episode_{output_episode_index:06d}",
            )
            episodes.append(
                {
                    "episode_index": output_episode_index,
                    "task": task,
                    "steps": steps,
                    "text_alignment": text_alignment,
                }
            )

        LOGGER.info(
            "Episode preprocessing complete: kept %d episode(s), skipped %d static segment(s).",
            len(episodes),
            skipped_static,
        )
        return episodes

    def _infer_fps(self) -> int:
        first_camera = next(iter(self.camera_frames.values()))
        if len(first_camera) < 2:
            return 30

        deltas = [
            first_camera[index + 1].timestamp_ns - first_camera[index].timestamp_ns
            for index in range(len(first_camera) - 1)
            if first_camera[index + 1].timestamp_ns > first_camera[index].timestamp_ns
        ]
        if not deltas:
            return 30

        median_delta_ns = float(np.median(np.asarray(deltas, dtype=np.float64)))
        return max(1, int(round(1_000_000_000.0 / median_delta_ns)))

    def _dataset_features(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        first_step = episodes[0]["steps"][0]
        state_dim = int(first_step["observation.state"].shape[0])
        action_dim = int(first_step["action"].shape[0])

        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": {"motors": [f"state_{index}" for index in range(state_dim)]},
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": {"motors": [f"action_{index}" for index in range(action_dim)]},
            },
        }

        for camera_name, frames in self.camera_frames.items():
            sample = frames[0]
            features[self._camera_feature_name(camera_name)] = {
                "dtype": "video",
                "shape": (3, sample.height, sample.width),
                "names": ["channels", "height", "width"],
            }

        return features

    def _load_video_frame(self, frame: CameraFrame, readers: dict[Path, VideoReaderState]) -> np.ndarray:
        reader = readers.get(frame.video_path)
        if reader is None:
            capture = cv2.VideoCapture(str(frame.video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video file: {frame.video_path}")
            reader = VideoReaderState(capture=capture)
            readers[frame.video_path] = reader

        requested_index = frame.video_frame_index

        if reader.last_frame_index == requested_index and reader.last_frame_rgb is not None:
            return reader.last_frame_rgb.copy()

        if reader.last_frame_index is None:
            reader.capture.set(cv2.CAP_PROP_POS_FRAMES, requested_index)
        elif requested_index < reader.last_frame_index:
            reader.capture.set(cv2.CAP_PROP_POS_FRAMES, requested_index)
        elif requested_index > reader.last_frame_index + 1:
            reader.capture.set(cv2.CAP_PROP_POS_FRAMES, requested_index)

        ok, image_bgr = reader.capture.read()
        if not ok or image_bgr is None:
            raise RuntimeError(
                f"Could not read frame {requested_index} from {frame.video_path}"
            )

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        reader.last_frame_index = requested_index
        reader.last_frame_rgb = image_rgb
        return image_rgb.copy()

    def export(self) -> Path:
        if self.output_dir.exists():
            if not self.force_overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {self.output_dir}. "
                    "Pass --force to overwrite it."
                )
            shutil.rmtree(self.output_dir)

        episodes = self.build_episodes()
        if not episodes:
            raise ValueError("No episodes were built from the dataset")

        LOGGER.info(
            "Starting dataset conversion for '%s' with %d episode(s).",
            self.dataset_dir.name,
            len(episodes),
        )

        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self._infer_fps(),
            root=self.output_dir,
            robot_type="franka",
            features=self._dataset_features(episodes),
            use_videos=True,
            vcodec=self.vcodec,
        )

        readers: dict[Path, VideoReaderState] = {}
        try:
            total_episodes = len(episodes)
            for episode_number, episode in enumerate(episodes, start=1):
                LOGGER.info(
                    "Processing episode %d/%d (task=%s, steps=%d).",
                    episode_number,
                    total_episodes,
                    episode["task"],
                    len(episode["steps"]),
                )
                for step in episode["steps"]:
                    frame_dict: dict[str, Any] = {
                        "task": episode["task"],
                        "observation.state": torch.from_numpy(step["observation.state"].copy()),
                        "action": torch.from_numpy(step["action"].copy()),
                    }

                    for camera_name, matched_frame in step["camera_matches"].items():
                        frame_dict[self._camera_feature_name(camera_name)] = self._load_video_frame(
                            matched_frame,
                            readers,
                        )

                    dataset.add_frame(frame_dict)

                dataset.save_episode(parallel_encoding=False)

            dataset.finalize()
        finally:
            for reader in readers.values():
                reader.capture.release()

        LOGGER.info(
            "Finished dataset conversion for '%s'. Output written to %s.",
            self.dataset_dir.name,
            self.output_dir,
        )
        return self.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw recording directory into a native LeRobotDataset v3 export."
    )
    parser.add_argument("dataset_name", help="Dataset directory name under --datasets-root.")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help="Directory containing raw datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where LeRobotDataset v3 exports will be created.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobot repo_id to store in dataset metadata. Defaults to local/<dataset_name>.",
    )
    parser.add_argument(
        "--primary-camera",
        type=str,
        default=None,
        help="Camera name to map to observation.images.top.",
    )
    parser.add_argument(
        "--camera-tolerance-ms",
        type=float,
        default=DEFAULT_CAMERA_TOLERANCE_NS / 1_000_000.0,
        help="Maximum robot-to-camera sync error in milliseconds.",
    )
    parser.add_argument(
        "--text-tolerance-ms",
        type=float,
        default=DEFAULT_TEXT_TOLERANCE_NS / 1_000_000.0,
        help="Maximum text-to-frame sync error in milliseconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing output directory if it already exists.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default=DEFAULT_VCODEC,
        help="Video codec used by LeRobotDataset.create(). Defaults to h264 for faster local exports.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit for the number of episodes to export.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=None,
        help="Optional limit for the number of steps exported from each episode.",
    )
    parser.add_argument(
        "--joint-motion-threshold",
        type=float,
        default=DEFAULT_JOINT_MOTION_THRESHOLD,
        help="Maximum joint delta considered stationary between consecutive steps.",
    )
    parser.add_argument(
        "--gripper-motion-threshold",
        type=float,
        default=DEFAULT_GRIPPER_MOTION_THRESHOLD,
        help="Maximum gripper-width delta considered stationary between consecutive steps.",
    )
    parser.add_argument(
        "--action-translation-threshold",
        type=float,
        default=DEFAULT_ACTION_TRANSLATION_THRESHOLD,
        help="Minimum executed_action translation norm considered movement.",
    )
    parser.add_argument(
        "--action-rotation-threshold",
        type=float,
        default=DEFAULT_ACTION_ROTATION_THRESHOLD,
        help="Minimum executed_action rotation norm considered movement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = args.datasets_root / args.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset '{args.dataset_name}' was not found at {dataset_dir}")

    output_dir = args.output_root / args.dataset_name
    repo_id = args.repo_id or f"{DEFAULT_REPO_OWNER}/{args.dataset_name}"

    converter = SmolVLADatasetConverter(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        repo_id=repo_id,
        primary_camera=args.primary_camera,
        camera_tolerance_ns=int(args.camera_tolerance_ms * 1_000_000.0),
        text_tolerance_ns=int(args.text_tolerance_ms * 1_000_000.0),
        force_overwrite=args.force,
        vcodec=args.vcodec,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        joint_motion_threshold=args.joint_motion_threshold,
        gripper_motion_threshold=args.gripper_motion_threshold,
        action_translation_threshold=args.action_translation_threshold,
        action_rotation_threshold=args.action_rotation_threshold,
    )
    LOGGER.info("Starting converter.")
    export_dir = converter.export()
    LOGGER.info("Converter finished.")
    print(f"LeRobotDataset v3 written to: {export_dir}")


if __name__ == "__main__":
    main()
