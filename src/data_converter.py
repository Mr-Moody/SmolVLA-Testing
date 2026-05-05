"""
Convert a cleaned dataset into a native LeRobotDataset v3 export.

Expected input layout (produced by data_cleaner.py):
    cleaned_datasets/<dataset_name>/
        session_metadata.json
        robot.jsonl
        episode_events.jsonl
        text.jsonl / language.jsonl  (optional)
        cameras/<camera_name>/
            frames.jsonl
            rgb.mp4

Run data_cleaner.py on raw_datasets first, then run this script on the output.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))

from dataset_utils import (
    CameraFrame,
    TextRecord,
    closest_index,
    find_episode_boundaries,
    infer_timestamp_ns,
    load_annotations,
    load_camera_frames,
    load_subtasks,
    load_text_rows,
    load_trims,
    read_json,
    read_jsonl,
)

DEFAULT_DATASETS_ROOT = Path("cleaned_datasets")
DEFAULT_OUTPUT_ROOT = Path("lerobot_datasets")
DEFAULT_CAMERA_TOLERANCE_NS = 150_000_000
DEFAULT_TEXT_TOLERANCE_NS = 2_000_000_000
DEFAULT_REPO_OWNER = "local"
DEFAULT_VCODEC = "h264"
# ~2s at 30fps per camera; matches lerobot-record / streaming encoding guide default.
DEFAULT_ENCODER_QUEUE_MAXSIZE = 60
DEFAULT_BLANK_MAX_STEPS = 1000
DEFAULT_MIN_GRIPPER_COMMAND = 0.1
DEFAULT_MIN_GRIPPER_WIDTH_SPAN = 0.002

# Natural-language task strings used per-step when subtask phases are annotated.
# Each key matches the phase id saved by the labeler; values are the task strings
# written into the LeRobotDataset frame and used for SmolVLA language conditioning.
SUBTASK_PHASE_TASKS: dict[str, str] = {
    "approach_can":  "Move the gripper to the soup can",
    "grasp_can":     "Close the gripper around the soup can",
    "lift_can":      "Lift the soup can off the table",
    "move_to_tray":  "Carry the soup can to the green tray",
    "release_can":   "Place the soup can into the tray",
    "retract_arm":   "Move the arm away from the tray",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class VideoReaderState:
    capture: cv2.VideoCapture
    last_frame_index: int | None = None
    last_frame_rgb: np.ndarray | None = None


@dataclass(frozen=True)
class EpisodeQuality:
    step_count: int
    duration_s: float
    max_abs_gripper_command: float
    gripper_width_span: float
    has_gripper_activity: bool
    blank_candidate: bool
    blank_reason: str | None


class SmolVLADatasetConverter:
    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        repo_id: str,
        primary_camera: str | None = None,
        cameras: list[str] | None = None,
        camera_tolerance_ns: int = DEFAULT_CAMERA_TOLERANCE_NS,
        text_tolerance_ns: int = DEFAULT_TEXT_TOLERANCE_NS,
        force_overwrite: bool = False,
        vcodec: str = DEFAULT_VCODEC,
        encoder_threads: int | None = None,
        encoder_queue_maxsize: int = DEFAULT_ENCODER_QUEUE_MAXSIZE,
        max_episodes: int | None = None,
        max_steps_per_episode: int | None = None,
        suppress_blank_episodes: bool = True,
        blank_max_steps: int = DEFAULT_BLANK_MAX_STEPS,
        min_gripper_command: float = DEFAULT_MIN_GRIPPER_COMMAND,
        min_gripper_width_span: float = DEFAULT_MIN_GRIPPER_WIDTH_SPAN,
        episode_report_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.repo_id = repo_id
        self.primary_camera = primary_camera
        self.cameras = cameras
        self.camera_tolerance_ns = camera_tolerance_ns
        self.text_tolerance_ns = text_tolerance_ns
        self.force_overwrite = force_overwrite
        self.vcodec = vcodec
        self.encoder_threads = encoder_threads
        self.encoder_queue_maxsize = encoder_queue_maxsize
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.suppress_blank_episodes = suppress_blank_episodes
        self.blank_max_steps = blank_max_steps
        self.min_gripper_command = min_gripper_command
        self.min_gripper_width_span = min_gripper_width_span
        self.episode_report_path = episode_report_path

        self._device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        LOGGER.info("Device: %s", self._device)
        LOGGER.info("Loading dataset: %s", dataset_dir)

        self.session_metadata = read_json(dataset_dir / "session_metadata.json")
        self.robot_rows = read_jsonl(dataset_dir / "robot.jsonl")
        self.episode_rows = read_jsonl(dataset_dir / "episode_events.jsonl")
        self.text_rows = load_text_rows(dataset_dir)
        self._annotations = load_annotations(dataset_dir)
        self._trims = load_trims(dataset_dir)
        self._subtasks = load_subtasks(dataset_dir)
        self.camera_frames = load_camera_frames(dataset_dir)
        if self.cameras is not None:
            unknown = set(self.cameras) - set(self.camera_frames)
            if unknown:
                raise ValueError(f"--cameras specified unknown camera(s): {sorted(unknown)}. Available: {sorted(self.camera_frames)}")
            self.camera_frames = {name: self.camera_frames[name] for name in self.cameras if name in self.camera_frames}
        self._camera_timestamps: dict[str, list[int]] = {
            name: [frame.timestamp_ns for frame in frames]
            for name, frames in self.camera_frames.items()
        }
        self._camera_video_seconds: dict[str, list[float]] = {
            name: self._camera_video_timeline_seconds(frames)
            for name, frames in self.camera_frames.items()
        }
        self._feature_name_cache: dict[str, str] = {
            name: self._camera_feature_name(name) for name in self.camera_frames
        }

        LOGGER.info(
            "Loaded %d robot row(s), %d camera(s): %s",
            len(self.robot_rows),
            len(self.camera_frames),
            ", ".join(self.camera_frames),
        )
        self._camera_ts_tensors: dict[str, torch.Tensor] = {
            name: torch.tensor(ts, dtype=torch.int64, device=self._device)
            for name, ts in self._camera_timestamps.items()
        }

    @staticmethod
    def sync_text_to_frames(
        frames: list[CameraFrame],
        text_records: list[TextRecord],
        max_delta_ns: int = DEFAULT_TEXT_TOLERANCE_NS,
    ) -> list[dict[str, Any]]:
        if not frames:
            return []
        text_timestamps = [r.timestamp_ns for r in text_records]
        aligned: list[dict[str, Any]] = []
        for frame in frames:
            match_index = closest_index(text_timestamps, frame.timestamp_ns)
            text: str | None = None
            delta_ns: int | None = None
            if match_index is not None:
                record = text_records[match_index]
                candidate_delta = abs(record.timestamp_ns - frame.timestamp_ns)
                if candidate_delta <= max_delta_ns:
                    text = record.text
                    delta_ns = candidate_delta
            aligned.append({
                "camera_name": frame.camera_name,
                "frame_index": frame.frame_index,
                "frame_timestamp_ns": frame.timestamp_ns,
                "text": text,
                "text_time_delta_ns": delta_ns,
            })
        return aligned

    def _camera_feature_name(self, camera_name: str) -> str:
        if self.primary_camera is not None:
            return "observation.images.top" if camera_name == self.primary_camera else f"observation.images.{camera_name}"
        default_camera = sorted(self.camera_frames)[0]
        return "observation.images.top" if camera_name == default_camera else f"observation.images.{camera_name}"

    @staticmethod
    def _camera_video_timeline_seconds(frames: list[CameraFrame]) -> list[float]:
        if not frames:
            return []
        if len(frames) < 2:
            fps = 30
        else:
            deltas = [
                frames[i + 1].timestamp_ns - frames[i].timestamp_ns
                for i in range(len(frames) - 1)
                if frames[i + 1].timestamp_ns > frames[i].timestamp_ns
            ]
            fps = max(1, round(1_000_000_000.0 / float(np.median(np.asarray(deltas, dtype=np.float64))))) if deltas else 30
        return [frame.video_frame_index / fps for frame in frames]

    def _trim_camera_name(self) -> str:
        if self.primary_camera is not None and self.primary_camera in self.camera_frames:
            return self.primary_camera
        return sorted(self.camera_frames)[0]

    def _timestamp_for_video_time(self, camera_name: str, video_time_s: float) -> int | None:
        video_seconds = self._camera_video_seconds.get(camera_name, [])
        frames = self.camera_frames.get(camera_name, [])
        if not video_seconds or not frames:
            return None

        insert_at = np.searchsorted(np.asarray(video_seconds, dtype=np.float64), float(video_time_s), side="left")
        candidate_indices: list[int] = []
        if insert_at < len(video_seconds):
            candidate_indices.append(int(insert_at))
        if insert_at > 0:
            candidate_indices.append(int(insert_at - 1))
        if not candidate_indices:
            return None

        best_index = min(candidate_indices, key=lambda idx: abs(video_seconds[idx] - float(video_time_s)))
        return int(frames[best_index].timestamp_ns)

    def _apply_episode_trim(
        self,
        episode_index: int,
        robot_slice: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        trim = self._trims.get(episode_index)
        if not trim or not robot_slice:
            return robot_slice

        camera_name = self._trim_camera_name()
        trim_start_s = float(trim.get("trim_start_s", 0.0))
        trim_end_s = float(trim.get("trim_end_s", 0.0))
        if trim_end_s <= trim_start_s:
            LOGGER.warning(
                "Ignoring invalid trim for episode %d: start=%.4fs end=%.4fs",
                episode_index,
                trim_start_s,
                trim_end_s,
            )
            return robot_slice

        start_ts = self._timestamp_for_video_time(camera_name, trim_start_s)
        end_ts = self._timestamp_for_video_time(camera_name, trim_end_s)
        if start_ts is None or end_ts is None:
            LOGGER.warning("Could not resolve trim timestamps for episode %d on camera '%s'", episode_index, camera_name)
            return robot_slice

        trimmed_rows = [
            row for row in robot_slice
            if start_ts <= infer_timestamp_ns(row) <= end_ts
        ]
        if not trimmed_rows:
            LOGGER.warning(
                "Trim removed every step from episode %d; keeping original episode bounds",
                episode_index,
            )
            return robot_slice
        LOGGER.info(
            "Episode %d trimmed: %d → %d steps (%.3f s – %.3f s)",
            episode_index, len(robot_slice), len(trimmed_rows), trim_start_s, trim_end_s,
        )
        return trimmed_rows

    def _resolve_subtask_phases(
        self,
        episode_index: int,
    ) -> list[dict[str, Any]] | None:
        """Convert subtask video-second boundaries to robot-timestamp ranges.

        Returns a list of phase dicts with keys: phase, start_ns, end_ns, start_s, end_s.
        Returns None if no subtask data exists or timestamps cannot be resolved.
        """
        phases = self._subtasks.get(episode_index)
        if not phases:
            return None
        camera_name = self._trim_camera_name()
        resolved: list[dict[str, Any]] = []
        for phase in phases:
            start_ts = self._timestamp_for_video_time(camera_name, float(phase["start_s"]))
            end_ts = self._timestamp_for_video_time(camera_name, float(phase["end_s"]))
            if start_ts is None or end_ts is None:
                LOGGER.warning(
                    "Could not resolve subtask timestamps for episode %d phase '%s'; skipping subtasks",
                    episode_index, phase.get("phase", "?"),
                )
                return None
            resolved.append({
                "phase": phase.get("phase", ""),
                "start_ns": start_ts,
                "end_ns": end_ts,
                "start_s": float(phase["start_s"]),
                "end_s": float(phase["end_s"]),
            })
        return resolved

    @staticmethod
    def _subtask_for_step(
        timestamp_ns: int,
        resolved_phases: list[dict[str, Any]],
    ) -> str:
        """Return the phase ID for a step timestamp, assigning the closest phase if between boundaries."""
        for phase in resolved_phases:
            if phase["start_ns"] <= timestamp_ns <= phase["end_ns"]:
                return phase["phase"]
        # Assign to closest phase when timestamp falls in a small inter-phase gap.
        return min(
            resolved_phases,
            key=lambda p: min(abs(p["start_ns"] - timestamp_ns), abs(p["end_ns"] - timestamp_ns)),
        )["phase"]

    def _match_camera_frames_batch(
        self, timestamps_ns: list[int], camera_name: str
    ) -> list[CameraFrame | None]:
        """Return the nearest CameraFrame (or None if outside tolerance) for each timestamp."""
        frames = self.camera_frames[camera_name]
        M = len(frames)
        if M == 0:
            return [None] * len(timestamps_ns)

        cam_ts = self._camera_ts_tensors[camera_name]
        robot_ts = torch.tensor(timestamps_ns, dtype=torch.int64, device=self._device)

        idx = torch.searchsorted(cam_ts, robot_ts)
        left_idx = (idx - 1).clamp(min=0)
        right_idx = idx.clamp(max=M - 1)

        left_delta = torch.where(
            idx > 0,
            (cam_ts[left_idx] - robot_ts).abs(),
            torch.full_like(robot_ts, 10 ** 18),
        )
        right_delta = torch.where(
            idx < M,
            (cam_ts[right_idx] - robot_ts).abs(),
            torch.full_like(robot_ts, 10 ** 18),
        )

        use_right = right_delta < left_delta
        best_idx = torch.where(use_right, right_idx, left_idx)
        best_delta = torch.minimum(left_delta, right_delta)

        best_idx_list = best_idx.cpu().tolist()
        valid_list = (best_delta <= self.camera_tolerance_ns).cpu().tolist()
        return [frames[i] if v else None for i, v in zip(best_idx_list, valid_list)]

    def _dominant_episode_text(self, synced_texts: list[str], fallback: str) -> str:
        if not synced_texts:
            return fallback
        return Counter(synced_texts).most_common(1)[0][0]

    def _state_vector(self, row: dict[str, Any]) -> np.ndarray:
        rs = row["robot_state"]
        q   = [float(v) for v in rs.get("q", [])]
        gripper    = [float(rs.get("gripper_width", 0.0))]
        if not q:
            raise KeyError("robot_state.q was missing; cannot build observation.state")
        state = q + gripper
        return np.asarray(state, dtype=np.float32)

    def _state_dimension_names(self) -> list[str]:
        """Build named dimension list from the first robot row's actual field sizes."""
        rs = self.robot_rows[0]["robot_state"]
        names: list[str] = []
        names += [f"q_{i}"  for i in range(len(rs.get("q",  [])))]
        names += ["gripper_width"]
        return names

    def _action_dimension_names(self) -> list[str]:
        """Build action dimension names for commanded joint targets plus gripper command."""
        rs = self.robot_rows[0]["robot_state"]
        names: list[str] = []
        names += [f"q_cmd_{i}" for i in range(len(rs.get("q_cmd", [])))]
        names += ["gripper_command"]
        return names

    def _warn_problematic_feature_std(self, episodes: list[dict[str, Any]]) -> None:
        """Warn when per-dataset feature std is zero/near-zero to flag normalization risks."""
        state_names = self._state_dimension_names()
        action_names = self._action_dimension_names()

        state_rows = [
            step["observation.state"]
            for episode in episodes
            for step in episode["steps"]
        ]
        action_rows = [
            step["action"]
            for episode in episodes
            for step in episode["steps"]
        ]
        if not state_rows or not action_rows:
            return

        state_std = np.std(np.stack(state_rows, axis=0), axis=0)
        action_std = np.std(np.stack(action_rows, axis=0), axis=0)

        near_zero_state_indices = np.where(state_std <= 1e-6)[0].tolist()
        if near_zero_state_indices:
            details = ", ".join(
                f"{state_names[idx]} (std={float(state_std[idx]):.3e})"
                for idx in near_zero_state_indices
            )
            LOGGER.warning(
                "State dimension(s) with near-zero std in dataset '%s': %s. "
                "This can amplify MEAN_STD normalization on single-dataset runs; "
                "cross-dataset stat aggregation is recommended.",
                self.dataset_dir.name,
                details,
            )

        zero_action_indices = np.where(action_std <= 1e-12)[0].tolist()
        if zero_action_indices:
            details = ", ".join(
                f"{action_names[idx]} (std={float(action_std[idx]):.3e})"
                for idx in zero_action_indices
            )
            LOGGER.warning(
                "Action dimension(s) with zero std in dataset '%s': %s. "
                "Keeping these action dimensions (expected for some subsets); "
                "cross-dataset stats typically restore non-zero variance.",
                self.dataset_dir.name,
                details,
            )

    def _action_vector(self, row: dict[str, Any], _next_row: dict[str, Any]) -> np.ndarray:
        q_cmd = [float(v) for v in row["robot_state"].get("q_cmd", [])]
        if not q_cmd:
            raise KeyError("robot_state.q_cmd was missing; cannot build action")
        gripper = [float(row.get("executed_action", {}).get("gripper_command", 0.0))]
        action = q_cmd + gripper
        return np.asarray(action, dtype=np.float32)

    def _episode_quality(self, steps: list[dict[str, Any]]) -> EpisodeQuality:
        if not steps:
            return EpisodeQuality(
                step_count=0,
                duration_s=0.0,
                max_abs_gripper_command=0.0,
                gripper_width_span=0.0,
                has_gripper_activity=False,
                blank_candidate=True,
                blank_reason="empty_episode",
            )

        timestamps = [int(step["timestamp_ns"]) for step in steps]
        duration_s = (timestamps[-1] - timestamps[0]) / 1_000_000_000.0 if len(timestamps) > 1 else 0.0

        gripper_commands = [float(step["action"][-1]) for step in steps]
        max_abs_gripper_command = max(abs(value) for value in gripper_commands)

        gripper_widths = [float(step["observation.state"][-1]) for step in steps]
        gripper_width_span = max(gripper_widths) - min(gripper_widths)

        has_gripper_activity = (
            max_abs_gripper_command >= self.min_gripper_command
            or gripper_width_span >= self.min_gripper_width_span
        )

        is_short_episode = len(steps) <= self.blank_max_steps
        blank_candidate = (not has_gripper_activity) and is_short_episode
        blank_reason = None
        if blank_candidate:
            blank_reason = (
                "no_gripper_activity_and_short_episode"
                f"(steps<={self.blank_max_steps})"
            )

        return EpisodeQuality(
            step_count=len(steps),
            duration_s=duration_s,
            max_abs_gripper_command=max_abs_gripper_command,
            gripper_width_span=gripper_width_span,
            has_gripper_activity=has_gripper_activity,
            blank_candidate=blank_candidate,
            blank_reason=blank_reason,
        )

    def _filter_episodes(self, episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.suppress_blank_episodes:
            return episodes, []

        kept: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        for episode in episodes:
            quality = episode["quality"]
            if quality.blank_candidate:
                suppressed.append(episode)
            else:
                kept.append(episode)
        return kept, suppressed

    def _write_episode_report(
        self,
        all_episodes: list[dict[str, Any]],
        kept_episodes: list[dict[str, Any]],
        suppressed_episodes: list[dict[str, Any]],
    ) -> None:
        if self.episode_report_path is None:
            return

        self.episode_report_path.parent.mkdir(parents=True, exist_ok=True)
        kept_indices = {episode["episode_index"] for episode in kept_episodes}

        payload = {
            "dataset": self.dataset_dir.name,
            "thresholds": {
                "blank_max_steps": self.blank_max_steps,
                "min_gripper_command": self.min_gripper_command,
                "min_gripper_width_span": self.min_gripper_width_span,
            },
            "summary": {
                "total_detected_episodes": len(all_episodes),
                "kept_episodes": len(kept_episodes),
                "suppressed_episodes": len(suppressed_episodes),
            },
            "episodes": [
                {
                    "episode_index": episode["episode_index"],
                    "task": episode["task"],
                    "step_count": episode["quality"].step_count,
                    "duration_s": episode["quality"].duration_s,
                    "max_abs_gripper_command": episode["quality"].max_abs_gripper_command,
                    "gripper_width_span": episode["quality"].gripper_width_span,
                    "has_gripper_activity": episode["quality"].has_gripper_activity,
                    "blank_candidate": episode["quality"].blank_candidate,
                    "blank_reason": episode["quality"].blank_reason,
                    "suppressed": episode["episode_index"] not in kept_indices,
                }
                for episode in all_episodes
            ],
        }

        with self.episode_report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        LOGGER.info("Wrote episode report to %s", self.episode_report_path)

    def _infer_fps(self) -> int:
        first_camera = next(iter(self.camera_frames.values()))
        if len(first_camera) < 2:
            return 30
        deltas = [
            first_camera[i + 1].timestamp_ns - first_camera[i].timestamp_ns
            for i in range(len(first_camera) - 1)
            if first_camera[i + 1].timestamp_ns > first_camera[i].timestamp_ns
        ]
        if not deltas:
            return 30
        return max(1, int(round(1_000_000_000.0 / float(np.median(np.asarray(deltas, dtype=np.float64))))))

    def _dataset_features(self, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        first_step = episodes[0]["steps"][0]
        state_dim = int(first_step["observation.state"].shape[0])
        action_dim = int(first_step["action"].shape[0])
        state_names = self._state_dimension_names()
        action_names = self._action_dimension_names()

        if len(state_names) != state_dim:
            raise ValueError(
                f"State feature name count ({len(state_names)}) does not match state dim ({state_dim})."
            )
        if len(action_names) != action_dim:
            raise ValueError(
                f"Action feature name count ({len(action_names)}) does not match action dim ({action_dim})."
            )

        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": {"motors": state_names},
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": {"motors": action_names},
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

        if (
            reader.last_frame_index is None
            or requested_index < reader.last_frame_index
            or requested_index > reader.last_frame_index + 1
        ):
            reader.capture.set(cv2.CAP_PROP_POS_FRAMES, requested_index)

        ok, image_bgr = reader.capture.read()
        if not ok or image_bgr is None:
            raise RuntimeError(f"Could not read frame {requested_index} from {frame.video_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        reader.last_frame_index = requested_index
        reader.last_frame_rgb = image_rgb
        return image_rgb.copy()

    def build_episodes(self) -> list[dict[str, Any]]:
        episodes: list[dict[str, Any]] = []
        boundaries = find_episode_boundaries(self.robot_rows, self.episode_rows)
        LOGGER.info("Found %d episode(s).", len(boundaries))
        if self.max_episodes is not None:
            boundaries = boundaries[: self.max_episodes]

        for episode_index, (start_idx, end_idx) in enumerate(boundaries):
            robot_slice = self.robot_rows[start_idx:end_idx]
            if not robot_slice:
                continue
            robot_slice = self._apply_episode_trim(episode_index, robot_slice)
            if self.max_steps_per_episode is not None:
                robot_slice = robot_slice[: self.max_steps_per_episode]
            if len(robot_slice) < 2:
                LOGGER.warning(
                    "Skipping episode %d: need at least 2 robot rows to build action targets, got %d.",
                    episode_index,
                    len(robot_slice),
                )
                continue

            step_rows = robot_slice[:-1]
            next_rows = robot_slice[1:]
            step_timestamps = [infer_timestamp_ns(row) for row in step_rows]

            # Batch-match all camera frames for this episode in one GPU pass per camera.
            camera_frame_matches: dict[str, list[CameraFrame | None]] = {
                cam: self._match_camera_frames_batch(step_timestamps, cam)
                for cam in self.camera_frames
            }

            steps: list[dict[str, Any]] = []
            matched_frames_by_camera: dict[str, list[CameraFrame]] = {name: [] for name in self.camera_frames}

            for step_index, (robot_row, next_row) in enumerate(zip(step_rows, next_rows, strict=True)):
                camera_matches: dict[str, CameraFrame] = {}
                for camera_name in self.camera_frames:
                    matched_frame = camera_frame_matches[camera_name][step_index]
                    if matched_frame is None:
                        raise ValueError(
                            f"No camera frame within {self.camera_tolerance_ns / 1e6:.1f} ms "
                            f"for episode {episode_index}, step {step_index}, camera '{camera_name}'."
                        )
                    camera_matches[camera_name] = matched_frame
                    matched_frames_by_camera[camera_name].append(matched_frame)

                steps.append({
                    "step_index": step_index,
                    "timestamp_ns": step_timestamps[step_index],
                    "observation.state": self._state_vector(robot_row),
                    "action": self._action_vector(robot_row, next_row),
                    "camera_matches": camera_matches,
                })

            synced_text_values: list[str] = []
            text_alignment: dict[str, list[dict[str, Any]]] = {}
            for camera_name, matched_frames in matched_frames_by_camera.items():
                alignment = self.sync_text_to_frames(matched_frames, self.text_rows, self.text_tolerance_ns)
                text_alignment[camera_name] = alignment
                synced_text_values.extend(item["text"] for item in alignment if item["text"])

            output_index = len(episodes)
            # Use episode_index (boundary position) so annotation/trim/subtask lookups
            # stay consistent when earlier episodes have empty robot slices and are skipped.
            task = self._annotations.get(
                episode_index,
                self._dominant_episode_text(synced_text_values, fallback=f"episode_{output_index:06d}"),
            )

            resolved_phases = self._resolve_subtask_phases(episode_index)
            if resolved_phases:
                ep_start_ns = step_timestamps[0]
                for step in steps:
                    step["subtask"] = self._subtask_for_step(step["timestamp_ns"], resolved_phases)
                # Rebase phase times to be episode-relative (seconds from first step).
                subtask_phases_output = [
                    {
                        "phase": p["phase"],
                        "start_s": round(max(0.0, (p["start_ns"] - ep_start_ns) / 1e9), 4),
                        "end_s": round(max(0.0, (p["end_ns"] - ep_start_ns) / 1e9), 4),
                    }
                    for p in resolved_phases
                ]
            else:
                subtask_phases_output = None

            quality = self._episode_quality(steps)
            episodes.append({
                "episode_index": output_index,
                "task": task,
                "steps": steps,
                "text_alignment": text_alignment,
                "quality": quality,
                "subtask_phases": subtask_phases_output,
            })

        LOGGER.info("Built %d episode(s).", len(episodes))
        return episodes

    def export(self) -> Path:
        if self.output_dir.exists():
            if not self.force_overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {self.output_dir}. Pass --force to overwrite it."
                )
            shutil.rmtree(self.output_dir)

        all_episodes = self.build_episodes()
        if not all_episodes:
            raise ValueError("No episodes were built from the dataset.")

        episodes, suppressed_episodes = self._filter_episodes(all_episodes)
        if not episodes:
            raise ValueError(
                "All episodes were classified as blank and suppressed. "
                "Try relaxing --blank-max-steps or disabling suppression with --keep-blank-episodes."
            )

        self._write_episode_report(all_episodes, episodes, suppressed_episodes)
        self._warn_problematic_feature_std(episodes)

        if suppressed_episodes:
            suppressed_indices = [episode["episode_index"] for episode in suppressed_episodes]
            tqdm.write(f"Suppressed {len(suppressed_episodes)} blank episode(s): {suppressed_indices}")
        tqdm.write(
            f"Episode selection: detected={len(all_episodes)}, kept={len(episodes)}, suppressed={len(suppressed_episodes)}."
        )

        tqdm.write(f"Device:  {self._device}")
        tqdm.write(f"Dataset: {self.dataset_dir}")
        tqdm.write(f"Output:  {self.output_dir}")
        tqdm.write(f"Starting conversion: {len(episodes)} episode(s).")
        tqdm.write(
            f"Streaming encode: vcodec={self.vcodec}, "
            f"encoder_threads={self.encoder_threads!r}, "
            f"encoder_queue_maxsize={self.encoder_queue_maxsize}"
        )

        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self._infer_fps(),
            root=self.output_dir,
            robot_type="franka",
            features=self._dataset_features(episodes),
            use_videos=True,
            vcodec=self.vcodec,
            streaming_encoding=True,
            encoder_queue_maxsize=self.encoder_queue_maxsize,
            encoder_threads=self.encoder_threads,
        )

        readers: dict[Path, VideoReaderState] = {}
        try:
            total = len(episodes)
            pbar = tqdm(episodes, desc="Converting", unit="ep", dynamic_ncols=True)
            for episode_number, episode in enumerate(pbar, start=1):
                ep_start = time.monotonic()
                pbar.set_postfix({"ep": f"{episode_number}/{total}", "steps": len(episode["steps"])})
                has_subtasks = bool(episode.get("subtask_phases"))
                if has_subtasks:
                    LOGGER.info(
                        "Episode %d: writing per-step subtask tasks (%d phases)",
                        episode["episode_index"], len(episode["subtask_phases"]),
                    )
                for step in episode["steps"]:
                    subtask_id: str | None = step.get("subtask") if has_subtasks else None
                    step_task = (
                        SUBTASK_PHASE_TASKS.get(subtask_id, episode["task"])
                        if subtask_id
                        else episode["task"]
                    )
                    frame_dict: dict[str, Any] = {
                        "task": step_task,
                        "observation.state": torch.from_numpy(step["observation.state"].copy()),
                        "action": torch.from_numpy(step["action"].copy()),
                    }
                    for camera_name, matched_frame in step["camera_matches"].items():
                        frame_dict[self._feature_name_cache[camera_name]] = self._load_video_frame(
                            matched_frame, readers
                        )
                    dataset.add_frame(frame_dict)
                dataset.save_episode(parallel_encoding=True)
                elapsed = time.monotonic() - ep_start
                task_desc = f"subtask phases" if has_subtasks else f"task: {episode['task']}"
                tqdm.write(
                    f"  Episode {episode_number}/{total} done in {elapsed:.1f}s"
                    f" — {len(episode['steps'])} steps, {task_desc}"
                )
            dataset.finalize()
        finally:
            for reader in readers.values():
                reader.capture.release()

        subtask_rows = [
            {"episode_index": ep["episode_index"], "phases": ep["subtask_phases"]}
            for ep in episodes
            if ep.get("subtask_phases")
        ]
        if subtask_rows:
            subtasks_path = self.output_dir / "subtasks.jsonl"
            with subtasks_path.open("w", encoding="utf-8") as f:
                for row in subtask_rows:
                    f.write(json.dumps(row) + "\n")
            LOGGER.info(
                "Wrote subtask phase metadata for %d/%d episode(s) to %s",
                len(subtask_rows), len(episodes), subtasks_path,
            )
        else:
            LOGGER.info("No subtask annotations found; subtasks.jsonl not written.")

        tqdm.write(f"Conversion complete. Output: {self.output_dir}")
        return self.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a cleaned dataset directory into a LeRobotDataset v3 export."
    )
    parser.add_argument("dataset_name", help="Dataset name under --datasets-root.")
    parser.add_argument(
        "--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT,
        help="Root containing cleaned datasets (default: cleaned_datasets).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--repo-id", type=str, default=None,
        help="LeRobot repo_id. Defaults to local/<dataset_name>.",
    )
    parser.add_argument("--primary-camera", type=str, default=None,
                        help="Camera name to map to observation.images.top.")
    parser.add_argument(
        "--cameras", type=str, default=None,
        help="Comma-separated list of camera names to include. Omit to use all cameras.",
    )
    parser.add_argument("--camera-tolerance-ms", type=float,
                        default=DEFAULT_CAMERA_TOLERANCE_NS / 1_000_000.0,
                        help="Maximum robot-to-camera sync error in milliseconds.")
    parser.add_argument("--text-tolerance-ms", type=float,
                        default=DEFAULT_TEXT_TOLERANCE_NS / 1_000_000.0,
                        help="Maximum text-to-frame sync error in milliseconds.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory.")
    parser.add_argument("--vcodec", type=str, default=DEFAULT_VCODEC,
                        help="Video codec for LeRobotDataset.create().")
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Threads per encoder instance (streaming encode). Omit for codec default.",
    )
    parser.add_argument(
        "--encoder-queue-maxsize",
        type=int,
        default=DEFAULT_ENCODER_QUEUE_MAXSIZE,
        help="Max buffered frames per camera during streaming encode (default: %(default)s).",
    )
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes to export.")
    parser.add_argument("--max-steps-per-episode", type=int, default=None,
                        help="Limit steps exported per episode.")
    parser.add_argument(
        "--keep-blank-episodes",
        action="store_true",
        help="Keep short episodes with no gripper activity instead of suppressing them.",
    )
    parser.add_argument(
        "--blank-max-steps",
        type=int,
        default=DEFAULT_BLANK_MAX_STEPS,
        help="Maximum episode length considered for blank suppression when no gripper activity is detected.",
    )
    parser.add_argument(
        "--min-gripper-command",
        type=float,
        default=DEFAULT_MIN_GRIPPER_COMMAND,
        help="Minimum absolute gripper command considered an active gripper event.",
    )
    parser.add_argument(
        "--min-gripper-width-span",
        type=float,
        default=DEFAULT_MIN_GRIPPER_WIDTH_SPAN,
        help="Minimum gripper width span (meters) considered an active gripper event.",
    )
    parser.add_argument(
        "--episode-report",
        type=Path,
        default=None,
        help="Optional JSON path for episode classification and suppression report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use (e.g. cuda, cpu). Defaults to cuda if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.datasets_root / args.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset '{args.dataset_name}' not found at {dataset_dir}")
    output_dir = args.output_root / args.dataset_name
    repo_id = args.repo_id or f"{DEFAULT_REPO_OWNER}/{args.dataset_name}"
    cameras = [c.strip() for c in args.cameras.split(",")] if args.cameras else None
    converter = SmolVLADatasetConverter(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        repo_id=repo_id,
        primary_camera=args.primary_camera,
        cameras=cameras,
        camera_tolerance_ns=int(args.camera_tolerance_ms * 1_000_000.0),
        text_tolerance_ns=int(args.text_tolerance_ms * 1_000_000.0),
        force_overwrite=args.force,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        encoder_queue_maxsize=args.encoder_queue_maxsize,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        suppress_blank_episodes=not args.keep_blank_episodes,
        blank_max_steps=args.blank_max_steps,
        min_gripper_command=args.min_gripper_command,
        min_gripper_width_span=args.min_gripper_width_span,
        episode_report_path=args.episode_report,
        device=args.device,
    )
    converter.export()


if __name__ == "__main__":
    main()
