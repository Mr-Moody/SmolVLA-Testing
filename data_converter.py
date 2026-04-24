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
import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dataset_utils import (
    CameraFrame,
    TextRecord,
    closest_index,
    find_episode_boundaries,
    infer_timestamp_ns,
    load_camera_frames,
    load_text_rows,
    read_json,
    read_jsonl,
)

DEFAULT_DATASETS_ROOT = Path("cleaned_datasets")
DEFAULT_OUTPUT_ROOT = Path("lerobot_datasets")
DEFAULT_CAMERA_TOLERANCE_NS = 150_000_000
DEFAULT_TEXT_TOLERANCE_NS = 2_000_000_000
DEFAULT_REPO_OWNER = "local"
DEFAULT_VCODEC = "h264"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class VideoReaderState:
    capture: cv2.VideoCapture
    last_frame_index: int | None = None
    last_frame_rgb: np.ndarray | None = None


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

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Device: %s", self._device)
        LOGGER.info("Loading dataset: %s", dataset_dir)

        self.session_metadata = read_json(dataset_dir / "session_metadata.json")
        self.robot_rows = read_jsonl(dataset_dir / "robot.jsonl")
        self.episode_rows = read_jsonl(dataset_dir / "episode_events.jsonl")
        self.text_rows = load_text_rows(dataset_dir)
        self.camera_frames = load_camera_frames(dataset_dir)
        self._camera_timestamps: dict[str, list[int]] = {
            name: [frame.timestamp_ns for frame in frames]
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
        robot_state = row["robot_state"]
        joint_positions = [float(v) for v in robot_state.get("q", [])]
        gripper_width = float(robot_state.get("gripper_width", 0.0))
        state = joint_positions + [gripper_width]
        if not state:
            raise KeyError("robot_state.q was missing; cannot build observation.state")
        return np.asarray(state, dtype=np.float32)

    def _action_vector(self, row: dict[str, Any]) -> np.ndarray:
        executed_action = row["executed_action"]
        translation = [float(v) for v in executed_action.get("cartesian_delta_translation", [])]
        rotation = [float(v) for v in executed_action.get("cartesian_delta_rotation", [])]
        gripper = [float(executed_action.get("gripper_command", 0.0))]
        action = translation + rotation + gripper
        if not action:
            raise KeyError("executed_action was missing; cannot build action")
        return np.asarray(action, dtype=np.float32)

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
        features: dict[str, Any] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": {"motors": [f"state_{i}" for i in range(state_dim)]},
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": {"motors": [f"action_{i}" for i in range(action_dim)]},
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
            if self.max_steps_per_episode is not None:
                robot_slice = robot_slice[: self.max_steps_per_episode]

            step_timestamps = [infer_timestamp_ns(row) for row in robot_slice]

            # Batch-match all camera frames for this episode in one GPU pass per camera.
            camera_frame_matches: dict[str, list[CameraFrame | None]] = {
                cam: self._match_camera_frames_batch(step_timestamps, cam)
                for cam in self.camera_frames
            }

            steps: list[dict[str, Any]] = []
            matched_frames_by_camera: dict[str, list[CameraFrame]] = {name: [] for name in self.camera_frames}

            for step_index, robot_row in enumerate(robot_slice):
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
                    "action": self._action_vector(robot_row),
                    "camera_matches": camera_matches,
                })

            synced_text_values: list[str] = []
            text_alignment: dict[str, list[dict[str, Any]]] = {}
            for camera_name, matched_frames in matched_frames_by_camera.items():
                alignment = self.sync_text_to_frames(matched_frames, self.text_rows, self.text_tolerance_ns)
                text_alignment[camera_name] = alignment
                synced_text_values.extend(item["text"] for item in alignment if item["text"])

            output_index = len(episodes)
            task = self._dominant_episode_text(synced_text_values, fallback=f"episode_{output_index:06d}")
            episodes.append({
                "episode_index": output_index,
                "task": task,
                "steps": steps,
                "text_alignment": text_alignment,
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

        episodes = self.build_episodes()
        if not episodes:
            raise ValueError("No episodes were built from the dataset.")

        LOGGER.info(
            "Starting conversion for '%s' with %d episode(s).",
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
            total = len(episodes)
            for episode_number, episode in enumerate(episodes, start=1):
                LOGGER.info(
                    "Processing episode %d/%d (task=%s, steps=%d).",
                    episode_number, total, episode["task"], len(episode["steps"]),
                )
                for step in episode["steps"]:
                    frame_dict: dict[str, Any] = {
                        "task": episode["task"],
                        "observation.state": torch.from_numpy(step["observation.state"].copy()),
                        "action": torch.from_numpy(step["action"].copy()),
                    }
                    for camera_name, matched_frame in step["camera_matches"].items():
                        frame_dict[self._feature_name_cache[camera_name]] = self._load_video_frame(
                            matched_frame, readers
                        )
                    dataset.add_frame(frame_dict)
                dataset.save_episode(parallel_encoding=True)
            dataset.finalize()
        finally:
            for reader in readers.values():
                reader.capture.release()

        LOGGER.info("Conversion complete. Output written to %s", self.output_dir)
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
    parser.add_argument("--camera-tolerance-ms", type=float,
                        default=DEFAULT_CAMERA_TOLERANCE_NS / 1_000_000.0,
                        help="Maximum robot-to-camera sync error in milliseconds.")
    parser.add_argument("--text-tolerance-ms", type=float,
                        default=DEFAULT_TEXT_TOLERANCE_NS / 1_000_000.0,
                        help="Maximum text-to-frame sync error in milliseconds.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory.")
    parser.add_argument("--vcodec", type=str, default=DEFAULT_VCODEC,
                        help="Video codec for LeRobotDataset.create().")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes to export.")
    parser.add_argument("--max-steps-per-episode", type=int, default=None,
                        help="Limit steps exported per episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.datasets_root / args.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset '{args.dataset_name}' not found at {dataset_dir}")
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
    )
    converter.export()


if __name__ == "__main__":
    main()
