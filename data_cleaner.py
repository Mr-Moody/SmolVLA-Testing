"""
Filter a raw recorded session to motion-only, camera-covered robot steps.

Expected input layout (raw_datasets/<dataset_name>/):
    session_metadata.json
    robot.jsonl
    episode_events.jsonl
    text.jsonl / language.jsonl  (optional)
    cameras/<camera_name>/
        frames.jsonl
        rgb.mp4

Output layout (cleaned_datasets/<dataset_name>/):
    session_metadata.json        (copied)
    robot.jsonl                  (filtered to motion + camera coverage only)
    episode_events.jsonl         (regenerated episode_start events)
    text.jsonl / language.jsonl  (copied if present)
    cameras/                     (symlinked from raw source — frames are unchanged)

Run data_converter.py on the output to produce a LeRobotDataset v3 export.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from dataset_utils import (
    DEFAULT_CAMERA_TOLERANCE_NS,
    closest_index,
    find_episode_boundaries,
    first_existing_path,
    infer_timestamp_ns,
    load_camera_frames,
    read_json,
    read_jsonl,
    write_jsonl,
)

DEFAULT_DATASETS_ROOT = Path("raw_datasets")
DEFAULT_OUTPUT_ROOT = Path("cleaned_datasets")
DEFAULT_JOINT_MOTION_THRESHOLD = 5e-4
DEFAULT_GRIPPER_MOTION_THRESHOLD = 2e-4
DEFAULT_ACTION_TRANSLATION_THRESHOLD = 5e-6
DEFAULT_ACTION_ROTATION_THRESHOLD = 5e-5

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


class DatasetCleaner:
    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        camera_tolerance_ns: int = DEFAULT_CAMERA_TOLERANCE_NS,
        force_overwrite: bool = False,
        max_episodes: int | None = None,
        joint_motion_threshold: float = DEFAULT_JOINT_MOTION_THRESHOLD,
        gripper_motion_threshold: float = DEFAULT_GRIPPER_MOTION_THRESHOLD,
        action_translation_threshold: float = DEFAULT_ACTION_TRANSLATION_THRESHOLD,
        action_rotation_threshold: float = DEFAULT_ACTION_ROTATION_THRESHOLD,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.camera_tolerance_ns = camera_tolerance_ns
        self.force_overwrite = force_overwrite
        self.max_episodes = max_episodes
        self.joint_motion_threshold = float(joint_motion_threshold)
        self.gripper_motion_threshold = float(gripper_motion_threshold)
        self.action_translation_threshold = float(action_translation_threshold)
        self.action_rotation_threshold = float(action_rotation_threshold)

        self.robot_rows = read_jsonl(dataset_dir / "robot.jsonl")
        self.episode_rows = read_jsonl(dataset_dir / "episode_events.jsonl")
        self.camera_frames = load_camera_frames(dataset_dir)
        self._camera_timestamps: dict[str, list[int]] = {
            name: [frame.timestamp_ns for frame in frames]
            for name, frames in self.camera_frames.items()
        }

    # --- motion detection ---

    @staticmethod
    def _extract_joint_positions(row: dict[str, Any]) -> np.ndarray:
        values = [float(v) for v in row.get("robot_state", {}).get("q", [])]
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _extract_gripper_width(row: dict[str, Any]) -> float:
        return float(row.get("robot_state", {}).get("gripper_width", 0.0))

    @staticmethod
    def _extract_action_norms(row: dict[str, Any]) -> tuple[float, float]:
        ea = row.get("executed_action", {})
        translation = np.asarray(ea.get("cartesian_delta_translation", []), dtype=np.float64)
        rotation = np.asarray(ea.get("cartesian_delta_rotation", []), dtype=np.float64)
        return (
            float(np.linalg.norm(translation)) if translation.size else 0.0,
            float(np.linalg.norm(rotation)) if rotation.size else 0.0,
        )

    def _is_moving_step(self, current: dict[str, Any], previous: dict[str, Any] | None) -> bool:
        t_norm, r_norm = self._extract_action_norms(current)
        if t_norm > self.action_translation_threshold:
            return True
        if r_norm > self.action_rotation_threshold:
            return True
        if previous is None:
            return False
        cur_q = self._extract_joint_positions(current)
        prev_q = self._extract_joint_positions(previous)
        if cur_q.size and prev_q.size and cur_q.shape == prev_q.shape:
            if float(np.max(np.abs(cur_q - prev_q))) > self.joint_motion_threshold:
                return True
        if abs(self._extract_gripper_width(current) - self._extract_gripper_width(previous)) > self.gripper_motion_threshold:
            return True
        return False

    def _trim_to_motion(self, robot_slice: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
        if not robot_slice:
            return [], 0, 0
        moving = [
            i for i in range(len(robot_slice))
            if self._is_moving_step(robot_slice[i], robot_slice[i - 1] if i > 0 else None)
        ]
        if not moving:
            return [], len(robot_slice), 0
        first, last = moving[0], moving[-1]
        return robot_slice[first : last + 1], first, len(robot_slice) - 1 - last

    # --- camera coverage ---

    def _has_camera_coverage(self, timestamp_ns: int) -> bool:
        for name, frames in self.camera_frames.items():
            idx = closest_index(self._camera_timestamps[name], timestamp_ns)
            if idx is None or abs(frames[idx].timestamp_ns - timestamp_ns) > self.camera_tolerance_ns:
                return False
        return True

    def _trim_to_camera_coverage(
        self, robot_slice: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int, int]:
        first = next(
            (i for i, row in enumerate(robot_slice) if self._has_camera_coverage(infer_timestamp_ns(row))),
            None,
        )
        if first is None:
            return [], len(robot_slice), 0
        last = next(
            (i for i in range(len(robot_slice) - 1, -1, -1) if self._has_camera_coverage(infer_timestamp_ns(robot_slice[i]))),
            first,
        )
        return robot_slice[first : last + 1], first, len(robot_slice) - 1 - last

    # --- main ---

    def clean(self) -> Path:
        if self.output_dir.exists():
            if not self.force_overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {self.output_dir}. Pass --force to overwrite."
                )
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        boundaries = find_episode_boundaries(self.robot_rows, self.episode_rows)
        LOGGER.info("Found %d raw episode segment(s).", len(boundaries))
        if self.max_episodes is not None:
            boundaries = boundaries[: self.max_episodes]

        kept_rows: list[dict[str, Any]] = []
        episode_events: list[dict[str, Any]] = []
        skipped_static = skipped_no_coverage = 0

        for raw_index, (start_idx, end_idx) in enumerate(boundaries):
            robot_slice = self.robot_rows[start_idx:end_idx]
            if not robot_slice:
                continue

            trimmed, removed_leading, removed_trailing = self._trim_to_motion(robot_slice)
            if not trimmed:
                skipped_static += 1
                LOGGER.info("Skipping segment %d (%d steps): no arm motion detected.", raw_index, len(robot_slice))
                continue
            if removed_leading or removed_trailing:
                LOGGER.info(
                    "Segment %d: removed %d leading and %d trailing static step(s).",
                    raw_index, removed_leading, removed_trailing,
                )

            covered, removed_pre, removed_post = self._trim_to_camera_coverage(trimmed)
            if not covered:
                skipped_no_coverage += 1
                LOGGER.info("Skipping segment %d: no steps fall within camera coverage.", raw_index)
                continue
            if removed_pre or removed_post:
                LOGGER.info(
                    "Segment %d: dropped %d leading and %d trailing step(s) outside camera coverage.",
                    raw_index, removed_pre, removed_post,
                )

            first_ts = infer_timestamp_ns(covered[0])
            episode_events.append({"event": "episode_start", "robot_timestamp_ns": first_ts})
            kept_rows.extend(covered)

            LOGGER.info(
                "Kept segment %d → output episode %d (%d steps).",
                raw_index, len(episode_events) - 1, len(covered),
            )

        LOGGER.info(
            "Cleaning complete: %d episode(s) kept, %d skipped (static), %d skipped (no camera coverage).",
            len(episode_events), skipped_static, skipped_no_coverage,
        )

        write_jsonl(self.output_dir / "robot.jsonl", kept_rows)
        write_jsonl(self.output_dir / "episode_events.jsonl", episode_events)
        shutil.copy2(self.dataset_dir / "session_metadata.json", self.output_dir / "session_metadata.json")

        for text_name in ("text.jsonl", "language.jsonl"):
            src = self.dataset_dir / text_name
            if src.exists():
                shutil.copy2(src, self.output_dir / text_name)

        # Symlink cameras directory — frame files are unchanged so no copying needed.
        cameras_link = self.output_dir / "cameras"
        cameras_link.symlink_to((self.dataset_dir / "cameras").resolve())

        LOGGER.info("Cleaned dataset written to %s", self.output_dir)
        return self.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a raw recording to motion-only, camera-covered steps."
    )
    parser.add_argument("dataset_name", help="Dataset directory name under --datasets-root.")
    parser.add_argument(
        "--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT,
        help="Root directory containing raw datasets.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for cleaned dataset output.",
    )
    parser.add_argument(
        "--camera-tolerance-ms", type=float, default=DEFAULT_CAMERA_TOLERANCE_NS / 1_000_000.0,
        help="Maximum robot-to-camera sync error in milliseconds.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes to process.")
    parser.add_argument(
        "--joint-motion-threshold", type=float, default=DEFAULT_JOINT_MOTION_THRESHOLD,
        help="Max joint delta (rad) considered stationary between consecutive steps.",
    )
    parser.add_argument(
        "--gripper-motion-threshold", type=float, default=DEFAULT_GRIPPER_MOTION_THRESHOLD,
        help="Max gripper-width delta (m) considered stationary between consecutive steps.",
    )
    parser.add_argument(
        "--action-translation-threshold", type=float, default=DEFAULT_ACTION_TRANSLATION_THRESHOLD,
        help="Min cartesian_delta_translation norm considered movement.",
    )
    parser.add_argument(
        "--action-rotation-threshold", type=float, default=DEFAULT_ACTION_ROTATION_THRESHOLD,
        help="Min cartesian_delta_rotation norm considered movement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.datasets_root / args.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset '{args.dataset_name}' not found at {dataset_dir}")
    output_dir = args.output_root / args.dataset_name
    cleaner = DatasetCleaner(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        camera_tolerance_ns=int(args.camera_tolerance_ms * 1_000_000.0),
        force_overwrite=args.force,
        max_episodes=args.max_episodes,
        joint_motion_threshold=args.joint_motion_threshold,
        gripper_motion_threshold=args.gripper_motion_threshold,
        action_translation_threshold=args.action_translation_threshold,
        action_rotation_threshold=args.action_rotation_threshold,
    )
    cleaner.clean()


if __name__ == "__main__":
    main()
