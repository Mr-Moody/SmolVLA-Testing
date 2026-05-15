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
    episode_events.jsonl         (regenerated episode_start and episode_end events)
    text.jsonl / language.jsonl  (copied if present)
    cameras/                     (symlinked from raw source — frames are unchanged)

Run data_converter.py on the output to produce a LeRobotDataset v3 export.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch

from src.labels.pick_place import generate_prompts
from src.data.utils import (
    DEFAULT_CAMERA_TOLERANCE_NS,
    find_episode_boundaries,
    infer_timestamp_ns,
    load_camera_frames,
    read_jsonl,
    write_jsonl,
)

DEFAULT_DATASETS_ROOT = Path("raw_datasets")
DEFAULT_OUTPUT_ROOT = Path("cleaned_datasets")
DEFAULT_MIN_EPISODE_DURATION_S = 2.0
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
        min_episode_duration_s: float = DEFAULT_MIN_EPISODE_DURATION_S,
        generate_tasks: bool = False,
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
        self.min_episode_duration_s = float(min_episode_duration_s)
        self.generate_tasks = generate_tasks

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Device: %s", self._device)
        LOGGER.info("Loading dataset: %s", dataset_dir)

        self.robot_rows = read_jsonl(dataset_dir / "robot.jsonl")
        self.episode_rows = read_jsonl(dataset_dir / "episode_events.jsonl")
        self.camera_frames = load_camera_frames(dataset_dir)
        self._camera_timestamps: dict[str, list[int]] = {
            name: [frame.timestamp_ns for frame in frames]
            for name, frames in self.camera_frames.items()
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

    # --- motion detection ---

    def _compute_moving_mask(self, robot_slice: list[dict[str, Any]]) -> torch.Tensor:
        """Return a bool tensor of shape (N,) indicating which steps involve motion."""
        N = len(robot_slice)
        dev = self._device

        t_vecs: list[list[float]] = []
        r_vecs: list[list[float]] = []
        q_vecs: list[list[float]] = []
        gw_list: list[float] = []
        for row in robot_slice:
            ea = row.get("executed_action", {})
            rs = row.get("robot_state", {})
            t_vecs.append([float(v) for v in ea.get("cartesian_delta_translation", [])])
            r_vecs.append([float(v) for v in ea.get("cartesian_delta_rotation", [])])
            q_vecs.append([float(v) for v in rs.get("q", [])])
            gw_list.append(float(rs.get("gripper_width", 0.0)))

        moving = torch.zeros(N, dtype=torch.bool, device=dev)

        for vecs, threshold in (
            (t_vecs, self.action_translation_threshold),
            (r_vecs, self.action_rotation_threshold),
        ):
            max_len = max((len(v) for v in vecs), default=0)
            if max_len == 0:
                continue
            if all(len(v) == max_len for v in vecs):
                tensor = torch.tensor(vecs, dtype=torch.float64, device=dev)
            else:
                tensor = torch.zeros(N, max_len, dtype=torch.float64, device=dev)
                for i, v in enumerate(vecs):
                    if v:
                        tensor[i, :len(v)] = torch.tensor(v, dtype=torch.float64)
            moving |= torch.linalg.norm(tensor, dim=1) > threshold

        first_q_len = len(q_vecs[0]) if q_vecs else 0
        if first_q_len > 0 and all(len(q) == first_q_len for q in q_vecs):
            q = torch.tensor(q_vecs, dtype=torch.float64, device=dev)
            moving[1:] |= (q[1:] - q[:-1]).abs().amax(dim=1) > self.joint_motion_threshold

        gw = torch.tensor(gw_list, dtype=torch.float64, device=dev)
        if N > 1:
            moving[1:] |= (gw[1:] - gw[:-1]).abs() > self.gripper_motion_threshold

        return moving

    def _trim_to_motion(self, robot_slice: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
        if not robot_slice:
            return [], 0, 0
        moving = self._compute_moving_mask(robot_slice)
        indices = moving.nonzero(as_tuple=False).view(-1)
        if indices.numel() == 0:
            return [], len(robot_slice), 0
        first, last = int(indices[0]), int(indices[-1])
        return robot_slice[first:last + 1], first, len(robot_slice) - 1 - last

    # --- camera coverage ---

    def _trim_to_camera_coverage(
        self, robot_slice: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int, int]:
        if not robot_slice:
            return [], 0, 0
        N = len(robot_slice)
        dev = self._device

        robot_ts = torch.tensor(
            [infer_timestamp_ns(row) for row in robot_slice],
            dtype=torch.int64, device=dev,
        )

        covered = torch.ones(N, dtype=torch.bool, device=dev)
        for cam_name, cam_ts in self._camera_ts_tensors.items():
            M = cam_ts.shape[0]
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
            covered &= torch.minimum(left_delta, right_delta) <= self.camera_tolerance_ns

        covered_idx = covered.nonzero(as_tuple=False).view(-1)
        if covered_idx.numel() == 0:
            return [], N, 0
        first, last = int(covered_idx[0]), int(covered_idx[-1])
        return robot_slice[first:last + 1], first, N - 1 - last

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
        skipped_static = skipped_no_coverage = skipped_too_short = 0

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
            last_ts = infer_timestamp_ns(covered[-1])
            duration_s = (last_ts - first_ts) / 1_000_000_000.0
            if duration_s < self.min_episode_duration_s:
                skipped_too_short += 1
                LOGGER.info(
                    "Skipping segment %d: duration %.2fs is below minimum %.2fs.",
                    raw_index, duration_s, self.min_episode_duration_s,
                )
                continue

            episode_events.append({"event": "episode_start", "robot_timestamp_ns": first_ts})
            kept_rows.extend(covered)
            episode_events.append({"event": "episode_end", "robot_timestamp_ns": last_ts})

            LOGGER.info(
                "Kept segment %d → output episode %d (%d steps).",
                raw_index, len(episode_events) - 1, len(covered),
            )

        n_kept = len(episode_events) // 2
        LOGGER.info(
            "Cleaning complete: %d episode(s) kept, %d skipped (static), %d skipped (no camera coverage),"
            " %d skipped (too short, < %.2fs).",
            n_kept, skipped_static, skipped_no_coverage, skipped_too_short, self.min_episode_duration_s,
        )

        write_jsonl(self.output_dir / "robot.jsonl", kept_rows)
        write_jsonl(self.output_dir / "episode_events.jsonl", episode_events)

        if self.generate_tasks and episode_events:
            num_kept = len(episode_events)
            prompts = generate_prompts(num_kept)
            annotations_path = self.output_dir / "annotations.jsonl"
            with annotations_path.open("w", encoding="utf-8") as fh:
                for ep_idx, task in enumerate(prompts):
                    fh.write(json.dumps({"episode_index": ep_idx, "task": task}) + "\n")
            LOGGER.info("Generated %d task annotation(s) → %s", num_kept, annotations_path)

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
    parser.add_argument(
        "--min-episode-duration", type=float, default=DEFAULT_MIN_EPISODE_DURATION_S,
        help="Minimum episode duration in seconds after trimming; shorter episodes are dropped (default: %(default)ss).",
    )
    parser.add_argument(
        "--generate-tasks", action="store_true",
        help="Auto-assign a unique global task prompt to each kept episode and write annotations.jsonl.",
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
        min_episode_duration_s=args.min_episode_duration,
        generate_tasks=args.generate_tasks,
    )
    cleaner.clean()


if __name__ == "__main__":
    main()
