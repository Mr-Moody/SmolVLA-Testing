"""Keyframe sampler for Qwen3-VL annotation.

Combines uniform 2 Hz sampling with event-triggered keyframes (gripper
transitions, F/T spikes, TCP velocity zero-crossings).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Sampling parameters
UNIFORM_INTERVAL_S = 0.5  # 2 Hz
MAX_KEYFRAMES = 60
DEDUP_WINDOW_S = 0.05  # 50 ms
CONTACT_FORCE_SPIKE_N = 5.0  # N
CONTACT_WINDOW_S = 0.1  # 100 ms


@dataclass
class Keyframe:
    timestamp: float  # seconds
    frame_idx: int
    reason: str  # "uniform" | "gripper" | "contact" | "velocity"


class KeyframeSampler:
    """Sample keyframes from a LeRobot-style episode dict.

    The episode dict is expected to have the following arrays (all indexed
    by frame):
        - ``timestamps``: 1-D float array, seconds
        - ``gripper_state``: 1-D float array (0 = open, 1 = closed)
        - ``wrench``: (N, 6) float array — Fx Fy Fz Tx Ty Tz
        - ``tcp_velocity``: (N, 6) float array — vx vy vz wx wy wz

    Arrays may be None/missing; rules that depend on them are silently skipped.
    """

    def sample(self, episode: Dict[str, Any]) -> List[Keyframe]:
        """Return a deduplicated, capped list of keyframes.

        Args:
            episode: Dict with arrays ``timestamps``, ``gripper_state``,
                     ``wrench``, ``tcp_velocity``.

        Returns:
            List of :class:`Keyframe` sorted by timestamp, at most
            :data:`MAX_KEYFRAMES` entries.
        """
        timestamps = np.asarray(episode.get("timestamps", []), dtype=float)
        if len(timestamps) == 0:
            return []

        keyframes: List[Keyframe] = []

        # (a) Uniform 2 Hz
        keyframes.extend(self._uniform(timestamps))

        # (b) Gripper transitions
        gripper = episode.get("gripper_state")
        if gripper is not None:
            gripper = np.asarray(gripper, dtype=float)
            keyframes.extend(self._gripper_transitions(timestamps, gripper))

        # (c) F/T spikes
        wrench = episode.get("wrench")
        if wrench is not None:
            wrench = np.asarray(wrench, dtype=float)
            keyframes.extend(self._contact_spikes(timestamps, wrench))

        # (d) TCP velocity zero-crossings
        tcp_vel = episode.get("tcp_velocity")
        if tcp_vel is not None:
            tcp_vel = np.asarray(tcp_vel, dtype=float)
            keyframes.extend(self._velocity_zero_crossings(timestamps, tcp_vel))

        keyframes = self._deduplicate(keyframes)
        keyframes = self._cap(keyframes)
        return keyframes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _uniform(self, timestamps: np.ndarray) -> List[Keyframe]:
        result = []
        next_t = timestamps[0]
        for i, t in enumerate(timestamps):
            if t >= next_t:
                result.append(Keyframe(timestamp=float(t), frame_idx=i, reason="uniform"))
                next_t = t + UNIFORM_INTERVAL_S
        return result

    def _gripper_transitions(
        self, timestamps: np.ndarray, gripper: np.ndarray
    ) -> List[Keyframe]:
        result = []
        binary = (gripper > 0.5).astype(int)
        for i in range(1, len(binary)):
            if binary[i] != binary[i - 1]:
                result.append(
                    Keyframe(timestamp=float(timestamps[i]), frame_idx=i, reason="gripper")
                )
        return result

    def _contact_spikes(
        self, timestamps: np.ndarray, wrench: np.ndarray
    ) -> List[Keyframe]:
        if wrench.ndim != 2 or wrench.shape[1] < 3:
            return []
        force_mag = np.linalg.norm(wrench[:, :3], axis=1)
        dt = np.diff(timestamps)
        result = []
        for i in range(1, len(force_mag)):
            window_s = min(dt[i - 1], CONTACT_WINDOW_S)
            n_frames = max(1, int(window_s / (dt[i - 1] + 1e-9)))
            start = max(0, i - n_frames)
            delta = abs(force_mag[i] - force_mag[start])
            if delta > CONTACT_FORCE_SPIKE_N:
                result.append(
                    Keyframe(timestamp=float(timestamps[i]), frame_idx=i, reason="contact")
                )
        return result

    def _velocity_zero_crossings(
        self, timestamps: np.ndarray, tcp_vel: np.ndarray
    ) -> List[Keyframe]:
        if tcp_vel.ndim != 2 or tcp_vel.shape[1] < 3:
            return []
        # Find the dominant translational axis
        mag = np.abs(tcp_vel[:, :3])
        dominant_axis = int(np.argmax(mag.sum(axis=0)))
        v = tcp_vel[:, dominant_axis]
        result = []
        for i in range(1, len(v)):
            if v[i - 1] * v[i] < 0:  # sign change
                result.append(
                    Keyframe(
                        timestamp=float(timestamps[i]), frame_idx=i, reason="velocity"
                    )
                )
        return result

    def _deduplicate(self, keyframes: List[Keyframe]) -> List[Keyframe]:
        """Remove duplicates within DEDUP_WINDOW_S; prefer event-triggered."""
        if not keyframes:
            return []
        keyframes.sort(key=lambda k: k.timestamp)
        kept: List[Keyframe] = [keyframes[0]]
        for kf in keyframes[1:]:
            if kf.timestamp - kept[-1].timestamp < DEDUP_WINDOW_S:
                # Prefer event-triggered over uniform
                if kf.reason != "uniform" and kept[-1].reason == "uniform":
                    kept[-1] = kf
                # else keep the earlier one
            else:
                kept.append(kf)
        return kept

    def _cap(self, keyframes: List[Keyframe]) -> List[Keyframe]:
        """Cap at MAX_KEYFRAMES, preferring event-triggered frames."""
        if len(keyframes) <= MAX_KEYFRAMES:
            return keyframes
        event = [kf for kf in keyframes if kf.reason != "uniform"]
        uniform = [kf for kf in keyframes if kf.reason == "uniform"]
        n_event = min(len(event), MAX_KEYFRAMES)
        n_uniform = MAX_KEYFRAMES - n_event
        selected = event[:n_event] + uniform[:n_uniform]
        selected.sort(key=lambda k: k.timestamp)
        return selected
