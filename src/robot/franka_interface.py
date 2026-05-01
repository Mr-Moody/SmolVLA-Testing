"""Franka Emika Panda robot interface.

FrankaInterface: real robot (requires frankx or franka-panda-py to be installed).
MockFrankaInterface: kinematic simulation for offline testing and CI.
"""
from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from src.common.phases import Phase
from src.fsm.runtime_fsm import Observation

logger = logging.getLogger(__name__)

SAFETY_LOG_PATH = Path("outputs/safety_log.jsonl")

# Safety limits
MAX_JOINT_VELOCITY_RAD_S = 2.0
MAX_EEF_FORCE_N = 30.0

# Workspace bounds (metres) — adjust to match your Franka cell
WORKSPACE = {
    "x": (-0.8, 0.8),
    "y": (-0.8, 0.8),
    "z": (-0.05, 1.2),
}


class RobotInterfaceBase(ABC):
    """Abstract base class for robot interfaces."""

    @abstractmethod
    def get_observation(self) -> Observation:
        ...

    @abstractmethod
    def execute_action(self, action: np.ndarray, current_phase: Phase = Phase.FREE_MOTION) -> None:
        ...

    @abstractmethod
    def home(self) -> None:
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        ...

    # ── Safety envelope ────────────────────────────────────────────────────

    def _check_action_safety(
        self,
        action: np.ndarray,
        current_phase: Phase,
        obs: Optional[Observation] = None,
    ) -> tuple[bool, str]:
        """Return (safe, reason). If not safe, the action must be rejected."""
        if action is None:
            return False, "action is None"

        if np.any(np.isnan(action)):
            return False, f"NaN in action: {action}"

        if np.any(np.isinf(action)):
            return False, f"Inf in action: {action}"

        # Workspace bounds (first 3 dims = xyz delta or target)
        if len(action) >= 3:
            pos = action[:3]
            for i, (ax, (lo, hi)) in enumerate(zip("xyz", [WORKSPACE[k] for k in "xyz"])):
                if not (lo - 0.5 <= pos[i] <= hi + 0.5):  # 0.5 m buffer for deltas
                    return False, f"Workspace violation on {ax}: {pos[i]:.3f} not in [{lo}, {hi}]"

        # Velocity check (dims 0-5 treated as velocity if shape matches)
        if len(action) >= 6:
            vel = action[:6]
            max_vel = np.max(np.abs(vel))
            if max_vel > MAX_JOINT_VELOCITY_RAD_S:
                return False, f"Joint velocity too high: {max_vel:.3f} rad/s > {MAX_JOINT_VELOCITY_RAD_S}"

        # Force check from current observation
        if obs is not None:
            fz = abs(float(obs.wrench[2]))
            if fz > MAX_EEF_FORCE_N:
                return False, f"EEF force too high: {fz:.1f} N > {MAX_EEF_FORCE_N}"

        # Defense: no gripper close in FREE_MOTION
        if current_phase == Phase.FREE_MOTION and len(action) >= 7:
            gripper_cmd = action[-1]
            if gripper_cmd > 0.7:  # closing command
                return False, "Gripper close command rejected during FREE_MOTION phase"

        return True, "OK"

    def _log_safety_rejection(self, action: np.ndarray, reason: str, phase: Phase):
        SAFETY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "phase": phase.name,
            "reason": reason,
            "action": action.tolist() if isinstance(action, np.ndarray) else str(action),
        }
        with open(SAFETY_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.warning("Safety rejection: %s", reason)


# ── Mock interface ─────────────────────────────────────────────────────────────

class MockFrankaInterface(RobotInterfaceBase):
    """Simulated Franka interface for offline testing.

    Maintains a simple kinematic state. Observations evolve based on applied
    actions so the FSM and safety envelope can be exercised.
    """

    def __init__(self, task: str = "pick_place", control_rate_hz: float = 30.0):
        self.task = task
        self.dt = 1.0 / control_rate_hz
        self._tcp_pose = np.array([0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=float)
        self._tcp_velocity = np.zeros(6)
        self._gripper_state = 0.0
        self._wrench = np.zeros(6)
        self._step = 0
        self._stopped = False

    def get_observation(self) -> Observation:
        return Observation(
            tcp_pose=self._tcp_pose.copy(),
            tcp_velocity=self._tcp_velocity.copy(),
            gripper_state=self._gripper_state,
            wrench=self._wrench.copy(),
        )

    def execute_action(self, action: np.ndarray, current_phase: Phase = Phase.FREE_MOTION) -> None:
        obs = self.get_observation()
        safe, reason = self._check_action_safety(action, current_phase, obs)
        if not safe:
            self._log_safety_rejection(action, reason, current_phase)
            return

        # Apply simple kinematic update
        if len(action) >= 3:
            self._tcp_pose[:3] += action[:3] * self.dt
        if len(action) >= 6:
            self._tcp_velocity = action[:6].copy()
        if len(action) >= 7:
            self._gripper_state = float(np.clip(action[-1], 0.0, 1.0))

        # Simulate contact force when gripper closes
        if self._gripper_state > 0.7:
            self._wrench[2] = 4.0
        else:
            self._wrench[2] = 0.0

        self._step += 1

    def home(self) -> None:
        self._tcp_pose = np.array([0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=float)
        self._tcp_velocity = np.zeros(6)
        self._gripper_state = 0.0
        self._wrench = np.zeros(6)

    def emergency_stop(self) -> None:
        self._stopped = True
        self._tcp_velocity = np.zeros(6)
        logger.warning("EMERGENCY STOP triggered on MockFrankaInterface")


# ── Real Franka interface ──────────────────────────────────────────────────────

class FrankaInterface(RobotInterfaceBase):
    """Real Franka Emika Panda interface.

    Requires frankx or franka-panda-py to be installed.
    Raises ImportError at instantiation if neither is available.
    """

    def __init__(
        self,
        robot_ip: str = "172.16.0.2",
        control_rate_hz: float = 100.0,
        dynamic_rel: float = 0.2,
    ):
        self._robot = self._connect(robot_ip, dynamic_rel)
        self.dt = 1.0 / control_rate_hz

    def _connect(self, ip: str, dynamic_rel: float):
        try:
            import frankx
            robot = frankx.Robot(ip)
            robot.set_default_behavior()
            robot.recover_from_errors()
            robot.velocity_rel = dynamic_rel
            robot.acceleration_rel = dynamic_rel
            logger.info("Connected to Franka via frankx at %s", ip)
            return robot
        except ImportError:
            pass
        try:
            import franka
            robot = franka.Robot(ip)
            logger.info("Connected to Franka via franka-panda-py at %s", ip)
            return robot
        except ImportError:
            pass
        raise ImportError(
            "Neither frankx nor franka-panda-py is installed. "
            "Install one of them to use FrankaInterface. "
            "For offline testing, use MockFrankaInterface."
        )

    def get_observation(self) -> Observation:
        state = self._robot.current_pose()  # adapter-specific; adjust to your library
        return Observation(
            tcp_pose=np.array(state.pose, dtype=float),
            tcp_velocity=np.zeros(6),  # fill from robot state
            gripper_state=0.0,         # fill from gripper state
            wrench=np.zeros(6),        # fill from F/T sensor
        )

    def execute_action(self, action: np.ndarray, current_phase: Phase = Phase.FREE_MOTION) -> None:
        obs = self.get_observation()
        safe, reason = self._check_action_safety(action, current_phase, obs)
        if not safe:
            self._log_safety_rejection(action, reason, current_phase)
            return
        # Send to robot — adapt to your control library
        logger.debug("Executing action: %s", action)

    def home(self) -> None:
        logger.info("Homing robot...")

    def emergency_stop(self) -> None:
        logger.critical("EMERGENCY STOP")
        self._robot.stop()
