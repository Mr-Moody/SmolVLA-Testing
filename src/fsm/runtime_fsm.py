"""Runtime FSM for phase estimation at control rate.

Converts proprioceptive observations into phase IDs deterministically and
at control rate (30–100 Hz). Only legal transitions (as defined in
src/common/phases.py) are allowed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set, Tuple

import numpy as np

from src.common.phases import Phase, LEGAL_TRANSITIONS, is_legal_transition

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Proprioceptive observation at a single timestep."""
    tcp_pose: np.ndarray        # (7,) position + quaternion
    tcp_velocity: np.ndarray    # (6,) linear + angular velocity
    gripper_state: float        # 0.0 = open, 1.0 = closed
    wrench: np.ndarray          # (6,) Fx Fy Fz Tx Ty Tz
    target_pose: Optional[np.ndarray] = None  # (7,) optional target


# A predicate takes (Observation, config_dict) → bool
Predicate = Callable[[Observation, Dict[str, Any]], bool]


def _dist_xy(obs: Observation, cfg: Dict[str, Any]) -> bool:
    if obs.target_pose is None:
        return False
    diff = obs.tcp_pose[:2] - obs.target_pose[:2]
    xy_dist = float(np.linalg.norm(diff))
    z_ok = obs.tcp_pose[2] < cfg.get("hover_z", 0.05)
    return xy_dist < cfg.get("align_xy_threshold", 0.005) and z_ok


def _contact_force_or_gripper_close(obs: Observation, cfg: Dict[str, Any]) -> bool:
    fz = abs(float(obs.wrench[2]))
    gripper_closing = obs.gripper_state > 0.5
    return fz > cfg.get("contact_force_threshold", 3.0) or gripper_closing


def _pose_error_grew(obs: Observation, cfg: Dict[str, Any]) -> bool:
    # Exposed via FSM internal counter; predicate checks stored flag
    return False  # handled via hysteresis on negative of _dist_xy


def _gripper_stable_and_force_in_band(obs: Observation, cfg: Dict[str, Any]) -> bool:
    gripper_closed = obs.gripper_state > 0.5
    fz = abs(float(obs.wrench[2]))
    lo, hi = cfg.get("mate_force_band", [0.5, 3.0])
    return gripper_closed and (lo <= fz <= hi)


def _target_z_and_seated_force(obs: Observation, cfg: Dict[str, Any]) -> bool:
    z_target = cfg.get("insertion_z_target", 0.012)
    if obs.target_pose is None:
        return False
    z_reached = obs.tcp_pose[2] <= obs.target_pose[2] + z_target
    fz = abs(float(obs.wrench[2]))
    lo, hi = cfg.get("seated_force_band", [3.0, 8.0])
    return z_reached and (lo <= fz <= hi)


def _gripper_open_and_retracting(obs: Observation, cfg: Dict[str, Any]) -> bool:
    gripper_open = obs.gripper_state < 0.5
    retracting = np.linalg.norm(obs.tcp_velocity[:3]) > 0.01
    return gripper_open and retracting


# Transition table: (current_phase, predicate) → next_phase
_TRANSITIONS: list[Tuple[Phase, Predicate, Phase]] = [
    (Phase.FREE_MOTION,        _dist_xy,                          Phase.FINE_ALIGN),
    (Phase.FINE_ALIGN,         _contact_force_or_gripper_close,   Phase.CONTACT_ESTABLISH),
    # FINE_ALIGN → FREE_MOTION fallback is handled separately via timeout
    (Phase.CONTACT_ESTABLISH,  _gripper_stable_and_force_in_band, Phase.CONSTRAINED_MOTION),
    (Phase.CONSTRAINED_MOTION, _target_z_and_seated_force,        Phase.VERIFY_RELEASE),
    (Phase.VERIFY_RELEASE,     _gripper_open_and_retracting,      Phase.VERIFY_RELEASE),  # terminal self
]


class RuntimeFSM:
    """Deterministic phase estimator running at control rate.

    Args:
        task: ``"pick_place"`` or ``"msd_plug"``.
        config: Dict of threshold values. Defaults are pick-and-place values.
        hysteresis_steps: Number of consecutive steps a predicate must be True
            before the transition is committed.
    """

    def __init__(
        self,
        task: str = "pick_place",
        config: Optional[Dict[str, Any]] = None,
        hysteresis_steps: int = 3,
    ):
        self.task = task
        self.config = config or {}
        self.hysteresis_steps = hysteresis_steps
        self._state = Phase.FREE_MOTION
        # Per-transition hysteresis counters: {(from_phase, to_phase): count}
        self._hysteresis: Dict[Tuple[Phase, Phase], int] = {}
        self._step_count = 0
        # For FINE_ALIGN → FREE_MOTION fallback timeout
        self._fine_align_entry_step: Optional[int] = None
        self._fine_align_timeout_steps = int(
            config.get("fine_align_timeout_s", 5.0) * config.get("control_rate_hz", 100.0)
            if config else 500
        )

    @property
    def current_phase(self) -> Phase:
        return self._state

    def step(self, observation: Observation) -> Phase:
        """Update FSM state from one observation and return current phase.

        Args:
            observation: Current robot observation.

        Returns:
            Current (possibly updated) Phase.
        """
        self._step_count += 1

        # Check FINE_ALIGN → FREE_MOTION timeout
        if self._state == Phase.FINE_ALIGN:
            if self._fine_align_entry_step is None:
                self._fine_align_entry_step = self._step_count
            elapsed = self._step_count - self._fine_align_entry_step
            if elapsed > self._fine_align_timeout_steps:
                logger.debug("FINE_ALIGN timeout → FREE_MOTION at step %d", self._step_count)
                self._try_transition(Phase.FREE_MOTION)
                self._fine_align_entry_step = None
                return self._state
        else:
            self._fine_align_entry_step = None

        # Evaluate transitions from current state
        for from_phase, predicate, to_phase in _TRANSITIONS:
            if from_phase != self._state:
                continue
            if to_phase == from_phase:
                continue  # skip self-loop transitions
            key = (from_phase, to_phase)
            if predicate(observation, self.config):
                self._hysteresis[key] = self._hysteresis.get(key, 0) + 1
                if self._hysteresis[key] >= self.hysteresis_steps:
                    self._try_transition(to_phase)
                    self._hysteresis.pop(key, None)
            else:
                # Reset hysteresis on predicate miss
                self._hysteresis.pop(key, None)

        return self._state

    def _try_transition(self, new_phase: Phase):
        if not is_legal_transition(self._state, new_phase):
            logger.warning(
                "FSM: Attempted illegal transition %s → %s; staying in %s",
                self._state.name,
                new_phase.name,
                self._state.name,
            )
            return
        logger.debug("FSM: %s → %s at step %d", self._state.name, new_phase.name, self._step_count)
        self._state = new_phase

    def reset(self):
        """Reset FSM to initial state."""
        self._state = Phase.FREE_MOTION
        self._hysteresis.clear()
        self._step_count = 0
        self._fine_align_entry_step = None
