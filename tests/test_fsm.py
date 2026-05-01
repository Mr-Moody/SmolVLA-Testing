"""Unit tests for the RuntimeFSM."""
import numpy as np
import pytest
from src.fsm.runtime_fsm import RuntimeFSM, Observation
from src.common.phases import Phase


def _obs(
    tcp_pose=None,
    tcp_velocity=None,
    gripper=0.0,
    wrench=None,
    target_pose=None,
):
    return Observation(
        tcp_pose=np.array(tcp_pose or [0.5, 0.0, 0.3, 0, 0, 0, 1], dtype=float),
        tcp_velocity=np.zeros(6) if tcp_velocity is None else np.array(tcp_velocity, dtype=float),
        gripper_state=gripper,
        wrench=np.zeros(6) if wrench is None else np.array(wrench, dtype=float),
        target_pose=np.array(target_pose, dtype=float) if target_pose is not None else None,
    )


_PICK_CFG = {
    "contact_force_threshold": 3.0,
    "align_xy_threshold": 0.005,
    "hover_z": 0.05,
    "gripper_stable_steps": 3,
    "mate_force_band": [0.5, 10.0],
    "seated_force_band": [3.0, 10.0],
    "insertion_z_target": 0.0,
    "fine_align_timeout_s": 100.0,
    "control_rate_hz": 100.0,
}


def _make_fsm():
    return RuntimeFSM(task="pick_place", config=_PICK_CFG, hysteresis_steps=1)


# ---------------------------------------------------------------
# Legal transition walkthrough
# ---------------------------------------------------------------

def test_starts_in_free_motion():
    fsm = _make_fsm()
    assert fsm.current_phase == Phase.FREE_MOTION


def test_free_motion_to_fine_align():
    fsm = _make_fsm()
    # Target close in xy and below hover_z
    obs = _obs(
        tcp_pose=[0.5, 0.0, 0.04, 0, 0, 0, 1],
        target_pose=[0.503, 0.002, 0.04, 0, 0, 0, 1],
    )
    phase = fsm.step(obs)
    assert phase == Phase.FINE_ALIGN


def test_fine_align_to_contact_establish_via_force():
    fsm = _make_fsm()
    # Move to fine_align first
    obs_close = _obs(
        tcp_pose=[0.5, 0.0, 0.04, 0, 0, 0, 1],
        target_pose=[0.503, 0.002, 0.04, 0, 0, 0, 1],
    )
    fsm.step(obs_close)
    assert fsm.current_phase == Phase.FINE_ALIGN

    # Now force spike
    obs_force = _obs(wrench=[0, 0, 5.0, 0, 0, 0])
    phase = fsm.step(obs_force)
    assert phase == Phase.CONTACT_ESTABLISH


def test_fine_align_to_contact_establish_via_gripper():
    fsm = _make_fsm()
    obs_close = _obs(
        tcp_pose=[0.5, 0.0, 0.04, 0, 0, 0, 1],
        target_pose=[0.503, 0.002, 0.04, 0, 0, 0, 1],
    )
    fsm.step(obs_close)
    obs_gripper = _obs(gripper=0.9)
    phase = fsm.step(obs_gripper)
    assert phase == Phase.CONTACT_ESTABLISH


def test_contact_establish_to_constrained_motion():
    fsm = _make_fsm()
    # Walk to CONTACT_ESTABLISH
    fsm.step(_obs(tcp_pose=[0.5, 0.0, 0.04, 0, 0, 0, 1],
                  target_pose=[0.503, 0.002, 0.04, 0, 0, 0, 1]))
    fsm.step(_obs(gripper=0.9))
    assert fsm.current_phase == Phase.CONTACT_ESTABLISH

    # Gripper closed + Fz in mate band
    obs = _obs(gripper=1.0, wrench=[0, 0, 1.5, 0, 0, 0])
    phase = fsm.step(obs)
    assert phase == Phase.CONSTRAINED_MOTION


def test_constrained_motion_to_verify_release():
    fsm = _make_fsm()
    # Walk to CONSTRAINED_MOTION
    fsm.step(_obs(tcp_pose=[0.5, 0.0, 0.04, 0, 0, 0, 1],
                  target_pose=[0.503, 0.002, 0.04, 0, 0, 0, 1]))
    fsm.step(_obs(gripper=0.9))
    fsm.step(_obs(gripper=1.0, wrench=[0, 0, 1.5, 0, 0, 0]))
    assert fsm.current_phase == Phase.CONSTRAINED_MOTION

    # Target z reached + seated force
    obs = _obs(
        gripper=1.0,
        wrench=[0, 0, 5.0, 0, 0, 0],
        tcp_pose=[0.5, 0.0, 0.01, 0, 0, 0, 1],
        target_pose=[0.5, 0.0, 0.01, 0, 0, 0, 1],
    )
    phase = fsm.step(obs)
    assert phase == Phase.VERIFY_RELEASE


# ---------------------------------------------------------------
# Illegal transitions
# ---------------------------------------------------------------

def test_illegal_transition_free_to_contact_rejected():
    """FSM cannot jump FREE_MOTION → CONTACT_ESTABLISH directly."""
    fsm = _make_fsm()
    # Manually force Fz spike without going through FINE_ALIGN
    obs = _obs(wrench=[0, 0, 10.0, 0, 0, 0])
    phase = fsm.step(obs)
    # Should stay in FREE_MOTION (no path from FREE_MOTION via force spike alone)
    assert phase == Phase.FREE_MOTION


def test_illegal_transition_free_to_constrained_rejected():
    fsm = _make_fsm()
    # Artificially set state to FREE_MOTION and try to get CONSTRAINED without path
    obs = _obs(gripper=1.0, wrench=[0, 0, 5.0, 0, 0, 0])
    phase = fsm.step(obs)
    assert phase == Phase.FREE_MOTION


def test_illegal_transition_verify_to_free_rejected():
    """Verify VERIFY_RELEASE cannot fall back to FREE_MOTION."""
    fsm = _make_fsm()
    # Walk to VERIFY_RELEASE
    fsm.step(_obs(tcp_pose=[0.5, 0.0, 0.04, 0, 0, 0, 1],
                  target_pose=[0.503, 0.002, 0.04, 0, 0, 0, 1]))
    fsm.step(_obs(gripper=0.9))
    fsm.step(_obs(gripper=1.0, wrench=[0, 0, 1.5, 0, 0, 0]))
    fsm.step(_obs(gripper=1.0, wrench=[0, 0, 5.0, 0, 0, 0],
                  tcp_pose=[0.5, 0.0, 0.01, 0, 0, 0, 1],
                  target_pose=[0.5, 0.0, 0.01, 0, 0, 0, 1]))
    assert fsm.current_phase == Phase.VERIFY_RELEASE

    # Now try to create conditions that would trigger FREE_MOTION — should stay
    obs_free = _obs(tcp_pose=[0.8, 0.0, 0.3, 0, 0, 0, 1])
    phase = fsm.step(obs_free)
    assert phase == Phase.VERIFY_RELEASE


# ---------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------

def test_hysteresis_prevents_premature_transition():
    fsm = RuntimeFSM(task="pick_place", config=_PICK_CFG, hysteresis_steps=3)
    obs_close = _obs(
        tcp_pose=[0.5, 0.0, 0.04, 0, 0, 0, 1],
        target_pose=[0.503, 0.002, 0.04, 0, 0, 0, 1],
    )
    # Step 1: predicate true but hysteresis not yet met
    fsm.step(obs_close)
    assert fsm.current_phase == Phase.FREE_MOTION  # still FREE_MOTION after 1 step
    fsm.step(obs_close)
    assert fsm.current_phase == Phase.FREE_MOTION  # still after 2
    fsm.step(obs_close)
    assert fsm.current_phase == Phase.FINE_ALIGN  # transitions at 3rd step


def test_reset():
    fsm = _make_fsm()
    fsm.step(_obs(tcp_pose=[0.5, 0.0, 0.04, 0, 0, 0, 1],
                  target_pose=[0.503, 0.002, 0.04, 0, 0, 0, 1]))
    assert fsm.current_phase == Phase.FINE_ALIGN
    fsm.reset()
    assert fsm.current_phase == Phase.FREE_MOTION
