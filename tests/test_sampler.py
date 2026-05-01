"""Unit tests for KeyframeSampler."""
import numpy as np
import pytest
from src.annotation.sampler import KeyframeSampler, MAX_KEYFRAMES


def _make_episode(n_frames=100, fps=30.0, gripper=None, wrench=None, tcp_vel=None):
    timestamps = np.arange(n_frames) / fps
    ep = {"timestamps": timestamps}
    if gripper is not None:
        ep["gripper_state"] = gripper
    if wrench is not None:
        ep["wrench"] = wrench
    if tcp_vel is not None:
        ep["tcp_velocity"] = tcp_vel
    return ep


def test_uniform_sampling():
    ep = _make_episode(n_frames=300, fps=30.0)
    sampler = KeyframeSampler()
    kfs = sampler.sample(ep)
    uniform = [k for k in kfs if k.reason == "uniform"]
    # 300/30 = 10 s episode → at 0.5 s interval → ~20 uniform frames
    assert len(uniform) >= 15


def test_gripper_transition_detected():
    n = 100
    gripper = np.zeros(n)
    gripper[40:] = 1.0  # open → closed at frame 40
    ep = _make_episode(n_frames=n, fps=30.0, gripper=gripper)
    sampler = KeyframeSampler()
    kfs = sampler.sample(ep)
    gripper_kfs = [k for k in kfs if k.reason == "gripper"]
    assert len(gripper_kfs) >= 1
    assert any(k.frame_idx == 40 for k in gripper_kfs)


def test_gripper_both_transitions():
    n = 200
    gripper = np.zeros(n)
    gripper[50:100] = 1.0  # close at 50, open at 100
    ep = _make_episode(n_frames=n, fps=30.0, gripper=gripper)
    sampler = KeyframeSampler()
    kfs = sampler.sample(ep)
    gripper_kfs = [k for k in kfs if k.reason == "gripper"]
    assert len(gripper_kfs) == 2


def test_contact_spike_detected():
    n = 100
    wrench = np.zeros((n, 6))
    # Create a force spike at frame 60: increase Fz by 10 N quickly
    wrench[60:, 2] = 10.0
    ep = _make_episode(n_frames=n, fps=30.0, wrench=wrench)
    sampler = KeyframeSampler()
    kfs = sampler.sample(ep)
    contact_kfs = [k for k in kfs if k.reason == "contact"]
    assert len(contact_kfs) >= 1


def test_velocity_zero_crossing_detected():
    n = 100
    tcp_vel = np.zeros((n, 6))
    # Dominant axis is 0: negative for first half, positive for second half
    tcp_vel[:50, 0] = -0.1
    tcp_vel[50:, 0] = 0.1  # zero-crossing at frame 50
    ep = _make_episode(n_frames=n, fps=30.0, tcp_vel=tcp_vel)
    sampler = KeyframeSampler()
    kfs = sampler.sample(ep)
    vel_kfs = [k for k in kfs if k.reason == "velocity"]
    assert len(vel_kfs) >= 1


def test_cap_at_max_keyframes():
    # Long episode with many events → should be capped
    n = 10000
    fps = 100.0
    gripper = np.zeros(n)
    # Lots of gripper transitions
    gripper[::50] = 1.0 - gripper[::50]
    ep = _make_episode(n_frames=n, fps=fps, gripper=gripper)
    sampler = KeyframeSampler()
    kfs = sampler.sample(ep)
    assert len(kfs) <= MAX_KEYFRAMES


def test_deduplication():
    # Two gripper transitions very close together should collapse to one
    n = 100
    gripper = np.zeros(n)
    gripper[40] = 0.6  # close
    gripper[41] = 1.0  # close more — same event
    ep = _make_episode(n_frames=n, fps=1000.0, gripper=gripper)  # high fps → close timestamps
    sampler = KeyframeSampler()
    kfs = sampler.sample(ep)
    # There should not be two gripper events within 50 ms
    gripper_kfs = sorted([k for k in kfs if k.reason == "gripper"], key=lambda x: x.timestamp)
    for i in range(1, len(gripper_kfs)):
        assert gripper_kfs[i].timestamp - gripper_kfs[i - 1].timestamp >= 0.045


def test_empty_episode():
    sampler = KeyframeSampler()
    assert sampler.sample({}) == []
    assert sampler.sample({"timestamps": []}) == []
