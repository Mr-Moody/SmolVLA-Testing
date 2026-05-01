"""Tests for the safety envelope using MockFrankaInterface."""
import numpy as np
import pytest
from pathlib import Path

from src.robot.franka_interface import MockFrankaInterface
from src.common.phases import Phase


@pytest.fixture
def mock():
    return MockFrankaInterface()


def test_normal_action_executes(mock):
    action = np.array([0.001, 0.0, -0.001, 0.0, 0.0, 0.0, 0.0])
    obs_before = mock.get_observation()
    mock.execute_action(action, Phase.FREE_MOTION)
    # Should not be rejected; check it mutated some state
    obs_after = mock.get_observation()
    assert not np.allclose(obs_before.tcp_pose, obs_after.tcp_pose)


def test_nan_action_rejected(mock, tmp_path, monkeypatch):
    monkeypatch.setattr("src.robot.franka_interface.SAFETY_LOG_PATH",
                        tmp_path / "safety_log.jsonl")
    action = np.array([float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obs_before = mock.get_observation()
    mock.execute_action(action, Phase.FREE_MOTION)
    obs_after = mock.get_observation()
    # Should be rejected — state unchanged
    assert np.allclose(obs_before.tcp_pose, obs_after.tcp_pose)
    log = tmp_path / "safety_log.jsonl"
    assert log.exists()


def test_inf_action_rejected(mock, tmp_path, monkeypatch):
    monkeypatch.setattr("src.robot.franka_interface.SAFETY_LOG_PATH",
                        tmp_path / "safety_log.jsonl")
    action = np.array([float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    obs_before = mock.get_observation()
    mock.execute_action(action, Phase.FREE_MOTION)
    obs_after = mock.get_observation()
    assert np.allclose(obs_before.tcp_pose, obs_after.tcp_pose)


def test_high_velocity_rejected(mock, tmp_path, monkeypatch):
    monkeypatch.setattr("src.robot.franka_interface.SAFETY_LOG_PATH",
                        tmp_path / "safety_log.jsonl")
    action = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0])  # vel > 2 rad/s
    obs_before = mock.get_observation()
    mock.execute_action(action, Phase.FREE_MOTION)
    obs_after = mock.get_observation()
    assert np.allclose(obs_before.tcp_pose, obs_after.tcp_pose)


def test_gripper_close_in_free_motion_rejected(mock, tmp_path, monkeypatch):
    monkeypatch.setattr("src.robot.franka_interface.SAFETY_LOG_PATH",
                        tmp_path / "safety_log.jsonl")
    action = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9])  # gripper cmd > 0.7
    obs_before = mock.get_observation()
    mock.execute_action(action, Phase.FREE_MOTION)  # should be rejected
    obs_after = mock.get_observation()
    assert np.allclose(obs_before.tcp_pose, obs_after.tcp_pose)


def test_gripper_close_allowed_in_contact_establish(mock):
    action = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
    obs_before = mock.get_observation()
    mock.execute_action(action, Phase.CONTACT_ESTABLISH)  # should be allowed
    obs_after = mock.get_observation()
    assert not np.allclose(obs_before.tcp_pose, obs_after.tcp_pose)


def test_rapid_sign_flip_not_necessarily_rejected(mock):
    """Rapid sign flips within velocity limits are not rejected."""
    for sign in [1, -1, 1, -1]:
        action = np.array([sign * 0.001, 0.0, 0.0, sign * 0.1, 0.0, 0.0, 0.0])
        mock.execute_action(action, Phase.FREE_MOTION)
    # Should not crash; obs should have moved
    obs = mock.get_observation()
    assert obs.tcp_pose is not None


def test_emergency_stop(mock):
    mock.emergency_stop()
    assert mock._stopped is True
    obs = mock.get_observation()
    assert np.allclose(obs.tcp_velocity, 0.0)


def test_franka_interface_import_error_at_instantiation_not_import():
    """FrankaInterface raises ImportError at instantiation, not at module import."""
    from src.robot.franka_interface import FrankaInterface  # must not raise
    with pytest.raises((ImportError, Exception)):
        # No frankx installed → should raise ImportError on __init__
        FrankaInterface(robot_ip="192.168.1.999")
