"""Tests for the LeRobot export feature contract."""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture()
def data_converter(monkeypatch):
    """Import data_converter with heavy optional runtime deps stubbed."""
    sys.modules.pop("src.data_converter", None)

    fake_torch = types.ModuleType("torch")
    fake_cv2 = types.ModuleType("cv2")
    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable=None, *args, **kwargs: iterable

    fake_lerobot = types.ModuleType("lerobot")
    fake_datasets = types.ModuleType("lerobot.datasets")
    fake_lerobot_dataset = types.ModuleType("lerobot.datasets.lerobot_dataset")

    class FakeLeRobotDataset:
        pass

    fake_lerobot_dataset.LeRobotDataset = FakeLeRobotDataset

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setitem(sys.modules, "lerobot", fake_lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", fake_lerobot_dataset)

    module = importlib.import_module("src.data_converter")
    yield module
    sys.modules.pop("src.data_converter", None)


def _row(q, gripper_width, gripper_command, q_cmd=None):
    robot_state = {
        "q": q,
        "dq": [99.0] * len(q),
        "tcp_position_xyz": [1.0, 2.0, 3.0],
        "tcp_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_width": gripper_width,
    }
    if q_cmd is not None:
        robot_state["q_cmd"] = q_cmd
    return {
        "robot_state": robot_state,
        "executed_action": {
            "cartesian_delta_translation": [10.0, 20.0, 30.0],
            "cartesian_delta_rotation": [40.0, 50.0, 60.0],
            "gripper_command": gripper_command,
        },
    }


def test_converter_uses_q_and_gripper_width_for_state(data_converter):
    converter = object.__new__(data_converter.SmolVLADatasetConverter)
    converter.robot_rows = [_row([0, 1, 2, 3, 4, 5, 6], 0.076, 0.0)]

    state = converter._state_vector(converter.robot_rows[0])

    np.testing.assert_allclose(state, np.array([0, 1, 2, 3, 4, 5, 6, 0.076], dtype=np.float32))
    assert converter._state_dimension_names() == [
        "q_0",
        "q_1",
        "q_2",
        "q_3",
        "q_4",
        "q_5",
        "q_6",
        "gripper_width",
    ]


def test_converter_uses_q_cmd_and_gripper_command_for_action(data_converter):
    converter = object.__new__(data_converter.SmolVLADatasetConverter)
    current = _row([0, 1, 2, 3, 4, 5, 6], 0.076, 1.0, q_cmd=[20, 21, 22, 23, 24, 25, 26])
    next_row = _row([10, 11, 12, 13, 14, 15, 16], 0.074, 0.0)
    converter.robot_rows = [current, next_row]

    action = converter._action_vector(current, next_row)

    np.testing.assert_allclose(action, np.array([20, 21, 22, 23, 24, 25, 26, 1.0], dtype=np.float32))
    assert converter._action_dimension_names() == [
        "q_cmd_0",
        "q_cmd_1",
        "q_cmd_2",
        "q_cmd_3",
        "q_cmd_4",
        "q_cmd_5",
        "q_cmd_6",
        "gripper_command",
    ]
