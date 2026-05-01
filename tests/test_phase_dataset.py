"""Tests for PhaseConditionedDataset and phase_collate_fn."""
import pytest
import torch
from torch.utils.data import DataLoader

from src.smolvla_fork.dataset import PhaseConditionedDataset, phase_collate_fn, _UNKNOWN_PHASE


class _FakeLeRobotDataset:
    """Minimal mock of LeRobotDataset for unit tests."""

    def __init__(self, n=20, has_phase=True):
        self._n = n
        self._has_phase = has_phase

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        sample = {
            "observation.state": torch.randn(7),
            "action": torch.randn(7),
            "frame_index": idx,
            "episode_index": 0,
        }
        if self._has_phase:
            sample["phase"] = torch.tensor(idx % 5, dtype=torch.long)
        return sample


def test_phase_returned_from_dataset():
    ds = PhaseConditionedDataset(_FakeLeRobotDataset(has_phase=True))
    sample = ds[0]
    assert "phase_id" in sample
    assert sample["phase_id"].item() == 0  # frame 0 → phase 0


def test_unknown_phase_when_no_phase_column():
    ds = PhaseConditionedDataset(_FakeLeRobotDataset(has_phase=False))
    sample = ds[3]
    assert "phase_id" in sample
    assert sample["phase_id"].item() == _UNKNOWN_PHASE


def test_phase_id_is_long_tensor():
    ds = PhaseConditionedDataset(_FakeLeRobotDataset(has_phase=True))
    sample = ds[2]
    assert sample["phase_id"].dtype == torch.long


def test_collate_fn_batches_phase_ids():
    ds = PhaseConditionedDataset(_FakeLeRobotDataset(has_phase=True))
    loader = DataLoader(ds, batch_size=4, collate_fn=phase_collate_fn)
    batch = next(iter(loader))
    assert "phase_id" in batch
    assert batch["phase_id"].shape == (4,)
    assert batch["phase_id"].dtype == torch.long


def test_collate_fn_without_phase_column():
    ds = PhaseConditionedDataset(_FakeLeRobotDataset(has_phase=False))
    loader = DataLoader(ds, batch_size=4, collate_fn=phase_collate_fn)
    batch = next(iter(loader))
    assert "phase_id" in batch
    assert (batch["phase_id"] == _UNKNOWN_PHASE).all()


def test_len():
    ds = PhaseConditionedDataset(_FakeLeRobotDataset(n=15))
    assert len(ds) == 15


def test_phase_label_smoothing_doesnt_crash():
    """Phase label smoothing should not raise; just produces varied phase IDs."""
    ds = PhaseConditionedDataset(
        _FakeLeRobotDataset(has_phase=True, n=50),
        phase_label_smoothing=0.5,
    )
    for i in range(0, 50, 5):
        sample = ds[i]
        assert 0 <= sample["phase_id"].item() <= 4
