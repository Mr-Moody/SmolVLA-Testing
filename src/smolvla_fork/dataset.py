"""Phase-conditioned dataset wrapper for SmolVLA fork training.

Wraps a LeRobotDataset and yields the ``phase`` column alongside standard
fields. If the dataset has no ``phase`` column, returns the unknown phase
index (5) for all samples.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


_UNKNOWN_PHASE = 5  # index for the "unknown" phase token


class PhaseConditionedDataset(Dataset):
    """Wraps a LeRobotDataset to expose a ``phase_id`` field.

    Args:
        lerobot_dataset: A LeRobotDataset instance.
        chunk_start_key: Key in the dataset sample that marks the start of the
            action chunk. The phase ID at this frame index is used.
        phase_label_smoothing: If > 0, sample the phase ID from a uniform
            window of ±``phase_label_smoothing`` frames around the chunk start.
            This regularises against annotation boundary noise.
    """

    def __init__(
        self,
        lerobot_dataset: Any,
        chunk_start_key: str = "frame_index",
        phase_label_smoothing: float = 0.0,
    ):
        self._ds = lerobot_dataset
        self._chunk_start_key = chunk_start_key
        self._phase_smoothing = phase_label_smoothing
        self._has_phase = self._check_has_phase()

    def _check_has_phase(self) -> bool:
        if len(self._ds) == 0:
            return False
        sample = self._ds[0]
        return "phase" in sample

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._ds[idx]

        if not self._has_phase:
            sample["phase_id"] = torch.tensor(_UNKNOWN_PHASE, dtype=torch.long)
            return sample

        # Get base phase ID at chunk start
        phase_raw = sample["phase"]
        if isinstance(phase_raw, torch.Tensor):
            base_phase = int(phase_raw.item())
        else:
            base_phase = int(phase_raw)

        # Phase label smoothing: sample from a window of ±3 frames
        if self._phase_smoothing > 0 and "frame_index" in sample:
            frame_idx = int(sample["frame_index"])
            window = 3  # ±3 frames
            # Collect phase IDs in the window (clamp to dataset bounds)
            episode_start = int(sample.get("episode_index", 0))
            lo = max(0, idx - window)
            hi = min(len(self._ds) - 1, idx + window)
            candidate_idx = random.randint(lo, hi)
            candidate = self._ds[candidate_idx]
            if "phase" in candidate:
                candidate_phase = candidate["phase"]
                if isinstance(candidate_phase, torch.Tensor):
                    base_phase = int(candidate_phase.item())
                else:
                    base_phase = int(candidate_phase)

        sample["phase_id"] = torch.tensor(base_phase, dtype=torch.long)
        return sample


def phase_collate_fn(batch):
    """Collate function that batches phase_id as a LongTensor of shape (B,).

    Use as the ``collate_fn`` argument to DataLoader when using
    ``PhaseConditionedDataset``.
    """
    from torch.utils.data.dataloader import default_collate

    # Separate phase_ids before default collate (avoids type issues)
    phase_ids = [sample.pop("phase_id", torch.tensor(_UNKNOWN_PHASE)) for sample in batch]
    collated = default_collate(batch)
    collated["phase_id"] = torch.stack(phase_ids, dim=0).long()
    return collated
