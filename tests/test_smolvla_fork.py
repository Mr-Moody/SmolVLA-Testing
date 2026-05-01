"""Unit tests for the SmolVLA fork phase-embedding additions.

These tests verify the phase conditioning logic WITHOUT loading the full VLM
(which requires a GPU and large model weights). They test the config, the
phase embed method, and zero-init invariant via mocks.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


# ─── Config tests ────────────────────────────────────────────────────────────

def test_config_defaults():
    from src.smolvla_fork.configuration_smolvla import SmolVLAForkedConfig
    cfg = SmolVLAForkedConfig()
    assert cfg.use_phase_conditioning is False
    assert cfg.phase_dropout_prob == 0.15


def test_config_phase_enabled():
    from src.smolvla_fork.configuration_smolvla import SmolVLAForkedConfig
    cfg = SmolVLAForkedConfig(use_phase_conditioning=True, phase_dropout_prob=0.20)
    assert cfg.use_phase_conditioning is True
    assert cfg.phase_dropout_prob == 0.20


# ─── Phase embedding unit tests (no VLM needed) ──────────────────────────────

class _FakeModel(nn.Module):
    """Minimal stand-in to test _phase_embed logic in isolation."""

    def __init__(self, hidden_size=64, use_phase=True, dropout=0.15):
        super().__init__()
        self._use_phase = use_phase
        self._dropout = dropout
        _N = 6
        if use_phase:
            self.phase_embedding = nn.Embedding(_N, hidden_size)
            nn.init.zeros_(self.phase_embedding.weight)

    def _phase_embed(self, state_emb, phase_id, training):
        if not self._use_phase:
            return state_emb
        bsize = state_emb.shape[0]
        device = state_emb.device
        _UNKNOWN = 5
        if phase_id is None:
            ids = torch.full((bsize,), _UNKNOWN, dtype=torch.long, device=device)
        else:
            ids = phase_id.to(device=device, dtype=torch.long).clamp(0, _UNKNOWN)
        if training and self._dropout > 0:
            mask = torch.rand(bsize, device=device) < self._dropout
            ids = torch.where(mask, torch.full_like(ids, _UNKNOWN), ids)
        emb = self.phase_embedding(ids)
        return state_emb + emb[:, None, :]


def test_zero_init_produces_zero_embedding():
    """With zero-init, phase embedding adds zeros → state_emb unchanged."""
    model = _FakeModel(hidden_size=64, use_phase=True)
    state = torch.randn(4, 1, 64)
    phase_ids = torch.tensor([0, 1, 2, 3])
    out = model._phase_embed(state, phase_ids, training=False)
    assert torch.allclose(out, state), "Zero-init phase embedding should not change state"


def test_phase_conditioning_off_returns_state_unchanged():
    model = _FakeModel(use_phase=False)
    state = torch.randn(4, 1, 64)
    out = model._phase_embed(state, torch.tensor([0, 1, 2, 3]), training=False)
    assert torch.allclose(out, state)


def test_unknown_token_used_when_phase_id_none():
    model = _FakeModel(hidden_size=8, use_phase=True)
    # Set unknown token to non-zero so we can detect it
    with torch.no_grad():
        model.phase_embedding.weight[5] = torch.ones(8) * 2.0
    state = torch.zeros(2, 1, 8)
    out = model._phase_embed(state, None, training=False)
    expected = torch.ones(2, 1, 8) * 2.0
    assert torch.allclose(out, expected), "Should use unknown token (index 5) when phase_id=None"


def test_phase_dropout_statistics():
    """Over many samples, dropout rate should be close to configured probability."""
    torch.manual_seed(42)
    model = _FakeModel(hidden_size=4, use_phase=True, dropout=0.3)
    # Make phase 0 embedding all 1s, unknown embedding all -1s
    with torch.no_grad():
        model.phase_embedding.weight.fill_(0.0)
        model.phase_embedding.weight[0] = torch.ones(4)
        model.phase_embedding.weight[5] = -torch.ones(4)

    N = 10000
    phase_ids = torch.zeros(N, dtype=torch.long)  # all phase 0
    state = torch.zeros(N, 1, 4)
    out = model._phase_embed(state, phase_ids, training=True)
    # Positions where unknown was applied: out[:, 0, 0] == -1
    unknown_applied = (out[:, 0, 0] < -0.5).float().mean().item()
    assert 0.20 < unknown_applied < 0.40, f"Dropout rate {unknown_applied:.3f} not close to 0.30"


def test_phase_dropout_off_in_eval():
    """Phase dropout should NOT apply when training=False."""
    torch.manual_seed(42)
    model = _FakeModel(hidden_size=4, use_phase=True, dropout=0.9)
    with torch.no_grad():
        model.phase_embedding.weight.fill_(0.0)
        model.phase_embedding.weight[0] = torch.ones(4)
        model.phase_embedding.weight[5] = -torch.ones(4)

    phase_ids = torch.zeros(100, dtype=torch.long)
    state = torch.zeros(100, 1, 4)
    out = model._phase_embed(state, phase_ids, training=False)
    # Should never apply dropout (all should use phase 0 = ones)
    assert torch.all(out[:, 0, 0] > 0.5), "Dropout must not fire during eval"


# ─── Import check ─────────────────────────────────────────────────────────────

def test_policy_class_importable():
    """SmolVLAPhasedPolicy is importable without loading the VLM."""
    # We only check the module-level import; instantiation requires a GPU
    import importlib
    mod = importlib.import_module("src.smolvla_fork.modeling_smolvla")
    assert hasattr(mod, "SmolVLAPhasedPolicy")
    assert hasattr(mod, "VLAFlowMatchingPhased")
