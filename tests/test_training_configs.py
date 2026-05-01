"""Tests that all training configs load cleanly and have required keys."""
import pytest
import yaml
from pathlib import Path

CONFIG_DIR = Path("configs/training")
REQUIRED_KEYS = {"model_type", "dataset_path", "output_dir"}


def _load(name):
    p = CONFIG_DIR / name
    assert p.exists(), f"Config not found: {p}"
    return yaml.safe_load(p.read_text())


def test_base_config_has_optimizer():
    cfg = _load("base.yaml")
    assert "optimizer" in cfg
    assert "lr" in cfg
    assert "steps" in cfg
    assert "batch_size" in cfg


def test_smolvla_baseline_keys():
    cfg = _load("smolvla_baseline.yaml")
    assert cfg.get("model_type") == "smolvla"
    assert cfg.get("use_phase_conditioning") is False
    for k in REQUIRED_KEYS:
        assert k in cfg, f"Missing key {k}"


def test_smolvla_fsm_keys():
    cfg = _load("smolvla_fsm.yaml")
    assert cfg.get("model_type") == "smolvla_fork"
    assert cfg.get("use_phase_conditioning") is True
    assert 0.0 <= cfg.get("phase_dropout_prob", 0) <= 1.0
    for k in REQUIRED_KEYS:
        assert k in cfg


def test_pi0_subtask_keys():
    cfg = _load("pi0_subtask.yaml")
    assert cfg.get("model_type") == "pi0"
    assert cfg.get("use_subtask_column") is True
    for k in REQUIRED_KEYS:
        assert k in cfg


def test_all_dataset_paths_use_template():
    for name in ["smolvla_baseline.yaml", "smolvla_fsm.yaml", "pi0_subtask.yaml"]:
        cfg = _load(name)
        path = cfg.get("dataset_path", "")
        assert "${dataset_name}" in path, f"{name}: dataset_path should use ${'{dataset_name}'} template"
