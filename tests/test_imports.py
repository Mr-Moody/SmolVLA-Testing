"""Smoke test: verify core dependencies import cleanly."""
import importlib
import pytest


def _importable(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _importable("torch"), reason="torch not installed")
def test_torch_imports():
    import torch
    assert torch.__version__


@pytest.mark.skipif(not _importable("transformers"), reason="transformers not installed")
def test_transformers_imports():
    import transformers
    assert transformers.__version__


@pytest.mark.skipif(not _importable("lerobot"), reason="lerobot not installed")
def test_lerobot_imports():
    import lerobot
    assert lerobot is not None


def test_core_imports():
    """At least one of torch/transformers/lerobot must be importable."""
    any_available = any(_importable(m) for m in ["torch", "transformers", "lerobot"])
    assert any_available, (
        "None of torch, transformers, or lerobot are importable. "
        "Run: uv --project ../lerobot run python -m pytest tests/test_imports.py"
    )
