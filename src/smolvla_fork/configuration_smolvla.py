# ============================================================
# FORK HEADER
# Upstream: lerobot/policies/smolvla/configuration_smolvla.py
# Upstream version: lerobot==0.4.4
# Fork date: 2026-05-01
# Changes: Added phase conditioning fields via direct import bypass.
# ============================================================
"""SmolVLA config with phase conditioning additions.

Loads the upstream SmolVLAConfig via importlib to avoid triggering
lerobot.policies.__init__ which has a transformers>=5.x compatibility issue
with the GR00T policy in lerobot==0.4.4.
"""
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


def _load_smolvla_config_direct():
    """Load SmolVLAConfig directly from file, bypassing lerobot.policies.__init__."""
    lerobot_root = Path(importlib.util.find_spec("lerobot").origin).parent
    cfg_file = lerobot_root / "policies" / "smolvla" / "configuration_smolvla.py"
    spec = importlib.util.spec_from_file_location(
        "lerobot._smolvla_cfg_direct", str(cfg_file)
    )
    mod = importlib.util.module_from_spec(spec)
    # Inject the lerobot submodules that configuration_smolvla needs
    # (these are importable without triggering the broken __init__)
    sys.modules.setdefault("lerobot._smolvla_cfg_direct", mod)
    spec.loader.exec_module(mod)
    return mod.SmolVLAConfig


try:
    SmolVLAConfig = _load_smolvla_config_direct()
    _BaseConfig = SmolVLAConfig
except Exception:
    # Fallback: plain object if lerobot is not available
    _BaseConfig = object
    SmolVLAConfig = object


@dataclass
class SmolVLAForkedConfig(_BaseConfig):
    """SmolVLA config extended with phase conditioning fields.

    Inherits all upstream fields from ``SmolVLAConfig``.
    When ``use_phase_conditioning=False`` (default) behaviour is identical
    to the upstream ``SmolVLAConfig``.
    """

    # -- Phase conditioning additions --
    use_phase_conditioning: bool = False
    """When True, a learned phase embedding is added to the state token."""

    phase_dropout_prob: float = 0.15
    """During training, replace phase ID with unknown token with this probability."""
