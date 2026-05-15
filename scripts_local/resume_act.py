#!/usr/bin/env python3
"""
Resume ACT training from the last checkpoint.

lerobot's parser.parse_arg("config_path") reads from sys.argv, not from cfg attributes.
This wrapper injects --config_path into sys.argv so validate() can find it, then
calls train_act with resume=True.

The lerobot/configs/train.py validate() was also patched on the GPU to set the
optimizer/scheduler preset even when resuming (when cfg.optimizer is None), because
train_act.py builds cfg manually without saving the optimizer config.

Usage (from smolvla_project/SmolVLA-Testing on the GPU node):
    source /scratch0/eredhead/smolvla_venv/bin/activate
    python scripts_local/resume_act.py
"""

import os
import sys
from pathlib import Path

OUTPUT_DIR = Path("/scratch0/eredhead/smolvla_outputs/msd_plug_200_204_act")
# config_path must point to train_config.json inside the checkpoint dir so that
# validate() sets policy_dir = checkpoints/last/pretrained_model/ (where model.safetensors is).
CONFIG_PATH = OUTPUT_DIR / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
DATASET_ROOT = Path("/scratch0/eredhead/lerobot_datasets/merged_msd_plug_200_204_act")
LEROBOT_ROOT = Path("/cs/student/ug/2024/eredhead/smolvla_project/lerobot")

os.environ["HF_HUB_OFFLINE"] = "1"

# Inject config_path into sys.argv so lerobot's parser.parse_arg("config_path") finds it.
sys.argv.append(f"--config_path={CONFIG_PATH}")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.act import train_act

train_act(
    dataset_root=DATASET_ROOT,
    lerobot_root=LEROBOT_ROOT,
    policy_path="lerobot/act_base",
    output_dir=OUTPUT_DIR,
    batch_size=8,
    steps=200000,
    device="cuda",
    num_workers=0,
    log_freq=50,
    save_freq=5000,
    eval_freq=0,
    seed=1000,
    use_amp=True,
    resume=True,
    episodes=None,
    job_name="msd_plug_200_204_act",
    push_to_hub=False,
    policy_repo_id=None,
    chunk_size=100,
    n_obs_steps=1,
    use_vae=True,
    vision_backbone="resnet18",
)
