#!/usr/bin/env bash
# =============================================================================
# params.sh  —  Local GPU training parameters
# =============================================================================
# Sourced by all scripts_local/ scripts. Edit this file to configure your run.
# Create scripts_local/params.local.sh for personal overrides (gitignored).

PARAMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${PARAMS_DIR}/.." && pwd)"

# -------- Project layout --------
LEROBOT_ROOT="${PROJECT_ROOT}/../lerobot"
RAW_DATASET_ROOT="${PROJECT_ROOT}/raw_datasets"
CLEANED_DATASET_ROOT="${PROJECT_ROOT}/cleaned_datasets"
LEROBOT_DATASET_ROOT="${PROJECT_ROOT}/lerobot_datasets"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs"

# Dedicated venv for SmolVLA training — keeps lerobot/.venv untouched so
# camera/data-collection packages installed there are not affected.
VENV_DIR="${PROJECT_ROOT}/.venv"

# -------- Run identity --------
MODEL_TYPE="act"              # smolvla | act
RUN_NAME="local_run"
DATASET_NAMES=(100 101 102 103)

# -------- Preprocessing --------
PRIMARY_CAMERA="ee_zed_m_left"
VCODEC="h264"                 # h264 (software) | h264_nvenc (NVIDIA encoder)
# Set to true to skip the clean step (raw → cleaned) if already done.
# NOTE: always set this to true if you have manually labelled annotations in
# cleaned_datasets/ — the clean step wipes the output dir, so labels would be
# lost unless they are also present in raw_datasets/ (see 02_run.sh which backs
# them up automatically, but SKIP_CLEAN=true is safest when labels already exist).
SKIP_CLEAN=true

# -------- Training hyperparameters --------
STEPS=20000
BATCH_SIZE=8
NUM_WORKERS=4
LOG_FREQ=50
SAVE_FREQ=1000
SEED=1000
USE_AMP=true
RESUME=false
DEVICE="cuda"

# -------- Model-specific knobs --------
SMOLVLA_POLICY_PATH="lerobot/smolvla_base"
ACT_POLICY_PATH="lerobot/act_base"
ACT_CHUNK_SIZE=100
ACT_N_OBS_STEPS=1
ACT_USE_VAE=true
ACT_VISION_BACKBONE="resnet18"

# -------- Local override (gitignored) --------
if [[ -f "${PARAMS_DIR}/params.local.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PARAMS_DIR}/params.local.sh"
fi
