#!/usr/bin/env bash
# Shared workflow parameters for sync, training, and extraction.

PARAMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOCAL_PROJECT_ROOT="$(cd "${PARAMS_DIR}/.." && pwd)"

# -------- Workflow identity --------
# Set this explicitly for UCL remote paths/SSH, instead of relying on shell USER.
WORKFLOW_USER="eredhead"

# -------- Project layout knobs --------
# Remote project root is discovered from remote $HOME, then this dirname is used.
REMOTE_PROJECT_DIRNAME="smolvla_project"

# -------- Core run controls --------
# MSD plug insertion run — 11 datasets (200-209), SmolVLA 25k steps.
MODEL_TYPE="smolvla"  # smolvla | act
RUN_NAME="msd_plug_200_209"
DATASET_NAMES=(200 201 201_1 202 203 204 205 206 207 208 209)
DATASET_ROOT="/scratch0/${WORKFLOW_USER}/lerobot_datasets"
SAVE_FREQ=2000

# -------- Preprocessing controls --------
# false: individual lerobot_datasets already on GPU scratch from SmolVLA run — skip re-conversion.
PREPROCESS_ON_GPU=false
REMOTE_CLEANED_DATASET_ROOT="/scratch0/${WORKFLOW_USER}/cleaned_datasets"
PREPROCESS_VCODEC="h264_nvenc"
# Primary camera for the convert step; wrist_d405 only (no third-person camera).
PRIMARY_CAMERA="wrist_d405"
# Default task description for MSD plug insertion task.
TASK_DESCRIPTION="Insert the MSD connector into the socket"
# -------- Training hyperparameters --------
STEPS=25000
BATCH_SIZE=8
NUM_WORKERS=4
LOG_FREQ=50
SEED=1000
USE_AMP=true
RESUME=false

# -------- Model-specific knobs --------
SMOLVLA_POLICY_PATH="lerobot/smolvla_base"
ACT_POLICY_PATH="lerobot/act_base"
ACT_CHUNK_SIZE=100
ACT_N_OBS_STEPS=1
ACT_USE_VAE=true
ACT_VISION_BACKBONE="resnet18"

# -------- Optional runtime patches --------
ALLOW_NEAREST_FRAME_FALLBACK=true
ALLOW_MISSING_TASK_FALLBACK=true

# -------- SSH topology --------
REMOTE_USER="${WORKFLOW_USER}"
GPU_NODE="bluestreak.cs.ucl.ac.uk"
JUMP_HOST="knuckles.cs.ucl.ac.uk"
# SSH key for passwordless access. Generate once with:
#   ssh-keygen -t ed25519 -f ~/.ssh/ucl_key -N ""
# Then install on the GPU node from an active session:
#   echo "$(cat ~/.ssh/ucl_key.pub)" >> ~/.ssh/authorized_keys
SSH_KEY_FILE="${HOME}/.ssh/ucl_key"

# -------- Remote project layout --------
REMOTE_SCRATCH_BASE="/scratch0/${REMOTE_USER}"

# -------- Local paths --------
LOCAL_PROJECT_ROOT="${DEFAULT_LOCAL_PROJECT_ROOT}"
LOCAL_DATA_PULL_SOURCE="${LOCAL_PROJECT_ROOT}/lerobot_datasets"
# Synced to REMOTE_CLEANED_DATASET_ROOT when PREPROCESS_ON_GPU=true.
LOCAL_CLEANED_DATA_SOURCE="${LOCAL_PROJECT_ROOT}/cleaned_datasets"
LOCAL_CHECKPOINTS_ROOT="${LOCAL_PROJECT_ROOT}/checkpoints"
EXTRACT_FOLDER_NAME="${RUN_NAME}_${MODEL_TYPE}_full"

# -------- Local override (gitignored) --------
# Create scripts/00_run_params.local.sh for personal paths/hosts/user values.
# It is sourced last and can override any variable defined above.
if [[ -f "${PARAMS_DIR}/00_run_params.local.sh" ]]; then
	# shellcheck source=/dev/null
	source "${PARAMS_DIR}/00_run_params.local.sh"
fi
