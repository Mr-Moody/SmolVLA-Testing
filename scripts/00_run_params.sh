#!/usr/bin/env bash
# Shared workflow parameters for sync, training, and extraction.

PARAMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LOCAL_PROJECT_ROOT="$(cd "${PARAMS_DIR}/.." && pwd)"

# -------- Workflow identity --------
# Set this explicitly for UCL remote paths/SSH, instead of relying on shell USER.
WORKFLOW_USER="your_ucl_username"

# -------- Project layout knobs --------
# Remote project root is discovered from remote $HOME, then this dirname is used.
REMOTE_PROJECT_DIRNAME="smolvla_project"

# -------- Core run controls --------
RUN_NAME="001_002_003"
DATASET_NAMES=(001 002 003)
DATASET_ROOT="/scratch0/${WORKFLOW_USER}/lerobot_datasets"
SAVE_FREQ=1000

# -------- Training hyperparameters --------
STEPS=20000
BATCH_SIZE=8
NUM_WORKERS=4
LOG_FREQ=50
SEED=1000
USE_AMP=true
RESUME=false

# -------- Optional runtime patches --------
ALLOW_NEAREST_FRAME_FALLBACK=true
ALLOW_MISSING_TASK_FALLBACK=true

# -------- SSH topology --------
REMOTE_USER="${WORKFLOW_USER}"
GPU_NODE="your-gpu-node.cs.ucl.ac.uk"
JUMP_HOST="knuckles.cs.ucl.ac.uk"

# -------- Remote project layout --------
REMOTE_SCRATCH_BASE="/scratch0/${REMOTE_USER}"

# -------- Local paths --------
LOCAL_PROJECT_ROOT="${DEFAULT_LOCAL_PROJECT_ROOT}"
LOCAL_DATA_PULL_SOURCE="${LOCAL_PROJECT_ROOT}/lerobot_datasets"
LOCAL_CHECKPOINTS_ROOT="${LOCAL_PROJECT_ROOT}/checkpoints"
EXTRACT_FOLDER_NAME="${RUN_NAME}_smolvla_full"

# -------- Local override (gitignored) --------
# Create scripts/00_run_params.local.sh for personal paths/hosts/user values.
# It is sourced last and can override any variable defined above.
if [[ -f "${PARAMS_DIR}/00_run_params.local.sh" ]]; then
	# shellcheck source=/dev/null
	source "${PARAMS_DIR}/00_run_params.local.sh"
fi
