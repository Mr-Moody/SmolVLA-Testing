#!/usr/bin/env bash
# =============================================================================
# 06_hf_prep_and_train.sh — Download HF datasets on GPU node → Validate → Train
# =============================================================================
#
# This script runs DIRECTLY ON THE GPU NODE (not via SSH from local).
# Launch it the same way as 04_start_training.sh — after syncing code via
# 01_sync_to_gpu.sh and setting up the environment via 03_setup_gpu.sh.
#
# Usage (on the GPU node):
#   bash scripts/06_hf_prep_and_train.sh [--resume] [--force-download]
#
# Or from local via SSH:
#   ssh -J $JUMP_HOST $GPU_NODE "cd ~/smolvla_project/SmolVLA-Testing && \
#       bash scripts/06_hf_prep_and_train.sh"
#
# Pipeline:
#   1. Download datasets from HuggingFace Hub (online, cached on scratch)
#   2. Validate each dataset (cleaned, converted, annotated)
#   3. Merge if multiple repos
#   4. Apply lerobot patches + launch training (offline, nohup)
#
# Configure HF_REPO_IDS in scripts/00_run_params.local.sh (gitignored):
#   HF_REPO_IDS=("NexusDwin/msd-connector-200-209")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOME_PROJECT_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
CODE_DIR="${REPO_ROOT}"
LEROBOT_DIR="${HOME_PROJECT_DIR}/lerobot"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
RESUME=false
FORCE_DOWNLOAD=false
for arg in "$@"; do
    [[ "${arg}" == "--resume" ]]         && RESUME=true
    [[ "${arg}" == "--force-download" ]] && FORCE_DOWNLOAD=true
done

# ---------------------------------------------------------------------------
# HF-specific configuration
# HF_REPO_IDS must be set in 00_run_params.local.sh
# ---------------------------------------------------------------------------
if [[ -z "${HF_REPO_IDS+x}" ]] || [[ "${#HF_REPO_IDS[@]}" -eq 0 ]]; then
    echo "ERROR: HF_REPO_IDS not set. Add to scripts/00_run_params.local.sh:"
    echo '  HF_REPO_IDS=("NexusDwin/msd-connector-200-209")'
    exit 1
fi

HF_CACHE_ROOT="${HF_CACHE_ROOT:-${REMOTE_SCRATCH_BASE}/hf_datasets}"
RUN_TAG="${RUN_NAME}_${MODEL_TYPE}"
SCRATCH_OUTPUT_DIR="${REMOTE_SCRATCH_BASE}/smolvla_outputs/${RUN_TAG}"
LOG_FILE="${REMOTE_SCRATCH_BASE}/smolvla_outputs/${RUN_TAG}.log"
ACTIVATE_SHIM="${REMOTE_SCRATCH_BASE}/activate_smolvla.sh"

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
if [[ ! -f "${ACTIVATE_SHIM}" ]]; then
    echo "ERROR: Missing ${ACTIVATE_SHIM}. Run 03_setup_gpu.sh first."
    exit 1
fi

if [[ ! -d "${LEROBOT_DIR}" ]]; then
    echo "ERROR: lerobot repo not found at ${LEROBOT_DIR}"
    exit 1
fi

source "${ACTIVATE_SHIM}"
mkdir -p "${REMOTE_SCRATCH_BASE}/smolvla_outputs"
mkdir -p "${HF_CACHE_ROOT}"

# ---------------------------------------------------------------------------
# Model-specific arguments (same as 04_start_training.sh)
# ---------------------------------------------------------------------------
case "${MODEL_TYPE}" in
    smolvla)
        POLICY_PATH="${POLICY_PATH:-${SMOLVLA_POLICY_PATH}}"
        TRAIN_MODEL_ARGS=(--model-type smolvla)
        ;;
    act)
        POLICY_PATH="${POLICY_PATH:-${ACT_POLICY_PATH}}"
        TRAIN_MODEL_ARGS=(
            --model-type act
            --chunk-size "${ACT_CHUNK_SIZE}"
            --n-obs-steps "${ACT_N_OBS_STEPS}"
            --vision-backbone "${ACT_VISION_BACKBONE}"
        )
        if [[ "${ACT_USE_VAE}" == "true" ]]; then
            TRAIN_MODEL_ARGS+=(--use-vae)
        else
            TRAIN_MODEL_ARGS+=(--no-vae)
        fi
        ;;
    *)
        echo "ERROR: Unsupported MODEL_TYPE='${MODEL_TYPE}'. Use smolvla or act."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Step 1: Download + validate HF datasets (ONLINE — do NOT set HF_HUB_OFFLINE)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 1: Downloading and validating HF datasets ==="
echo "  Repos: ${HF_REPO_IDS[*]}"
echo "  Cache: ${HF_CACHE_ROOT}"

PREPARE_ARGS=("${HF_REPO_IDS[@]}")
PREPARE_ARGS+=(--cache-root "${HF_CACHE_ROOT}")

if [[ "${FORCE_DOWNLOAD}" == "true" ]]; then
    PREPARE_ARGS+=(--force)
fi

if [[ "${#HF_REPO_IDS[@]}" -gt 1 ]]; then
    MERGED_ROOT="${DATASET_ROOT}/merged_${RUN_TAG}"
    PREPARE_ARGS+=(--merge-output "${MERGED_ROOT}")
fi

cd "${LEROBOT_DIR}"
TRAIN_DATASET_ROOT=$(uv run python "${CODE_DIR}/src/hf/dataset_hub.py" prepare \
    "${PREPARE_ARGS[@]}")

echo "  Dataset ready: ${TRAIN_DATASET_ROOT}"

# ---------------------------------------------------------------------------
# Step 2: Checkpoint collision check
# ---------------------------------------------------------------------------
if [[ -d "${SCRATCH_OUTPUT_DIR}/checkpoints" ]] && \
   [[ -n "$(ls -A "${SCRATCH_OUTPUT_DIR}/checkpoints" 2>/dev/null)" ]] && \
   [[ "${RESUME}" != "true" ]]; then
    echo "ERROR: Existing checkpoints found in ${SCRATCH_OUTPUT_DIR}."
    echo "Use: bash scripts/06_hf_prep_and_train.sh --resume"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Apply lerobot patches
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Applying lerobot patches ==="

if [[ "${ALLOW_NEAREST_FRAME_FALLBACK:-true}" == "true" ]]; then
    python "${CODE_DIR}/src/patch_frame_tolerance.py"
    echo "  patch_frame_tolerance applied."
fi

if [[ "${ALLOW_MISSING_TASK_FALLBACK:-true}" == "true" ]]; then
    python "${CODE_DIR}/src/patch_task_none.py"
    echo "  patch_task_none applied."
fi

# ---------------------------------------------------------------------------
# Step 4: Launch training (OFFLINE — set HF_HUB_OFFLINE=1 to prevent lookups)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Launching training ==="

RESUME_FLAG=""
[[ "${RESUME}" == "true" ]] && RESUME_FLAG="--resume"

AMP_FLAG=""
[[ "${USE_AMP}" == "true" ]] && AMP_FLAG="--use-amp"

TRAIN_CMD="cd ${LEROBOT_DIR} && uv run python ${CODE_DIR}/main.py train \
    ${TRAIN_MODEL_ARGS[*]} \
    --dataset-root ${TRAIN_DATASET_ROOT} \
    --lerobot-root ${LEROBOT_DIR} \
    --policy-path ${POLICY_PATH} \
    --output-dir ${SCRATCH_OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --steps ${STEPS} \
    --num-workers ${NUM_WORKERS} \
    --save-freq ${SAVE_FREQ} \
    --log-freq ${LOG_FREQ} \
    --seed ${SEED} \
    --device cuda \
    --eval-freq 0 \
    ${AMP_FLAG} \
    ${RESUME_FLAG}"

echo "Run name: ${RUN_TAG}"
echo "Dataset:  ${TRAIN_DATASET_ROOT}"
echo "Output:   ${SCRATCH_OUTPUT_DIR}"
echo "Log:      ${LOG_FILE}"

nohup bash -c "source '${ACTIVATE_SHIM}'; \
    export HF_HUB_OFFLINE=1; \
    export TORCH_HOME='${REMOTE_SCRATCH_BASE}/.cache/torch'; \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
    ${TRAIN_CMD}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

echo ""
echo "Training started. PID=${TRAIN_PID}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "GPU:     watch -n 2 nvidia-smi"
