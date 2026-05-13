#!/usr/bin/env bash
# =============================================================================
# 03_hf_train.sh  —  Download HF datasets → Validate → Merge → Train (local GPU)
# =============================================================================
#
# Usage:
#   bash scripts_local/03_hf_train.sh [--resume] [--force-download]
#
# Pipeline:
#   1. Download datasets from HuggingFace (cached in .hf_cache/)
#   2. Validate each dataset (cleaned, converted, annotated)
#   3. Merge if multiple repos specified
#   4. Apply lerobot patches + launch training
#
# Edit scripts_local/hf_params.sh (or hf_params.local.sh) to configure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hf_params.sh"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
RESUME=false
FORCE_DOWNLOAD=false
for arg in "$@"; do
    [[ "${arg}" == "--resume" ]]         && RESUME=true
    [[ "${arg}" == "--force-download" ]] && FORCE_DOWNLOAD=true
done

RUN_TAG="${RUN_NAME}_${MODEL_TYPE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_FILE="${OUTPUT_ROOT}/${RUN_TAG}.log"

UV="UV_PROJECT_ENVIRONMENT=${VENV_DIR} uv --project ${LEROBOT_ROOT} run python"

# ---------------------------------------------------------------------------
# Validate pre-conditions
# ---------------------------------------------------------------------------
if [[ ! -d "${LEROBOT_ROOT}/src" ]]; then
    echo "ERROR: lerobot not found at ${LEROBOT_ROOT}. Run 01_setup.sh first."
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: training venv not found at ${VENV_DIR}. Run 01_setup.sh first."
    exit 1
fi

if [[ "${#HF_REPO_IDS[@]}" -eq 0 ]]; then
    echo "ERROR: HF_REPO_IDS is empty. Set it in hf_params.sh or hf_params.local.sh."
    exit 1
fi

mkdir -p "${LEROBOT_DATASET_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

# ---------------------------------------------------------------------------
# Step 1: Download + validate HF datasets
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Downloading and validating HF datasets..."
echo "  Repos: ${HF_REPO_IDS[*]}"
echo "  Cache: ${HF_CACHE_ROOT}"

PREPARE_ARGS=("${HF_REPO_IDS[@]}")
PREPARE_ARGS+=(--cache-root "${HF_CACHE_ROOT}")

if [[ "${FORCE_DOWNLOAD}" == "true" ]]; then
    PREPARE_ARGS+=(--force)
fi

# If multiple repos, set up merge output
if [[ "${#HF_REPO_IDS[@]}" -gt 1 ]]; then
    MERGED_ROOT="${LEROBOT_DATASET_ROOT}/merged_${RUN_TAG}"
    PREPARE_ARGS+=(--merge-output "${MERGED_ROOT}")
fi

TRAIN_DATASET_ROOT=$(${UV} "${PROJECT_ROOT}/src/hf/dataset_hub.py" prepare \
    "${PREPARE_ARGS[@]}")

echo "  Dataset ready: ${TRAIN_DATASET_ROOT}"

# ---------------------------------------------------------------------------
# Step 2: Apply lerobot patches
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Applying lerobot patches..."

${UV} "${PROJECT_ROOT}/src/patch_frame_tolerance.py"
echo "  patch_frame_tolerance applied."

${UV} "${PROJECT_ROOT}/src/patch_task_none.py"
echo "  patch_task_none applied."

# ---------------------------------------------------------------------------
# Step 3: Train
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Launching training..."

case "${MODEL_TYPE}" in
    smolvla)
        POLICY_PATH="${SMOLVLA_POLICY_PATH}"
        TRAIN_MODEL_ARGS=(--model-type smolvla)
        ;;
    act)
        POLICY_PATH="${ACT_POLICY_PATH}"
        TRAIN_MODEL_ARGS=(
            --model-type      act
            --chunk-size      "${ACT_CHUNK_SIZE}"
            --n-obs-steps     "${ACT_N_OBS_STEPS}"
            --vision-backbone "${ACT_VISION_BACKBONE}"
        )
        if [[ "${ACT_USE_VAE}" == "true" ]]; then
            TRAIN_MODEL_ARGS+=(--use-vae)
        else
            TRAIN_MODEL_ARGS+=(--no-vae)
        fi
        ;;
    *)
        echo "ERROR: Unknown MODEL_TYPE='${MODEL_TYPE}'. Use smolvla or act."
        exit 1
        ;;
esac

RESUME_FLAG=""
[[ "${RESUME}" == "true" ]] && RESUME_FLAG="--resume"

AMP_FLAG=""
[[ "${USE_AMP}" == "true" ]] && AMP_FLAG="--use-amp"

if [[ -d "${OUTPUT_DIR}/checkpoints" ]] && \
   [[ -n "$(ls -A "${OUTPUT_DIR}/checkpoints" 2>/dev/null)" ]] && \
   [[ "${RESUME}" != "true" ]]; then
    echo ""
    echo "ERROR: Existing checkpoints found in ${OUTPUT_DIR}."
    echo "Use: bash scripts_local/03_hf_train.sh --resume"
    exit 1
fi

echo ""
echo "Training: ${TRAIN_DATASET_ROOT} → ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo ""

${UV} "${PROJECT_ROOT}/main.py" train \
    "${TRAIN_MODEL_ARGS[@]}" \
    --dataset-root  "${TRAIN_DATASET_ROOT}" \
    --lerobot-root  "${LEROBOT_ROOT}" \
    --policy-path   "${POLICY_PATH}" \
    --output-dir    "${OUTPUT_DIR}" \
    --batch-size    "${BATCH_SIZE}" \
    --steps         "${STEPS}" \
    --num-workers   "${NUM_WORKERS}" \
    --save-freq     "${SAVE_FREQ}" \
    --log-freq      "${LOG_FREQ}" \
    --seed          "${SEED}" \
    --device        "${DEVICE}" \
    --eval-freq     0 \
    ${AMP_FLAG} \
    ${RESUME_FLAG} \
    2>&1 | tee "${LOG_FILE}"
