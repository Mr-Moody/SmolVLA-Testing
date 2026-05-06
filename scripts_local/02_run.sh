#!/usr/bin/env bash
# =============================================================================
# 02_run.sh  —  Convert datasets and launch training on the local GPU
# =============================================================================
#
# Usage:
#   bash scripts_local/02_run.sh [--resume]
#
# Pipeline:
#   cleaned_datasets/<name>/  →  lerobot_datasets/<name>/  →  outputs/<run>/
#
# Edit scripts_local/params.sh (or params.local.sh) to configure the run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/params.sh"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
RESUME=false
for arg in "$@"; do
    [[ "${arg}" == "--resume" ]] && RESUME=true
done

RUN_TAG="${RUN_NAME}_${MODEL_TYPE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_FILE="${OUTPUT_ROOT}/${RUN_TAG}.log"

mkdir -p "${OUTPUT_ROOT}"

# Shorthand: run python via the lerobot project (matches CLAUDE.md convention)
UV="uv --project ${LEROBOT_ROOT} run python"

# ---------------------------------------------------------------------------
# Create local data directories if they don't exist yet
# ---------------------------------------------------------------------------
mkdir -p "${PROJECT_ROOT}/raw_datasets"
mkdir -p "${CLEANED_DATASET_ROOT}"
mkdir -p "${LEROBOT_DATASET_ROOT}"
mkdir -p "${PROJECT_ROOT}/checkpoints"
mkdir -p "${OUTPUT_ROOT}"

# ---------------------------------------------------------------------------
# Validate pre-conditions
# ---------------------------------------------------------------------------
if [[ ! -d "${LEROBOT_ROOT}/src" ]]; then
    echo "ERROR: lerobot not found at ${LEROBOT_ROOT}. Run 01_setup.sh first."
    exit 1
fi

for ds in "${DATASET_NAMES[@]}"; do
    if [[ ! -d "${CLEANED_DATASET_ROOT}/${ds}" ]]; then
        echo "ERROR: cleaned dataset '${ds}' not found at ${CLEANED_DATASET_ROOT}/${ds}"
        echo "       Copy it into ${CLEANED_DATASET_ROOT}/ before running."
        exit 1
    fi
done

echo "================================================================="
echo "  SmolVLA Local Training"
echo "================================================================="
echo "  Run       : ${RUN_TAG}"
echo "  Model     : ${MODEL_TYPE}"
echo "  Datasets  : ${DATASET_NAMES[*]}"
echo "  Device    : ${DEVICE}"
echo "  Steps     : ${STEPS}  |  Batch: ${BATCH_SIZE}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Log       : ${LOG_FILE}"
echo "  Resume    : ${RESUME}"
echo "================================================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1: Convert each cleaned dataset → lerobot_datasets/
# ---------------------------------------------------------------------------
echo "[1/3] Converting cleaned datasets..."

for ds in "${DATASET_NAMES[@]}"; do
    echo "  Converting '${ds}'..."
    ${UV} "${PROJECT_ROOT}/main.py" convert "${ds}" \
        --datasets-root  "${CLEANED_DATASET_ROOT}" \
        --output-root    "${LEROBOT_DATASET_ROOT}" \
        --primary-camera "${PRIMARY_CAMERA}" \
        --vcodec         "${VCODEC}" \
        --force
    echo "  Done: ${LEROBOT_DATASET_ROOT}/${ds}"
done

# ---------------------------------------------------------------------------
# Step 2: Merge datasets if more than one
# ---------------------------------------------------------------------------
TRAIN_DATASET_ROOT="${LEROBOT_DATASET_ROOT}/${DATASET_NAMES[0]}"

if [[ "${#DATASET_NAMES[@]}" -gt 1 ]]; then
    DATASET_NAME_JOINED=$(IFS="_"; echo "${DATASET_NAMES[*]}")
    MERGED_ROOT="${LEROBOT_DATASET_ROOT}/merged_${DATASET_NAME_JOINED}"
    echo ""
    echo "[2/3] Merging ${#DATASET_NAMES[@]} datasets into ${MERGED_ROOT}..."

    DATASET_PATHS=()
    for ds in "${DATASET_NAMES[@]}"; do
        DATASET_PATHS+=("${LEROBOT_DATASET_ROOT}/${ds}")
    done

    ${UV} "${PROJECT_ROOT}/src/merge_datasets.py" \
        "${DATASET_PATHS[@]}" \
        --output "${MERGED_ROOT}" \
        --force

    TRAIN_DATASET_ROOT="${MERGED_ROOT}"
    echo "  Merge complete: ${TRAIN_DATASET_ROOT}"
else
    echo ""
    echo "[2/3] Single dataset — skipping merge."
fi

# ---------------------------------------------------------------------------
# Step 3: Apply lerobot runtime patches
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Applying lerobot patches..."

${UV} "${PROJECT_ROOT}/src/patch_frame_tolerance.py"
echo "  patch_frame_tolerance applied."

${UV} "${PROJECT_ROOT}/src/patch_task_none.py"
echo "  patch_task_none applied."

# ---------------------------------------------------------------------------
# Build training command
# ---------------------------------------------------------------------------
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
    echo "Use: bash scripts_local/02_run.sh --resume"
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
echo ""
echo "Launching training..."
echo "  Log: ${LOG_FILE}"
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
