#!/usr/bin/env bash
# =============================================================================
# 02_run.sh  —  Clean → Convert → Train on the local GPU
# =============================================================================
#
# Usage:
#   bash scripts_local/02_run.sh [--resume] [--skip-clean]
#
# Pipeline:
#   raw_datasets/<name>/      (step 1: clean)
#   cleaned_datasets/<name>/  (step 2: convert)
#   lerobot_datasets/<name>/  (step 3: merge if needed, step 4: train)
#   outputs/<run>/
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
    [[ "${arg}" == "--resume" ]]     && RESUME=true
    [[ "${arg}" == "--skip-clean" ]] && SKIP_CLEAN=true
done

RUN_TAG="${RUN_NAME}_${MODEL_TYPE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
LOG_FILE="${OUTPUT_ROOT}/${RUN_TAG}.log"

# Shorthand: run python via the lerobot project using our dedicated venv.
# UV_PROJECT_ENVIRONMENT keeps lerobot/.venv untouched (camera packages safe).
UV="UV_PROJECT_ENVIRONMENT=${VENV_DIR} uv --project ${LEROBOT_ROOT} run python"

# ---------------------------------------------------------------------------
# Create local data directories if they don't exist yet
# ---------------------------------------------------------------------------
mkdir -p "${RAW_DATASET_ROOT}"
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

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: training venv not found at ${VENV_DIR}. Run 01_setup.sh first."
    exit 1
fi

if [[ "${SKIP_CLEAN}" != "true" ]]; then
    for ds in "${DATASET_NAMES[@]}"; do
        if [[ ! -d "${RAW_DATASET_ROOT}/${ds}" ]]; then
            echo "ERROR: raw dataset '${ds}' not found at ${RAW_DATASET_ROOT}/${ds}"
            echo "       Copy raw recordings there, or set SKIP_CLEAN=true in params.sh"
            echo "       if cleaned_datasets/ is already populated."
            exit 1
        fi
    done
else
    for ds in "${DATASET_NAMES[@]}"; do
        if [[ ! -d "${CLEANED_DATASET_ROOT}/${ds}" ]]; then
            echo "ERROR: --skip-clean set but cleaned dataset '${ds}' not found"
            echo "       at ${CLEANED_DATASET_ROOT}/${ds}"
            exit 1
        fi
    done
fi

echo "================================================================="
echo "  SmolVLA Local Training"
echo "================================================================="
echo "  Run       : ${RUN_TAG}"
echo "  Model     : ${MODEL_TYPE}"
echo "  Datasets  : ${DATASET_NAMES[*]}"
echo "  Device    : ${DEVICE}"
echo "  Steps     : ${STEPS}  |  Batch: ${BATCH_SIZE}"
echo "  Skip clean: ${SKIP_CLEAN}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Log       : ${LOG_FILE}"
echo "  Resume    : ${RESUME}"
echo "================================================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1: Clean raw recordings → cleaned_datasets/
# ---------------------------------------------------------------------------
if [[ "${SKIP_CLEAN}" == "true" ]]; then
    echo "[1/4] Skipping clean step (SKIP_CLEAN=true)."
else
    echo "[1/4] Cleaning raw datasets..."
    for ds in "${DATASET_NAMES[@]}"; do
        echo "  Cleaning '${ds}'..."

        # annotations.jsonl canonical home is raw_datasets/<ds>/annotations.jsonl.
        # If it only exists in cleaned_datasets/ (e.g. first run), seed it there now
        # so it survives future --force cleans without needing the cleaned copy.
        ANN_RAW="${RAW_DATASET_ROOT}/${ds}/annotations.jsonl"
        ANN_CLEANED="${CLEANED_DATASET_ROOT}/${ds}/annotations.jsonl"
        if [[ ! -f "${ANN_RAW}" && -f "${ANN_CLEANED}" ]]; then
            cp "${ANN_CLEANED}" "${ANN_RAW}"
            echo "  Seeded annotations.jsonl into raw_datasets/${ds}/."
        fi

        ${UV} "${PROJECT_ROOT}/main.py" clean "${ds}" \
            --datasets-root "${RAW_DATASET_ROOT}" \
            --output-root   "${CLEANED_DATASET_ROOT}" \
            --force

        # Restore annotations from raw_datasets/ if the cleaner didn't generate its own.
        if [[ ! -f "${ANN_CLEANED}" && -f "${ANN_RAW}" ]]; then
            cp "${ANN_RAW}" "${ANN_CLEANED}"
            echo "  Restored annotations.jsonl for '${ds}'."
        fi

        echo "  Done: ${CLEANED_DATASET_ROOT}/${ds}"
    done
fi

# ---------------------------------------------------------------------------
# Step 2: Convert cleaned datasets → lerobot_datasets/
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Converting cleaned datasets..."

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
# Step 3: Merge datasets if more than one
# ---------------------------------------------------------------------------
TRAIN_DATASET_ROOT="${LEROBOT_DATASET_ROOT}/${DATASET_NAMES[0]}"

if [[ "${#DATASET_NAMES[@]}" -gt 1 ]]; then
    DATASET_NAME_JOINED=$(IFS="_"; echo "${DATASET_NAMES[*]}")
    MERGED_ROOT="${LEROBOT_DATASET_ROOT}/merged_${DATASET_NAME_JOINED}"
    echo ""
    echo "[3/4] Merging ${#DATASET_NAMES[@]} datasets into ${MERGED_ROOT}..."

    DATASET_PATHS=()
    for ds in "${DATASET_NAMES[@]}"; do
        DATASET_PATHS+=("${LEROBOT_DATASET_ROOT}/${ds}")
    done

    ${UV} "${PROJECT_ROOT}/src/data/merge.py" \
        "${DATASET_PATHS[@]}" \
        --output "${MERGED_ROOT}" \
        --force

    TRAIN_DATASET_ROOT="${MERGED_ROOT}"
    echo "  Merge complete: ${TRAIN_DATASET_ROOT}"
else
    echo ""
    echo "[3/4] Single dataset — skipping merge."
fi

# ---------------------------------------------------------------------------
# Step 4: Apply lerobot patches + train
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Applying lerobot patches and launching training..."

${UV} "${PROJECT_ROOT}/src/patch_frame_tolerance.py"
echo "  patch_frame_tolerance applied."

${UV} "${PROJECT_ROOT}/src/patch_task_none.py"
echo "  patch_task_none applied."

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
