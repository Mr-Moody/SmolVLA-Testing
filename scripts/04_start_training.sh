#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOME_PROJECT_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
CODE_DIR="${REPO_ROOT}"
LEROBOT_DIR="${HOME_PROJECT_DIR}/lerobot"

if [[ "${1:-}" == "--resume" ]]; then
  RESUME=true
fi

SCRATCH_OUTPUT_DIR="${REMOTE_SCRATCH_BASE}/smolvla_outputs/${RUN_NAME}_smolvla"
LOG_FILE="${REMOTE_SCRATCH_BASE}/smolvla_outputs/${RUN_NAME}.log"
POLICY_PATH="lerobot/smolvla_base"
ACTIVATE_SHIM="${REMOTE_SCRATCH_BASE}/activate_smolvla.sh"

mkdir -p "${REMOTE_SCRATCH_BASE}/smolvla_outputs"

if [[ ! -f "${ACTIVATE_SHIM}" ]]; then
  echo "ERROR: Missing ${ACTIVATE_SHIM}. Run 03_setup_gpu.sh first."
  exit 1
fi

if [[ ! -d "${LEROBOT_DIR}" ]]; then
  echo "ERROR: lerobot repo not found at ${LEROBOT_DIR}"
  exit 1
fi

if [[ ! -d "${CODE_DIR}" ]]; then
  echo "ERROR: code dir not found at ${CODE_DIR}"
  exit 1
fi

DATASET_PATHS=()
for ds in "${DATASET_NAMES[@]}"; do
  p="${DATASET_ROOT}/${ds}"
  if [[ ! -d "${p}/meta" ]]; then
    echo "ERROR: Dataset missing at ${p}"
    exit 1
  fi
  DATASET_PATHS+=("${p}")
done

TRAIN_DATASET_ROOT="${DATASET_PATHS[0]}"
if [[ "${#DATASET_PATHS[@]}" -gt 1 ]]; then
  MERGED_DATASET_ROOT="${DATASET_ROOT}/merged_${RUN_NAME}"
  echo "Merging datasets into ${MERGED_DATASET_ROOT}..."
  source "${ACTIVATE_SHIM}"
  cd "${LEROBOT_DIR}" && uv run python "${CODE_DIR}/merge_datasets.py" \
    "${DATASET_PATHS[@]}" \
    --output "${MERGED_DATASET_ROOT}" \
    --force
  TRAIN_DATASET_ROOT="${MERGED_DATASET_ROOT}"
fi

if [[ -d "${SCRATCH_OUTPUT_DIR}/checkpoints" ]] && [[ -n "$(ls -A "${SCRATCH_OUTPUT_DIR}/checkpoints" 2>/dev/null)" ]] && [[ "${RESUME}" != "true" ]]; then
  echo "ERROR: Existing checkpoints found in ${SCRATCH_OUTPUT_DIR}."
  echo "Use: bash 04_start_training.sh --resume"
  exit 1
fi

if [[ "${ALLOW_NEAREST_FRAME_FALLBACK}" == "true" ]]; then
  python "${CODE_DIR}/patch_frame_tolerance.py"
fi

if [[ "${ALLOW_MISSING_TASK_FALLBACK}" == "true" ]]; then
  python "${CODE_DIR}/patch_task_none.py"
fi

RESUME_FLAG=""
if [[ "${RESUME}" == "true" ]]; then
  RESUME_FLAG="--resume"
fi

AMP_FLAG=""
if [[ "${USE_AMP}" == "true" ]]; then
  AMP_FLAG="--use-amp"
fi

TRAIN_CMD="cd ${LEROBOT_DIR} && uv run python ${CODE_DIR}/main.py \
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

echo "Launching training with nohup..."
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"

nohup bash -c "source '${ACTIVATE_SHIM}'; ${TRAIN_CMD}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

echo "Training started. PID=${TRAIN_PID}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "GPU: watch -n 2 nvidia-smi"
