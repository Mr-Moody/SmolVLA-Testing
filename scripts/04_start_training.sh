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

if [[ -z "${MODEL_TYPE:-}" ]]; then
  MODEL_TYPE="smolvla"
fi

RUN_TAG="${RUN_NAME}_${MODEL_TYPE}"
SCRATCH_OUTPUT_DIR="${REMOTE_SCRATCH_BASE}/smolvla_outputs/${RUN_TAG}"
LOG_FILE="${REMOTE_SCRATCH_BASE}/smolvla_outputs/${RUN_TAG}.log"
PREPROCESS_LOG_DIR="${REMOTE_SCRATCH_BASE}/smolvla_outputs/preprocess_logs/${RUN_TAG}"
ACTIVATE_SHIM="${REMOTE_SCRATCH_BASE}/activate_smolvla.sh"

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

mkdir -p "${REMOTE_SCRATCH_BASE}/smolvla_outputs"
mkdir -p "${PREPROCESS_LOG_DIR}"

run_preprocess_job() {
  local ds="$1"
  local codec="$2"
  local log_path="$3"

  nohup bash -c "source '${ACTIVATE_SHIM}'; cd '${LEROBOT_DIR}' && uv run python '${CODE_DIR}/src/data_converter.py' '${ds}' \
    --datasets-root '${REMOTE_CLEANED_DATASET_ROOT}' \
    --output-root '${DATASET_ROOT}' \
    --vcodec '${codec}' \
    --force" >> "${log_path}" 2>&1 &
  local preprocess_pid=$!
  echo "  PID=${preprocess_pid}  log=${log_path}"
  wait "${preprocess_pid}"
}

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

if [[ "${PREPROCESS_ON_GPU}" == "true" ]]; then
  echo "GPU preprocessing enabled: converting cleaned datasets before training..."
  if [[ ! -d "${REMOTE_CLEANED_DATASET_ROOT}" ]]; then
    echo "ERROR: cleaned dataset root not found at ${REMOTE_CLEANED_DATASET_ROOT}"
    exit 1
  fi

  source "${ACTIVATE_SHIM}"
  for ds in "${DATASET_NAMES[@]}"; do
    if [[ ! -d "${REMOTE_CLEANED_DATASET_ROOT}/${ds}" ]]; then
      echo "ERROR: cleaned dataset ${ds} missing at ${REMOTE_CLEANED_DATASET_ROOT}/${ds}"
      exit 1
    fi

    PREPROCESS_LOG_FILE="${PREPROCESS_LOG_DIR}/${ds}.log"
    : > "${PREPROCESS_LOG_FILE}"

    echo "Preprocessing dataset ${ds} on GPU with vcodec=${PREPROCESS_VCODEC} (nohup)..."
    if ! run_preprocess_job "${ds}" "${PREPROCESS_VCODEC}" "${PREPROCESS_LOG_FILE}"; then
      if [[ "${PREPROCESS_VCODEC}" == "h264_nvenc" ]]; then
        echo "WARN: NVENC preprocessing failed for ${ds}; retrying with software codec h264..."
        if ! run_preprocess_job "${ds}" "h264" "${PREPROCESS_LOG_FILE}"; then
          echo "ERROR: preprocessing failed for ${ds} with fallback codec h264"
          echo "See log: ${PREPROCESS_LOG_FILE}"
          exit 1
        fi
      else
        echo "ERROR: preprocessing failed for ${ds} with vcodec=${PREPROCESS_VCODEC}"
        echo "See log: ${PREPROCESS_LOG_FILE}"
        exit 1
      fi
    fi

    echo "Preprocessing finished for ${ds}. Log: ${PREPROCESS_LOG_FILE}"
  done
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
  MERGED_DATASET_ROOT="${DATASET_ROOT}/merged_${RUN_TAG}"
  echo "Merging datasets into ${MERGED_DATASET_ROOT}..."
  source "${ACTIVATE_SHIM}"
  cd "${LEROBOT_DIR}" && uv run python "${CODE_DIR}/src/merge_datasets.py" \
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
  python "${CODE_DIR}/src/patch_frame_tolerance.py"
fi

if [[ "${ALLOW_MISSING_TASK_FALLBACK}" == "true" ]]; then
  python "${CODE_DIR}/src/patch_task_none.py"
fi

RESUME_FLAG=""
if [[ "${RESUME}" == "true" ]]; then
  RESUME_FLAG="--resume"
fi

AMP_FLAG=""
if [[ "${USE_AMP}" == "true" ]]; then
  AMP_FLAG="--use-amp"
fi

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

echo "Launching training with nohup..."
echo "Run name: ${RUN_NAME}"
echo "Log file: ${LOG_FILE}"

nohup bash -c "source '${ACTIVATE_SHIM}'; export HF_HUB_OFFLINE=1; export TORCH_HOME='${REMOTE_SCRATCH_BASE}/.cache/torch'; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ${TRAIN_CMD}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

echo "Training started. PID=${TRAIN_PID}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "GPU: watch -n 2 nvidia-smi"
