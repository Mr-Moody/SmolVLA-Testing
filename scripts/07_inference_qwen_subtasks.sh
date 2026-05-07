#!/usr/bin/env bash
# RUN ON: REMOTE SERVER
#
# Launcher for subtask labeling with Qwen3-VL.
# This wraps scripts/07_inference_qwen_subtasks.py with the workflow defaults
# from 00_run_params.sh while still allowing command-line overrides.
#
# Usage:
#   ./scripts/07_inference_qwen_subtasks.sh
#   ./scripts/07_inference_qwen_subtasks.sh --video-path /path/to/video.mp4
#   ./scripts/07_inference_qwen_subtasks.sh --subtasks-file /path/to/subtasks.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SCRATCH_VENV="/scratch0/${WORKFLOW_USER}/smolvla_venv"
if [[ ! -d "${SCRATCH_VENV}" ]]; then
  echo "ERROR: Scratch venv not found at ${SCRATCH_VENV}"
  echo "Run: INSTALL_DEPS=true bash scripts/03_setup_gpu.sh"
  exit 1
fi

# Activate the venv and keep caches on scratch.
source "${SCRATCH_VENV}/bin/activate"
export PIP_CACHE_DIR="/scratch0/${WORKFLOW_USER}/.cache/pip"
export HF_HOME="/scratch0/${WORKFLOW_USER}/.cache/huggingface"
export TORCH_HOME="/scratch0/${WORKFLOW_USER}/.cache/torch"
export UV_CACHE_DIR="/scratch0/${WORKFLOW_USER}/.cache/uv"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

DEFAULT_DATASET_NAME="${DATASET_NAMES[0]}"
DEFAULT_OUTPUT="${REMOTE_CLEANED_DATASET_ROOT}/${DEFAULT_DATASET_NAME}/subtasks.jsonl"
DEFAULT_SUBTASKS_FILE="${SUBTASKS_FILE:-}"

PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/07_inference_qwen_subtasks.py"
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "ERROR: Python script not found at ${PYTHON_SCRIPT}"
  exit 1
fi

echo "=========================================="
echo "Qwen3-VL Subtask Labeling"
echo "=========================================="
echo "Scratch venv: ${SCRATCH_VENV}"
echo "Python: $(which python3)"
echo "Output: ${DEFAULT_OUTPUT}"
echo "Dataset: ${DEFAULT_DATASET_NAME}"
echo ""

CMD=(
  python3 "${PYTHON_SCRIPT}"
  --data-name "${DEFAULT_DATASET_NAME}"
  --output "${DEFAULT_OUTPUT}"
  --sampling-hz 5.0
)

if [[ -n "${DEFAULT_SUBTASKS_FILE}" ]]; then
  CMD+=(--subtasks-file "${DEFAULT_SUBTASKS_FILE}")
fi

CMD+=("$@")

"${CMD[@]}"

