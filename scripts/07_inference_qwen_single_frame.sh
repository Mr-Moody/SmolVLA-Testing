#!/usr/bin/env bash
# RUN ON: REMOTE SERVER
#
# Launcher for the single-timestamp subtask label debugger.
# Wraps scripts/07_debug_subtask_label.py with the same env setup as
# 07_inference_qwen_subtasks.sh, but expects a --time argument to pinpoint
# a single moment for two-stage diagnosis (describe + label).
#
# Usage:
#   ./scripts/07_debug_subtask_label.sh --time 12.5
#   ./scripts/07_debug_subtask_label.sh --time 12.5 --context-window 2
#   ./scripts/07_debug_subtask_label.sh --time 12.5 --save-frames /tmp/debug_frames
#   ./scripts/07_debug_subtask_label.sh --video-path /path/to/dataset --time 12.5

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
DEFAULT_SUBTASKS_FILE="${SUBTASKS_FILE:-}"

PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/07_inference_qwen_single_frame.py"
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "ERROR: Python script not found at ${PYTHON_SCRIPT}"
  exit 1
fi

# The debug script imports helpers from 07_inference_qwen_subtasks.py via the
# module name `label_subtasks`. Make sure both are importable; the debug
# script itself adds PROJECT_ROOT and src to sys.path, but if the production
# script lives at scripts/07_inference_qwen_subtasks.py we also need that on
# the path so `import label_subtasks` resolves. Symlink approach below keeps
# the debug script's import line stable regardless of the production
# filename.
PRODUCTION_SCRIPT="${PROJECT_ROOT}/scripts/07_inference_qwen_subtasks.py"
LABEL_SUBTASKS_LINK="${PROJECT_ROOT}/scripts/label_subtasks.py"
if [[ -f "${PRODUCTION_SCRIPT}" && ! -e "${LABEL_SUBTASKS_LINK}" ]]; then
  ln -s "${PRODUCTION_SCRIPT}" "${LABEL_SUBTASKS_LINK}"
fi
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"

echo "=========================================="
echo "Qwen3-VL Subtask Label Debugger (single timestamp)"
echo "=========================================="
echo "Scratch venv: ${SCRATCH_VENV}"
echo "Python: $(which python3)"
echo "Dataset: ${DEFAULT_DATASET_NAME}"
echo ""
echo "Pass --time <seconds> to pick a frame. Stage 1 prints a description,"
echo "Stage 2 prints the predicted label."
echo ""

CMD=(
  python3 "${PYTHON_SCRIPT}"
  --data-name "${DEFAULT_DATASET_NAME}"
  --sampling-hz 5.0
  --max-model-len 25000
)

if [[ -n "${DEFAULT_SUBTASKS_FILE}" ]]; then
  CMD+=(--subtasks-file "${DEFAULT_SUBTASKS_FILE}")
fi

CMD+=("$@")

"${CMD[@]}"