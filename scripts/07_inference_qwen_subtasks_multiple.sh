#!/usr/bin/env bash
# RUN ON: REMOTE SERVER
#
# Launcher for subtask labeling with Qwen3-VL across ALL datasets in
# DATASET_NAMES from 00_run_params.sh.
#
# A failure on one dataset does not stop the batch — the script continues,
# records which datasets failed, and exits non-zero at the end if any did.
#
# Usage:
#   ./scripts/07_inference_qwen_subtasks.sh
#   ./scripts/07_inference_qwen_subtasks.sh --sampling-hz 2.0
#   ./scripts/07_inference_qwen_subtasks.sh --subtasks-file /path/to/subtasks.json
#
# Per-dataset flags (--video-path, --data-name, --output) are intentionally
# NOT accepted here — they're determined per dataset inside the loop.

set -uo pipefail   # NOTE: no -e; we want to capture per-dataset failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

# ---------------------------------------------------------------------------
# Parse and validate batch-level overrides
# ---------------------------------------------------------------------------
EXTRA_ARGS=()
SAMPLING_HZ="5.0"
SUBTASKS_FILE="${SUBTASKS_FILE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sampling-hz)
      SAMPLING_HZ="$2"; shift 2 ;;
    --sampling-hz=*)
      SAMPLING_HZ="${1#*=}"; shift ;;
    --subtasks-file)
      SUBTASKS_FILE="$2"; shift 2 ;;
    --subtasks-file=*)
      SUBTASKS_FILE="${1#*=}"; shift ;;
    --video-path|--video-path=*|--data-name|--data-name=*|--output|--output=*)
      echo "ERROR: '$1' is per-dataset and cannot be set at the batch level."
      echo "       Run the per-dataset script directly if you need to override these."
      exit 2 ;;
    -h|--help)
      sed -n '2,17p' "$0"; exit 0 ;;
    *)
      echo "ERROR: Unknown flag '$1'"
      echo "       Allowed batch-level flags: --sampling-hz, --subtasks-file"
      exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Environment setup (once, before the loop)
# ---------------------------------------------------------------------------
SCRATCH_VENV="/scratch0/${WORKFLOW_USER}/smolvla_venv"
if [[ ! -d "${SCRATCH_VENV}" ]]; then
  echo "ERROR: Scratch venv not found at ${SCRATCH_VENV}"
  echo "Run: INSTALL_DEPS=true bash scripts/03_setup_gpu.sh"
  exit 1
fi

# shellcheck source=/dev/null
source "${SCRATCH_VENV}/bin/activate"
export PIP_CACHE_DIR="/scratch0/${WORKFLOW_USER}/.cache/pip"
export HF_HOME="/scratch0/${WORKFLOW_USER}/.cache/huggingface"
export TORCH_HOME="/scratch0/${WORKFLOW_USER}/.cache/torch"
export UV_CACHE_DIR="/scratch0/${WORKFLOW_USER}/.cache/uv"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

PYTHON_SCRIPT="${PROJECT_ROOT}/scripts/07_inference_qwen_subtasks.py"
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "ERROR: Python script not found at ${PYTHON_SCRIPT}"
  exit 1
fi

if [[ ${#DATASET_NAMES[@]} -eq 0 ]]; then
  echo "ERROR: DATASET_NAMES is empty in 00_run_params.sh"
  exit 1
fi

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Qwen3-VL Subtask Labeling — Batch Mode"
echo "=========================================="
echo "Scratch venv  : ${SCRATCH_VENV}"
echo "Python        : $(which python3)"
echo "Datasets (${#DATASET_NAMES[@]}): ${DATASET_NAMES[*]}"
echo "Sampling Hz   : ${SAMPLING_HZ}"
[[ -n "${SUBTASKS_FILE}" ]] && echo "Subtasks file : ${SUBTASKS_FILE}"
echo ""

# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
SUCCEEDED=()
FAILED=()
declare -A FAIL_REASON

total=${#DATASET_NAMES[@]}
i=0
for ds in "${DATASET_NAMES[@]}"; do
  i=$((i + 1))
  output="${REMOTE_CLEANED_DATASET_ROOT}/${ds}/subtasks.jsonl"

  echo "------------------------------------------"
  echo "[${i}/${total}] Dataset: ${ds}"
  echo "         Output: ${output}"
  echo "------------------------------------------"

  CMD=(
    python3 "${PYTHON_SCRIPT}"
    --data-name "${ds}"
    --output "${output}"
    --sampling-hz "${SAMPLING_HZ}"
  )
  if [[ -n "${SUBTASKS_FILE}" ]]; then
    CMD+=(--subtasks-file "${SUBTASKS_FILE}")
  fi

  if "${CMD[@]}"; then
    SUCCEEDED+=("${ds}")
    echo "[OK]   ${ds}"
  else
    rc=$?
    FAILED+=("${ds}")
    FAIL_REASON["${ds}"]="exit ${rc}"
    echo "[FAIL] ${ds} (exit ${rc}) — continuing with remaining datasets"
  fi
  echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Batch summary"
echo "=========================================="
echo "Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-<none>}"
echo "Failed    (${#FAILED[@]}): ${FAILED[*]:-<none>}"
for ds in "${FAILED[@]}"; do
  echo "  - ${ds}: ${FAIL_REASON[$ds]}"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
exit 0