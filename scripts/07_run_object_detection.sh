#!/usr/bin/env bash
# RUN ON: REMOTE SERVER
#
# Quick test to identify objects in a video using Qwen3-VL-4B.
# Extracts frames from the video and asks the model to identify each object.
#
# Usage:
#   ./scripts/07_run_object_detection.sh
#
# Optional overrides:
#   ./scripts/07_run_object_detection.sh --frames 0 --gpu-mem 0.95 --max-len 512
#
# Expected output:
#   [Frame 1/10] /path/to/frame
#   Identified: [object description]
#   ...

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure we're using the exact scratch venv created by setup_scratch.sh
SCRATCH_VENV="/scratch0/xparker/smolvla_venv"
if [ ! -d "${SCRATCH_VENV}" ]; then
    echo "ERROR: Scratch venv not found at ${SCRATCH_VENV}"
    echo "Run: INSTALL_DEPS=true bash scripts/03_setup_gpu.sh"
    exit 1
fi

# Activate the venv and set ALL cache directories to scratch (critical!)
source "${SCRATCH_VENV}/bin/activate"
export PIP_CACHE_DIR="/scratch0/xparker/.cache/pip"
export HF_HOME="/scratch0/xparker/.cache/huggingface"
export TORCH_HOME="/scratch0/xparker/.cache/torch"
export UV_CACHE_DIR="/scratch0/xparker/.cache/uv"

# Parse command line args
FRAMES="${1:-0}"
GPU_MEM="${2:-0.98}"
MAX_LEN="${3:-1024}"

# Handle named args
while [[ $# -gt 0 ]]; do
    case $1 in
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM="$2"
            shift 2
            ;;
        --max-len)
            MAX_LEN="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "=========================================="
echo "Qwen3-VL Object Detection Test"
echo "=========================================="
if [ "$FRAMES" -le 0 ]; then
    echo "Frames to extract: all frames"
else
    echo "Frames to extract: $FRAMES"
fi
echo "GPU memory utilization: $GPU_MEM"
echo "Max model length: $MAX_LEN"
echo ""

# Run the inference script
python3 scripts/07_inference_qwen_test.py \
    --frames-to-extract "$FRAMES" \
    --gpu-mem-util "$GPU_MEM" \
    --max-model-len "$MAX_LEN"

exit $?
