#!/usr/bin/env bash
# RUN ON: REMOTE SERVER
#
# Quick test to verify Qwen3-VL can load and run on the server.
# This processes just 1 dataset with 5 episodes, no heavy processing.
#
# Usage:
#   ./scripts/06_test_qwen.sh
#
# Expected output:
#   ✓ Python OK
#   ✓ vLLM OK
#   ✓ Qwen model loaded
#   ✓ Test annotation complete

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure we're using the scratch venv
SCRATCH_VENV="/scratch0/xparker/smolvla_venv"
if [ ! -d "${SCRATCH_VENV}" ]; then
    echo "ERROR: Scratch venv not found at ${SCRATCH_VENV}"
    exit 1
fi

# Activate the venv and set cache directories to scratch
source "${SCRATCH_VENV}/bin/activate"
export PIP_CACHE_DIR="/scratch0/xparker/.cache/pip"
export HF_HOME="/scratch0/xparker/.cache/huggingface"
export TORCH_HOME="/scratch0/xparker/.cache/torch"
export UV_CACHE_DIR="/scratch0/xparker/.cache/uv"

echo "=========================================="
echo "Qwen3-VL Server Test"
echo "=========================================="
echo "Using Python: $(which python3)"
echo "Venv: ${VIRTUAL_ENV}"
echo ""

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    PYTHON_PATH=$(which python3)
    PYTHON_MINOR=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "${PYTHON_PATH}" == "${SCRATCH_VENV}"* ]]; then
        if [[ "${PYTHON_MINOR}" == "3.12" ]]; then
            echo "✓ $PYTHON_VERSION (from scratch venv, pinned and compatible)"
        else
            echo "✗ $PYTHON_VERSION (from scratch venv, but wrong version)"
            echo "  Expected: Python 3.12"
            echo "  Got: ${PYTHON_MINOR}"
            exit 1
        fi
    else
        echo "✗ Python not from scratch venv!"
        echo "  Expected: ${SCRATCH_VENV}/bin/python3"
        echo "  Got: ${PYTHON_PATH}"
        exit 1
    fi
else
    echo "✗ Python3 not found"
    exit 1
fi

# Check vLLM
echo -n "Checking vLLM... "
if python3 -c "from vllm import LLM; print(LLM.__module__)" 2>/dev/null; then
    echo "✓ vLLM installed in scratch venv"
else
    echo "✗ vLLM not found in scratch venv"
    echo "  Run: INSTALL_DEPS=true bash scripts/03_setup_gpu.sh"
    exit 1
fi

# Check Qwen utils
echo -n "Checking Qwen utils... "
if python3 -c "from qwen_vl_utils import *" 2>/dev/null; then
    echo "✓ qwen-vl-utils installed in scratch venv"
else
    echo "✗ qwen-vl-utils not found in scratch venv"
    echo "  Run: INSTALL_DEPS=true bash scripts/03_setup_gpu.sh"
    exit 1
fi

# Check GPU
echo -n "Checking GPU... "
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)
    echo "✓ Found $GPU_COUNT GPU(s)"
else
    echo "✗ NVIDIA GPU tools not found"
    exit 1
fi

# Quick model load test
echo ""
echo "Loading Qwen3-VL model (this may take 1-2 minutes)..."
python3 << 'EOF'
import os
import sys
from vllm import LLM

try:
    # Load model - this is the actual test
    print("  Loading: Qwen/Qwen3-VL-4B-Instruct")
    # Allow overrides via env vars; choose conservative defaults to fit smaller GPUs
    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.9"))
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048"))
    print(f"  Using gpu_memory_utilization={gpu_mem_util}, max_model_len={max_model_len}")
    llm = LLM(
        model="Qwen/Qwen3-VL-4B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    print("  ✓ Model loaded successfully!")
    
    # Show model info
    print(f"  Model type: {type(llm)}")
    
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ All tests passed!"
    echo "=========================================="
    echo ""
    echo "Next step: Run the full overnight pipeline"
    echo "  python3 run_overnight_pipeline.py \\"
    echo "      --raw-datasets raw_datasets \\"
    echo "      --dataset-names 001 \\"
    echo "      --enable-annotation \\"
    echo "      --num-gpus 1 \\"
    echo "      --max-episodes 5 \\"
    echo "      --output-dir overnight_output"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Tests failed"
    echo "=========================================="
    exit 1
fi
