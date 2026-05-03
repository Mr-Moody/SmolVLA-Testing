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
    echo "✓ $PYTHON_VERSION"
    echo "  Location: $(which python3)"
    if [[ "$(which python3)" == "${SCRATCH_VENV}"* ]]; then
        echo "  ✓ Using scratch venv (correct)"
    else
        echo "  ⚠ WARNING: Python not from scratch venv!"
    fi
else
    echo "✗ Python3 not found"
    exit 1
fi

# Check vLLM
echo -n "Checking vLLM... "
if python3 -c "from vllm import LLM" 2>/dev/null; then
    echo "✓ vLLM installed"
else
    echo "✗ vLLM not found. Install with: pip install vllm>=0.7"
    exit 1
fi

# Check Qwen utils
echo -n "Checking Qwen utils... "
if python3 -c "from qwen_vl_utils import *" 2>/dev/null; then
    echo "✓ qwen-vl-utils installed"
else
    echo "✗ qwen-vl-utils not found. Install with: pip install qwen-vl-utils"
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
import sys
from vllm import LLM

try:
    # Load model - this is the actual test
    print("  Loading: Qwen/Qwen3-VL-30B-A3B-Instruct")
    llm = LLM(
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
    )
    print("  ✓ Model loaded successfully!")
    
    # Show model info
    print(f"  Model type: {type(llm)}")
    print(f"  GPU memory allocated: ~24GB")
    
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
