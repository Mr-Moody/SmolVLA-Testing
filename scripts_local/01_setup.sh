#!/usr/bin/env bash
# =============================================================================
# 01_setup.sh  —  One-time local environment setup
# =============================================================================
# Run this once before your first training run, or after updating lerobot.
#
# Usage:
#   bash scripts_local/01_setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/params.sh"

echo "================================================================="
echo "  SmolVLA Local Environment Setup"
echo "================================================================="
echo "  Project  : ${PROJECT_ROOT}"
echo "  lerobot  : ${LEROBOT_ROOT}"
echo "================================================================="
echo ""

# ---------------------------------------------------------------------------
# 1. Pre-conditions
# ---------------------------------------------------------------------------
echo "[1/3] Checking pre-conditions..."

if [[ ! -d "${LEROBOT_ROOT}/src" ]]; then
    echo "ERROR: lerobot not found at ${LEROBOT_ROOT}"
    echo ""
    echo "Clone it as a sibling of this repo:"
    echo "  git clone https://github.com/huggingface/lerobot ${LEROBOT_ROOT}"
    exit 1
fi
echo "  lerobot found: ${LEROBOT_ROOT}"

if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  uv found: $(uv --version)"

if command -v nvidia-smi &>/dev/null; then
    echo "  GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    /'
else
    echo "  WARNING: nvidia-smi not found — training will run on CPU."
fi

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Installing dependencies (this may take a few minutes)..."

uv --project "${LEROBOT_ROOT}" sync --extra smolvla --extra dataset

echo "  Dependencies installed into ${LEROBOT_ROOT}/.venv"

# ---------------------------------------------------------------------------
# 3. Verify torch + CUDA
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Verifying PyTorch installation..."

uv --project "${LEROBOT_ROOT}" run python - <<'PY'
import torch
print(f"  torch version  : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device    : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version   : {torch.version.cuda}")
    print(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  WARNING: CUDA not available — check your PyTorch build and drivers.")
PY

echo ""
echo "================================================================="
echo "  Setup complete."
echo ""
echo "  Next: bash scripts_local/02_run.sh"
echo "================================================================="
