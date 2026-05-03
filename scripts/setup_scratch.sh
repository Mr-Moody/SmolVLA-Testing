#!/usr/bin/env bash
# =============================================================================
# setup_scratch.sh  —  UCL TSG GPU Workstation Environment Bootstrap
# =============================================================================
#
# PURPOSE
# -------
# Provisions a Python virtual environment and ALL dependency caches in the
# local scratch disk (/scratch0/$USER/), completely bypassing the 10GB network
# quota on your persistent home directory (~/).
#
# UCL TSG QUOTA ARCHITECTURE
# --------------------------
# Your ~/  is a network-mounted NFS share with a hard 10GB quota.
# PyTorch alone is ~2-3GB, the HuggingFace SmolVLA base model is ~4GB.
# Installing them into ~/  would instantly exhaust your quota and lock you out.
#
# /scratch0/$USER/ is a local NVME disk (typically 1TB+) with NO quota limit.
# It is wiped when your 72-hour GPU booking expires, so we only store
# re-creatable artefacts here (venv, pip cache, HF model cache).
#
# PERSISTENT HOME LAYOUT (rsync'd before running this script)
# -----------------------------------------------------------
#   ~/smolvla_project/
#     lerobot/            <- sibling lerobot repo (source code only, ~MB)
#     SmolVLA-Testing/    <- this repo (source code only, ~MB)
#     checkpoints/        <- rescued checkpoints land here (persistent)
#
# SCRATCH LAYOUT (created by this script, wiped after booking)
# ------------------------------------------------------------
#   /scratch0/$USER/
#     smolvla_venv/       <- Python venv with all deps (~5-8GB)
#     .cache/
#       uv/               <- uv package cache (avoids re-downloading wheels)
#       pip/              <- pip fallback cache
#       huggingface/      <- HF Hub model weights (SmolVLA base ~4GB)
#
# USAGE
# -----
#   bash ~/smolvla_project/SmolVLA-Testing/setup_scratch.sh
#
# Run this ONCE per GPU booking. If your booking expires and you get a new
# machine, run it again — the scratch space will be empty.
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, or pipe failures

# ---------------------------------------------------------------------------
# 0. CONFIGURATION — adjust these paths if your layout differs
# ---------------------------------------------------------------------------

# The UCL scratch disk — this is local to the GPU node, no quota
SCRATCH_BASE="/scratch0/${USER}"

# Where the uv-managed venv will live (on fast local disk)
VENV_DIR="${SCRATCH_BASE}/smolvla_venv"

# Python version pinned for vllm/numba compatibility.
PYTHON_VERSION="3.12"

# All caches redirected to scratch — CRITICAL to set before any install step
CACHE_DIR="${SCRATCH_BASE}/.cache"

# Your persistent home project root (must already exist, rsync'd beforehand)
HOME_PROJECT="${HOME}/smolvla_project"

# The lerobot sibling repo — uv reads its pyproject.toml/uv.lock from here
LEROBOT_ROOT="${HOME_PROJECT}/lerobot"

# ---------------------------------------------------------------------------
# 1. REDIRECT ALL CACHES TO SCRATCH — set BEFORE touching pip/uv/HuggingFace
# ---------------------------------------------------------------------------
# These variables must be exported so every child process (pip, uv, transformers)
# inherits them. If HF_HOME is not set here, the SmolVLA base checkpoint (~4GB)
# will download into ~/.cache/huggingface and immediately bust your quota.

export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export UV_CACHE_DIR="${CACHE_DIR}/uv"

# HuggingFace Hub — model weights, tokenizers, datasets metadata
export HF_HOME="${CACHE_DIR}/huggingface"
# Legacy variable respected by older versions of transformers/datasets
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"

# PyTorch / torchvision hub cache — ResNet and other backbone weights land here.
# Without this, torch.hub defaults to ~/.cache/torch which is on the 10GB NFS
# home quota and will immediately exhaust it when the vision backbone downloads.
export TORCH_HOME="${CACHE_DIR}/torch"

# Tell uv to create the project's venv in scratch rather than inside lerobot/
# Without this, uv would default to lerobot/.venv which is on the network quota.
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

# Tell main.py where to find the lerobot source tree
export LEROBOT_ROOT="${LEROBOT_ROOT}"

echo "================================================================="
echo "  SmolVLA Scratch Setup — UCL TSG GPU Workstation"
echo "================================================================="
echo "  User       : ${USER}"
echo "  Scratch    : ${SCRATCH_BASE}"
echo "  Venv       : ${VENV_DIR}"
echo "  Python     : ${PYTHON_VERSION}"
echo "  HF cache   : ${HF_HOME}"
echo "  lerobot    : ${LEROBOT_ROOT}"
echo "================================================================="

# ---------------------------------------------------------------------------
# 2. CREATE SCRATCH DIRECTORY TREE
# ---------------------------------------------------------------------------

echo ""
echo "[1/4] Creating scratch directory structure..."

mkdir -p "${VENV_DIR}"
mkdir -p "${CACHE_DIR}/pip"
mkdir -p "${CACHE_DIR}/uv"
mkdir -p "${CACHE_DIR}/huggingface/hub"
mkdir -p "${CACHE_DIR}/huggingface/datasets"
mkdir -p "${CACHE_DIR}/torch/hub/checkpoints"

# Also ensure the persistent checkpoint rescue directory exists in home.
# This is small (just model weights) so it is safe to keep in ~/
mkdir -p "${HOME_PROJECT}/checkpoints"

echo "    OK — scratch directories created."

# If a venv already exists but is not using Python 3.13, remove it so uv can
# recreate it with the pinned interpreter.
if [[ -x "${VENV_DIR}/bin/python3" ]]; then
    CURRENT_PYTHON_VERSION="$(${VENV_DIR}/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ "${CURRENT_PYTHON_VERSION}" != "${PYTHON_VERSION}" ]]; then
        echo "    Existing venv uses Python ${CURRENT_PYTHON_VERSION:-unknown}; recreating with Python ${PYTHON_VERSION}..."
        rm -rf "${VENV_DIR}"
        mkdir -p "${VENV_DIR}"
    else
        echo "    Existing venv already uses Python ${PYTHON_VERSION}."
    fi
fi

# ---------------------------------------------------------------------------
# 3. VERIFY PRE-CONDITIONS
# ---------------------------------------------------------------------------

echo ""
echo "[2/4] Verifying pre-conditions..."

# Check that the lerobot sibling repo exists in home
if [[ ! -d "${LEROBOT_ROOT}/src" ]]; then
    echo ""
    echo "ERROR: Cannot find lerobot source at ${LEROBOT_ROOT}/src"
    echo ""
    echo "You must rsync the full project layout to UCL first. From your"
    echo "local machine run:"
    echo ""
    echo "  rsync -avz /path/to/Industrial-Project/lerobot/ \\"
    echo "      ${USER}@knuckles.cs.ucl.ac.uk:${HOME_PROJECT}/lerobot/"
    echo ""
    echo "  rsync -avz /path/to/Industrial-Project/SmolVLA-Testing/ \\"
    echo "      ${USER}@knuckles.cs.ucl.ac.uk:${HOME_PROJECT}/SmolVLA-Testing/"
    echo ""
    exit 1
fi

# Check uv is available — it is pre-installed on UCL TSG machines, or install it
if ! command -v uv &>/dev/null; then
    echo "    uv not found in PATH. Installing uv to scratch space..."
    # Install uv binary to scratch so it doesn't consume home quota
    # This is idempotent — safe to run multiple times
    export CARGO_HOME="${SCRATCH_BASE}/.cargo"
    curl -LsSf https://astral.sh/uv/install.sh | \
        env INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR="${SCRATCH_BASE}/bin" sh
    export PATH="${SCRATCH_BASE}/bin:${PATH}"
    echo "    uv installed to ${SCRATCH_BASE}/bin/uv"
else
    echo "    uv found: $(which uv) — $(uv --version)"
fi

# Ensure uv has the required Python version available locally.
if ! uv python find "${PYTHON_VERSION}" >/dev/null 2>&1; then
    echo "    Python ${PYTHON_VERSION} not found by uv. Installing it to scratch..."
    uv python install "${PYTHON_VERSION}"
fi

# Confirm CUDA is visible to the system
if command -v nvidia-smi &>/dev/null; then
    echo "    GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | \
        sed 's/^/        /'
else
    echo "    WARNING: nvidia-smi not found. Ensure you are on a GPU node."
fi

# ---------------------------------------------------------------------------
# 4. INSTALL DEPENDENCIES INTO SCRATCH VENV
# ---------------------------------------------------------------------------
# uv reads pyproject.toml and uv.lock from the lerobot project directory.
# UV_PROJECT_ENVIRONMENT (set above) redirects the venv to scratch.
# All downloaded wheels are cached in UV_CACHE_DIR (also scratch).
#
# The --extra flags below install the optional dependency groups defined in
# lerobot's pyproject.toml that are needed for SmolVLA + CUDA training.
# Adjust the extras if lerobot's pyproject.toml uses different group names.

echo ""
echo "[3/4] Installing dependencies into scratch venv..."
echo "      (This downloads PyTorch ~2GB + SmolVLA deps — takes 5-15 min)"
echo "      Venv target : ${VENV_DIR}"
echo "      Cache target: ${UV_CACHE_DIR}"
echo ""

cd "${LEROBOT_ROOT}"

# Sync the lerobot project with all extras needed for SmolVLA fine-tuning.
# Use Python 3.13 explicitly (compatible with numba, vllm, PyTorch, etc.)
# --extra smolvla  : SmolVLA policy and its vision-language deps
# --extra dev      : training utilities (wandb, tensorboard, etc.) if present
# Omit extras that don't exist in the lerobot version you have cloned.
uv sync \
    --python "${PYTHON_VERSION}" \
    --extra smolvla \
    --extra dataset \
    --no-editable \
    2>&1 | tee "${SCRATCH_BASE}/setup_install.log"

# Verify the critical imports resolved correctly
echo ""
echo "    Verifying PyTorch + CUDA inside the new venv..."
uv run python -c "
import torch
print(f'    torch version  : {torch.__version__}')
print(f'    CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    CUDA device    : {torch.cuda.get_device_name(0)}')
    print(f'    CUDA version   : {torch.version.cuda}')
else:
    print('    WARNING: CUDA not available — training will fall back to CPU')
"

# ---------------------------------------------------------------------------
# 5. WRITE ENVIRONMENT ACTIVATION HELPERS
# ---------------------------------------------------------------------------
# UCL machines default to tcsh. We write two shims:
#   activate_smolvla.sh   — for bash  (used by run_training.sh and nohup)
#   activate_smolvla.csh  — for tcsh  (for interactive terminal sessions)

ACTIVATE_SHIM="${SCRATCH_BASE}/activate_smolvla.sh"
ACTIVATE_CSH="${SCRATCH_BASE}/activate_smolvla.csh"

# Resolve the uv binary location so the shims can add it to PATH.
# setup_scratch.sh may have installed uv to scratch, or it may be on the
# system PATH. We capture whichever was found during setup.
UV_BIN_DIR="$(dirname "$(command -v uv)")"

# --- bash shim ---
cat > "${ACTIVATE_SHIM}" << SHIM_EOF
# Auto-generated by setup_scratch.sh — source in bash
# Usage:  bash; source /scratch0/${USER}/activate_smolvla.sh

# Add uv to PATH (may not be on bash's default PATH on UCL TSG machines)
export PATH="${UV_BIN_DIR}:\$PATH"

export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export HF_HOME="${CACHE_DIR}/huggingface"
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"
export TORCH_HOME="${CACHE_DIR}/torch"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export LEROBOT_ROOT="${LEROBOT_ROOT}"

echo "SmolVLA scratch environment activated."
echo "  uv      : ${UV_BIN_DIR}/uv"
echo "  Venv    : ${VENV_DIR}"
echo "  HF Home : ${HF_HOME}"
echo "  Torch   : ${CACHE_DIR}/torch"
SHIM_EOF

# --- tcsh shim ---
cat > "${ACTIVATE_CSH}" << CSH_EOF
# Auto-generated by setup_scratch.sh — source in tcsh
# Usage:  source /scratch0/${USER}/activate_smolvla.csh

setenv PATH "${UV_BIN_DIR}:\$PATH"

setenv PIP_CACHE_DIR "${CACHE_DIR}/pip"
setenv UV_CACHE_DIR "${CACHE_DIR}/uv"
setenv HF_HOME "${CACHE_DIR}/huggingface"
setenv TRANSFORMERS_CACHE "${CACHE_DIR}/huggingface/hub"
setenv HF_DATASETS_CACHE "${CACHE_DIR}/huggingface/datasets"
setenv TORCH_HOME "${CACHE_DIR}/torch"
setenv UV_PROJECT_ENVIRONMENT "${VENV_DIR}"
setenv LEROBOT_ROOT "${LEROBOT_ROOT}"

echo "SmolVLA scratch environment activated."
echo "  uv      : ${UV_BIN_DIR}/uv"
echo "  Venv    : ${VENV_DIR}"
echo "  HF Home : ${HF_HOME}"
CSH_EOF

chmod +x "${ACTIVATE_SHIM}" "${ACTIVATE_CSH}"

echo ""
echo "[4/5] Environment activation shims written:"
echo "      bash  — ${ACTIVATE_SHIM}"
echo "      tcsh  — ${ACTIVATE_CSH}"

# ---------------------------------------------------------------------------
# 6. PATCH LEROBOT VIDEO UTILS FOR NVENC
# ---------------------------------------------------------------------------
# UCL TSG trailbreaker exposes NVENC API 12.2. PyAV 15.x (bundled with
# lerobot) requires NVENC API 13.0 and fails at codec open time.
# The system ffmpeg (Lavc59 / ffmpeg 5.x) was compiled against the older
# NVENC SDK and works fine. patch_nvenc.py replaces the PyAV call inside
# lerobot's encode_video_frames with a subprocess call to system ffmpeg
# when vcodec='h264_nvenc' is requested. Safe to re-run (idempotent).

echo ""
echo "[5/5] Patching lerobot video_utils.py for h264_nvenc compatibility..."

PATCH_SCRIPT="${HOME_PROJECT}/SmolVLA-Testing/src/patch_nvenc.py"

if [[ -f "${PATCH_SCRIPT}" ]]; then
    uv run python "${PATCH_SCRIPT}"
else
    echo "    WARNING: patch_nvenc.py not found at ${PATCH_SCRIPT}"
    echo "    Skipping NVENC patch — h264_nvenc encoding may fail if you use it."
fi

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------

echo ""
echo "================================================================="
echo "  Setup complete."
echo "================================================================="
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. Convert raw recordings to LeRobotDataset format:"
echo "        bash ~/smolvla_project/SmolVLA-Testing/scripts/restart_conversion.sh"
echo ""
echo "  2. Then launch training:"
echo "        bash ~/smolvla_project/SmolVLA-Testing/scripts/run_training.sh <dataset_name>"
echo ""
echo "  To activate the env interactively (useful for manual commands):"
echo "    bash shell : source ${ACTIVATE_SHIM}"
echo "    tcsh shell : source ${ACTIVATE_CSH}"
echo ""
echo "  REMINDER: This scratch space will be wiped when your 72-hour"
echo "  GPU booking expires. Re-run setup_scratch.sh on your next booking."
echo "  Your checkpoints in ~/smolvla_project/checkpoints/ are PERSISTENT."
echo "================================================================="
