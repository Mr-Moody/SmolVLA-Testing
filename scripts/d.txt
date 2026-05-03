#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOME_PROJECT_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
CODE_DIR="${REPO_ROOT}"
SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

echo "Running GPU environment setup..."
echo "Code dir (home): ${CODE_DIR}"
echo "Home project dir: ${HOME_PROJECT_DIR}"
echo "Scratch base: ${REMOTE_SCRATCH_BASE}"
echo "Python version: 3.12 (compatible with vllm, numba, PyTorch)"
echo ""

# Run remote setup and ensure dependencies are installed in scratch venv using uv.
# Dependency installation is opt-in: set INSTALL_DEPS=true when calling this script to install vllm/qwen-vl-utils.

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s "${REMOTE_SCRATCH_BASE}" "${REMOTE_USER}" "${INSTALL_DEPS:-false}" <<'REMOTE_EOF'
set -euo pipefail

# Arguments passed from local script
REMOTE_SCRATCH_BASE="$1"
REMOTE_USER="$2"
INSTALL_DEPS="$3"
UV_BIN="${REMOTE_SCRATCH_BASE}/bin/uv"

echo "Running remote setup_scratch.sh..."
bash ~/smolvla_project/SmolVLA-Testing/scripts/setup_scratch.sh

# Locate and use the scratch venv for dependency installation.
# Note: setup_scratch.sh creates VENV_DIR="${SCRATCH_BASE}/smolvla_venv"
SCRATCH_VENV="${REMOTE_SCRATCH_BASE}/smolvla_venv"
if [ -d "${SCRATCH_VENV}" ]; then
	echo "Scratch venv found at: ${SCRATCH_VENV}"
	"${SCRATCH_VENV}/bin/python3" -c 'import sys; print("  python:", sys.executable)'
	if [ "${INSTALL_DEPS:-false}" = "true" ]; then
		echo "INSTALL_DEPS=true - installing Qwen VL dependencies with uv into scratch venv..."
		if [ ! -x "${UV_BIN}" ]; then
			echo "ERROR: uv not found at ${UV_BIN}"
			exit 1
		fi
		export PIP_CACHE_DIR="${REMOTE_SCRATCH_BASE}/.cache/pip"
		export HF_HOME="${REMOTE_SCRATCH_BASE}/.cache/huggingface"
		export TORCH_HOME="${REMOTE_SCRATCH_BASE}/.cache/torch"
		export UV_CACHE_DIR="${REMOTE_SCRATCH_BASE}/.cache/uv"
		export UV_PYTHON_INSTALL_DIR="${REMOTE_SCRATCH_BASE}/.cache/uv/python"
		mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"
		"${UV_BIN}" pip install --python "${SCRATCH_VENV}/bin/python" --only-binary :all: vllm==0.7.2 || {
			echo "Partial/failed install - check disk quota or logs"
			exit 1
		}
		"${UV_BIN}" pip install --python "${SCRATCH_VENV}/bin/python" qwen-vl-utils || {
			echo "Partial/failed install - check disk quota or logs"
			exit 1
		}
		echo "Verifying installation..."
		"${SCRATCH_VENV}/bin/python3" -c "from vllm import LLM; from qwen_vl_utils import *; print('✓ Qwen dependencies installed in scratch venv')" || echo "Warning: verification failed"
	else
		echo "INSTALL_DEPS not set to true; skipping dependency installation."
		echo "To install deps, re-run with: INSTALL_DEPS=true bash scripts/03_setup_gpu.sh"
	fi
else
	echo "ERROR: Expected scratch venv at ${SCRATCH_VENV} but not found."
	echo "Check that setup_scratch.sh ran successfully and created the venv."
	exit 1
fi
REMOTE_EOF