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

TRANSFORMERS_FROM_GIT="${TRANSFORMERS_FROM_GIT:-false}"

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s "${REMOTE_SCRATCH_BASE}" "${REMOTE_USER}" "${INSTALL_DEPS:-false}" "${TRANSFORMERS_FROM_GIT}" <<'REMOTE_EOF'
set -euo pipefail

# Arguments passed from local script
REMOTE_SCRATCH_BASE="$1"
REMOTE_USER="$2"
INSTALL_DEPS="$3"
TRANSFORMERS_FROM_GIT="$4"
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
		# Optionally install latest transformers from the Hugging Face repo
		if [ "${TRANSFORMERS_FROM_GIT:-false}" = "true" ]; then
			echo "TRANSFORMERS_FROM_GIT=true — installing transformers from the HF GitHub main branch"
			"${UV_BIN}" pip install --python "${SCRATCH_VENV}/bin/python" git+https://github.com/huggingface/transformers || {
				echo "Failed to install transformers from git"
				exit 1
			}
			# still pin tokenizers & huggingface_hub to known-good versions
			"${UV_BIN}" pip install --python "${SCRATCH_VENV}/bin/python" tokenizers==0.22.2 huggingface_hub==1.11.0 || {
				echo "Failed to install tokenizers/huggingface_hub"
				exit 1
			}
		fi

		# Pinned dependency set known to work with the target Python 3.12/vLLM setup
		# prefer binary wheels for vllm to avoid source builds requiring nvcc
		"${UV_BIN}" pip install --python "${SCRATCH_VENV}/bin/python" --only-binary :all: vllm==0.7.2 qwen-vl-utils==0.0.14 || {
			echo "Partial/failed install - check disk quota or logs"
			exit 1
		}
		echo "Verifying installation..."
			"${SCRATCH_VENV}/bin/python3" -c "import sys; from vllm import LLM; import qwen_vl_utils as qvu; import transformers, tokenizers, huggingface_hub; print('✓ Qwen dependencies installed in scratch venv', sys.version)" || echo "Warning: verification failed"
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