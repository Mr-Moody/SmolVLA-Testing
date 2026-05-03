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

# Run remote setup and ensure dependencies are installed in scratch venv using uv.
# Dependency installation is opt-in: set INSTALL_DEPS=true when calling this script to install vllm/qwen-vl-utils.

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s "${REMOTE_SCRATCH_BASE}" "${REMOTE_USER}" <<'REMOTE_EOF'
set -euo pipefail

# Arguments passed from local script
REMOTE_SCRATCH_BASE="$1"
REMOTE_USER="$2"

echo "Running remote setup_scratch.sh..."
bash ~/smolvla_project/SmolVLA-Testing/scripts/setup_scratch.sh

# Locate and use the scratch venv for dependency installation.
SCRATCH_VENV="${REMOTE_SCRATCH_BASE}/lerobot/.venv"
if [ -d "${SCRATCH_VENV}" ]; then
	echo "Scratch venv found at: ${SCRATCH_VENV}"
	"${SCRATCH_VENV}/bin/python3" -c 'import sys; print("  python:", sys.executable)'
	if [ "${INSTALL_DEPS:-false}" = "true" ]; then
		echo "INSTALL_DEPS=true - installing Qwen VL dependencies using uv..."
		/scratch0/${REMOTE_USER}/bin/uv pip install --python "${SCRATCH_VENV}/bin/python" vllm>=0.7 qwen-vl-utils || echo "Partial/failed install - check disk quota or logs"
	else
		echo "INSTALL_DEPS not set to true; skipping dependency installation."
		echo "To install deps, re-run with: INSTALL_DEPS=true bash scripts/03_setup_gpu.sh"
	fi
else
	echo "WARNING: Expected scratch venv at ${SCRATCH_VENV} but not found."
	echo "Check that setup_scratch.sh ran successfully and created the venv."
	exit 1
fi
REMOTE_EOF