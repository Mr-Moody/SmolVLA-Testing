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

# Run remote setup and ensure a venv exists in the scratch area.
# Dependency installation is opt-in: set INSTALL_DEPS=true when calling this script to install vllm/qwen-vl-utils into the scratch venv.

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" "bash -s" <<REMOTE_EOF
set -euo pipefail

# Local-supplied value for remote scratch base
REMOTE_SCRATCH_BASE="${REMOTE_SCRATCH_BASE}"

echo "Running remote setup_scratch.sh..."
bash ~/smolvla_project/SmolVLA-Testing/scripts/setup_scratch.sh

# Create or inspect a project venv inside the scratch area.
SCRATCH_VENV="${REMOTE_SCRATCH_BASE}/lerobot/.venv"
if [ -d "${SCRATCH_VENV}" ]; then
	echo "Scratch venv already exists at: ${SCRATCH_VENV}"
	"${SCRATCH_VENV}/bin/python3" -c 'import sys; print("venv:", sys.executable)'
	if [ "${INSTALL_DEPS:-false}" = "true" ]; then
		echo "INSTALL_DEPS=true - installing/ensuring required packages in existing scratch venv..."
		"${SCRATCH_VENV}/bin/pip" install --upgrade pip setuptools wheel
		"${SCRATCH_VENV}/bin/pip" install vllm>=0.7 qwen-vl-utils || echo "Partial/failed install - check disk quota or logs"
	else
		echo "INSTALL_DEPS not set to true; skipping dependency installation into existing venv."
	fi
else
	echo "Creating scratch venv at: ${SCRATCH_VENV}"
	mkdir -p "$(dirname "${SCRATCH_VENV}")"
	python3 -m venv "${SCRATCH_VENV}"
	"${SCRATCH_VENV}/bin/pip" install --upgrade pip setuptools wheel
	if [ "${INSTALL_DEPS:-false}" = "true" ]; then
		echo "Installing required packages into scratch venv..."
		"${SCRATCH_VENV}/bin/pip" install vllm>=0.7 qwen-vl-utils || echo "Partial/failed install - check disk quota or logs"
	else
		echo "INSTALL_DEPS not set to true; skipping heavy dependency installs."
		echo "To install deps, re-run with: INSTALL_DEPS=true bash scripts/03_setup_gpu.sh"
	fi
fi
REMOTE_EOF
