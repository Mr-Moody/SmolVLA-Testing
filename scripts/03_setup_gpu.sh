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

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" "bash ~/smolvla_project/SmolVLA-Testing/scripts/setup_scratch.sh"

echo ""
echo "Installing Qwen dependencies on remote server..."
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" bash << 'EOF'
set -e
echo "Activating lerobot environment..."
if [[ -f ~/lerobot/.venv/bin/activate ]]; then
    source ~/lerobot/.venv/bin/activate
fi

echo "Installing vllm and qwen-vl-utils..."
pip install --upgrade pip
pip install vllm>=0.7 qwen-vl-utils

echo "Verifying installation..."
python3 -c "from vllm import LLM; from qwen_vl_utils import *; print('✓ Qwen dependencies installed successfully')"
EOF

echo "✓ Remote environment ready for Qwen annotation"
