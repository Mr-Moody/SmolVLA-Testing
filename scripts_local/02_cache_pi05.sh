#!/usr/bin/env bash
# Cache lerobot/pi05_base on the GPU node (required before HF_HUB_OFFLINE training).
# Run once per GPU booking after scripts/03_setup_gpu.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

SCRATCH="/scratch0/${REMOTE_USER}"
ACTIVATE_SHIM="${SCRATCH}/activate_smolvla.sh"

echo "Caching pi05_base on ${GPU_NODE} (requires internet, ~GB download)..."

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'bash -s' << REMOTE
set -euo pipefail
source "${ACTIVATE_SHIM}"

VENV="${SCRATCH}/smolvla_venv/bin/python"

echo "Downloading lerobot/pi05_base ..."
\${VENV} -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/pi05_base')
print('lerobot/pi05_base cached OK')
"

echo ""
echo "HF cache contents:"
du -sh "${SCRATCH}/.cache/huggingface/hub/"* 2>/dev/null | sort -h || true
REMOTE

echo ""
echo "pi05_base cache done on ${GPU_NODE}."
