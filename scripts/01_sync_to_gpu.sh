#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"

REMOTE_HOME_DIR="$(ssh -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
REMOTE_HOME_PROJECT="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}"
REMOTE_CODE_DIR="${REMOTE_HOME_PROJECT}/SmolVLA-Testing"

echo "Syncing code to home and converted datasets to scratch..."
echo "Remote home detected as: ${REMOTE_HOME_DIR}"

# Ensure destination directories exist.
ssh -J "${SSH_JUMP}" "${SSH_REMOTE}" \
  "mkdir -p '${REMOTE_CODE_DIR}' '${REMOTE_SCRATCH_BASE}/lerobot_datasets'"

# Code to persistent home (lightweight only).
rsync -avz --progress \
  -e "ssh -J ${SSH_JUMP}" \
  --exclude '.git' \
  --exclude 'checkpoints' \
  --exclude 'lerobot_datasets' \
  --exclude 'raw_recordings' \
  --exclude 'cleaned_datasets' \
  "${LOCAL_PROJECT_ROOT}/" \
  "${SSH_REMOTE}:${REMOTE_CODE_DIR}/"

# Converted training data to scratch.
rsync -avzP \
  -e "ssh -J ${SSH_JUMP}" \
  "${LOCAL_DATA_PULL_SOURCE}/" \
  "${SSH_REMOTE}:${REMOTE_SCRATCH_BASE}/lerobot_datasets/"

echo "Sync complete."
