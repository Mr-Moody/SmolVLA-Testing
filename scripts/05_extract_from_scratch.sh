#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

RUN_TAG="${RUN_NAME}_${MODEL_TYPE}"
REMOTE_RUN_DIR="${REMOTE_SCRATCH_BASE}/smolvla_outputs/${RUN_TAG}"
REMOTE_LOG_FILE="${REMOTE_SCRATCH_BASE}/smolvla_outputs/${RUN_TAG}.log"
LOCAL_DEST="${LOCAL_CHECKPOINTS_ROOT}/${EXTRACT_FOLDER_NAME}"

# Stop if destination already exists and is non-empty.
if [[ -e "${LOCAL_DEST}" ]] && [[ -n "$(ls -A "${LOCAL_DEST}" 2>/dev/null || true)" ]]; then
  echo "ERROR: Extraction target already exists and is not empty: ${LOCAL_DEST}"
  echo "Refusing to overwrite. Change EXTRACT_FOLDER_NAME in scripts/00_run_params.sh."
  exit 1
fi

mkdir -p "${LOCAL_DEST}" "${LOCAL_DEST}/logs"

echo "Checking remote run directory exists..."
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" "test -d '${REMOTE_RUN_DIR}'"

echo "Pulling checkpoints from scratch..."
rsync -avP --partial --inplace --timeout=120 \
  -e "ssh ${SSH_OPTS} -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -J ${SSH_JUMP}" \
  "${SSH_REMOTE}:${REMOTE_RUN_DIR}/" \
  "${LOCAL_DEST}/"

echo "Pulling launcher log..."
rsync -avP \
  -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
  "${SSH_REMOTE}:${REMOTE_LOG_FILE}" \
  "${LOCAL_DEST}/logs/" || true

echo "Extraction complete: ${LOCAL_DEST}"
