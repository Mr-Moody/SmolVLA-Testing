#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

LOCAL_CLEANED_ROOT="${LOCAL_DATA_PULL_SOURCE}"
REMOTE_CLEANED_ROOT="${REMOTE_CLEANED_DATASET_ROOT}"

mkdir -p "${LOCAL_CLEANED_ROOT}"

echo "Pulling subtasks.jsonl files from scratch into local cleaned_datasets..."
echo "Local root:  ${LOCAL_CLEANED_ROOT}"
echo "Remote root: ${REMOTE_CLEANED_ROOT}"
echo ""

for ds in "${DATASET_NAMES[@]}"; do
  local_dataset_dir="${LOCAL_CLEANED_ROOT}/${ds}"
  remote_subtasks="${SSH_REMOTE}:${REMOTE_CLEANED_ROOT}/${ds}/subtasks.jsonl"

  mkdir -p "${local_dataset_dir}"

  echo "Syncing ${ds}..."
  rsync -avzP \
    -e "ssh ${SSH_OPTS} -J ${SSH_JUMP}" \
    "${remote_subtasks}" \
    "${local_dataset_dir}/" \
    || {
      echo "WARNING: Could not pull subtasks.jsonl for ${ds}"
      continue
    }

done

echo ""
echo "Subtask extraction complete."
