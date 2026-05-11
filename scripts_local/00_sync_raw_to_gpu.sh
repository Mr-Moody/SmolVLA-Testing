#!/usr/bin/env bash
# Sync raw_datasets/200-209 from local to /scratch0/$USER/raw_datasets on GPU.
# Run this once before scripts_local/01_prep_on_gpu.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/00_run_params.sh"

DATASETS=(200 201 201_1 202 203 204 205 206 207 208 209)
SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"
RAW_SCRATCH="/scratch0/${REMOTE_USER}/raw_datasets"
LOCAL_RAW="${SCRIPT_DIR}/../raw_datasets"

echo "Creating raw_datasets directory on GPU scratch..."
ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" "mkdir -p '${RAW_SCRATCH}'"

SSH_E="ssh ${SSH_OPTS} -J ${SSH_JUMP} -o ServerAliveInterval=30 -o ServerAliveCountMax=20"

for ds in "${DATASETS[@]}"; do
    echo "Syncing raw_datasets/${ds} ..."
    rsync -avzP \
        -e "${SSH_E}" \
        "${LOCAL_RAW}/${ds}/" \
        "${SSH_REMOTE}:${RAW_SCRATCH}/${ds}/"
done

echo ""
echo "Raw dataset sync complete → ${GPU_NODE}:${RAW_SCRATCH}"
