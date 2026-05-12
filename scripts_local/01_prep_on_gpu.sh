#!/usr/bin/env bash
# Full GPU-side prep: clean → annotate → convert → merge for datasets 200-209.
#
# Prerequisites (run in order):
#   bash scripts/01_sync_to_gpu.sh          (code to home)
#   bash scripts/03_setup_gpu.sh            (venv + shim)
#   bash scripts_local/00_sync_raw_to_gpu.sh (raw data to scratch)
#
# What this does on the GPU:
#   1. main.py clean        — raw_datasets → cleaned_datasets (all 11)
#   2. create_labels_msd.py — generate annotations.jsonl in each cleaned dir
#   3. main.py convert      — cleaned_datasets → lerobot_datasets (wrist_d405 only)
#                             tries h264_nvenc first, falls back to h264
#   4. merge_datasets.py    — merge all 11 → lerobot_datasets/merged_msd_200_209
#
# Output: /scratch0/$USER/lerobot_datasets/merged_msd_200_209
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

# Detect remote home (same pattern as 01_sync_to_gpu.sh)
REMOTE_HOME="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
CODE_DIR="${REMOTE_HOME}/smolvla_project/SmolVLA-Testing"
LEROBOT_DIR="${REMOTE_HOME}/smolvla_project/lerobot"
SCRATCH="/scratch0/${REMOTE_USER}"
ACTIVATE_SHIM="${SCRATCH}/activate_smolvla.sh"
RAW_SCRATCH="${SCRATCH}/raw_datasets"
CLEANED_SCRATCH="${SCRATCH}/cleaned_datasets"
LEROBOT_SCRATCH="${SCRATCH}/lerobot_datasets"
MERGED_OUTPUT="${LEROBOT_SCRATCH}/merged_msd_200_209"

echo "Running full prep on ${GPU_NODE}..."
echo "  Code    : ${CODE_DIR}"
echo "  Scratch : ${SCRATCH}"
echo "  Output  : ${MERGED_OUTPUT}"
echo ""

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'bash -s' << REMOTE
set -euo pipefail
source "${ACTIVATE_SHIM}"

mkdir -p "${CLEANED_SCRATCH}" "${LEROBOT_SCRATCH}"

# -----------------------------------------------------------------------
# Step 1: Clean all datasets (raw → cleaned)
# -----------------------------------------------------------------------
echo ""
echo "=== Step 1: Cleaning ==="
for ds in 200 201 201_1 202 203 204 205 206 207 208 209; do
    echo "--- Cleaning \${ds} ---"
    cd "${LEROBOT_DIR}"
    uv run python "${CODE_DIR}/main.py" clean "\${ds}" \
        --datasets-root "${RAW_SCRATCH}" \
        --output-root "${CLEANED_SCRATCH}" \
        --force
done

# -----------------------------------------------------------------------
# Step 2: Generate annotations.jsonl for all datasets
# -----------------------------------------------------------------------
echo ""
echo "=== Step 2: Generating annotations ==="
CLEANED_DIRS=""
for ds in 200 201 201_1 202 203 204 205 206 207 208 209; do
    CLEANED_DIRS="\${CLEANED_DIRS} ${CLEANED_SCRATCH}/\${ds}"
done
cd "${LEROBOT_DIR}"
uv run python "${CODE_DIR}/src/create_labels_msd.py" \${CLEANED_DIRS} --force

# -----------------------------------------------------------------------
# Step 3: Convert cleaned → lerobot_datasets (wrist_d405 as primary camera)
# -----------------------------------------------------------------------
echo ""
echo "=== Step 3: Converting ==="
for ds in 200 201 201_1 202 203 204 205 206 207 208 209; do
    echo "--- Converting \${ds} ---"
    cd "${LEROBOT_DIR}"
    if uv run python "${CODE_DIR}/main.py" convert "\${ds}" \
        --datasets-root "${CLEANED_SCRATCH}" \
        --output-root "${LEROBOT_SCRATCH}" \
        --primary-camera wrist_d405 \
        --cameras wrist_d405,third_person_d405 \
        --vcodec h264_nvenc \
        --force; then
        echo "  \${ds}: converted with h264_nvenc"
    else
        echo "  \${ds}: h264_nvenc failed, retrying with h264..."
        uv run python "${CODE_DIR}/main.py" convert "\${ds}" \
            --datasets-root "${CLEANED_SCRATCH}" \
            --output-root "${LEROBOT_SCRATCH}" \
            --primary-camera wrist_d405 \
            --cameras wrist_d405,third_person_d405 \
            --vcodec h264 \
            --force
        echo "  \${ds}: converted with h264"
    fi
done

# -----------------------------------------------------------------------
# Step 4: Merge all 11 datasets
# -----------------------------------------------------------------------
echo ""
echo "=== Step 4: Merging ==="
DATASET_PATHS=""
for ds in 200 201 201_1 202 203 204 205 206 207 208 209; do
    DATASET_PATHS="\${DATASET_PATHS} ${LEROBOT_SCRATCH}/\${ds}"
done
cd "${LEROBOT_DIR}"
uv run python "${CODE_DIR}/src/merge_datasets.py" \${DATASET_PATHS} \
    --output "${MERGED_OUTPUT}" \
    --force

echo ""
echo "=== Prep complete ==="
echo "Merged dataset: ${MERGED_OUTPUT}"
uv run python -c "
import json, pathlib
info = json.loads((pathlib.Path('${MERGED_OUTPUT}') / 'meta' / 'info.json').read_text())
print(f'  Episodes : {info[\"total_episodes\"]}')
print(f'  Frames   : {info[\"total_frames\"]}')
print(f'  Tasks    : {info[\"total_tasks\"]}')
print(f'  Features : {list(info[\"features\"].keys())}')
"
REMOTE

echo ""
echo "Prep finished on ${GPU_NODE}."
