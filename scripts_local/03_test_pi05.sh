#!/usr/bin/env bash
# Run a single training step of pi0.5 on the merged dataset to validate the
# full pipeline and confirm the rename mapping before uploading to HuggingFace
# and submitting to the UCL supercomputer.
#
# Prerequisites:
#   bash scripts_local/01_prep_on_gpu.sh   (merged dataset must exist)
#   bash scripts_local/02_cache_pi05.sh    (pi05_base must be cached)
#
# Training runs in the foreground so you see output immediately.
# batch_size=1, steps=1, save_freq=1 — just checks one forward+backward pass.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
SSH_OPTS="-i ${SSH_KEY_FILE} -o IdentitiesOnly=yes -o IdentityAgent=none"

REMOTE_HOME="$(ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"')"
CODE_DIR="${REMOTE_HOME}/smolvla_project/SmolVLA-Testing"
LEROBOT_DIR="${REMOTE_HOME}/smolvla_project/lerobot"
SCRATCH="/scratch0/${REMOTE_USER}"
ACTIVATE_SHIM="${SCRATCH}/activate_smolvla.sh"
MERGED_DATASET="${SCRATCH}/lerobot_datasets/merged_msd_200_209"
OUTPUT_DIR="${SCRATCH}/smolvla_outputs/msd_plug_200_209_pi05_test"

echo "Running 1-step pi0.5 validation on ${GPU_NODE}..."
echo "  Dataset : ${MERGED_DATASET}"
echo "  Output  : ${OUTPUT_DIR}"
echo ""

ssh ${SSH_OPTS} -J "${SSH_JUMP}" "${SSH_REMOTE}" 'bash -s' << REMOTE
set -euo pipefail
source "${ACTIVATE_SHIM}"

# Verify merged dataset exists
if [[ ! -f "${MERGED_DATASET}/meta/info.json" ]]; then
    echo "ERROR: merged dataset not found at ${MERGED_DATASET}"
    echo "Run scripts_local/01_prep_on_gpu.sh first."
    exit 1
fi

echo "Dataset info:"
uv run python -c "
import json, pathlib
info = json.loads((pathlib.Path('${MERGED_DATASET}') / 'meta' / 'info.json').read_text())
print(f'  Episodes : {info[\"total_episodes\"]}')
print(f'  Frames   : {info[\"total_frames\"]}')
print(f'  Tasks    : {info[\"total_tasks\"]}')
print(f'  Features : {list(info[\"features\"].keys())}')
"

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "Launching 1-step pi0.5 test..."
cd "${LEROBOT_DIR}"

export HF_HUB_OFFLINE=1
export TORCH_HOME="${SCRATCH}/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python "${CODE_DIR}/main.py" train \
    --model-type pi05 \
    --dataset-root "${MERGED_DATASET}" \
    --lerobot-root "${LEROBOT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --policy-path lerobot/pi05_base \
    --batch-size 1 \
    --steps 1 \
    --save-freq 1 \
    --num-workers 2 \
    --log-freq 1 \
    --seed 1000 \
    --device cuda \
    --gradient-checkpointing \
    --dtype bfloat16 \
    --eval-freq 0

echo ""
echo "=== 1-step pi0.5 test PASSED ==="
echo "Output: ${OUTPUT_DIR}"
ls "${OUTPUT_DIR}/" 2>/dev/null || true
REMOTE

echo ""
echo "Test complete. If no errors above, the merged dataset and rename mapping are valid."
echo "Next steps:"
echo "  1. rsync merged dataset back: rsync -avzP --ignore-existing \\"
echo "       -e 'ssh -J ucl-knuckles' \\"
echo "       ucl-bumblebee:${SCRATCH}/lerobot_datasets/merged_msd_200_209/ \\"
echo "       lerobot_datasets/merged_msd_200_209/"
echo "  2. Upload to HuggingFace and submit to supercomputer."
