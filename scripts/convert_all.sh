#!/usr/bin/env bash
# =============================================================================
# convert_all.sh  —  Batch Raw-Recording → LeRobotDataset Converter
# =============================================================================
#
# PURPOSE
# -------
# Iterates over a list of raw recording sessions and converts each one into
# a LeRobotDataset v3 format using data_converter.py. Runs sequentially so
# that GPU memory and I/O are not contended. Logs timestamped episode-level
# progress to stdout (redirect to a file via restart_conversion.sh).
#
# DATA LAYOUT ASSUMPTION
# ----------------------
# Raw recordings live on SCRATCH (not home) to avoid the 10GB NFS quota:
#
#   /scratch0/$USER/raw_recordings/
#     001/
#       episode_events.jsonl
#       frames/
#       ...
#     002/
#     003/
#
# Converted datasets are also written to scratch:
#
#   /scratch0/$USER/lerobot_datasets/
#     001/
#     002/
#     003/
#
# USAGE
# -----
#   DO NOT run this directly — use restart_conversion.sh which launches it
#   under nohup so it survives SSH disconnection:
#
#     bash ~/smolvla_project/SmolVLA-Testing/restart_conversion.sh
#
#   To run interactively (will stop on SSH disconnect):
#     bash ~/smolvla_project/SmolVLA-Testing/convert_all.sh
#
# SHELL REQUIREMENT
# -----------------
# UCL TSG machines use tcsh as the default interactive shell. This script is
# bash and MUST be invoked via:   bash convert_all.sh
# Do NOT run it from a tcsh prompt with:  ./convert_all.sh  or  source ...
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURATION — edit DATASETS and paths to match your session IDs
# ---------------------------------------------------------------------------

# List of raw recording session IDs to convert (directory names under RAW_ROOT)
DATASETS=(001 002 003)

SCRATCH_BASE="/scratch0/${USER}"

# Source the activation shim so uv and all cache env vars are set correctly
ACTIVATE_SHIM="${SCRATCH_BASE}/activate_smolvla.sh"

if [[ ! -f "${ACTIVATE_SHIM}" ]]; then
    echo "ERROR: Activation shim not found at ${ACTIVATE_SHIM}"
    echo "Run setup_scratch.sh first:"
    echo "  bash ~/smolvla_project/SmolVLA-Testing/setup_scratch.sh"
    exit 1
fi

# shellcheck source=/dev/null
source "${ACTIVATE_SHIM}"

RAW_ROOT="${SCRATCH_BASE}/raw_recordings"
OUTPUT_ROOT="${SCRATCH_BASE}/lerobot_datasets"
REPORT_ROOT="${SCRATCH_BASE}/lerobot_datasets"

# Camera label used as the primary viewpoint for the LeRobotDataset
PRIMARY_CAMERA="ee_zed_m"

# Video codec — h264_nvenc uses the GPU encoder (fast); fallback to libx264 if
# the NVENC patch was not applied or the GPU driver doesn't support it.
VCODEC="h264_nvenc"

REPO_DIR="${HOME}/smolvla_project/SmolVLA-Testing"

# ---------------------------------------------------------------------------
# MAIN CONVERSION LOOP
# ---------------------------------------------------------------------------

TOTAL_DATASETS=${#DATASETS[@]}
DS_INDEX=0

echo "================================================================="
echo "  SmolVLA Batch Conversion — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Datasets : ${DATASETS[*]}"
echo "  Input    : ${RAW_ROOT}"
echo "  Output   : ${OUTPUT_ROOT}"
echo "  Codec    : ${VCODEC}"
echo "================================================================="

for ds in "${DATASETS[@]}"; do
    DS_INDEX=$((DS_INDEX + 1))

    # Count total episodes in this session from the events file
    events="${RAW_ROOT}/${ds}/episode_events.jsonl"
    if [[ -f "${events}" ]]; then
        total=$(grep -c "episode_start" "${events}" 2>/dev/null || echo "?")
    else
        total="?"
    fi

    echo ""
    echo "================================================================="
    echo "  Dataset ${ds} (${DS_INDEX}/${TOTAL_DATASETS}) | Episodes: ${total} | $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================="

    uv run --project "${HOME}/smolvla_project/lerobot" \
        python "${REPO_DIR}/data_converter.py" "${ds}" \
        --datasets-root "${RAW_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --primary-camera "${PRIMARY_CAMERA}" \
        --episode-report "${REPORT_ROOT}/${ds}_report.json" \
        --vcodec "${VCODEC}" \
        --force

    echo "================================================================="
    echo "  Done: dataset ${ds} (${DS_INDEX}/${TOTAL_DATASETS}) | $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================="
done

echo ""
echo "================================================================="
echo "  All datasets converted. $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================="
