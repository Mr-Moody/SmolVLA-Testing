#!/usr/bin/env bash
# Personal overrides for scripts/00_run_params.sh
# Copy to scripts/00_run_params.local.sh and edit values.

WORKFLOW_USER="eredhead"
REMOTE_USER="${WORKFLOW_USER}"
GPU_NODE="trailbreaker.cs.ucl.ac.uk"

# Optional overrides
# RUN_NAME="001_002_003"
# DATASET_NAMES=(001 002 003)
# SAVE_FREQ=1000
# DATASET_ROOT="/scratch0/${WORKFLOW_USER}/lerobot_datasets"

# Optional local path overrides
# LOCAL_PROJECT_ROOT="/Volumes/ROS2_SSD/SmolVLA-Testing"
# LOCAL_DATA_PULL_SOURCE="${LOCAL_PROJECT_ROOT}/lerobot_datasets"
# LOCAL_CHECKPOINTS_ROOT="${LOCAL_PROJECT_ROOT}/checkpoints"
# EXTRACT_FOLDER_NAME="${RUN_NAME}_smolvla_full"
