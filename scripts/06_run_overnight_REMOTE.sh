#!/usr/bin/env bash
#
# RUN ON: REMOTE SERVER (GPU node)
#
# This script runs DIRECTLY on the remote GPU server.
# Copy this script to the remote server and run it there:
#
#   # On local machine:
#   scp scripts/06_run_overnight_REMOTE.sh user@server:~/SmolVLA-Testing/scripts/
#
#   # Then SSH to server and run:
#   ssh user@server
#   cd ~/SmolVLA-Testing
#   chmod +x scripts/06_run_overnight_REMOTE.sh
#   ./scripts/06_run_overnight_REMOTE.sh
#

set -uo pipefail

# ============================================================================
# Configuration (edit these for remote execution)
# ============================================================================

# Dataset names
DATASET_NAMES="001 002 003"

# Directory structure (relative to this script's location)
RAW_DATASETS_ROOT="raw_datasets"
CLEANED_DATASETS_ROOT="cleaned_datasets"
LEROBOT_DATASETS_ROOT="lerobot_datasets"
OVERNIGHT_OUTPUT_DIR="overnight_output"

# Qwen3-VL configuration
ENABLE_ANNOTATION=true
NUM_GPUS=1
BATCH_SIZE_ANNOTATION=4
QWEN_MODEL=""  # Leave empty for auto-detect

# Processing options
MAX_EPISODES_PER_DATASET=""
CONTINUE_ON_ERROR=false
ENABLE_CHECKPOINT=true

# ============================================================================
# Logging
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/${OVERNIGHT_OUTPUT_DIR}/logs"
MAIN_LOG="${LOG_DIR}/overnight_run_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Redirect output
exec 1> >(tee -a "$MAIN_LOG")
exec 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

# ============================================================================
# Dependency Checks
# ============================================================================

check_dependencies() {
    log_info "Checking dependencies on remote server..."

    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found"
        return 1
    fi

    # Check CUDA/GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found (GPU/CUDA not available)"
        return 1
    fi

    log_info "Dependencies OK"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    return 0
}

# ============================================================================
# Build Command
# ============================================================================

build_command() {
    local cmd="python3 run_overnight_pipeline.py"

    cmd="$cmd --raw-datasets '${PROJECT_ROOT}/${RAW_DATASETS_ROOT}'"
    cmd="$cmd --cleaned-datasets '${PROJECT_ROOT}/${CLEANED_DATASETS_ROOT}'"
    cmd="$cmd --lerobot-datasets '${PROJECT_ROOT}/${LEROBOT_DATASETS_ROOT}'"
    cmd="$cmd --dataset-names $DATASET_NAMES"
    cmd="$cmd --output-dir '${PROJECT_ROOT}/${OVERNIGHT_OUTPUT_DIR}'"
    cmd="$cmd --num-gpus $NUM_GPUS"
    cmd="$cmd --batch-size-annotation $BATCH_SIZE_ANNOTATION"

    if [[ "$ENABLE_ANNOTATION" != "true" ]]; then
        cmd="$cmd --skip-annotation"
    fi

    if [[ -n "${QWEN_MODEL:-}" ]]; then
        cmd="$cmd --qwen-model '$QWEN_MODEL'"
    fi

    if [[ -n "${MAX_EPISODES_PER_DATASET:-}" ]]; then
        cmd="$cmd --max-episodes $MAX_EPISODES_PER_DATASET"
    fi

    if [[ "$ENABLE_CHECKPOINT" == "true" ]]; then
        cmd="$cmd --checkpoint '${PROJECT_ROOT}/${OVERNIGHT_OUTPUT_DIR}/checkpoint.json'"
    fi

    echo "$cmd"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    local exit_code=0
    local start_time=$(date)

    log ""
    log "================================================================================"
    log "OVERNIGHT ANNOTATION PIPELINE - REMOTE SERVER EXECUTION"
    log "================================================================================"
    log "Start time: $start_time"
    log "Server: $(hostname)"
    log "Working directory: ${PROJECT_ROOT}"
    log "Log file: $MAIN_LOG"
    log ""
    log "Datasets: $DATASET_NAMES"
    log "Enable annotation: $ENABLE_ANNOTATION"
    log "Number of GPUs: $NUM_GPUS"
    log "Batch size: $BATCH_SIZE_ANNOTATION"
    log "================================================================================"
    log ""

    # Check dependencies
    if ! check_dependencies; then
        log_error "Dependency check failed"
        exit 1
    fi

    # Create output directories
    mkdir -p "${PROJECT_ROOT}/${OVERNIGHT_OUTPUT_DIR}"
    mkdir -p "$LOG_DIR"

    log_info ""
    log_info "Starting pipeline on remote server..."
    log_info ""

    # Build and execute command
    cd "${PROJECT_ROOT}"

    local cmd=$(build_command)
    log_info "Running: $cmd"
    log ""

    eval "$cmd"
    exit_code=$?

    # Handle errors
    if [[ $exit_code -ne 0 ]]; then
        if [[ "$CONTINUE_ON_ERROR" == "true" ]]; then
            log "Pipeline exited with code $exit_code but continuing (CONTINUE_ON_ERROR=true)"
            exit_code=0
        else
            log_error "Pipeline exited with code $exit_code"
        fi
    fi

    # Final summary
    local end_time=$(date)
    log ""
    log "================================================================================"
    log "OVERNIGHT ANNOTATION PIPELINE COMPLETED"
    log "================================================================================"
    log "End time: $end_time"
    log "Output directory: ${PROJECT_ROOT}/${OVERNIGHT_OUTPUT_DIR}"
    log "Log file: $MAIN_LOG"
    log "================================================================================"

    return $exit_code
}

# ============================================================================
# Entry Point
# ============================================================================

main
exit $?
