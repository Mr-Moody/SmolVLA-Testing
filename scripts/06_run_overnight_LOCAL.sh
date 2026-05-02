#!/usr/bin/env bash
#
# RUN ON: LOCAL MACHINE
#
# This script runs on your local machine and can either:
#   1. Execute the pipeline directly on your local machine, OR
#   2. SSH to a remote GPU server and execute it there
#
# Configuration in scripts/06_overnight_params.sh controls execution mode:
#   - REMOTE_EXECUTION=false → runs locally
#   - REMOTE_EXECUTION=true  → runs on remote server via SSH
#
# Usage:
#   ./scripts/06_run_overnight.sh                              # Uses default params
#   ./scripts/06_run_overnight.sh scripts/06_overnight_params.sh  # Custom config
#   ./scripts/06_run_overnight.sh scripts/06_example_small_batch.sh
#

set -uo pipefail

# ============================================================================
# Configuration Loading
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load parameters
CONFIG_FILE="${1:-${SCRIPT_DIR}/06_overnight_params.sh}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file]"
    exit 1
fi

source "$CONFIG_FILE"

# Defaults
OVERNIGHT_OUTPUT_DIR="${OVERNIGHT_OUTPUT_DIR:-.overnight_output}"
REMOTE_EXECUTION="${REMOTE_EXECUTION:-false}"
ENABLE_ANNOTATION="${ENABLE_ANNOTATION:-true}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE_ANNOTATION="${BATCH_SIZE_ANNOTATION:-4}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
ENABLE_CHECKPOINT="${ENABLE_CHECKPOINT:-true}"

# ============================================================================
# Logging Setup
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${OVERNIGHT_OUTPUT_DIR}/logs"
MAIN_LOG="${LOG_DIR}/overnight_run_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Redirect output to log file
exec 1> >(tee -a "$MAIN_LOG")
exec 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

# ============================================================================
# Dependency Checks
# ============================================================================

check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found"
        return 1
    fi

    if [[ "$REMOTE_EXECUTION" == "true" ]]; then
        if ! command -v ssh &> /dev/null; then
            log_error "ssh not found (required for remote execution)"
            return 1
        fi
    fi

    log_info "Dependencies OK"
    return 0
}

validate_local_datasets() {
    log_info "Validating datasets..."

    for ds in $DATASET_NAMES; do
        raw_path="${PROJECT_ROOT}/${RAW_DATASETS_ROOT}/${ds}"
        if [[ ! -d "$raw_path" ]]; then
            log_error "Raw dataset not found: $raw_path"
            return 1
        fi
    done

    log_info "All datasets found"
    return 0
}

# ============================================================================
# Path Resolution
# ============================================================================

resolve_local_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        echo "$path"
    else
        echo "${PROJECT_ROOT}/${path}"
    fi
}

# ============================================================================
# Build Commands
# ============================================================================

build_local_command() {
    local cmd="python3 run_overnight_pipeline.py"

    cmd="$cmd --raw-datasets '$(resolve_local_path "$RAW_DATASETS_ROOT")'"
    cmd="$cmd --cleaned-datasets '$(resolve_local_path "$CLEANED_DATASETS_ROOT")'"
    cmd="$cmd --lerobot-datasets '$(resolve_local_path "$LEROBOT_DATASETS_ROOT")'"
    cmd="$cmd --dataset-names $DATASET_NAMES"
    cmd="$cmd --output-dir '$(resolve_local_path "$OVERNIGHT_OUTPUT_DIR")'"
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
        cmd="$cmd --checkpoint '$(resolve_local_path "$OVERNIGHT_OUTPUT_DIR")/checkpoint.json'"
    fi

    echo "$cmd"
}

build_remote_command() {
    # This command runs on the remote server
    # Note: Uses absolute paths based on remote project root
    local cmd="cd '${REMOTE_PROJECT_ROOT}' && python3 run_overnight_pipeline.py"

    cmd="$cmd --raw-datasets '${REMOTE_PROJECT_ROOT}/${RAW_DATASETS_ROOT}'"
    cmd="$cmd --cleaned-datasets '${REMOTE_PROJECT_ROOT}/${CLEANED_DATASETS_ROOT}'"
    cmd="$cmd --lerobot-datasets '${REMOTE_PROJECT_ROOT}/${LEROBOT_DATASETS_ROOT}'"
    cmd="$cmd --dataset-names $DATASET_NAMES"
    cmd="$cmd --output-dir '${REMOTE_PROJECT_ROOT}/${OVERNIGHT_OUTPUT_DIR}'"
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
        cmd="$cmd --checkpoint '${REMOTE_PROJECT_ROOT}/${OVERNIGHT_OUTPUT_DIR}/checkpoint.json'"
    fi

    echo "$cmd"
}

# ============================================================================
# Execution Modes
# ============================================================================

execute_locally() {
    log_info "EXECUTION MODE: LOCAL MACHINE"
    log_info "Executing pipeline on this machine..."
    log ""

    cd "${PROJECT_ROOT}"

    local cmd=$(build_local_command)
    log_info "Running: $cmd"
    log ""

    eval "$cmd"
    return $?
}

execute_remotely() {
    log_info "EXECUTION MODE: REMOTE SERVER"
    log_info "Server: ${GPU_NODE} (user: ${REMOTE_USER})"
    log_info "Project root: ${REMOTE_PROJECT_ROOT}"
    log ""

    local cmd=$(build_remote_command)
    log_info "Remote command:"
    log_info "  $cmd"
    log ""

    # Execute via SSH
    ssh -i "${SSH_KEY_FILE}" "${REMOTE_USER}@${GPU_NODE}" "$cmd"
    return $?
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    local exit_code=0
    local start_time=$(date)

    log ""
    log "================================================================================"
    log "OVERNIGHT ANNOTATION PIPELINE"
    log "================================================================================"
    log "Start time: $start_time"
    log "Config file: $CONFIG_FILE"
    log "Log file: $MAIN_LOG"
    log ""
    log "Execution mode: ${REMOTE_EXECUTION:=false}"
    if [[ "$REMOTE_EXECUTION" == "true" ]]; then
        log "  Remote server: $GPU_NODE"
        log "  Remote user: $REMOTE_USER"
        log "  Remote path: $REMOTE_PROJECT_ROOT"
    fi
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

    # Validate datasets (local mode only)
    if [[ "$REMOTE_EXECUTION" != "true" ]]; then
        if ! validate_local_datasets; then
            log_error "Dataset validation failed"
            exit 1
        fi
    else
        log_info "Skipping local validation (executing on remote server)"
    fi

    # Create output directories
    mkdir -p "$OVERNIGHT_OUTPUT_DIR"
    mkdir -p "$LOG_DIR"

    log_info ""
    log_info "Starting pipeline..."
    log_info ""

    # Execute based on mode
    if [[ "$REMOTE_EXECUTION" == "true" ]]; then
        execute_remotely
        exit_code=$?
    else
        execute_locally
        exit_code=$?
    fi

    # Handle errors
    if [[ $exit_code -ne 0 ]]; then
        if [[ "$CONTINUE_ON_ERROR" == "true" ]]; then
            log_warn "Pipeline exited with code $exit_code but continuing (CONTINUE_ON_ERROR=true)"
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
    log "Output directory: $OVERNIGHT_OUTPUT_DIR"
    log "Main log: $MAIN_LOG"
    log "================================================================================"

    if [[ "$REMOTE_EXECUTION" == "true" ]] && [[ $exit_code -eq 0 ]]; then
        log ""
        log "Results are on remote server at: ${REMOTE_PROJECT_ROOT}/${OVERNIGHT_OUTPUT_DIR}/"
        log ""
        log "To retrieve results locally, run:"
        log "  rsync -avz -e \"ssh -i ${SSH_KEY_FILE}\" \\"
        log "    \"${REMOTE_USER}@${GPU_NODE}:${REMOTE_PROJECT_ROOT}/${OVERNIGHT_OUTPUT_DIR}/\" \\"
        log "    \"./${OVERNIGHT_OUTPUT_DIR}/remote_results/\""
    fi

    return $exit_code
}

# ============================================================================
# Entry Point
# ============================================================================

main
exit $?
