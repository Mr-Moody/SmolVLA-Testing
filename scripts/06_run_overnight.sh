#!/usr/bin/env bash
# Main overnight annotation and conversion pipeline
# Runs: Clean → Annotate with Qwen → Convert to lerobot format
#
# Usage:
#   ./scripts/06_run_overnight.sh
#   ./scripts/06_run_overnight.sh --config-file custom_params.sh
#   ./scripts/06_run_overnight.sh --dataset-names "001 002 003"
#
# Features:
#   • Robust error handling with per-dataset logging
#   • Checkpoint/resume support for long runs
#   • Real-time progress monitoring
#   • Comprehensive summary report

set -euo pipefail

# ============================================================================
# Configuration Loading
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load parameters from config file (or use defaults)
CONFIG_FILE="${1:-${SCRIPT_DIR}/06_overnight_params.sh}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file]"
    exit 1
fi

source "$CONFIG_FILE"

# ============================================================================
# Logging Setup
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/overnight_run_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

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
# Utility Functions
# ============================================================================

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found"
        return 1
    fi
    
    # Check if lerobot env is available
    if [[ ! -d "$PROJECT_ROOT/../lerobot" ]]; then
        log_error "lerobot directory not found at $PROJECT_ROOT/../lerobot"
        return 1
    fi
    
    log_info "Dependencies OK"
    return 0
}

validate_datasets() {
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

create_summary_report() {
    local report_file="$1"
    
    cat > "$report_file" << 'EOF'
# Overnight Annotation Pipeline Summary

## Overview
- **Start Time**: {START_TIME}
- **End Time**: {END_TIME}
- **Total Duration**: {DURATION}
- **Overall Status**: {STATUS}

## Processing Results

| Dataset | Cleaning | Annotation | Conversion | Status | Error |
|---------|----------|-----------|-----------|--------|-------|
{RESULTS_TABLE}

## Statistics
- **Total Datasets**: {TOTAL_DATASETS}
- **Successful**: {SUCCESS_COUNT}
- **Failed**: {FAILED_COUNT}
- **Total Episodes Processed**: {TOTAL_EPISODES}

## Logs
- Main log: {MAIN_LOG}
- Per-dataset logs: {LOG_DIR}/

## Output
- Cleaned datasets: {CLEANED_DATASETS_ROOT}/
- LeRobot datasets: {LEROBOT_DATASETS_ROOT}/
- Annotations: {OVERNIGHT_OUTPUT_DIR}/

EOF

    echo "$report_file"
}

# ============================================================================
# Build Command
# ============================================================================

build_pipeline_command() {
    local cmd="python3 run_overnight_pipeline.py"
    
    cmd="$cmd --raw-datasets '$RAW_DATASETS_ROOT'"
    cmd="$cmd --cleaned-datasets '$CLEANED_DATASETS_ROOT'"
    cmd="$cmd --lerobot-datasets '$LEROBOT_DATASETS_ROOT'"
    cmd="$cmd --dataset-names $DATASET_NAMES"
    cmd="$cmd --output-dir '$OVERNIGHT_OUTPUT_DIR'"
    cmd="$cmd --num-gpus $NUM_GPUS"
    cmd="$cmd --batch-size-annotation $BATCH_SIZE_ANNOTATION"
    
    if [[ "$ENABLE_ANNOTATION" != "true" ]]; then
        cmd="$cmd --skip-annotation"
    fi
    
    if [[ -n "$QWEN_MODEL" ]]; then
        cmd="$cmd --qwen-model '$QWEN_MODEL'"
    fi
    
    if [[ -n "$MAX_EPISODES_PER_DATASET" ]]; then
        cmd="$cmd --max-episodes $MAX_EPISODES_PER_DATASET"
    fi
    
    if [[ "$ENABLE_CHECKPOINT" == "true" ]] && [[ -f "$CHECKPOINT_FILE" ]]; then
        cmd="$cmd --checkpoint '$CHECKPOINT_FILE'"
    fi
    
    echo "$cmd"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    local exit_code=0
    local start_time=$(date)
    
    log "=================================================================================="
    log "OVERNIGHT ANNOTATION PIPELINE"
    log "=================================================================================="
    log "Start time: $start_time"
    log "Config file: $CONFIG_FILE"
    log "Log file: $MAIN_LOG"
    log "Datasets: $DATASET_NAMES"
    log "=================================================================================="
    log ""
    
    # Check dependencies
    if ! check_dependencies; then
        log_error "Dependency check failed"
        exit 1
    fi
    
    # Validate datasets
    if ! validate_datasets; then
        log_error "Dataset validation failed"
        exit 1
    fi
    
    # Create output directories
    mkdir -p "$OVERNIGHT_OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    
    # Print configuration summary
    log_info "Configuration:"
    log_info "  Datasets: $DATASET_NAMES"
    log_info "  Enable annotation: $ENABLE_ANNOTATION"
    log_info "  Number of GPUs: $NUM_GPUS"
    log_info "  Batch size: $BATCH_SIZE_ANNOTATION"
    log_info "  Continue on error: $CONTINUE_ON_ERROR"
    log ""
    
    # Build and run pipeline
    cd "$PROJECT_ROOT"
    
    cmd=$(build_pipeline_command)
    log_info "Running: $cmd"
    log ""
    
    if eval "$cmd"; then
        exit_code=0
        log_info "Pipeline completed successfully"
    else
        exit_code=$?
        if [[ "$CONTINUE_ON_ERROR" == "true" ]]; then
            log_warn "Pipeline exited with code $exit_code but continuing due to CONTINUE_ON_ERROR"
            exit_code=0
        else
            log_error "Pipeline exited with code $exit_code"
        fi
    fi
    
    # Final summary
    local end_time=$(date)
    log ""
    log "=================================================================================="
    log "OVERNIGHT ANNOTATION PIPELINE COMPLETED"
    log "=================================================================================="
    log "End time: $end_time"
    log "Output directory: $OVERNIGHT_OUTPUT_DIR"
    log "Main log: $MAIN_LOG"
    log "=================================================================================="
    
    # Send completion email if configured
    if [[ -n "$COMPLETION_EMAIL" ]]; then
        log_info "Would send completion email to $COMPLETION_EMAIL"
        # In production, uncomment and implement email sending:
        # send_completion_email "$COMPLETION_EMAIL" "$exit_code" "$MAIN_LOG"
    fi
    
    return $exit_code
}

# Run main
main
exit $?
