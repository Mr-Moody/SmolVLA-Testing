# RUN ON: LOCAL MACHINE (Windows)
# Execute overnight annotation pipeline locally or on remote server via SSH

param(
    [string]$ConfigFile = "scripts/06_overnight_params.sh"
)

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

# ============================================================================
# Configuration Loading
# ============================================================================

if (-not (Test-Path $ConfigFile)) {
    Write-Host "ERROR: Config file not found: $ConfigFile" -ForegroundColor Red
    Write-Host "Usage: $($MyInvocation.MyCommand.Name) [config_file]"
    exit 1
}

# Parse bash config file
$config = @{}
Get-Content $ConfigFile | ForEach-Object {
    if ($_ -match '^([A-Z_]+)=(.+)$') {
        $key = $matches[1]
        $value = $matches[2] -replace "^[`\"']+|[`\"']+`$", ''
        $config[$key] = $value
    }
}

$OVERNIGHT_OUTPUT_DIR = if ($config["OVERNIGHT_OUTPUT_DIR"]) { $config["OVERNIGHT_OUTPUT_DIR"] } else { ".overnight_output" }
$REMOTE_EXECUTION = if ($config["REMOTE_EXECUTION"]) { $config["REMOTE_EXECUTION"] } else { "false" }
$ENABLE_ANNOTATION = if ($config["ENABLE_ANNOTATION"]) { $config["ENABLE_ANNOTATION"] } else { "true" }
$NUM_GPUS = if ($config["NUM_GPUS"]) { $config["NUM_GPUS"] } else { 1 }
$BATCH_SIZE_ANNOTATION = if ($config["BATCH_SIZE_ANNOTATION"]) { $config["BATCH_SIZE_ANNOTATION"] } else { 4 }
$CONTINUE_ON_ERROR = if ($config["CONTINUE_ON_ERROR"]) { $config["CONTINUE_ON_ERROR"] } else { "false" }
$ENABLE_CHECKPOINT = if ($config["ENABLE_CHECKPOINT"]) { $config["ENABLE_CHECKPOINT"] } else { "true" }

# Parse dataset names
$DATASET_NAMES = ""
if ($config["DATASET_NAMES"]) {
    $DATASET_NAMES = $config["DATASET_NAMES"]
}

# Remote config
$GPU_NODE = $config["GPU_NODE"]
$REMOTE_USER = $config["REMOTE_USER"]
$REMOTE_PROJECT_ROOT = $config["REMOTE_PROJECT_ROOT"]
$SSH_KEY_FILE = $config["SSH_KEY_FILE"]

# Paths
$RAW_DATASETS_ROOT = if ($config["RAW_DATASETS_ROOT"]) { $config["RAW_DATASETS_ROOT"] } else { "raw_datasets" }
$CLEANED_DATASETS_ROOT = if ($config["CLEANED_DATASETS_ROOT"]) { $config["CLEANED_DATASETS_ROOT"] } else { "cleaned_datasets" }
$LEROBOT_DATASETS_ROOT = if ($config["LEROBOT_DATASETS_ROOT"]) { $config["LEROBOT_DATASETS_ROOT"] } else { "lerobot_datasets" }

# Optional
$MAX_EPISODES_PER_DATASET = $config["MAX_EPISODES_PER_DATASET"]
$QWEN_MODEL = $config["QWEN_MODEL"]

# ============================================================================
# Logging Setup
# ============================================================================

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_DIR = Join-Path $OVERNIGHT_OUTPUT_DIR "logs"
$MAIN_LOG = Join-Path $LOG_DIR "overnight_run_$TIMESTAMP.log"

New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null

function Log {
    param([string]$msg)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $entry = "[$timestamp] $msg"
    Add-Content -Path $MAIN_LOG -Value $entry
    Write-Host $entry
}

function LogInfo {
    param([string]$msg)
    Log "[INFO] $msg"
}

function LogWarn {
    param([string]$msg)
    Log "[WARN] $msg"
}

function LogError {
    param([string]$msg)
    Log "[ERROR] $msg"
}

# ============================================================================
# Dependency Checks
# ============================================================================

function CheckDependencies {
    LogInfo "Checking dependencies..."
    
    if (-not (Get-Command python3 -ErrorAction SilentlyContinue)) {
        LogError "python3 not found"
        return $false
    }
    
    if ($REMOTE_EXECUTION -eq "true") {
        if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
            LogError "ssh not found (required for remote execution)"
            return $false
        }
    }
    
    LogInfo "Dependencies OK"
    return $true
}

function ValidateLocalDatasets {
    LogInfo "Validating local datasets..."
    
    $datasets = $DATASET_NAMES -split '\s+'
    foreach ($ds in $datasets) {
        $raw_path = Join-Path $PROJECT_ROOT $RAW_DATASETS_ROOT $ds
        if (-not (Test-Path $raw_path -PathType Container)) {
            LogError "Raw dataset not found: $raw_path"
            return $false
        }
    }
    
    LogInfo "All local datasets found"
    return $true
}

# ============================================================================
# Path Resolution
# ============================================================================

function ResolvePath {
    param([string]$path)
    if ([System.IO.Path]::IsPathRooted($path)) {
        return $path
    }
    return Join-Path $PROJECT_ROOT $path
}

# ============================================================================
# Build Commands
# ============================================================================

function BuildLocalCommand {
    $cmd = "python3 run_overnight_pipeline.py"
    
    $cmd += " --raw-datasets '$(ResolvePath $RAW_DATASETS_ROOT)'"
    $cmd += " --cleaned-datasets '$(ResolvePath $CLEANED_DATASETS_ROOT)'"
    $cmd += " --lerobot-datasets '$(ResolvePath $LEROBOT_DATASETS_ROOT)'"
    $cmd += " --dataset-names $DATASET_NAMES"
    $cmd += " --output-dir '$(ResolvePath $OVERNIGHT_OUTPUT_DIR)'"
    $cmd += " --num-gpus $NUM_GPUS"
    $cmd += " --batch-size-annotation $BATCH_SIZE_ANNOTATION"
    
    if ($ENABLE_ANNOTATION -ne "true") {
        $cmd += " --skip-annotation"
    }
    
    if ($QWEN_MODEL) {
        $cmd += " --qwen-model '$QWEN_MODEL'"
    }
    
    if ($MAX_EPISODES_PER_DATASET) {
        $cmd += " --max-episodes $MAX_EPISODES_PER_DATASET"
    }
    
    if ($ENABLE_CHECKPOINT -eq "true") {
        $checkpoint_path = Join-Path (ResolvePath $OVERNIGHT_OUTPUT_DIR) "checkpoint.json"
        $cmd += " --checkpoint '$checkpoint_path'"
    }
    
    return $cmd
}

function BuildRemoteCommand {
    $cmd = "cd '$REMOTE_PROJECT_ROOT' && python3 run_overnight_pipeline.py"
    
    $cmd += " --raw-datasets '$REMOTE_PROJECT_ROOT/$RAW_DATASETS_ROOT'"
    $cmd += " --cleaned-datasets '$REMOTE_PROJECT_ROOT/$CLEANED_DATASETS_ROOT'"
    $cmd += " --lerobot-datasets '$REMOTE_PROJECT_ROOT/$LEROBOT_DATASETS_ROOT'"
    $cmd += " --dataset-names $DATASET_NAMES"
    $cmd += " --output-dir '$REMOTE_PROJECT_ROOT/$OVERNIGHT_OUTPUT_DIR'"
    $cmd += " --num-gpus $NUM_GPUS"
    $cmd += " --batch-size-annotation $BATCH_SIZE_ANNOTATION"
    
    if ($ENABLE_ANNOTATION -ne "true") {
        $cmd += " --skip-annotation"
    }
    
    if ($QWEN_MODEL) {
        $cmd += " --qwen-model '$QWEN_MODEL'"
    }
    
    if ($MAX_EPISODES_PER_DATASET) {
        $cmd += " --max-episodes $MAX_EPISODES_PER_DATASET"
    }
    
    if ($ENABLE_CHECKPOINT -eq "true") {
        $cmd += " --checkpoint '$REMOTE_PROJECT_ROOT/$OVERNIGHT_OUTPUT_DIR/checkpoint.json'"
    }
    
    return $cmd
}

# ============================================================================
# Execution Modes
# ============================================================================

function ExecuteLocally {
    LogInfo "EXECUTION MODE: LOCAL MACHINE"
    LogInfo "Executing pipeline on this machine..."
    Log ""
    
    Push-Location $PROJECT_ROOT
    try {
        $cmd = BuildLocalCommand
        LogInfo "Running: $cmd"
        Log ""
        
        Invoke-Expression $cmd
        return $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
}

function ExecuteRemotely {
    LogInfo "EXECUTION MODE: REMOTE SERVER"
    LogInfo "Server: $GPU_NODE (user: $REMOTE_USER)"
    LogInfo "Project root: $REMOTE_PROJECT_ROOT"
    Log ""
    
    $cmd = BuildRemoteCommand
    LogInfo "Remote command:"
    LogInfo "  $cmd"
    Log ""
    
    # Execute via SSH
    & ssh -i $SSH_KEY_FILE "$REMOTE_USER@$GPU_NODE" $cmd
    return $LASTEXITCODE
}

# ============================================================================
# Main Execution
# ============================================================================

$start_time = Get-Date

Log ""
Log "================================================================================"
Log "OVERNIGHT ANNOTATION PIPELINE"
Log "================================================================================"
Log "Start time: $start_time"
Log "Config file: $ConfigFile"
Log "Log file: $MAIN_LOG"
Log ""
Log "Execution mode: $REMOTE_EXECUTION"
if ($REMOTE_EXECUTION -eq "true") {
    Log "  Remote server: $GPU_NODE"
    Log "  Remote user: $REMOTE_USER"
    Log "  Remote path: $REMOTE_PROJECT_ROOT"
}
Log ""
Log "Datasets: $DATASET_NAMES"
Log "Enable annotation: $ENABLE_ANNOTATION"
Log "Number of GPUs: $NUM_GPUS"
Log "Batch size: $BATCH_SIZE_ANNOTATION"
Log "================================================================================"
Log ""

# Check dependencies
if (-not (CheckDependencies)) {
    LogError "Dependency check failed"
    exit 1
}

# Validate datasets (local mode only)
if ($REMOTE_EXECUTION -ne "true") {
    if (-not (ValidateLocalDatasets)) {
        LogError "Dataset validation failed"
        exit 1
    }
}
else {
    LogInfo "Skipping local validation (executing on remote server)"
}

# Create output directories
New-Item -ItemType Directory -Path $OVERNIGHT_OUTPUT_DIR -Force | Out-Null
New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null

LogInfo ""
LogInfo "Starting pipeline..."
LogInfo ""

# Execute based on mode
$exit_code = 0
if ($REMOTE_EXECUTION -eq "true") {
    ExecuteRemotely
    $exit_code = $LASTEXITCODE
}
else {
    ExecuteLocally
    $exit_code = $LASTEXITCODE
}

# Handle errors
if ($exit_code -ne 0) {
    if ($CONTINUE_ON_ERROR -eq "true") {
        LogWarn "Pipeline exited with code $exit_code but continuing (CONTINUE_ON_ERROR=true)"
        $exit_code = 0
    }
    else {
        LogError "Pipeline exited with code $exit_code"
    }
}

# Final summary
$end_time = Get-Date
Log ""
Log "================================================================================"
Log "OVERNIGHT ANNOTATION PIPELINE COMPLETED"
Log "================================================================================"
Log "End time: $end_time"
Log "Output directory: $OVERNIGHT_OUTPUT_DIR"
Log "Main log: $MAIN_LOG"
Log "================================================================================"

if ($REMOTE_EXECUTION -eq "true" -and $exit_code -eq 0) {
    Log ""
    Log "Results are on remote server at: $REMOTE_PROJECT_ROOT/$OVERNIGHT_OUTPUT_DIR/"
    Log ""
    Log "To retrieve results locally, run:"
    Log "  rsync -avz -e `"ssh -i $SSH_KEY_FILE`" \"
    Log "    `"$REMOTE_USER@$GPU_NODE`:$REMOTE_PROJECT_ROOT/$OVERNIGHT_OUTPUT_DIR/`" \"
    Log "    `"./$OVERNIGHT_OUTPUT_DIR/remote_results/`""
}

exit $exit_code
