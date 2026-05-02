# RUN ON: LOCAL MACHINE (Windows)
# Check remote GPU server is ready for training

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

# Load config
$config = @{}
Get-Content "$SCRIPT_DIR/00_run_params.sh" | ForEach-Object {
    if ($_ -match '^([A-Z_]+)=(.+)$') {
        $key = $matches[1]
        $value = $matches[2] -replace "^[`\"']+|[`\"']+`$", ''
        $config[$key] = $value
    }
}

$REMOTE_USER = $config["REMOTE_USER"]
$GPU_NODE = $config["GPU_NODE"]
$JUMP_HOST = $config["JUMP_HOST"]
$SSH_KEY_FILE = $config["SSH_KEY_FILE"]
$REMOTE_PROJECT_DIRNAME = $config["REMOTE_PROJECT_DIRNAME"]
$DATASET_NAMES = @()
if ($config["DATASET_NAMES"] -match '\((.+?)\)') {
    $DATASET_NAMES = $matches[1] -split '\s+' | Where-Object { $_ }
}
$PREPROCESS_ON_GPU = $config["PREPROCESS_ON_GPU"]
$DATASET_ROOT = $config["DATASET_ROOT"]
$REMOTE_CLEANED_DATASET_ROOT = $config["REMOTE_CLEANED_DATASET_ROOT"]

$FAILED = 0

function Pass {
    param([string]$msg)
    Write-Host "[PASS] $msg" -ForegroundColor Green
}

function Warn {
    param([string]$msg)
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

function Fail {
    param([string]$msg)
    Write-Host "[FAIL] $msg" -ForegroundColor Red
    $global:FAILED = 1
}

function RequireLocalDir {
    param([string]$path, [string]$label)
    if (Test-Path $path -PathType Container) {
        Pass "$label : $path"
    }
    else {
        Fail "$label missing: $path"
    }
}

Write-Host "=== SmolVLA Preflight Check ===" -ForegroundColor Cyan
Write-Host "Run name      : $($config["RUN_NAME"])"
Write-Host "Remote target : $REMOTE_USER@$GPU_NODE (via $JUMP_HOST)"
Write-Host ""

# Check local prerequisites
Write-Host "[1/4] Local Prerequisites"
RequireLocalDir $PROJECT_ROOT "Local project root"
if ($PREPROCESS_ON_GPU -eq "true") {
    RequireLocalDir "$PROJECT_ROOT/cleaned_datasets" "Local cleaned dataset root"
    foreach ($ds in $DATASET_NAMES) {
        RequireLocalDir "$PROJECT_ROOT/cleaned_datasets/$ds" "Local cleaned dataset $ds"
    }
}
else {
    RequireLocalDir "$PROJECT_ROOT/lerobot_datasets" "Local converted dataset root"
    foreach ($ds in $DATASET_NAMES) {
        RequireLocalDir "$PROJECT_ROOT/lerobot_datasets/$ds" "Local dataset $ds"
    }
}

# Check SSH connectivity
Write-Host ""
Write-Host "[2/4] SSH Reachability"
$REMOTE_HOME_DIR = & ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -o ConnectTimeout=12 -J "$REMOTE_USER@$JUMP_HOST" "$REMOTE_USER@$GPU_NODE" "printf %s `$HOME" 2>$null

if ($REMOTE_HOME_DIR) {
    Pass "SSH/jump connectivity to $GPU_NODE"
    Pass "Remote home detected: $REMOTE_HOME_DIR"
}
else {
    Fail "Cannot connect to $GPU_NODE via jump host $JUMP_HOST"
    Write-Host "      Verify booking is active and credentials/keys are valid."
}

# Check remote layout
Write-Host ""
Write-Host "[3/4] Remote Layout"
if ($REMOTE_HOME_DIR) {
    $REMOTE_HOME_PROJECT = "$REMOTE_HOME_DIR/$REMOTE_PROJECT_DIRNAME"
    Pass "Remote home project: $REMOTE_HOME_PROJECT"
    Pass "Remote scratch base: $($config["REMOTE_SCRATCH_BASE"])"
}

if ($FAILED -eq 0) {
    Write-Host ""
    Write-Host "✓ All checks passed!" -ForegroundColor Green
    exit 0
}
else {
    Write-Host ""
    Write-Host "✗ Some checks failed" -ForegroundColor Red
    exit 1
}
