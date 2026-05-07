# RUN ON: LOCAL MACHINE (Windows)
# Sync code and datasets to remote GPU server

param(
    [string]$ConfigFile = "scripts/06_overnight_params.sh"
)

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

# Load bash config by parsing it
$config = @{}
Get-Content "$SCRIPT_DIR/00_run_params.sh" | ForEach-Object {
    if ($_ -match '^([A-Z_]+)=(.+)$') {
        $key = $matches[1]
        $value = $matches[2] -replace "^[`"']+|[`"']+$", ''
        $config[$key] = $value
    }
}

$REMOTE_USER = $config["REMOTE_USER"]
$GPU_NODE = $config["GPU_NODE"]
$JUMP_HOST = $config["JUMP_HOST"]
$SSH_KEY_FILE = $config["SSH_KEY_FILE"]
$REMOTE_PROJECT_DIRNAME = $config["REMOTE_PROJECT_DIRNAME"]
$REMOTE_SCRATCH_BASE = $config["REMOTE_SCRATCH_BASE"]
$REMOTE_CLEANED_DATASET_ROOT = $config["REMOTE_CLEANED_DATASET_ROOT"]
$LOCAL_CLEANED_DATA_SOURCE = if ($config["LOCAL_CLEANED_DATA_SOURCE"]) { $config["LOCAL_CLEANED_DATA_SOURCE"] } else { "$PROJECT_ROOT/cleaned_datasets" }
$PREPROCESS_ON_GPU = $config["PREPROCESS_ON_GPU"]
$LOCAL_DATA_PULL_SOURCE = if ($config["LOCAL_DATA_PULL_SOURCE"]) { $config["LOCAL_DATA_PULL_SOURCE"] } else { "$PROJECT_ROOT/lerobot_datasets" }

# Parse DATASET_NAMES array
$DATASET_NAMES = @()
if ($config["DATASET_NAMES"] -match '\((.+?)\)') {
    $DATASET_NAMES = $matches[1] -split '\s+' | Where-Object { $_ }
}

Write-Host "Syncing code to remote GPU server..."
Write-Host "Remote target: $REMOTE_USER@$GPU_NODE (via $JUMP_HOST)"

# Get remote home directory
Write-Host "Detecting remote home directory..."
$REMOTE_HOME_DIR = & ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J "$REMOTE_USER@$JUMP_HOST" "$REMOTE_USER@$GPU_NODE" "printf %s \`$HOME"

if (-not $REMOTE_HOME_DIR) {
    Write-Error "Failed to detect remote home directory"
    exit 1
}

$REMOTE_HOME_PROJECT = "$REMOTE_HOME_DIR/$REMOTE_PROJECT_DIRNAME"
$REMOTE_CODE_DIR = "$REMOTE_HOME_PROJECT/SmolVLA-Testing"

Write-Host "Remote home: $REMOTE_HOME_DIR"
Write-Host "Remote project: $REMOTE_HOME_PROJECT"

# Create directories on remote
Write-Host "Creating remote directories..."
$CREATE_DIRS_CMD = "mkdir -p '$REMOTE_CODE_DIR' '$REMOTE_SCRATCH_BASE/lerobot_datasets' '$REMOTE_CLEANED_DATASET_ROOT'"
& ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J "$REMOTE_USER@$JUMP_HOST" "$REMOTE_USER@$GPU_NODE" $CREATE_DIRS_CMD

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create remote directories"
    exit 1
}

# Sync code
Write-Host "Syncing code to remote..."
& rsync -avz --progress `
    -e "ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J `"$REMOTE_USER@$JUMP_HOST`"" `
    --exclude '.git' `
    --exclude 'checkpoints' `
    --exclude 'lerobot_datasets' `
    --exclude 'raw_datasets' `
    --exclude 'raw_recordings' `
    --exclude 'cleaned_datasets' `
    --exclude 'train_*.log' `
    --exclude '*_loss.png' `
    "$PROJECT_ROOT/" `
    "$REMOTE_USER@$GPU_NODE`:$REMOTE_CODE_DIR/"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to sync code to remote"
    exit 1
}

# Sync datasets
if ($PREPROCESS_ON_GPU -eq "true") {
    Write-Host "GPU preprocessing enabled: syncing cleaned datasets..."
    foreach ($ds in $DATASET_NAMES) {
        Write-Host "  Syncing cleaned dataset: $ds"
        & rsync -avzP `
            -e "ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J `"$REMOTE_USER@$JUMP_HOST`"" `
            "$LOCAL_CLEANED_DATA_SOURCE/$ds/" `
            "$REMOTE_USER@$GPU_NODE`:$REMOTE_CLEANED_DATASET_ROOT/$ds/"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to sync cleaned dataset: $ds"
            exit 1
        }
    }
}
else {
    Write-Host "GPU preprocessing disabled: syncing converted datasets..."
    foreach ($ds in $DATASET_NAMES) {
        Write-Host "  Syncing dataset: $ds"
        & rsync -avzP `
            -e "ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J `"$REMOTE_USER@$JUMP_HOST`"" `
            "$LOCAL_DATA_PULL_SOURCE/$ds/" `
            "$REMOTE_USER@$GPU_NODE`:$REMOTE_SCRATCH_BASE/lerobot_datasets/$ds/"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to sync dataset: $ds"
            exit 1
        }
    }
}

Write-Host "✓ Sync complete."

# Install Qwen dependencies on remote
Write-Host ""
Write-Host "Installing Qwen dependencies on remote server..."

$install_script = @"
set -e
echo "Activating lerobot environment..."
if [[ -f ~/lerobot/.venv/bin/activate ]]; then
    source ~/lerobot/.venv/bin/activate
elif [[ -f ~/miniconda3/envs/lerobot/bin/activate ]]; then
    source ~/miniconda3/envs/lerobot/bin/activate
else
    echo "Warning: Could not find lerobot virtual environment, proceeding anyway..."
fi

echo "Installing vllm and qwen-vl-utils..."
pip install --upgrade pip
pip install vllm>=0.7 qwen-vl-utils

echo "Verifying installation..."
python3 -c "from vllm import LLM; from qwen_vl_utils import *; print('OK - Qwen dependencies installed successfully')"
"@

& ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J "$REMOTE_USER@$JUMP_HOST" "$REMOTE_USER@$GPU_NODE" bash -c $install_script

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Remote environment ready for Qwen annotation" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install Qwen dependencies" -ForegroundColor Red
    exit 1
}