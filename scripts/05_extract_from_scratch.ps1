# RUN ON: LOCAL MACHINE (Windows)
# Extract training results from remote GPU server

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
$REMOTE_SCRATCH_BASE = $config["REMOTE_SCRATCH_BASE"]
$RUN_NAME = $config["RUN_NAME"]
$MODEL_TYPE = $config["MODEL_TYPE"]
$LOCAL_CHECKPOINTS_ROOT = if ($config["LOCAL_CHECKPOINTS_ROOT"]) { $config["LOCAL_CHECKPOINTS_ROOT"] } else { "$PROJECT_ROOT/checkpoints" }
$EXTRACT_FOLDER_NAME = $config["EXTRACT_FOLDER_NAME"]

$RUN_TAG = "${RUN_NAME}_${MODEL_TYPE}"
$REMOTE_RUN_DIR = "$REMOTE_SCRATCH_BASE/smolvla_outputs/$RUN_TAG"
$REMOTE_LOG_FILE = "$REMOTE_SCRATCH_BASE/smolvla_outputs/$RUN_TAG.log"
$LOCAL_DEST = "$LOCAL_CHECKPOINTS_ROOT/$EXTRACT_FOLDER_NAME"

Write-Host "Extracting results from remote GPU server..."
Write-Host "Remote: $REMOTE_USER@$GPU_NODE"
Write-Host "Source: $REMOTE_RUN_DIR"
Write-Host "Destination: $LOCAL_DEST"

# Check if destination exists and has content
if ((Test-Path $LOCAL_DEST) -and ((Get-ChildItem $LOCAL_DEST -Force | Measure-Object).Count -gt 0)) {
    Write-Host "ERROR: Extraction target already exists and is not empty: $LOCAL_DEST" -ForegroundColor Red
    Write-Host "Refusing to overwrite. Change EXTRACT_FOLDER_NAME in scripts/00_run_params.sh."
    exit 1
}

# Create local destination
New-Item -ItemType Directory -Path "$LOCAL_DEST/logs" -Force | Out-Null

# Check remote run directory exists
Write-Host "Checking remote run directory..."
& ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J "$REMOTE_USER@$JUMP_HOST" "$REMOTE_USER@$GPU_NODE" `
    "test -d '$REMOTE_RUN_DIR'" | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Remote directory not found: $REMOTE_RUN_DIR" -ForegroundColor Red
    exit 1
}

# Pull checkpoints
Write-Host "Pulling checkpoints from remote..."
& rsync -avP --partial --inplace --timeout=120 `
    -e "ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -J $REMOTE_USER@$JUMP_HOST" `
    "$REMOTE_USER@$GPU_NODE`:$REMOTE_RUN_DIR/" `
    "$LOCAL_DEST/"

# Pull launcher log
Write-Host "Pulling launcher log..."
& rsync -avP `
    -e "ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J $REMOTE_USER@$JUMP_HOST" `
    "$REMOTE_USER@$GPU_NODE`:$REMOTE_LOG_FILE" `
    "$LOCAL_DEST/logs/" 2>$null | Out-Null

Write-Host "✓ Extraction complete: $LOCAL_DEST" -ForegroundColor Green
