# RUN ON: LOCAL MACHINE (Windows)
# Set up remote GPU environment

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

Write-Host "Running GPU environment setup..."
Write-Host "Remote: $REMOTE_USER@$GPU_NODE"

# Run setup script on remote
Write-Host "Executing setup_scratch.sh on remote..."
& ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J "$REMOTE_USER@$JUMP_HOST" "$REMOTE_USER@$GPU_NODE" `
    "bash ~/smolvla_project/SmolVLA-Testing/scripts/setup_scratch.sh"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Setup complete." -ForegroundColor Green
}
else {
    Write-Host "✗ Setup failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
