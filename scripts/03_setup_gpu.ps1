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

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Setup failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# Install Qwen dependencies
Write-Host ""
Write-Host "Installing Qwen dependencies on remote server..."

$install_script = @'
set -e
echo "Activating lerobot environment..."
if [[ -f ~/lerobot/.venv/bin/activate ]]; then
    source ~/lerobot/.venv/bin/activate
fi

echo "Installing vllm and qwen-vl-utils..."
pip install --upgrade pip
pip install vllm>=0.7 qwen-vl-utils

echo "Verifying installation..."
python3 -c "from vllm import LLM; from qwen_vl_utils import *; print('✓ Qwen dependencies installed successfully')"
'@

& ssh -i $SSH_KEY_FILE -o IdentitiesOnly=yes -o IdentityAgent=none -J "$REMOTE_USER@$JUMP_HOST" "$REMOTE_USER@$GPU_NODE" bash << "EOF"
$install_script
EOF

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Remote environment ready for Qwen annotation" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install Qwen dependencies" -ForegroundColor Red
    exit 1
}
