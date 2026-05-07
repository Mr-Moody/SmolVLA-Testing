# RUN ON: LOCAL MACHINE (Windows)
# Preflight check: verifies local prerequisites, SSH reachability, and remote layout
# before launching training on the GPU node.

$ErrorActionPreference = 'Stop'
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# ---------------------------------------------------------------------------
# Load run parameters
# Expects 00_run_params.ps1 (sibling file) to define the following variables:
#   $REMOTE_USER, $GPU_NODE, $JUMP_HOST, $SSH_KEY_FILE
#   $REMOTE_PROJECT_DIRNAME, $REMOTE_SCRATCH_BASE
#   $DATASET_ROOT, $REMOTE_CLEANED_DATASET_ROOT
#   $LOCAL_PROJECT_ROOT, $LOCAL_CLEANED_DATA_SOURCE, $LOCAL_DATA_PULL_SOURCE
#   $PREPROCESS_ON_GPU            ('true'/'false' string, kept for parity)
#   $RUN_NAME
#   $DATASET_NAMES                (PowerShell array, e.g. @('ds1','ds2'))
# ---------------------------------------------------------------------------
. "$SCRIPT_DIR/00_run_params.ps1"

$CHECK_TIMEOUT = 12
$script:FAILED = 0

# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
function Pass { param([string]$Msg) Write-Host "[PASS] $Msg" -ForegroundColor Green }
function Warn { param([string]$Msg) Write-Host "[WARN] $Msg" -ForegroundColor Yellow }
function Fail {
    param([string]$Msg)
    Write-Host "[FAIL] $Msg" -ForegroundColor Red
    $script:FAILED = 1
}

function Test-LocalDir {
    param([string]$Path, [string]$Label)
    if (Test-Path -LiteralPath $Path -PathType Container) {
        Pass "${Label}: $Path"
    } else {
        Fail "$Label missing: $Path"
    }
}

# ---------------------------------------------------------------------------
# SSH helpers (mirrors style of the sync script)
# ---------------------------------------------------------------------------
$SshRemote = "${REMOTE_USER}@${GPU_NODE}"
$SshJump   = "${REMOTE_USER}@${JUMP_HOST}"
$SshOpts   = @(
    '-i', $SSH_KEY_FILE,
    '-o', 'IdentitiesOnly=yes',
    '-o', 'IdentityAgent=none',
    '-o', "ConnectTimeout=$CHECK_TIMEOUT",
    '-J', $SshJump
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
Write-Host '=== SmolVLA preflight check ==='
Write-Host "Run name      : $RUN_NAME"
Write-Host "Remote target : $SshRemote (jump $SshJump)"
Write-Host ''

# ---------------------------------------------------------------------------
# [1/4] Local prerequisites
# ---------------------------------------------------------------------------
Write-Host '[1/4] Local prerequisites'
Test-LocalDir $LOCAL_PROJECT_ROOT 'Local project root'

if ($PREPROCESS_ON_GPU -eq 'true') {
    Test-LocalDir $LOCAL_CLEANED_DATA_SOURCE 'Local cleaned dataset root'
} else {
    Test-LocalDir $LOCAL_DATA_PULL_SOURCE 'Local converted dataset root'
}

foreach ($ds in $DATASET_NAMES) {
    if ($PREPROCESS_ON_GPU -eq 'true') {
        Test-LocalDir "$LOCAL_CLEANED_DATA_SOURCE/$ds" "Local cleaned dataset $ds"
    } else {
        Test-LocalDir "$LOCAL_DATA_PULL_SOURCE/$ds" "Local dataset $ds"
    }
}

# ---------------------------------------------------------------------------
# [2/4] SSH reachability
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '[2/4] SSH reachability'

$REMOTE_HOME_DIR = ''
$homeOutput = & ssh @SshOpts $SshRemote 'printf %s "$HOME"' 2>$null
if ($LASTEXITCODE -eq 0 -and $homeOutput) {
    $REMOTE_HOME_DIR = $homeOutput.Trim()
    Pass "SSH/jump connectivity to $GPU_NODE"
    Pass "Remote home detected: $REMOTE_HOME_DIR"
} else {
    Fail "Cannot connect to $GPU_NODE via jump host $JUMP_HOST"
    Write-Host '      Verify booking is active and credentials/keys are valid.'
}

# ---------------------------------------------------------------------------
# [3/4] + [4/4] Remote layout & training readiness
# Combined into one SSH call (matches the original bash heredoc).
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '[3/4] Remote layout'

if ($REMOTE_HOME_DIR) {
    $REMOTE_HOME_PROJECT = "$REMOTE_HOME_DIR/$REMOTE_PROJECT_DIRNAME"
    $REMOTE_LEROBOT_DIR  = "$REMOTE_HOME_PROJECT/lerobot"
    $REMOTE_CODE_DIR     = "$REMOTE_HOME_PROJECT/SmolVLA-Testing"
    $DATASET_NAMES_STR   = ($DATASET_NAMES -join ' ')

    # Remote bash payload — read positional args from $1..$8 (passed after `bash -s --`).
    # Piped over stdin so we don't have to wrestle with PowerShell heredoc quoting.
    $remoteScript = @'
set -u
status=0
home_project="$1"
lerobot="$2"
code="$3"
scratch="$4"
data_root="$5"
cleaned_root="$6"
preprocess_on_gpu="$7"
datasets="$8"

check_dir() {
  label="$1"
  path="$2"
  if [[ -d "$path" ]]; then
    echo "[PASS] $label: $path"
  else
    echo "[FAIL] $label missing on remote: $path"
    status=1
  fi
}

check_file() {
  label="$1"
  path="$2"
  if [[ -f "$path" ]]; then
    echo "[PASS] $label: $path"
  else
    echo "[WARN] $label missing on remote: $path"
  fi
}

check_dir "Remote home project" "$home_project"
check_dir "Remote lerobot" "$lerobot"
check_dir "Remote SmolVLA-Testing code" "$code"
check_dir "Remote scratch base" "$scratch"
if [[ "$preprocess_on_gpu" == "true" ]]; then
  check_dir "Remote cleaned dataset root" "$cleaned_root"
  for ds in $datasets; do
    check_dir "Remote cleaned dataset $ds" "$cleaned_root/$ds"
  done
else
  check_dir "Remote dataset root" "$data_root"
  for ds in $datasets; do
    check_dir "Remote dataset $ds" "$data_root/$ds/meta"
  done
fi

echo "[4/4] Training readiness"
check_file "Scratch activation shim exists (setup complete)" "$scratch/activate_smolvla.sh"
if [[ ! -f "$scratch/activate_smolvla.sh" ]]; then
  echo "[WARN] This is expected before first setup; run scripts/03_setup_gpu.sh on the GPU node."
fi

if [[ -w "$scratch" ]]; then
  echo "[PASS] Scratch is writable: $scratch"
else
  echo "[FAIL] Scratch is not writable: $scratch"
  status=1
fi

exit "$status"
'@

    # Normalise to LF — bash chokes on CRLF
    $remoteScript = $remoteScript -replace "`r`n", "`n"

    # Pipe script over stdin; positional args follow `bash -s --`.
    # We DO NOT want PowerShell to throw on a non-zero exit here — a FAIL is data,
    # not an error. Capture output and inspect it.
    $remoteOutput = $remoteScript | & ssh @SshOpts $SshRemote `
        'bash' '-s' '--' `
        $REMOTE_HOME_PROJECT `
        $REMOTE_LEROBOT_DIR `
        $REMOTE_CODE_DIR `
        $REMOTE_SCRATCH_BASE `
        $DATASET_ROOT `
        $REMOTE_CLEANED_DATASET_ROOT `
        $PREPROCESS_ON_GPU `
        $DATASET_NAMES_STR 2>&1

    # Print remote output with colour-coded tags
    foreach ($line in $remoteOutput) {
        $text = [string]$line
        if     ($text -match '^\[PASS\]') { Write-Host $text -ForegroundColor Green }
        elseif ($text -match '^\[FAIL\]') { Write-Host $text -ForegroundColor Red }
        elseif ($text -match '^\[WARN\]') { Write-Host $text -ForegroundColor Yellow }
        else                              { Write-Host $text }
    }

    if ($remoteOutput -match '\[FAIL\]') {
        $script:FAILED = 1
    }
} else {
    $script:FAILED = 1
    Write-Host '      Skipping remote layout checks (no SSH connection).'
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ''
if ($script:FAILED -eq 0) {
    Write-Host 'Preflight result: PASS' -ForegroundColor Green
    Write-Host 'You can proceed with:'
    Write-Host "  1) ssh -l $REMOTE_USER -J $REMOTE_USER@$JUMP_HOST $GPU_NODE"
    Write-Host '  2) cd ~/smolvla_project/SmolVLA-Testing/scripts && bash ./04_start_training.sh'
    exit 0
}

Write-Host 'Preflight result: FAIL' -ForegroundColor Red
Write-Host 'Fix the failed checks above before launching training.'
exit 1