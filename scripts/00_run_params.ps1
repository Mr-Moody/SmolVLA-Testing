# Shared workflow parameters for sync, training, and extraction.
# Dot-source this from other scripts:  . "$SCRIPT_DIR/00_run_params.ps1"
# No need to run it directly

$PARAMS_DIR                 = Split-Path -Parent $MyInvocation.MyCommand.Path
$DEFAULT_LOCAL_PROJECT_ROOT = (Resolve-Path (Join-Path $PARAMS_DIR '..')).Path

# -------- Workflow identity --------
# Set this explicitly for UCL remote paths/SSH, instead of relying on $env:USERNAME.
$WORKFLOW_USER = 'xparker'

# -------- Project layout knobs --------
# Remote project root is discovered from remote $HOME, then this dirname is used.
$REMOTE_PROJECT_DIRNAME = 'smolvla_project'

# -------- Core run controls --------
$MODEL_TYPE    = 'act'   # smolvla | act
$RUN_NAME      = 'qwen'
$DATASET_NAMES = @('201_1')
$SAVE_FREQ     = 1000

# -------- Preprocessing controls --------
# If 'true', convert cleaned_datasets -> lerobot_datasets on GPU right before training.
$PREPROCESS_ON_GPU         = 'false'
$LOCAL_CLEANED_DATA_SOURCE = Join-Path $DEFAULT_LOCAL_PROJECT_ROOT 'cleaned_datasets'
$PREPROCESS_VCODEC         = 'h264_nvenc'
# Primary camera for the convert step; mapped to observation.images.top in LeRobotDataset.
$PRIMARY_CAMERA            = 'ee_zed_m_left'
# Default task description written into annotations.jsonl for every episode.
$TASK_DESCRIPTION          = 'Looking at objects'

# -------- Training hyperparameters --------
# STEPS is intentionally tiny — this run is a pipeline/data-feed validation only.
$STEPS       = 50
$BATCH_SIZE  = 4
$NUM_WORKERS = 4
$LOG_FREQ    = 50
$SEED        = 1000
$USE_AMP     = 'true'
$RESUME      = 'false'

# -------- Model-specific knobs --------
$SMOLVLA_POLICY_PATH  = 'lerobot/smolvla_base'
$ACT_POLICY_PATH      = 'lerobot/act_base'
$ACT_CHUNK_SIZE       = 100
$ACT_N_OBS_STEPS      = 1
$ACT_USE_VAE          = 'true'
$ACT_VISION_BACKBONE  = 'resnet18'

# -------- Optional runtime patches --------
$ALLOW_NEAREST_FRAME_FALLBACK = 'true'
$ALLOW_MISSING_TASK_FALLBACK  = 'true'

# -------- SSH topology --------
$GPU_NODE  = 'prowl.cs.ucl.ac.uk'
$JUMP_HOST = 'knuckles.cs.ucl.ac.uk'
# SSH key for passwordless access. Generate once with:
#   ssh-keygen -t ed25519 -f $HOME/.ssh/ucl_key -N '""'
# Then install on the GPU node from an active session:
#   Get-Content $HOME/.ssh/ucl_key.pub | ssh ... "cat >> ~/.ssh/authorized_keys"
$SSH_KEY_FILE = Join-Path $HOME '.ssh/ucl_key'

# -------- Local paths --------
$LOCAL_PROJECT_ROOT     = $DEFAULT_LOCAL_PROJECT_ROOT
$LOCAL_DATA_PULL_SOURCE = Join-Path $LOCAL_PROJECT_ROOT 'cleaned_datasets'
$LOCAL_CHECKPOINTS_ROOT = Join-Path $LOCAL_PROJECT_ROOT 'checkpoints'

# ---------------------------------------------------------------------------
# Local override (gitignored)
# Create 00_run_params.local.ps1 alongside this file for personal paths/hosts.
# It is dot-sourced before deriving WORKFLOW_USER-dependent paths, so overriding
# $WORKFLOW_USER there will correctly propagate to remote paths below.
# (NOTE: this differs from the bash version, where overriding WORKFLOW_USER in
#  the local file would NOT propagate. Treating that as a latent bug; fixing here.)
# ---------------------------------------------------------------------------
$localOverride = Join-Path $PARAMS_DIR '00_run_params.local.ps1'
if (Test-Path -LiteralPath $localOverride) {
    . $localOverride
}

# -------- Derived values (computed AFTER local override) --------
$REMOTE_USER                 = $WORKFLOW_USER
$REMOTE_SCRATCH_BASE         = "/scratch0/$WORKFLOW_USER"
$DATASET_ROOT                = "/scratch0/$WORKFLOW_USER/cleaned_datasets"
$REMOTE_CLEANED_DATASET_ROOT = "/scratch0/$WORKFLOW_USER/cleaned_datasets"
$EXTRACT_FOLDER_NAME         = "${RUN_NAME}_${MODEL_TYPE}_full"