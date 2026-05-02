#!/usr/bin/env bash
# Overnight annotation pipeline parameters
# Edit these settings for your overnight run

# ========================================
# Dataset Configuration
# ========================================

# List of dataset names to process (space-separated)
DATASET_NAMES="001 002 003"

# Root directories
RAW_DATASETS_ROOT="raw_datasets"
CLEANED_DATASETS_ROOT="cleaned_datasets"
LEROBOT_DATASETS_ROOT="lerobot_datasets"

# Remote server configuration (if running on remote GPU server)
REMOTE_EXECUTION=false                    # true = run on remote server; false = run locally
REMOTE_USER="xparker"                     # SSH user
GPU_NODE="bluestreak.cs.ucl.ac.uk"        # GPU server hostname
REMOTE_PROJECT_ROOT="/home/xparker/smolvla_project"  # Path on remote server
SSH_KEY_FILE="${HOME}/.ssh/id_ed25519"    # SSH key for authentication

# Local paths (used only if REMOTE_EXECUTION=false)
LOCAL_PROJECT_ROOT="."                    # Current directory

# If executing remotely, paths relative to REMOTE_PROJECT_ROOT
# If executing locally, paths relative to LOCAL_PROJECT_ROOT

# Output directories
OVERNIGHT_OUTPUT_DIR="overnight_output"
LOG_DIR="${OVERNIGHT_OUTPUT_DIR}/logs"

# ========================================
# Qwen3-VL Configuration
# ========================================

# Whether to annotate with Qwen (disable to just clean + convert)
ENABLE_ANNOTATION=true

# Qwen model ID (leave empty for auto-detect AWQ or base)
QWEN_MODEL=""

# Number of GPUs for tensor parallelism
NUM_GPUS=1

# Batch size for annotation
BATCH_SIZE_ANNOTATION=4

# ========================================
# Processing Controls
# ========================================

# Max episodes per dataset (None = all)
MAX_EPISODES_PER_DATASET=""

# Cleaning parameters
CAMERA_TOLERANCE_MS=150.0
JOINT_MOTION_THRESHOLD=5e-4
GRIPPER_MOTION_THRESHOLD=2e-4

# Primary camera for video extraction
PRIMARY_CAMERA="ee_zed_m"

# ========================================
# Runtime Controls
# ========================================

# Stop on first error or continue (true = continue)
CONTINUE_ON_ERROR=false

# Enable checkpoint/resume (true = resume from last checkpoint)
ENABLE_CHECKPOINT=true
CHECKPOINT_FILE="${OVERNIGHT_OUTPUT_DIR}/checkpoint.json"

# Monitor script interval (seconds)
MONITOR_INTERVAL=60

# ========================================
# Housekeeping
# ========================================

# Send completion email (leave empty to disable)
COMPLETION_EMAIL=""

# Keep temporary files after completion
KEEP_TEMP_FILES=false

# Verbosity level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL="INFO"

# ========================================
# Email Configuration (optional)
# ========================================

SMTP_SERVER="smtp.gmail.com"
SMTP_PORT="587"
