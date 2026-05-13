#!/usr/bin/env bash
# =============================================================================
# hf_params.sh  —  HuggingFace dataset parameters for HF-based training
# =============================================================================
# Sourced by scripts_local/03_hf_train.sh. Sources params.sh for training
# hyperparameters, then adds HF-specific configuration.
#
# Override in scripts_local/hf_params.local.sh (gitignored).

PARAMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Import base training parameters (model type, batch size, steps, etc.)
# shellcheck source=/dev/null
source "${PARAMS_DIR}/params.sh"

# -------- HF dataset repo IDs --------
# List of HuggingFace dataset repos to download and train on.
# Single repo: trains directly.  Multiple repos: merged before training.
HF_REPO_IDS=(
    "NexusDwin/msd-connector-200-209"
)

# -------- HF cache --------
# Local directory for cached HF dataset downloads.
HF_CACHE_ROOT="${PROJECT_ROOT}/.hf_cache"

# -------- Override run identity for HF runs --------
# RUN_NAME can be overridden here to distinguish HF-sourced runs.
# RUN_NAME="hf_msd_200_209"

# -------- Local override (gitignored) --------
if [[ -f "${PARAMS_DIR}/hf_params.local.sh" ]]; then
    # shellcheck source=/dev/null
    source "${PARAMS_DIR}/hf_params.local.sh"
fi
