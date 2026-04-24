#!/usr/bin/env bash
# =============================================================================
# run_training.sh  —  SmolVLA Fine-Tuning Launcher (UCL TSG GPU Workstation)
# =============================================================================
#
# PURPOSE
# -------
# Launches main.py under nohup so training survives SSH disconnection and
# browser tab closure. Registers a trap that rsync's all outputs from the
# fast scratch disk back to your persistent home directory when the run
# finishes (or crashes).
#
# PREREQUISITE
# ------------
# Run setup_scratch.sh FIRST on each new GPU booking to build the venv.
# This script will abort early if the venv is missing.
#
# UCL SSH / NOHUP PATTERN
# -----------------------
# When you close an SSH terminal or lose connection, the kernel sends SIGHUP
# to the foreground process group, which terminates the training run.
# Running via nohup detaches the process from the terminal's signal group so
# it continues running even after you disconnect.
#
# The output is redirected to training.log on the scratch disk (fast local
# NVMe write) and simultaneously tee'd — you can tail it at any time with:
#   tail -f /scratch0/$USER/smolvla_outputs/training.log
#
# CHECKPOINT RESCUE STRATEGY
# --------------------------
# Scratch is wiped when your booking expires. This script registers a bash
# EXIT trap (fires on both normal completion AND unhandled errors/kills) that
# copies all outputs from scratch back to the persistent NFS home directory.
# You can then pull them to your local machine via rsync at any time:
#
#   rsync -avP username@knuckles.cs.ucl.ac.uk:~/smolvla_project/checkpoints/ \
#         /local/path/smolvla_project/checkpoints/
#
# USAGE
# -----
#   bash ~/smolvla_project/SmolVLA-Testing/run_training.sh 001 002 003
#
#   Pass one or more dataset basenames. All are trained jointly.
#   Defaults to "example" if no argument is given.
#
# To resume from a checkpoint, set RESUME_CHECKPOINT below before running.
# =============================================================================

set -uo pipefail
# NOTE: We intentionally do NOT use 'set -e' here. We want the EXIT trap
# to fire even when the training Python process returns a non-zero exit code
# (e.g. on OOM or a user Ctrl+C), not abort the shell script mid-trap.

# ---------------------------------------------------------------------------
# 0. CONFIGURATION — edit these variables to match your run
# ---------------------------------------------------------------------------

# Dataset name(s) — one or more basenames of converted LeRobotDataset directories.
# Pass multiple to train across all datasets jointly, e.g.:
#   bash run_training.sh 001 002 003
# To resume from the last checkpoint in the output dir:
#   bash run_training.sh --resume 001 002 003
# Defaults to "example" if no argument is given.
RESUME=false
ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--resume" ]]; then
        RESUME=true
    else
        ARGS+=("${arg}")
    fi
done
DATASET_NAMES=("${ARGS[@]:-example}")

# UCL scratch disk — local NVMe, no quota, wiped after booking
SCRATCH_BASE="/scratch0/${USER}"

# Persistent home project root — survives booking expiry
HOME_PROJECT="${HOME}/smolvla_project"

# lerobot sibling repo root (source code, stored in persistent home)
LEROBOT_ROOT="${HOME_PROJECT}/lerobot"

# SmolVLA-Testing repo root — on scratch since there is no usable home quota
SMOLVLA_REPO="${SCRATCH_BASE}/SmolVLA-Testing"

# Dataset root — stored in SCRATCH, not home, because image frames exceed the
# 10GB home quota. Rsynced here at the start of each booking from your local
# machine with:
#   rsync -avz --progress lerobot_datasets/ \
#       -e "ssh -J eredhead@knuckles.cs.ucl.ac.uk" \
#       eredhead@trailbreaker.cs.ucl.ac.uk:/scratch0/eredhead/lerobot_datasets/
SCRATCH_DATASET_ROOT="${SCRATCH_BASE}/lerobot_datasets"

# Join dataset names for use in output directory / job names
DATASET_JOINED=$(IFS="_"; echo "${DATASET_NAMES[*]}")

# All outputs (checkpoints, logs) written here DURING training (fast scratch disk)
SCRATCH_OUTPUT_DIR="${SCRATCH_BASE}/smolvla_outputs/${DATASET_JOINED}_smolvla"

# Where to rescue outputs when training ends or crashes (persistent NFS home)
# These will survive a 72-hour booking expiry and can be rsync'd to your laptop.
RESCUE_DIR="${HOME_PROJECT}/checkpoints"

# uv virtual environment in scratch (created by setup_scratch.sh)
VENV_DIR="${SCRATCH_BASE}/smolvla_venv"

# Cache dirs — must match what setup_scratch.sh configured
CACHE_DIR="${SCRATCH_BASE}/.cache"

# Log file — sits one level above the output dir so we don't need to
# pre-create SCRATCH_OUTPUT_DIR (lerobot creates it; pre-creating it
# triggers a FileExistsError in lerobot's validate() when resume=False).
LOG_FILE="${SCRATCH_BASE}/smolvla_outputs/${DATASET_JOINED}.log"

# SmolVLA base model checkpoint from HuggingFace Hub.
# Once downloaded, HF_HOME cache on scratch stores it — no re-download needed
# until the booking expires.
POLICY_PATH="lerobot/smolvla_base"

# To resume from a specific checkpoint instead of the base model, set this:
#   POLICY_PATH="/scratch0/${USER}/smolvla_outputs/example_smolvla/checkpoints/step_005000"
# or point to a rescued checkpoint in your home dir:
#   POLICY_PATH="${HOME_PROJECT}/checkpoints/example_smolvla/checkpoints/step_005000"

# Training hyperparameters — tune for your dataset and GPU VRAM
BATCH_SIZE=8
STEPS=20000
NUM_WORKERS=4
SAVE_FREQ=500     # checkpoint every N steps
LOG_FREQ=50
SEED=1000

# Set to "--use-amp" to enable Automatic Mixed Precision (recommended for >=A5000)
# Leave empty to disable AMP
AMP_FLAG="--use-amp"

# Experimental: patch lerobot video decoding to keep nearest-frame fallback
# when timestamp tolerance is violated instead of raising FrameTimestampError.
# Set to false to disable and restore strict behavior.
ALLOW_NEAREST_FRAME_FALLBACK=true

# Experimental: patch tokenizer to replace missing task strings with a fallback
# label instead of raising ValueError("Task cannot be None").
ALLOW_MISSING_TASK_FALLBACK=true

# ---------------------------------------------------------------------------
# 1. RE-EXPORT CACHE ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
# These MUST be set before uv/Python loads any HuggingFace or torch code.
# Even if you already ran setup_scratch.sh in this shell session, re-exporting
# here is a safety net — nohup launches a new process environment, and these
# vars must be visible to the training subprocess.

export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export HF_HOME="${CACHE_DIR}/huggingface"
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
export LEROBOT_ROOT="${LEROBOT_ROOT}"

# ---------------------------------------------------------------------------
# 2. PRE-FLIGHT CHECKS
# ---------------------------------------------------------------------------

echo "================================================================="
echo "  SmolVLA Training Launcher — UCL TSG GPU Workstation"
echo "================================================================="
echo "  User        : ${USER}"
echo "  Dataset(s)  : ${DATASET_NAMES[*]}"
echo "  Scratch out : ${SCRATCH_OUTPUT_DIR}"
echo "  Rescue to   : ${RESCUE_DIR}"
echo "  Log file    : ${LOG_FILE}"
echo "  Steps       : ${STEPS}  |  Batch: ${BATCH_SIZE}  |  Seed: ${SEED}"
echo "================================================================="
echo ""

# Abort if the scratch venv is missing — user must run setup_scratch.sh first
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: Scratch venv not found at ${VENV_DIR}"
    echo ""
    echo "Run setup first:"
    echo "  bash ${SMOLVLA_REPO}/setup_scratch.sh"
    exit 1
fi

# Validate all datasets exist
DATASET_PATHS=()
for ds in "${DATASET_NAMES[@]}"; do
    ds_path="${SCRATCH_DATASET_ROOT}/${ds}"
    if [[ ! -d "${ds_path}/meta" ]]; then
        echo "ERROR: LeRobotDataset not found at ${ds_path}"
        echo ""
        echo "The dataset lives in SCRATCH (not home) to avoid the 10GB quota."
        echo "Rsync it from your local machine each new booking:"
        echo ""
        echo "  rsync -avz --progress /local/lerobot_datasets/${ds}/ \\"
        echo "      -e 'ssh -J ${USER}@knuckles.cs.ucl.ac.uk' \\"
        echo "      ${USER}@trailbreaker.cs.ucl.ac.uk:/scratch0/${USER}/lerobot_datasets/${ds}/"
        exit 1
    fi
    DATASET_PATHS+=("${ds_path}")
done

# ---------------------------------------------------------------------------
# 2b. MERGE DATASETS (if more than one)
# ---------------------------------------------------------------------------
# lerobot's LeRobotMultiDataset is not implemented in this version, so we
# merge at the file level with merge_datasets.py before training.

if [[ "${#DATASET_NAMES[@]}" -gt 1 ]]; then
    MERGED_DATASET_ROOT="${SCRATCH_DATASET_ROOT}/merged_${DATASET_JOINED}"
    echo "Multiple datasets detected — merging into ${MERGED_DATASET_ROOT} ..."

    source "${SCRATCH_BASE}/activate_smolvla.sh"

    cd "${LEROBOT_ROOT}" && uv run python "${SMOLVLA_REPO}/merge_datasets.py" \
        "${DATASET_PATHS[@]}" \
        --output "${MERGED_DATASET_ROOT}" \
        --force

    if [[ $? -ne 0 ]]; then
        echo "ERROR: merge_datasets.py failed. Aborting."
        exit 1
    fi

    TRAIN_DATASET_ROOT="${MERGED_DATASET_ROOT}"
    echo "Merge complete. Training on: ${TRAIN_DATASET_ROOT}"
else
    TRAIN_DATASET_ROOT="${DATASET_PATHS[0]}"
fi

# Handle existing output directory
if [[ -d "${SCRATCH_OUTPUT_DIR}" ]]; then
    HAS_CHECKPOINTS=false
    if [[ -d "${SCRATCH_OUTPUT_DIR}/checkpoints" ]] && \
       [[ -n "$(ls -A "${SCRATCH_OUTPUT_DIR}/checkpoints" 2>/dev/null)" ]]; then
        HAS_CHECKPOINTS=true
    fi

    if [[ "${RESUME}" == "true" ]]; then
        if [[ "${HAS_CHECKPOINTS}" == "false" ]]; then
            echo "WARNING: --resume passed but no checkpoints found in ${SCRATCH_OUTPUT_DIR}."
            echo "Starting fresh."
        else
            echo "Resuming from last checkpoint in ${SCRATCH_OUTPUT_DIR}/checkpoints/"
        fi
    elif [[ "${HAS_CHECKPOINTS}" == "true" ]]; then
        echo "ERROR: Output directory already contains checkpoints:"
        echo "  ${SCRATCH_OUTPUT_DIR}/checkpoints/"
        echo ""
        echo "To resume training from the last checkpoint, run:"
        echo "  bash ${SMOLVLA_REPO}/run_training.sh --resume ${DATASET_NAMES[*]}"
        exit 1
    else
        echo "Removing stale output directory (no checkpoints found)..."
        rm -rf "${SCRATCH_OUTPUT_DIR}"
    fi
fi

mkdir -p "${SCRATCH_BASE}/smolvla_outputs"  # parent only — lerobot creates the output subdir

# Ensure the persistent rescue directory exists in home
mkdir -p "${RESCUE_DIR}"

if [[ "${ALLOW_NEAREST_FRAME_FALLBACK}" == "true" ]]; then
    echo "Applying nearest-frame fallback patch (experimental)..."
    if ! python "${SMOLVLA_REPO}/patch_frame_tolerance.py"; then
        echo "ERROR: Failed to patch lerobot timestamp tolerance behavior."
        echo "Set ALLOW_NEAREST_FRAME_FALLBACK=false to continue with strict decoding."
        exit 1
    fi

    PATCH_TARGET="${LEROBOT_ROOT}/src/lerobot/datasets/video_utils.py"
    if ! grep -q "SmolVLA-Testing patch: tolerate timestamp violations by using nearest frame." "${PATCH_TARGET}"; then
        echo "ERROR: Nearest-frame fallback marker not found in ${PATCH_TARGET}"
        echo "Aborting to avoid running with strict timestamp checks by accident."
        exit 1
    fi
    echo "Nearest-frame fallback patch verified in ${PATCH_TARGET}"
fi

if [[ "${ALLOW_MISSING_TASK_FALLBACK}" == "true" ]]; then
    echo "Applying missing-task fallback patch (experimental)..."
    if ! python "${SMOLVLA_REPO}/patch_task_none.py"; then
        echo "ERROR: Failed to patch tokenizer missing-task behavior."
        echo "Set ALLOW_MISSING_TASK_FALLBACK=false to continue with strict task checks."
        exit 1
    fi

    TASK_PATCH_TARGET="${LEROBOT_ROOT}/src/lerobot/processor/tokenizer_processor.py"
    if ! grep -q "SmolVLA-Testing patch: allow missing task with fallback label." "${TASK_PATCH_TARGET}"; then
        echo "ERROR: Missing-task fallback marker not found in ${TASK_PATCH_TARGET}"
        echo "Aborting to avoid running with strict task checks by accident."
        exit 1
    fi
    echo "Missing-task fallback patch verified in ${TASK_PATCH_TARGET}"
fi

# Validate that the lerobot checkout has the symbols used by main.py.
# This catches version/API mismatches early with a clear error instead of an
# immediate nohup exit code 1 with no output directory.
echo "Checking lerobot training imports..."
cd "${LEROBOT_ROOT}" && uv run python - <<'PY'
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_train import train

print("lerobot import check passed")
PY

if [[ $? -ne 0 ]]; then
    echo "ERROR: lerobot import preflight failed."
    echo "Your lerobot checkout or environment is likely out-of-sync with main.py."
    echo "Fix by updating lerobot and rebuilding the scratch environment:"
    echo "  cd ${LEROBOT_ROOT} && git pull"
    echo "  bash ${SMOLVLA_REPO}/setup_scratch.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. CHECKPOINT RESCUE TRAP
# ---------------------------------------------------------------------------
# bash's EXIT pseudo-signal fires whenever this script exits — whether that is
# because:
#   (a) training completed normally           (exit 0)
#   (b) training crashed with a Python error  (exit != 0)
#   (c) SIGTERM was sent (e.g. booking expired)
#
# The trap runs rsync to copy all outputs from the volatile scratch disk to
# your persistent home directory BEFORE the booking daemon wipes /scratch0/.
#
# NOTE: If the machine is hard-killed (SIGKILL, power loss) the trap cannot
# fire. Enable periodic mid-run sync by setting SYNC_FREQ below.

rescue_checkpoints() {
    local exit_code="${1:-$?}"
    echo ""
    echo "================================================================="
    echo "  EXIT TRAP FIRED (exit code: ${exit_code})"
    echo "  Rescuing outputs from scratch to persistent home..."
    echo "================================================================="

    if [[ -d "${SCRATCH_OUTPUT_DIR}" ]]; then
        # rsync flags:
        #   -a  : archive mode (preserves timestamps, permissions, symlinks)
        #   -v  : verbose — log what is being copied
        #   -z  : compress data in transit (NFS write is slower than scratch read)
        #   --partial : keep partially-transferred files on interrupt (rsync-safe resume)
        rsync -avz --partial \
            "${SCRATCH_OUTPUT_DIR}/" \
            "${RESCUE_DIR}/${DATASET_JOINED}_smolvla/" \
            2>&1 | tee -a "${LOG_FILE}"

        local rsync_status=$?
        if [[ ${rsync_status} -eq 0 ]]; then
            echo ""
            echo "  Rescue complete. Checkpoints are safe at:"
            echo "    ${RESCUE_DIR}/${DATASET_JOINED}_smolvla/"
            echo ""
            echo "  Pull them to your local machine with:"
            echo "    rsync -avP ${USER}@knuckles.cs.ucl.ac.uk:${RESCUE_DIR}/ \\"
            echo "        /your/local/checkpoints/"
        else
            echo ""
            echo "  WARNING: rsync exited with code ${rsync_status}."
            echo "  Some files may not have been rescued. Check disk space with:"
            echo "    df -h ~/"
        fi
    else
        echo "  WARNING: Output directory ${SCRATCH_OUTPUT_DIR} not found."
        echo "  Training may have failed before writing any outputs."
    fi

    echo "================================================================="
}

# Register the rescue function — runs on any exit from this script
trap rescue_checkpoints EXIT

# ---------------------------------------------------------------------------
# 4. PERIODIC MID-RUN SYNC
# ---------------------------------------------------------------------------
# Syncs checkpoints from scratch to persistent home every SYNC_INTERVAL_SECONDS.
# This is the safety net against SIGKILL (hard node failure, booking daemon),
# where the EXIT trap below cannot fire. Without this, a hard kill leaves all
# checkpoints on scratch which will be wiped with the booking.

SYNC_INTERVAL_SECONDS=1800  # sync every 30 minutes

(
  while true; do
    sleep "${SYNC_INTERVAL_SECONDS}"
    rsync -az --partial "${SCRATCH_OUTPUT_DIR}/" "${RESCUE_DIR}/${DATASET_JOINED}_smolvla/"
    echo "$(date '+%Y-%m-%d %H:%M:%S') | periodic sync to ${RESCUE_DIR}/${DATASET_JOINED}_smolvla/ complete" >> "${LOG_FILE}"
  done
) &
SYNC_PID=$!

# Override the EXIT trap to also kill the sync loop on clean exit.
# Preserve the script's real exit code; otherwise the preceding `kill` can
# overwrite `$?` and make failures look like success in the rescue log.
trap 'exit_code=$?; kill "${SYNC_PID}" 2>/dev/null || true; rescue_checkpoints "${exit_code}"' EXIT

# ---------------------------------------------------------------------------
# 5. BUILD THE TRAINING COMMAND
# ---------------------------------------------------------------------------
# We call 'uv run' from the lerobot project directory so uv uses the correct
# pyproject.toml/uv.lock. UV_PROJECT_ENVIRONMENT redirects the venv to scratch.
#
# main.py will:
#   1. Detect LEROBOT_ROOT and add lerobot/src to sys.path
#   2. Load the dataset from DATASET_ROOT (persistent home)
#   3. Download SmolVLA base weights to HF_HOME (scratch cache)
#   4. Write checkpoints and logs to SCRATCH_OUTPUT_DIR (fast scratch NVMe)

RESUME_FLAG=""
[[ "${RESUME}" == "true" ]] && RESUME_FLAG="--resume"

TRAIN_CMD="cd ${LEROBOT_ROOT} && uv run python ${SMOLVLA_REPO}/main.py \
    --dataset-root ${TRAIN_DATASET_ROOT} \
    --lerobot-root ${LEROBOT_ROOT} \
    --policy-path ${POLICY_PATH} \
    --output-dir ${SCRATCH_OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --steps ${STEPS} \
    --num-workers ${NUM_WORKERS} \
    --save-freq ${SAVE_FREQ} \
    --log-freq ${LOG_FREQ} \
    --seed ${SEED} \
    --device cuda \
    --eval-freq 0 \
    ${RESUME_FLAG} \
    ${AMP_FLAG}"

echo "Training command:"
echo "  ${TRAIN_CMD}" | fold -s -w 78
echo ""

# ---------------------------------------------------------------------------
# 6. LAUNCH WITH NOHUP
# ---------------------------------------------------------------------------
# nohup immunises the process against SIGHUP (the signal sent when an SSH
# connection drops or you close a browser tab on Apache Guacamole).
#
# The 2>&1 at the end merges stderr into stdout so ALL output (Python
# tracebacks, tqdm progress bars, loss values) goes into LOG_FILE.
#
# The trailing & backgrounds the process so this script can continue
# monitoring and the EXIT trap can register properly.
#
# CRITICAL: Do NOT "Log Out" from the Guacamole desktop — that sends SIGTERM
# to your entire session. Instead, simply close the browser tab.

echo "Launching training under nohup..."
echo "Log file: ${LOG_FILE}"
echo ""

nohup bash -c "
    # Source the activation shim — this sets PATH (including uv), all cache
    # dirs, and UV_PROJECT_ENVIRONMENT in one shot. The nohup subshell starts
    # with a clean environment and will not find uv otherwise.
    source '${SCRATCH_BASE}/activate_smolvla.sh'

    ${TRAIN_CMD}
" >> "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!

echo "================================================================="
echo "  Training launched."
echo "  PID : ${TRAIN_PID}"
echo "================================================================="
echo ""
echo "  Monitor live output:"
echo "    tail -f ${LOG_FILE}"
echo ""
echo "  Check GPU utilisation:"
echo "    watch -n 2 nvidia-smi"
echo ""
echo "  Check process is still running:"
echo "    ps aux | grep main.py"
echo "    # or:"
echo "    kill -0 ${TRAIN_PID} && echo 'still running'"
echo ""
echo "  SAFE TO DISCONNECT — close the SSH terminal or browser tab."
echo "  Do NOT click 'Log Out' in the Guacamole desktop menu."
echo ""
echo "  When the run finishes, checkpoints will be rescued to:"
echo "    ${RESCUE_DIR}/${DATASET_JOINED}_smolvla/"
echo "================================================================="

# ---------------------------------------------------------------------------
# 7. WAIT FOR THE BACKGROUND PROCESS
# ---------------------------------------------------------------------------
# 'wait' blocks this shell script until the nohup'd training job completes.
# This is what ensures the EXIT trap (and therefore the rsync rescue) fires
# only AFTER training is truly done — not immediately after the '&' above.
#
# If you disconnect and reconnect later, the nohup'd process runs on without
# this wait. The rescue trap will have already fired when the booking daemon
# kills this parent shell — at which point the nohup subprocess continues.
# Re-attach to the log with:  tail -f <LOG_FILE>

wait ${TRAIN_PID}
FINAL_EXIT=$?

echo ""
echo "Training process exited with code: ${FINAL_EXIT}"

if [[ ${FINAL_EXIT} -ne 0 ]]; then
    echo ""
    echo "================================================================="
    echo "  Training failed (exit ${FINAL_EXIT})."
    echo "  Extracting latest traceback/error block from ${LOG_FILE}:"
    echo "================================================================="
    if [[ -f "${LOG_FILE}" ]]; then
        python - "${LOG_FILE}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
lines = log_path.read_text(errors="replace").splitlines()

# Prefer the last traceback block (until next blank separator) if present.
tb_start = None
for i, line in enumerate(lines):
    if "Traceback (most recent call last):" in line:
        tb_start = i

if tb_start is not None:
    end = len(lines)
    for j in range(tb_start + 1, len(lines)):
        if lines[j].startswith("================================================================="):
            end = j
            break
    for line in lines[tb_start:end]:
        print(line)
    raise SystemExit(0)

# Fallback: print last 40 likely error lines.
err_pat = re.compile(r"(error|exception|traceback|killed|oom|cuda)", re.IGNORECASE)
err_lines = [line for line in lines if err_pat.search(line)]
if err_lines:
    for line in err_lines[-40:]:
        print(line)
else:
    # Final fallback if no explicit error markers are found.
    for line in lines[-200:]:
        print(line)
PY
        echo ""
        echo "-----------------------------------------------------------------"
        echo "  Last 120 lines of raw log (context):"
        echo "-----------------------------------------------------------------"
        tail -n 120 "${LOG_FILE}"
    else
        echo "  Log file not found."
    fi
    echo "================================================================="
fi

# EXIT trap (rescue_checkpoints) fires here automatically as the script exits.
exit ${FINAL_EXIT}
