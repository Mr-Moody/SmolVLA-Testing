#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/00_run_params.sh"

SSH_REMOTE="${REMOTE_USER}@${GPU_NODE}"
SSH_JUMP="${REMOTE_USER}@${JUMP_HOST}"
CHECK_TIMEOUT=12

FAILED=0

pass() {
  echo "[PASS] $*"
}

fail() {
  echo "[FAIL] $*"
  FAILED=1
}

require_local_dir() {
  local p="$1"
  local label="$2"
  if [[ -d "${p}" ]]; then
    pass "${label}: ${p}"
  else
    fail "${label} missing: ${p}"
  fi
}

echo "=== SmolVLA preflight check ==="
echo "Run name      : ${RUN_NAME}"
echo "Remote target : ${SSH_REMOTE} (jump ${SSH_JUMP})"
echo ""

echo "[1/4] Local prerequisites"
require_local_dir "${LOCAL_PROJECT_ROOT}" "Local project root"
require_local_dir "${LOCAL_DATA_PULL_SOURCE}" "Local converted dataset root"
for ds in "${DATASET_NAMES[@]}"; do
  require_local_dir "${LOCAL_DATA_PULL_SOURCE}/${ds}" "Local dataset ${ds}"
done

echo ""
echo "[2/4] SSH reachability"
REMOTE_HOME_DIR=""
if REMOTE_HOME_DIR="$(ssh -o ConnectTimeout="${CHECK_TIMEOUT}" -J "${SSH_JUMP}" "${SSH_REMOTE}" 'printf %s "$HOME"' 2>/dev/null)"; then
  pass "SSH/jump connectivity to ${GPU_NODE}"
  pass "Remote home detected: ${REMOTE_HOME_DIR}"
else
  fail "Cannot connect to ${GPU_NODE} via jump host ${JUMP_HOST}"
  echo "      Verify booking is active and credentials/keys are valid."
fi

echo ""
echo "[3/4] Remote layout"
if [[ -n "${REMOTE_HOME_DIR}" ]]; then
  REMOTE_HOME_PROJECT="${REMOTE_HOME_DIR}/${REMOTE_PROJECT_DIRNAME}"
  REMOTE_LEROBOT_DIR="${REMOTE_HOME_PROJECT}/lerobot"
  REMOTE_CODE_DIR="${REMOTE_HOME_PROJECT}/SmolVLA-Testing"
  DATASET_NAMES_STR="${DATASET_NAMES[*]}"

  remote_checks_output="$(ssh -o ConnectTimeout="${CHECK_TIMEOUT}" -J "${SSH_JUMP}" "${SSH_REMOTE}" bash -s -- "${REMOTE_HOME_PROJECT}" "${REMOTE_LEROBOT_DIR}" "${REMOTE_CODE_DIR}" "${REMOTE_SCRATCH_BASE}" "${DATASET_ROOT}" "${DATASET_NAMES_STR}" <<'EOS'
set -u
status=0
home_project="$1"
lerobot="$2"
code="$3"
scratch="$4"
data_root="$5"
datasets="$6"

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
    echo "[FAIL] $label missing on remote: $path"
    status=1
  fi
}

check_dir "Remote home project" "$home_project"
check_dir "Remote lerobot" "$lerobot"
check_dir "Remote SmolVLA-Testing code" "$code"
check_dir "Remote scratch base" "$scratch"
check_dir "Remote dataset root" "$data_root"

for ds in $datasets; do
  check_dir "Remote dataset $ds" "$data_root/$ds/meta"
done

echo "[4/4] Training readiness"
check_file "Scratch activation shim exists (setup complete)" "$scratch/activate_smolvla.sh"

if [[ -w "$scratch" ]]; then
  echo "[PASS] Scratch is writable: $scratch"
else
  echo "[FAIL] Scratch is not writable: $scratch"
  status=1
fi

exit "$status"
EOS
)" || true

  echo "${remote_checks_output}"
  if echo "${remote_checks_output}" | grep -q "\[FAIL\]"; then
    FAILED=1
  fi
else
  FAILED=1
fi

echo ""
if [[ "${FAILED}" -eq 0 ]]; then
  echo "Preflight result: PASS"
  echo "You can proceed with:"
  echo "  1) ssh -l ${REMOTE_USER} -J ${REMOTE_USER}@${JUMP_HOST} ${GPU_NODE}"
  echo "  2) cd ~/smolvla_project/SmolVLA-Testing/scripts && bash ./04_start_training.sh"
  exit 0
fi

echo "Preflight result: FAIL"
echo "Fix the failed checks above before launching training."
exit 1
