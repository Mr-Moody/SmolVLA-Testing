#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
if [[ -f "${SCRIPT_DIR}/00_run_params.sh" ]]; then
  # we source params only to get REMOTE_SCRATCH_BASE and REMOTE_PROJECT_DIRNAME
  source "${SCRIPT_DIR}/00_run_params.sh"
else
  echo "Cannot find 00_run_params.sh in ${SCRIPT_DIR}; please run this from the repository scripts directory." >&2
  exit 2
fi

echo "Inspecting REMOTE_SCRATCH_BASE: ${REMOTE_SCRATCH_BASE}"

declare -a candidates=(
  "${REMOTE_SCRATCH_BASE}/lerobot/.venv"
  "${REMOTE_SCRATCH_BASE}/${REMOTE_PROJECT_DIRNAME}/lerobot/.venv"
  "${REMOTE_SCRATCH_BASE}/SmolVLA-Testing/../lerobot/.venv"
)

found=0
for p in "${candidates[@]}"; do
  if [ -d "${p}" ]; then
    echo "FOUND: ${p}"
    if [ -x "${p}/bin/python3" ]; then
      echo "  python: $(${p}/bin/python3 -c 'import sys; print(sys.executable)')"
    else
      echo "  (no python binary at ${p}/bin/python3)"
    fi
    du -sh "${p}" 2>/dev/null || true
    found=$((found+1))
  else
    echo "Not found: ${p}"
  fi
done

echo
echo "Searching for any .venv directories under ${REMOTE_SCRATCH_BASE} (maxdepth=4)..."
mapfile -t search_results < <(find "${REMOTE_SCRATCH_BASE}" -maxdepth 4 -type d -name ".venv" 2>/dev/null || true)

if [ ${#search_results[@]} -eq 0 ]; then
  echo "No .venv directories found under ${REMOTE_SCRATCH_BASE}"
else
  echo "Found ${#search_results[@]} .venv directory(ies):"
  for v in "${search_results[@]}"; do
    echo "  ${v}"
    if [ -x "${v}/bin/python3" ]; then
      echo "    python: $(${v}/bin/python3 -c 'import sys; print(sys.executable)')"
    fi
    du -sh "${v}" 2>/dev/null || true
  done
  found=$((found + ${#search_results[@]}))
fi

echo
echo "Summary: total candidate hits + find results: ${found}"

echo "Done. This script only discovers and reports venv locations; it will not modify or install anything."
