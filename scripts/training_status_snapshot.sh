#!/usr/bin/env bash
# Print a compact SmolVLA training status snapshot.
# Usage:
#   bash training_status_snapshot.sh [DATASET_JOINED] [TOTAL_STEPS]
# Example:
#   bash training_status_snapshot.sh 001_002_003 20000

set -euo pipefail

DATASET_JOINED="${1:-001_002_003}"
TOTAL_STEPS="${2:-20000}"
BASE="/scratch0/${USER}/smolvla_outputs"
MODEL_TYPE="${MODEL_TYPE:-smolvla}"
RUN_TAG="${DATASET_JOINED}_${MODEL_TYPE}"
LOG_FILE="${BASE}/${RUN_TAG}.log"
CKPT_DIR="${BASE}/${RUN_TAG}/checkpoints"

printf "=================================================\n"
printf "SmolVLA Status Snapshot\n"
printf "User:         %s\n" "${USER}"
printf "Dataset tag:  %s\n" "${DATASET_JOINED}"
printf "Model type:   %s\n" "${MODEL_TYPE}"
printf "Target steps: %s\n" "${TOTAL_STEPS}"
printf "Log file:     %s\n" "${LOG_FILE}"
printf "Checkpoint:   %s\n" "${CKPT_DIR}"
printf "Timestamp:    %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "=================================================\n"

if [[ -f "${LOG_FILE}" ]]; then
    echo "Last log lines:"
    tail -n 5 "${LOG_FILE}" || true
else
    echo "Log file not found yet."
fi

echo
latest=""
prev=""
if [[ -d "${CKPT_DIR}" ]]; then
    mapfile -t ckpts < <(ls -1 "${CKPT_DIR}" 2>/dev/null | grep -E '^[0-9]+$' | sort -n)
    count="${#ckpts[@]}"
    if [[ "${count}" -gt 0 ]]; then
        latest="${ckpts[$((count - 1))]}"
        latest_num=$((10#${latest}))
        echo "Latest checkpoint step: ${latest}"
        awk -v s="${latest_num}" -v t="${TOTAL_STEPS}" 'BEGIN {
            if (t > 0) {
                printf("Progress: %.1f%% (%d/%d)\n", (s/t)*100.0, s, t)
            }
        }'

        if [[ "${count}" -gt 1 ]]; then
            prev="${ckpts[$((count - 2))]}"
            prev_num=$((10#${prev}))
            t_latest=$(stat -c %Y "${CKPT_DIR}/${latest}" 2>/dev/null || echo "")
            t_prev=$(stat -c %Y "${CKPT_DIR}/${prev}" 2>/dev/null || echo "")
            if [[ -n "${t_latest}" && -n "${t_prev}" ]]; then
                dt=$((t_latest - t_prev))
                ds=$((latest_num - prev_num))
                if [[ "${dt}" -gt 0 && "${ds}" -gt 0 ]]; then
                    rate=$(awk -v ds="${ds}" -v dt="${dt}" 'BEGIN { printf("%.6f", ds/dt) }')
                    rem=$((TOTAL_STEPS - latest_num))
                    if [[ "${rem}" -gt 0 ]]; then
                        eta_sec=$(awk -v rem="${rem}" -v r="${rate}" 'BEGIN { if (r>0) printf("%.0f", rem/r); else print 0 }')
                        eta_h=$(awk -v s="${eta_sec}" 'BEGIN { printf("%.2f", s/3600.0) }')
                        echo "Recent speed: ${ds} steps in ${dt}s (${rate} steps/s)"
                        echo "ETA: ~${eta_h} hours"
                    else
                        echo "Training has reached or exceeded target steps."
                    fi
                fi
            fi
        fi
    else
        echo "No numeric checkpoints yet under ${CKPT_DIR}."
    fi
else
    echo "Checkpoint directory not found yet."
fi

echo
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU summary:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
else
    echo "nvidia-smi not found in PATH."
fi

echo
if pgrep -af "main.py" >/dev/null 2>&1; then
    echo "Training process candidates:"
    pgrep -af "main.py"
else
    echo "No main.py process found by pgrep."
fi
