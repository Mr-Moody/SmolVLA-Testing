#!/usr/bin/env bash
# =============================================================================
# restart_conversion.sh  —  Kill existing conversion and relaunch under nohup
# =============================================================================
#
# PURPOSE
# -------
# Stops any currently running data_converter.py or convert_all.sh processes,
# then relaunches convert_all.sh under nohup so the job survives SSH
# disconnection. Tails the log so you can watch progress and then Ctrl+C
# to detach (the conversion keeps running).
#
# SHELL REQUIREMENT
# -----------------
# UCL TSG machines use tcsh as the default interactive shell. Run this as:
#   bash ~/smolvla_project/SmolVLA-Testing/restart_conversion.sh
# Do NOT run it with:  ./restart_conversion.sh  (tcsh will reject 'export')
#
# USAGE
# -----
#   bash ~/smolvla_project/SmolVLA-Testing/restart_conversion.sh
#
# MONITORING (after detaching with Ctrl+C)
# -----------------------------------------
#   tail -f /scratch0/$USER/conversion.log
#   grep "Episode\|Done\|Error" /scratch0/$USER/conversion.log
# =============================================================================

SCRATCH_BASE="/scratch0/${USER}"
LOG_FILE="${SCRATCH_BASE}/conversion.log"
CONVERT_SCRIPT="${HOME}/smolvla_project/SmolVLA-Testing/convert_all.sh"

# Stop any currently running conversion jobs
echo "Stopping any existing conversion jobs..."
pkill -f convert_all.sh 2>/dev/null || true
pkill -f data_converter.py 2>/dev/null || true
sleep 2

# Confirm they are gone
if pgrep -f data_converter.py > /dev/null 2>&1; then
    echo "WARNING: data_converter.py still running — sending SIGKILL..."
    pkill -9 -f data_converter.py 2>/dev/null || true
    sleep 1
fi

echo "Relaunching conversion with NVENC under nohup..."
nohup bash "${CONVERT_SCRIPT}" >> "${LOG_FILE}" 2>&1 &

CONV_PID=$!
echo ""
echo "================================================================="
echo "  Conversion launched."
echo "  PID : ${CONV_PID}"
echo "  Log : ${LOG_FILE}"
echo "================================================================="
echo ""
echo "  Tailing log — Ctrl+C to stop watching (conversion keeps running)."
echo "  To reattach later:  tail -f ${LOG_FILE}"
echo ""

tail -f "${LOG_FILE}"
