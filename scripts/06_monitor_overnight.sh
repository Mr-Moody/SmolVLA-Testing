#!/usr/bin/env bash
# Monitor overnight pipeline progress
#
# Usage:
#   ./scripts/06_monitor_overnight.sh overnight_output/
#   ./scripts/06_monitor_overnight.sh overnight_output/ --interval 30

set -uo pipefail

OUTPUT_DIR="${1:-.overnight_output}"
MONITOR_INTERVAL="${2:-60}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: Output directory not found: $OUTPUT_DIR"
    exit 1
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

show_status() {
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║        OVERNIGHT PIPELINE PROGRESS MONITOR                    ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Output Directory: $OUTPUT_DIR"
    echo "Last Updated: $(date)"
    echo ""
    
    # Check for checkpoint
    if [[ -f "$OUTPUT_DIR/checkpoint.json" ]]; then
        echo "━━ CHECKPOINT STATUS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python3 << 'PYEOF'
import json
import sys
from pathlib import Path

cp_file = Path(sys.argv[1]) / "checkpoint.json"
if cp_file.exists():
    data = json.loads(cp_file.read_text())
    print(f"Last checkpoint: {data.get('timestamp', 'unknown')}")
    print()
    print("Dataset Status:")
    for ds in data.get('datasets', []):
        status = ds.get('status', 'unknown')
        name = ds.get('name', '?')
        error = ds.get('error', '')
        if error:
            error = f"  ERROR: {error[:50]}"
        status_icon = "✓" if status == "done" else "✗" if status == "failed" else "▸"
        print(f"  {status_icon} {name:20s} {status:20s} {error}")
PYEOF
        echo ""
    fi
    
    # Show log summary
    if [[ -d "$OUTPUT_DIR/logs" ]]; then
        echo "━━ LATEST LOG ENTRIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        latest_log=$(ls -t "$OUTPUT_DIR/logs"/*.log 2>/dev/null | head -1)
        if [[ -n "$latest_log" ]]; then
            echo "Log: $(basename $latest_log)"
            echo ""
            tail -20 "$latest_log" | sed 's/^/  /'
        fi
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Monitoring interval: ${MONITOR_INTERVAL}s | Press Ctrl+C to exit"
    echo "Next update: $(date -d "+${MONITOR_INTERVAL} seconds" 2>/dev/null || echo 'in ${MONITOR_INTERVAL}s')"
}

# Initial display
show_status

# Monitor loop
while true; do
    sleep "$MONITOR_INTERVAL"
    show_status
done
