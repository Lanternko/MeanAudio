#!/usr/bin/env bash
# Durable five-minute fallback watcher used when Grok scheduling is unavailable.
# Monitor classification is authoritative. A stop requires a fresh Codex SOL
# verdict and a second current incident observation; this script never uses kill.

set -u

ROOT=/home/kojiek/MeanAudio
STATE=/home/kojiek/logs/phase8_fixedq_attm_monitor
LOG=/home/kojiek/logs/phase8_fixedq_attm_durable_watch.log
MONITOR="$ROOT/scripts/monitor_phase8_fixedq_attm_ft.py"
ADJUDICATE="$ROOT/scripts/adjudicate_phase8_fixedq_attm_stop_with_codex.sh"
VERDICT="$STATE/codex_sol_verdict.json"

source /home/kojiek/venvs/dac/bin/activate
mkdir -p "$STATE"

while true; do
    echo "[WATCH] $(date --iso-8601=seconds)" >>"$LOG"
    "$MONITOR" --once >>"$LOG" 2>&1
    rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "[WATCH] incident candidate; requesting Codex SOL" >>"$LOG"
        if "$ADJUDICATE" >>"$LOG" 2>&1; then
            # Require a second current monitor observation after the verdict.
            "$MONITOR" --once >>"$LOG" 2>&1
            confirm_rc=$?
            if [ "$confirm_rc" -ne 0 ] && python - "$VERDICT" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
fresh = time.time() - path.stat().st_mtime < 600
raise SystemExit(
    0
    if fresh
    and payload.get("decision") == "stop"
    and payload.get("stop_authorized") is True
    else 1
)
PY
            then
                echo "[WATCH] fresh Codex SOL stop authorization; sending one Ctrl-C" >>"$LOG"
                tmux send-keys -t p8_fixedq_attm C-c
                exit 0
            fi
        else
            echo "[WATCH] adjudication failed; preserving experiment" >>"$LOG"
        fi
    fi

    # Five one-minute waits keep each individual wait bounded.
    for _ in 1 2 3 4 5; do
        sleep 60
    done
done
