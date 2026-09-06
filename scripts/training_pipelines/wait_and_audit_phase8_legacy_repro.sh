#!/usr/bin/env bash
# Wait for the guarded Phase-8 run, then execute the independent strict audit.

set -euo pipefail

WORK_DIR=/home/kojiek/MeanAudio
STATE_DIR=/home/kojiek/logs/phase8_legacy_repro_guard
STATE_FILE="$STATE_DIR/state.json"
ALERT_FILE="$STATE_DIR/ALERT.json"
STATUS_FILE="$STATE_DIR/final_audit_loop_status.json"
LOG_FILE="$STATE_DIR/final_audit.log"
LOCK_FILE="$STATE_DIR/final_audit_loop.lock"
AUDIT="$WORK_DIR/scripts/audit_phase8_legacy_repro.py"

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[STOP] final-audit loop already owns $LOCK_FILE" >&2
    exit 2
fi

exec > >(tee -a "$LOG_FILE") 2>&1
source /home/kojiek/venvs/dac/bin/activate
cd "$WORK_DIR"

write_status() {
    local phase="$1"
    local detail="$2"
    python - "$STATUS_FILE" "$phase" "$detail" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, phase, detail = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
payload = {
    "phase": phase,
    "detail": detail,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

read_phase() {
    python - "$STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text()).get("phase", "UNKNOWN")
except Exception:
    value = "UNKNOWN"
print(value)
PY
}

on_exit() {
    local status=$?
    if [ "$status" -ne 0 ]; then
        write_status FAILED "final-audit loop exited with status $status" || true
    fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

write_status WAITING "waiting for guarded pipeline DONE state"
echo "[$(date -Is)] final-audit loop started"

while true; do
    if [ -s "$ALERT_FILE" ]; then
        echo "[FAIL] guarded watcher alert is present: $ALERT_FILE" >&2
        exit 2
    fi
    phase=$(read_phase)
    case "$phase" in
        DONE)
            break
            ;;
        FAILED)
            echo "[FAIL] guarded pipeline entered FAILED state" >&2
            exit 2
            ;;
        *)
            write_status WAITING "guard phase=$phase; waiting for DONE"
            sleep 55
            ;;
    esac
done

write_status AUDITING "guard is DONE; running strict completion audit"
echo "[$(date -Is)] guarded pipeline is DONE; starting strict audit"
python "$AUDIT"

write_status PASSED "strict completion audit passed"
echo "[$(date -Is)] strict completion audit passed"
trap - EXIT INT TERM
