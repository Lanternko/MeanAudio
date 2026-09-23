#!/bin/bash
# Retrofit a Discord notification onto a direct run that is already going and has no
# notify_lib.sh trap (bash buffered the script at launch, so editing it does nothing).
# The exit code of a non-child PID is not observable, so success = the done marker
# appears in the log after the PID exits.
#
#   bash scripts/notify_when_pid_exits.sh <pid> <experiment> <log> <done_marker> [started_epoch]
set -uo pipefail
PID="$1"; EXP="$2"; LOG="$3"; MARKER="$4"; T0="${5:-}"
source "$HOME/MeanAudio/scripts/notify_lib.sh"

kill -0 "$PID" 2>/dev/null || { echo "pid $PID not running" >&2; exit 2; }
echo "[$(date -u +%FT%TZ)] watching pid $PID for $EXP"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] pid $PID exited"

if grep -qF -- "$MARKER" "$LOG"; then
  notify_send success "$EXP" "direct run finished (marker '$MARKER' found)" "$LOG" 0 "$T0"
else
  notify_send failure "$EXP" "pid $PID exited without '$MARKER' in log" "$LOG" "" "$T0"
fi
