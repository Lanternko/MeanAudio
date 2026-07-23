#!/usr/bin/env bash
# Run one complete experiment command and report success/failure/interruption.
#
# Usage:
#   scripts/run_with_experiment_report.sh \
#     --experiment NAME --report /path/to/final.json --log /path/to/run.log \
#     -- bash path/to/sequence.sh

set -uo pipefail

ROOT=/home/kojiek/MeanAudio
NOTIFIER="$ROOT/scripts/notify_experiment_webhook.py"
EXPERIMENT=
REPORT=
LOG=
SUMMARY=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --experiment)
            EXPERIMENT="${2:?--experiment requires a value}"
            shift 2
            ;;
        --report)
            REPORT="${2:?--report requires a value}"
            shift 2
            ;;
        --log)
            LOG="${2:?--log requires a value}"
            shift 2
            ;;
        --summary)
            SUMMARY="${2:?--summary requires a value}"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "[FAIL] unknown wrapper argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$EXPERIMENT" ] || [ "$#" -eq 0 ]; then
    echo "[FAIL] usage: $0 --experiment NAME [--report FILE] [--log FILE] -- COMMAND..." >&2
    exit 2
fi
if [ ! -x "$NOTIFIER" ]; then
    echo "[FAIL] notifier is missing or not executable: $NOTIFIER" >&2
    exit 2
fi

started_epoch=$(date +%s)
child_pid=
notification_sent=false

notify_once() {
    local status="$1"
    local exit_code="$2"
    local default_summary="$3"
    local args=(
        --status "$status"
        --experiment "$EXPERIMENT"
        --exit-code "$exit_code"
        --started-epoch "$started_epoch"
        --summary "${SUMMARY:-$default_summary}"
    )
    [ -n "$REPORT" ] && args+=(--report "$REPORT")
    [ -n "$LOG" ] && args+=(--log "$LOG")
    if [ "$notification_sent" = false ]; then
        notification_sent=true
        "$NOTIFIER" "${args[@]}" || {
            echo "[WARN] Discord experiment notification failed" >&2
        }
    fi
}

handle_signal() {
    local signal="$1"
    local code="$2"
    trap - HUP INT TERM
    if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
        kill -s "$signal" -- "-$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    notify_once interrupted "$code" "experiment interrupted by SIG$signal"
    exit "$code"
}

trap 'handle_signal HUP 129' HUP
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

setsid --wait "$@" &
child_pid=$!
wait "$child_pid"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
    notify_once success 0 "experiment and registered evaluations completed"
elif [ "$exit_code" -eq 129 ] || [ "$exit_code" -eq 130 ] || \
     [ "$exit_code" -eq 137 ] || [ "$exit_code" -eq 143 ]; then
    notify_once interrupted "$exit_code" "experiment process was interrupted"
else
    notify_once failure "$exit_code" "experiment stopped with a non-zero exit"
fi

exit "$exit_code"
