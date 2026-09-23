#!/bin/bash
# Discord notifications for GPU jobs launched outside ~/gpu_queue.
# The queue host notifies on seat/done/failure; a direct run gets nothing unless it
# sources this file. Usage, near the top of a direct-run script (after set -eo pipefail):
#
#   source "$HOME/MeanAudio/scripts/notify_lib.sh"
#   notify_on_exit "075_d2_chain" "$LOG"      # sends start now, success/failure on exit
#
# Notifier failures are logged to stderr and never kill the job.

NOTIFY_PY="${NOTIFY_PY:-$HOME/venvs/dac/bin/python}"
NOTIFY_SCRIPT="${NOTIFY_SCRIPT:-$HOME/MeanAudio/scripts/notify_experiment_webhook.py}"

notify_send() {  # notify_send <status> <experiment> <summary> [log] [exit_code] [started_epoch]
  local args=(--status "$1" --experiment "$2" --summary "$3")
  [ -n "${4:-}" ] && args+=(--log "$4")
  [ -n "${5:-}" ] && args+=(--exit-code "$5")
  [ -n "${6:-}" ] && args+=(--started-epoch "$6")
  "$NOTIFY_PY" "$NOTIFY_SCRIPT" "${args[@]}" >&2 || echo "[notify_lib] NOTIFY_FAIL $2 $1" >&2
}

notify_on_exit() {  # notify_on_exit <experiment> [log]
  _NOTIFY_EXP="$1"
  _NOTIFY_LOG="${2:-}"
  _NOTIFY_T0="$(date +%s)"
  notify_send start "$_NOTIFY_EXP" "direct run (outside gpu_queue), pid $$" "$_NOTIFY_LOG"
  trap '_notify_exit_handler $?' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

_notify_exit_handler() {
  local rc="$1"
  if [ "$rc" -eq 0 ]; then
    notify_send success "$_NOTIFY_EXP" "direct run finished" "$_NOTIFY_LOG" "$rc" "$_NOTIFY_T0"
  elif [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
    notify_send interrupted "$_NOTIFY_EXP" "direct run interrupted by signal" "$_NOTIFY_LOG" "$rc" "$_NOTIFY_T0"
  else
    notify_send failure "$_NOTIFY_EXP" "direct run exited rc=$rc" "$_NOTIFY_LOG" "$rc" "$_NOTIFY_T0"
  fi
}
