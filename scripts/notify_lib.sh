#!/bin/bash
# Discord notifications for GPU jobs launched outside ~/gpu_queue.
# The queue host notifies on seat/done/failure; a direct run gets nothing unless it
# sources this file. Usage, near the top of a direct-run script (after set -eo pipefail):
#
#   source "$HOME/MeanAudio/scripts/notify_lib.sh"
#   notify_on_exit "075_d2_chain" "$LOG"      # sends start now, success/failure on exit
#
# Notifier failures are logged to stderr and never kill the job.
#
# The queue host's "gpu-queue-idle" message fires once per empty-queue period, so a direct
# run that ends while the queue is already empty would leave the GPU idle silently. The exit
# message therefore says whether the GPU is now idle, and when it is, a separate IDLE message
# goes out through the same queue-status notifier the host uses.

NOTIFY_PY="${NOTIFY_PY:-$HOME/venvs/dac/bin/python}"
NOTIFY_SCRIPT="${NOTIFY_SCRIPT:-$HOME/MeanAudio/scripts/notify_experiment_webhook.py}"
NOTIFY_QUEUE_STATUS="${NOTIFY_QUEUE_STATUS:-$HOME/MeanAudio/scripts/notify_queue_status_webhook.py}"

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

_notify_gpu_state() {  # one-line GPU/queue status for the exit message
  local q="${GPU_QUEUE_ROOT:-$HOME/gpu_queue}" queued=0 procs
  if LC_ALL=C find "$q"/p1/pending "$q"/p1/running "$q"/p2/pending "$q"/p2/running \
      -maxdepth 1 -name '*.sh' -type f -print -quit 2>/dev/null | grep -q .; then
    queued=1
  fi
  # Resident services (e.g. the arale-persona-bot TTS server, ~1.2 GB) are not experiments;
  # only count compute processes above NOTIFY_GPU_BUSY_MIB.
  procs=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -v t="${NOTIFY_GPU_BUSY_MIB:-4096}" '$1+0 > t {n++} END {print n+0}')
  if [ "$queued" -eq 0 ] && [ "${procs:-0}" -eq 0 ]; then
    echo "GPU now IDLE: queue empty, no other GPU job"
  elif [ "$queued" -eq 1 ]; then
    echo "queue has pending/running jobs"
  else
    echo "queue empty, ${procs} other GPU job(s) > ${NOTIFY_GPU_BUSY_MIB:-4096} MiB still running"
  fi
}

_notify_exit_handler() {
  local rc="$1" gpu
  gpu="$(_notify_gpu_state)"
  if [ "$rc" -eq 0 ]; then
    notify_send success "$_NOTIFY_EXP" "direct run finished; $gpu" "$_NOTIFY_LOG" "$rc" "$_NOTIFY_T0"
  elif [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then
    notify_send interrupted "$_NOTIFY_EXP" "direct run interrupted by signal; $gpu" "$_NOTIFY_LOG" "$rc" "$_NOTIFY_T0"
  else
    notify_send failure "$_NOTIFY_EXP" "direct run exited rc=$rc; $gpu" "$_NOTIFY_LOG" "$rc" "$_NOTIFY_T0"
  fi
  case "$gpu" in
    "GPU now IDLE"*)
      "$NOTIFY_PY" "$NOTIFY_QUEUE_STATUS" --status idle --experiment gpu-idle \
        --summary "direct run $_NOTIFY_EXP ended; p1/p2 queue empty, no other GPU job" --exit-code 0 >&2 \
        || echo "[notify_lib] NOTIFY_FAIL gpu-idle" >&2 ;;
  esac
}
