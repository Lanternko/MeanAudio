#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
LOG_DIR="$HOME/logs"
LOG="$LOG_DIR/mfexpanded3cap100k_queue.log"
MONITOR_LOG="$LOG_DIR/mfexpanded3cap100k_monitor.log"
SESSION_NAME="${SESSION_NAME:-mfexpanded3cap100k_queue}"
IDLE_GPU_UTIL_MAX="${IDLE_GPU_UTIL_MAX:-10}"
MAX_BUSY_MEMORY_MB="${MAX_BUSY_MEMORY_MB:-8192}"
POLL_SECONDS="${POLL_SECONDS:-300}"
WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-600}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

ts() {
    date "+[%F %T %Z]"
}

gpu_util() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

gpu_compute_processes() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
}

gpu_memory_used_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

wait_for_gpu_idle() {
    while true; do
        local util
        util="$(gpu_util)"
        local memory_used
        memory_used="$(gpu_memory_used_mb)"
        local compute
        compute="$(gpu_compute_processes)"

        if [ "${util:-100}" -le "$IDLE_GPU_UTIL_MAX" ] && [ "${memory_used:-999999}" -le "$MAX_BUSY_MEMORY_MB" ]; then
            echo "$(ts) GPU available: util=${util}% memory=${memory_used}MiB"
            if [ -n "$compute" ]; then
                echo "$compute"
            fi
            return 0
        fi

        echo "$(ts) GPU busy: util=${util}% memory=${memory_used}MiB"
        if [ -n "$compute" ]; then
            echo "$compute"
        fi
        sleep "$POLL_SECONDS"
    done
}

start_watchdog() {
    (
        while true; do
            {
                echo "$(ts) [watchdog] session=$SESSION_NAME"
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
                nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
                tail -n 20 "$HOME/logs/mfexpanded3cap100k_noq_pipeline.log" 2>/dev/null || true
                for exp in \
                    mfexpanded3cap100k_noq_stage2_100000_musiccaps \
                    mfexpanded3cap100k_noq_stage2_100000_jamendo_holdout2048
                do
                    if [ -f "$WORK_DIR/eval_output/metrics/$exp/metrics.txt" ]; then
                        echo "[metrics] $exp"
                        tail -n 8 "$WORK_DIR/eval_output/metrics/$exp/metrics.txt"
                    fi
                done
                echo
            } >> "$MONITOR_LOG" 2>&1
            sleep "$WATCHDOG_SECONDS"
        done
    ) &
    WATCHDOG_PID=$!
    echo "$(ts) watchdog started: pid=$WATCHDOG_PID log=$MONITOR_LOG"
}

cleanup_watchdog() {
    if [ -n "${WATCHDOG_PID:-}" ]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
}
trap cleanup_watchdog EXIT

echo "$(ts) MF-expanded-3cap 100k queue started"
echo "$(ts) [WAIT] GPU idle"
wait_for_gpu_idle

cd "$WORK_DIR"
start_watchdog
echo "$(ts) [START] A6 MF-expanded-3cap 100k-audio / 300k-caption"
bash "$WORK_DIR/scripts/training_pipelines/train_pipeline_music_flamingo_expanded_3cap_100k.sh"
status=$?
echo "$(ts) [DONE] A6 status=$status"

if [ "$status" -eq 0 ]; then
    echo "$(ts) [NEXT] A6 short-direct MF Jamendo eval-only"
    bash "$WORK_DIR/scripts/eval/eval_mfexpanded3cap100k_shortdirect_jamendo.sh"
    next_status=$?
    echo "$(ts) [DONE] A6 short-direct MF Jamendo eval-only status=$next_status"
    exit "$next_status"
else
    echo "$(ts) [ALERT] A6 stopped or failed; inspect $LOG and $MONITOR_LOG"
fi
exit "$status"
