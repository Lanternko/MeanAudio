#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
LOG_DIR="$HOME/logs"
LOG="$LOG_DIR/music_flamingo_open_todos_queue.log"
WAIT_SESSION="${WAIT_SESSION:-mfstatic3cap100k_queue}"
IDLE_GPU_UTIL_MAX="${IDLE_GPU_UTIL_MAX:-10}"
POLL_SECONDS="${POLL_SECONDS:-300}"

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

wait_for_tmux_session_done() {
    if [ -z "$WAIT_SESSION" ]; then
        return 0
    fi

    while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
        echo "$(ts) waiting for tmux session to finish: $WAIT_SESSION"
        sleep "$POLL_SECONDS"
    done
    echo "$(ts) tmux session finished: $WAIT_SESSION"
}

wait_for_gpu_idle() {
    while true; do
        local util
        util="$(gpu_util)"
        local compute
        compute="$(gpu_compute_processes)"

        if [ -z "$compute" ] && [ "${util:-100}" -le "$IDLE_GPU_UTIL_MAX" ]; then
            echo "$(ts) GPU idle: util=${util}%"
            return 0
        fi

        echo "$(ts) GPU busy: util=${util}%"
        if [ -n "$compute" ]; then
            echo "$compute"
        fi
        sleep "$POLL_SECONDS"
    done
}

run_when_needed() {
    local name="$1"
    local metric_file="$2"
    shift 2

    if [ -f "$metric_file" ]; then
        echo "$(ts) [SKIP] $name already has metrics: $metric_file"
        return 0
    fi

    wait_for_gpu_idle
    echo "$(ts) [START] $name"
    "$@"
    local status=$?
    echo "$(ts) [DONE] $name status=$status"
    return "$status"
}

echo "$(ts) Music Flamingo open TODO queue started"
echo "$(ts) [WAIT] session=$WAIT_SESSION, then GPU idle between tasks"
wait_for_tmux_session_done

cd "$WORK_DIR"

run_when_needed \
    "C1 LPMC100k on short-direct MF Jamendo prompts" \
    "$WORK_DIR/eval_output/metrics/lpmc100k_noq_stage2_100000_jamendo_holdout2048_shortdirect_mfcap/metrics.txt" \
    bash "$WORK_DIR/scripts/eval/eval_lpmc100k_shortdirect_mfcap_jamendo.sh"

run_when_needed \
    "A4b MF-static-random-3cap 10k on original MF Jamendo prompts" \
    "$WORK_DIR/eval_output/metrics/mfstatic3cap10k_noq_fast_stage2_50000_jamendo_holdout2048_original_mfcap/metrics.txt" \
    bash "$WORK_DIR/scripts/eval/eval_mfstatic3cap10k_original_mfcap_jamendo.sh"

echo "$(ts) [DONE] Music Flamingo open TODO queue complete"
