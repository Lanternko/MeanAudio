#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
LOG_DIR="$HOME/logs"
LOG="$LOG_DIR/music_flamingo_todo_queue.log"
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

busy_processes() {
    ps -u "$USER" -o pid=,args= | awk '
        $2 ~ /(python|torchrun)$/ &&
        $0 ~ /(train[.]py|eval[.]py|phase4_eval[.]py|peav_eval[.]py|music_flamingo_jamendo_slice_caption[.]py|torchrun)/ {
            print
        }
    '
}

wait_for_gpu_idle() {
    while true; do
        local busy
        busy="$(busy_processes)"
        local util
        util="$(gpu_util)"
        local compute
        compute="$(gpu_compute_processes)"

        if [ -z "$busy" ] && [ -z "$compute" ] && [ "${util:-100}" -le "$IDLE_GPU_UTIL_MAX" ]; then
            echo "$(ts) GPU idle: util=${util}%"
            return 0
        fi

        echo "$(ts) GPU busy: util=${util}%"
        if [ -n "$busy" ]; then
            echo "$busy"
        fi
        if [ -n "$compute" ]; then
            echo "$compute"
        fi
        sleep "$POLL_SECONDS"
    done
}

run_a2_short_direct_10k() {
    local ema="$WORK_DIR/exps/mfshort10k_direct_noq_fast_stage2_50000/mfshort10k_direct_noq_fast_stage2_50000_ema_final.pth"
    if [ -f "$ema" ]; then
        echo "$(ts) [SKIP] A2 MF-short-direct 10k already complete: $ema"
        return 0
    fi

    echo "$(ts) [START] A2 MF-short-direct 10k"
    wait_for_gpu_idle
    cd "$WORK_DIR"
    bash "$WORK_DIR/scripts/training_pipelines/train_pipeline_music_flamingo_short_direct_10k.sh"
    echo "$(ts) [DONE] A2 MF-short-direct 10k"

    for d in \
        "$WORK_DIR/eval_output/metrics/mfshort10k_direct_noq_fast_stage2_50000_musiccaps" \
        "$WORK_DIR/eval_output/metrics/mfshort10k_direct_noq_fast_stage2_50000_jamendo_holdout2048"; do
        if [ -f "$d/metrics.txt" ]; then
            echo "$(ts) metrics: $d/metrics.txt"
            cat "$d/metrics.txt"
        fi
    done
}

echo "$(ts) Music Flamingo TODO queue started"
run_a2_short_direct_10k
echo "$(ts) Music Flamingo TODO queue finished; no remaining scheduled TODOs"
