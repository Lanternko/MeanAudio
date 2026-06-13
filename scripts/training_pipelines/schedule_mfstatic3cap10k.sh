#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
LOG_DIR="$HOME/logs"
LOG="$LOG_DIR/mfstatic3cap10k_queue.log"
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

echo "$(ts) MF static-random-3cap 10k queue started"
echo "$(ts) [WAIT] current GPU work must finish first"
wait_for_gpu_idle

cd "$WORK_DIR"
echo "$(ts) [START] MF-static-random-3cap 10k"
bash "$WORK_DIR/scripts/training_pipelines/train_pipeline_music_flamingo_static_random_3cap_10k.sh"
status=$?
echo "$(ts) [DONE] MF-static-random-3cap 10k status=$status"
exit "$status"
