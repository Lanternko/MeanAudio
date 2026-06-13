#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
LOG_DIR="$HOME/logs"
MONITOR_LOG="$LOG_DIR/monitor_mf10k_then_next.log"
MF_SESSION="${MF_SESSION:-mf10k_train}"
IDLE_UTIL_MAX="${IDLE_UTIL_MAX:-10}"
IDLE_MEM_MAX_MB="${IDLE_MEM_MAX_MB:-2500}"
IDLE_CHECKS="${IDLE_CHECKS:-5}"
IDLE_SLEEP_SEC="${IDLE_SLEEP_SEC:-60}"
POLL_SEC="${POLL_SEC:-300}"
SUCCESS_MC_CLAP="${SUCCESS_MC_CLAP:-0.15}"
SUCCESS_JM_CLAP="${SUCCESS_JM_CLAP:-0.12}"

MF_EXP_S2="${MF_EXP_S2:-mf10k_noq_fast_stage2_50000}"
MF_PIPELINE_LOG="$LOG_DIR/mf10k_noq_fast_pipeline.log"
MC_METRICS="$WORK_DIR/eval_output/metrics/${MF_EXP_S2}_musiccaps/metrics.txt"
JM_METRICS="$WORK_DIR/eval_output/metrics/${MF_EXP_S2}_jamendo_holdout2048/metrics.txt"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$MONITOR_LOG") 2>&1

ts() {
    date '+%Y-%m-%d %H:%M:%S'
}

metric_value() {
    local path="$1"
    local key="$2"
    awk -F': *' -v key="$key" '$1 == key {print $2; exit}' "$path"
}

gpu_idle() {
    local line util mem
    line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1)
    util=$(printf '%s\n' "$line" | awk -F',' '{gsub(/ /,"",$1); print $1}')
    mem=$(printf '%s\n' "$line" | awk -F',' '{gsub(/ /,"",$2); print $2}')
    echo "  gpu util=${util}% mem=${mem}MiB"
    [ "$util" -le "$IDLE_UTIL_MAX" ] && [ "$mem" -le "$IDLE_MEM_MAX_MB" ]
}

wait_gpu_idle() {
    local ok=0
    while [ "$ok" -lt "$IDLE_CHECKS" ]; do
        if gpu_idle; then
            ok=$((ok + 1))
            echo "  idle check $ok/$IDLE_CHECKS"
        else
            ok=0
            echo "  GPU busy; idle counter reset"
        fi
        [ "$ok" -lt "$IDLE_CHECKS" ] && sleep "$IDLE_SLEEP_SEC"
    done
    return 0
}

launch_tmux_once() {
    local session="$1"
    local cmd="$2"
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "[$(ts)] session already exists: $session"
        return 0
    fi
    tmux new-session -d -s "$session" "cd '$WORK_DIR' && $cmd"
    echo "[$(ts)] launched $session"
}

echo "======================================================"
echo "  Monitor mf10k -> next experiment"
echo "  MF session      : $MF_SESSION"
echo "  success gates   : MusicCaps CLAP >= $SUCCESS_MC_CLAP, Jamendo CLAP >= $SUCCESS_JM_CLAP"
echo "  idle definition : util <= ${IDLE_UTIL_MAX}%, mem <= ${IDLE_MEM_MAX_MB}MiB for ${IDLE_CHECKS} checks"
echo "======================================================"

while tmux has-session -t "$MF_SESSION" 2>/dev/null; do
    echo "[$(ts)] waiting for $MF_SESSION to finish"
    sleep "$POLL_SEC"
done

echo "[$(ts)] $MF_SESSION is no longer running"

if ! grep -q "Music Flamingo slice10 10k pipeline complete" "$MF_PIPELINE_LOG" 2>/dev/null; then
    echo "[ABORT] mf10k pipeline did not finish cleanly; not launching follow-up"
    echo "        check: $MF_PIPELINE_LOG"
    exit 1
fi

for path in "$MC_METRICS" "$JM_METRICS"; do
    if [ ! -f "$path" ]; then
        echo "[ABORT] missing metrics: $path"
        exit 1
    fi
done

mc_clap=$(metric_value "$MC_METRICS" clap_score)
jm_clap=$(metric_value "$JM_METRICS" clap_score)
echo "[$(ts)] mf10k metrics: MusicCaps clap=$mc_clap Jamendo clap=$jm_clap"

echo "[$(ts)] waiting for GPU idle before next job"
wait_gpu_idle

decision=$(python - <<PY
mc = float("$mc_clap")
jm = float("$jm_clap")
mc_gate = float("$SUCCESS_MC_CLAP")
jm_gate = float("$SUCCESS_JM_CLAP")
print("mf100k" if (mc >= mc_gate and jm >= jm_gate) else "lpmc10k")
PY
)

if [ "$decision" = "mf100k" ]; then
    echo "[$(ts)] Flamingo 10k passed gates; launching Flamingo 100k"
    launch_tmux_once mf100k_train "bash scripts/training_pipelines/train_pipeline_music_flamingo_100k.sh"
else
    echo "[$(ts)] Flamingo 10k did not pass gates; launching LP-MC 10k control"
    launch_tmux_once lpmc10k_train "bash scripts/training_pipelines/train_pipeline_lpmc_10k_control.sh"
fi

echo "[$(ts)] monitor complete"
