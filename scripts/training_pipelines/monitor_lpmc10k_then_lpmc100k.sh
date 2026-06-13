#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
LOG_DIR="$HOME/logs"
LOG="$LOG_DIR/monitor_lpmc10k_then_lpmc100k.log"
WAIT_SESSION="${WAIT_SESSION:-lpmc10k_control}"
NEXT_SESSION="${NEXT_SESSION:-lpmc100k_control}"

S2_10K="$WORK_DIR/exps/lpmc10k_noq_fast_stage2_50000/lpmc10k_noq_fast_stage2_50000_ema_final.pth"
MC_10K="$WORK_DIR/eval_output/metrics/lpmc10k_noq_fast_stage2_50000_musiccaps/metrics.txt"
JM_10K="$WORK_DIR/eval_output/metrics/lpmc10k_noq_fast_stage2_50000_jamendo_holdout2048/metrics.txt"
S2_100K="$WORK_DIR/exps/lpmc100k_noq_stage2_100000/lpmc100k_noq_stage2_100000_ema_final.pth"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

date "+[%F %T %Z] monitor started"
echo "waiting for tmux session: $WAIT_SESSION"
echo "next session: $NEXT_SESSION"

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    date "+[%F %T %Z] $WAIT_SESSION still running"
    sleep 300
done

date "+[%F %T %Z] $WAIT_SESSION is no longer running; checking success markers"
for path in "$S2_10K" "$MC_10K" "$JM_10K"; do
    if [ ! -f "$path" ]; then
        echo "[ABORT] LPMC10k did not finish cleanly; missing: $path"
        exit 20
    fi
done

if [ -f "$S2_100K" ]; then
    echo "[SKIP] LPMC100k already finished: $S2_100K"
    exit 0
fi

if tmux has-session -t "$NEXT_SESSION" 2>/dev/null; then
    echo "[SKIP] $NEXT_SESSION is already running"
    exit 0
fi

date "+[%F %T %Z] launching LPMC100k control"
tmux new-session -d -s "$NEXT_SESSION" \
    "cd '$WORK_DIR' && bash scripts/training_pipelines/train_pipeline_lpmc_100k_control.sh"
tmux ls
date "+[%F %T %Z] launched $NEXT_SESSION"
