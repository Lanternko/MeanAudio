#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
LOG_DIR="$HOME/logs"
LOG="$LOG_DIR/monitor_lpmc100k_then_mfshort10k.log"
WAIT_SESSION="${WAIT_SESSION:-lpmc100k_control}"
NEXT_SESSION="${NEXT_SESSION:-mfshort10k_rewrite}"

NEXT_EMA="$WORK_DIR/exps/mfshort10k_rewrite_noq_fast_stage2_50000/mfshort10k_rewrite_noq_fast_stage2_50000_ema_final.pth"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

date "+[%F %T %Z] monitor started"
echo "waiting for tmux session: $WAIT_SESSION"
echo "next session: $NEXT_SESSION"

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
    date "+[%F %T %Z] $WAIT_SESSION still running"
    sleep 300
done

date "+[%F %T %Z] $WAIT_SESSION is no longer running"

if [ -f "$NEXT_EMA" ]; then
    echo "[SKIP] MF-short-rewrite 10k already finished: $NEXT_EMA"
    exit 0
fi

if tmux has-session -t "$NEXT_SESSION" 2>/dev/null; then
    echo "[SKIP] $NEXT_SESSION is already running"
    exit 0
fi

date "+[%F %T %Z] launching MF-short-rewrite 10k"
tmux new-session -d -s "$NEXT_SESSION" \
    "cd '$WORK_DIR' && bash scripts/training_pipelines/train_pipeline_music_flamingo_short_rewrite_10k.sh"
tmux ls
date "+[%F %T %Z] launched $NEXT_SESSION"
