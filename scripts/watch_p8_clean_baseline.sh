#!/usr/bin/env bash
# Structured snapshot for p8_clean_baseline watcher.
set -u
REPO=/home/kojiek/MeanAudio
LOG_DIR=/home/kojiek/logs
EXP_S1=phase8_bugfix_rerun_stage1_400000
EXP_S2=phase8_bugfix_rerun_stage2_200000
S1_ITERS=400000
S2_ITERS=200000

ts=$(date -Is)
echo "=== p8_clean_baseline status @ $ts ==="

echo "-- tmux --"
if tmux has-session -t p8_clean_baseline 2>/dev/null; then
  echo "tmux: ALIVE"
  tmux capture-pane -t p8_clean_baseline -p 2>/dev/null | tail -20
else
  echo "tmux: DEAD"
fi

echo "-- processes --"
ps -eo pid,etime,pcpu,pmem,args | awk "/[n]ode/ && /train|eval/ {print} /[p]ython/ && /(train|eval|reextract)/ {print}" | head -20

echo "-- gpu --"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"

echo "-- checkpoints --"
for f in \
  "$REPO/exps/$EXP_S1/${EXP_S1}_ckpt_last.pth" \
  "$REPO/exps/$EXP_S1/${EXP_S1}_ema_final.pth" \
  "$REPO/exps/$EXP_S2/${EXP_S2}_ckpt_last.pth" \
  "$REPO/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
do
  if [ -f "$f" ]; then
    sz=$(stat -c%s "$f" 2>/dev/null)
    mt=$(stat -c%y "$f" 2>/dev/null | cut -c1-19)
    it=$(python3 -c "import torch; c=torch.load(\"$f\", map_location=\"cpu\", weights_only=False); print(c.get(\"it\", \"?\"))" 2>/dev/null || echo "?")
    echo "CKPT $(basename "$f") it=$it size=$sz mtime=$mt"
  else
    echo "CKPT missing: $(basename "$f")"
  fi
done

echo "-- pipeline log tail --"
if [ -f "$LOG_DIR/phase8_bugfix_rerun_pipeline.log" ]; then
  tail -30 "$LOG_DIR/phase8_bugfix_rerun_pipeline.log"
  # derive latest iter/loss if present
  grep -E "it\s+[0-9]+:" "$LOG_DIR/phase8_bugfix_rerun_pipeline.log" | tail -3
else
  echo "pipeline log missing"
fi

echo "-- stage heuristics --"
if [ -f "$REPO/exps/$EXP_S2/${EXP_S2}_ema_final.pth" ]; then
  echo "STAGE: S2_DONE (ema_final present)"
elif [ -f "$REPO/exps/$EXP_S2/${EXP_S2}_ckpt_last.pth" ]; then
  echo "STAGE: S2_TRAINING_OR_PARTIAL"
elif [ -f "$REPO/exps/$EXP_S1/${EXP_S1}_ema_final.pth" ] || [ -f "$REPO/exps/$EXP_S1/${EXP_S1}_ckpt_last.pth" ]; then
  it=$(python3 -c "import torch; c=torch.load(\"$REPO/exps/$EXP_S1/${EXP_S1}_ckpt_last.pth\", map_location=\"cpu\", weights_only=False); print(c.get(\"it\", 0))" 2>/dev/null || echo 0)
  echo "STAGE: S1 it=$it / $S1_ITERS"
else
  echo "STAGE: PRE_S1_OR_UNKNOWN"
fi

echo "-- eval outputs --"
ls -la "$REPO/eval_output/${EXP_S2}_musiccaps" 2>/dev/null | head -10 || echo "no MusicCaps eval dir yet"
find "$REPO/eval_output" -name "*phase8_bugfix*" -type d 2>/dev/null | head
find /home/kojiek/research/meanaudio_eval /home/kojiek/MeanAudio/eval_output -name "*phase8_bugfix*metrics*" 2>/dev/null | head
ls /home/kojiek/MeanAudio/eval_output/metrics 2>/dev/null | grep -i phase8_bugfix || true

echo "-- disk --"
df -h /home/kojiek | tail -1

echo "=== end status ==="
