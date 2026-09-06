#!/usr/bin/env bash
# Approved launch surface: outer reporter survives child OOM/SIGKILL and handles
# tmux HUP/INT/TERM.  Do not launch the sequence directly.
set -euo pipefail

SESSION=ms_quarter
ROOT=/home/kojiek/MeanAudio
SEQUENCE=/home/kojiek/research/meanaudio_training/caption10s_pipeline/sequence_caption10s_multisent_quarter.sh
STATE=/home/kojiek/logs/caption10s_multisent_noq_quarter
REPORT=/home/kojiek/logs/phase8_qwen_caption10s_multisent_noq_quarter_FINAL_METRICS.json
WRAPPER="$ROOT/scripts/run_with_experiment_report.sh"
DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=true
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "[FAIL] usage: $0 [--dry-run]" >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[FAIL] tmux session already exists: $SESSION" >&2
  exit 3
fi
if pgrep -u "$(id -u)" -f "$SEQUENCE" >/dev/null; then
  echo "[FAIL] sequence process already exists: $SEQUENCE" >&2
  exit 3
fi

mkdir -p "$STATE"
COMMAND="$WRAPPER --experiment caption10s_multisent_noq_quarter --report $REPORT --log $STATE/sequence.log --gpu-released -- bash $SEQUENCE"
if [ "$DRY_RUN" = true ]; then
  echo "tmux new-session -d -s $SESSION $COMMAND"
  exit 0
fi
tmux new-session -d -s "$SESSION" \
  "$COMMAND"
echo "[STARTED] $SESSION via run_with_experiment_report.sh"
