#!/usr/bin/env bash
# Approved launch surface; the outer reporter survives child OOM/SIGKILL.
set -euo pipefail

SESSION=rich_shared_gate
ROOT=/home/kojiek/MeanAudio
SEQUENCE="$ROOT/scripts/training_pipelines/sequence_rich_shared_then_matched_full.sh"
STATE=/home/kojiek/logs/rich_shared_then_matched_full
REPORT=/home/kojiek/logs/rich_shared_then_matched_full_FINAL_METRICS.json
WRAPPER="$ROOT/scripts/run_with_experiment_report.sh"
DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=true; shift; fi
[ "$#" -eq 0 ] || { echo "usage: $0 [--dry-run]" >&2; exit 2; }

tmux has-session -t "$SESSION" 2>/dev/null && { echo "[FAIL] tmux session exists: $SESSION" >&2; exit 3; }
pgrep -u "$(id -u)" -f "$SEQUENCE" >/dev/null && { echo "[FAIL] sequence process exists" >&2; exit 3; }
mkdir -p "$STATE"
COMMAND="$WRAPPER --experiment rich_shared_then_matched_full --report $REPORT --log $STATE/sequence.log --gpu-released -- bash $SEQUENCE"
if [ "$DRY_RUN" = true ]; then echo "tmux new-session -d -s $SESSION $COMMAND"; exit 0; fi
tmux new-session -d -s "$SESSION" "$COMMAND"
echo "[STARTED] $SESSION via run_with_experiment_report.sh"
