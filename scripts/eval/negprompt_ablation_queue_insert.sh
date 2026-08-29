#!/bin/bash
# Insert the CFG x negative-prompt ablation matrix ahead of the P2 pending queue.
#
# Waits for the currently running job to finish -- it does NOT interrupt it.
# 025_true_random_full is S1 400k + S2 200k + eval, so expect a long wait before
# this seats. Nothing here pauses or modifies a live run.
#
# What the matrix settles, which the 12-arm negprompt_reeval sweep could not:
#   * how much of the +0.49 mean PQ gain is CFG itself rather than the negative
#     wording (the 'none' cells are textbook CFG against the stored null), and
#   * which negative wording is best, and whether that depends on the arm.
#
# Insertion mechanics are unchanged from negprompt_reeval_queue_insert.sh:
# a hold marker in p1/pending makes p2_host stand down after its current job,
# we take gpu0.lock ourselves, and both are released on any exit path.
set -eo pipefail

QROOT=/home/kojiek/gpu_queue
MARKER="$QROOT/p1/pending/000_hold_for_negprompt_ablation.sh"
PY=/home/kojiek/venvs/dac/bin/python
LOCK_STATUS=$(mktemp /tmp/negprompt_gpu_lock_status.XXXXXX)
LOCK_PID=""
LOG=/home/kojiek/logs/negprompt_ablation_matrix.log

cleanup() {
  [ -n "$LOCK_PID" ] && kill "$LOCK_PID" 2>/dev/null || true
  rm -f "$MARKER" "$LOCK_STATUS"
  echo "RELEASED p2 queue marker and gpu0.lock $(date -u +%FT%TZ)" | tee -a "$LOG"
}
trap cleanup EXIT INT TERM

cat > "$MARKER" <<'MARKER_BODY'
#!/bin/bash
# Hold marker, not a real job. Present only to keep p2_host idle while the
# negative-prompt ablation matrix occupies the GPU. Removed automatically by
# MeanAudio/scripts/eval/negprompt_ablation_queue_insert.sh when the sweep exits.
echo "hold marker for negprompt re-eval sweep; not runnable" >&2
exit 1
MARKER_BODY
chmod +x "$MARKER"
echo "HOLD_MARKER placed $MARKER $(date -u +%FT%TZ)" | tee -a "$LOG"

# Block until whatever P2 is currently running releases the GPU.
"$PY" "$QROOT/hold_lock.py" --block --watch-pid $$ "$QROOT/gpu0.lock" > "$LOCK_STATUS" &
LOCK_PID=$!
echo "WAITING for gpu0.lock $(date -u +%FT%TZ)" | tee -a "$LOG"
while ! grep -q LOCKED "$LOCK_STATUS" 2>/dev/null; do
  if ! kill -0 "$LOCK_PID" 2>/dev/null; then
    echo "FAIL hold_lock died before acquiring gpu0.lock" >&2
    exit 2
  fi
  sleep 5
done
echo "GPU_ACQUIRED $(date -u +%FT%TZ)" | tee -a "$LOG"

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
cd /home/kojiek/MeanAudio
python scripts/eval/negprompt_ablation_matrix.py "$@" 2>&1 | tee -a "$LOG"
