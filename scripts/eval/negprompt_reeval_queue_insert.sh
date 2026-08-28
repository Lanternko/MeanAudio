#!/bin/bash
# Insert the negative-prompt re-eval sweep ahead of the P2 pending queue.
#
# Operator decision 2026-08-29: run this right after arm 024 finishes, before
# 001_modular_template_quarter / 025_true_random_full / 026_fake_random_full.
# Reason: 025/026's decision rule is written in CFG 0 terms, so knowing whether
# the arm ordering survives the negative-prompt protocol is worth ~a day of
# queue delay.
#
# How the insertion works, without touching P2's own bookkeeping:
#   * p2_host.sh stands down whenever lib_scheduler.p1_has_work() is true, which
#     is just "any *.sh in p1/pending or p1/running". Dropping a hold marker
#     there makes P2 go idle after it finishes the job it is already running.
#   * We then take gpu0.lock ourselves (blocking) so nothing else can seat.
#   * On exit -- success, failure, or Ctrl-C -- the marker and the lock are both
#     released and P2 resumes with 001.
#
# p1_host.sh is NOT running on this machine. If someone starts it while this is
# in flight, the marker exits non-zero on purpose so it lands in p1/failed and
# is visible, rather than silently completing and letting P2 back in early.
set -eo pipefail

QROOT=/home/kojiek/gpu_queue
MARKER="$QROOT/p1/pending/000_hold_for_negprompt_reeval.sh"
PY=/home/kojiek/venvs/dac/bin/python
LOCK_STATUS=$(mktemp /tmp/negprompt_gpu_lock_status.XXXXXX)
LOCK_PID=""
LOG=/home/kojiek/logs/negprompt_reeval_full_arms.log

cleanup() {
  [ -n "$LOCK_PID" ] && kill "$LOCK_PID" 2>/dev/null || true
  rm -f "$MARKER" "$LOCK_STATUS"
  echo "RELEASED p2 queue marker and gpu0.lock $(date -u +%FT%TZ)" | tee -a "$LOG"
}
trap cleanup EXIT INT TERM

cat > "$MARKER" <<'MARKER_BODY'
#!/bin/bash
# Hold marker, not a real job. Present only to keep p2_host idle while the
# negative-prompt re-eval sweep occupies the GPU. Removed automatically by
# MeanAudio/scripts/eval/negprompt_reeval_queue_insert.sh when the sweep exits.
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
python scripts/eval/negprompt_reeval_full_arms.py "$@" 2>&1 | tee -a "$LOG"
