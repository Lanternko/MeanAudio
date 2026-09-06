#!/usr/bin/bash
# Zero-GPU migration hook for the three exact fair013 evaluator call tuples.
# Every non-zero coordinator outcome remains at this seam; the set -e parent is
# released only after one exact contract and its hold event are durable.
set -uo pipefail

readonly COORDINATOR=/home/kojiek/MeanAudio/scripts/eval/register_slot3_fair013_cfg0.py
readonly PYTHON=/usr/bin/python3.12
readonly ENV=/usr/bin/env
readonly SLEEP=/usr/bin/sleep
readonly CALLER_PID="$PPID"
readonly HOOK_PID="$$"

while :; do
  "$ENV" -i \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/usr/bin:/bin \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    "$PYTHON" "$COORDINATOR" \
      --hook-pid "$HOOK_PID" \
      --caller-pid "$CALLER_PID" \
      -- "$@"
  coordinator_status=$?
  if [ "$coordinator_status" -eq 0 ]; then
    exit 0
  fi
  "$SLEEP" 5
done
