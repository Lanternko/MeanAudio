#!/bin/bash
# Full-scale entry point for the c2p0 slot4 (no-digits) line.
#
# Same shared action as the quarter arm, only the budget changes: S1 400k / S2
# 200k, which is the budget every other c2p0 single-slot full arm was trained at
# (slot0, slot2, slot3). The corpus, overlay, seed, lr, batch and both eval
# cells are whatever Step 1-7 of the shared action already do.
#
# The quarter arm came in a tie (CFG0 CLAP 0.2050 vs slot0 0.2029, 0.5x the
# training-seed floor), so this arm exists to answer the same question at the
# scale the board's headline numbers live at, not because the quarter cleared a
# gate.
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/caption2p0_slot4_action.sh"
EXPECTED="4e15b289a20586d62d02507b5d283bc7d65e566aed1fdace6e81986c57a858a7"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" full
