#!/bin/bash
# Thin per-scale entry point. accept_guest hashes commands.run[-1].
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/caption2p0_slot4v2_action.sh"
EXPECTED="fccb2df452d04f5286dbd1ac094b3ce7850bd9dea57bd9c9467e052af886c914"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" quarter
