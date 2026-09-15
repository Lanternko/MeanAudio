#!/bin/bash
# Thin per-scale entry point. accept_guest hashes commands.run[-1].
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/caption2p0_slot0clean_action.sh"
EXPECTED="ecee47a862d9a203991c3c0760012f585f96fc918c2c7f55889785286b1c41f5"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" quarter
