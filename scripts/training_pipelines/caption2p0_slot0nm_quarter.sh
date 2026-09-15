#!/bin/bash
# Thin per-scale entry point. accept_guest hashes commands.run[-1].
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/caption2p0_slot0nm_action.sh"
EXPECTED="d6383a8ebb097016c61f71238f3d177910eb59f459a96c52d87a86fc0c9a59c4"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" quarter
