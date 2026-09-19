#!/bin/bash
# Thin per-arm entry point. accept_guest hashes commands.run[-1].
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/caption2p0_nmv2pair_action.sh"
EXPECTED="e21e312be22655dba72a09fd465f039fd26b3afe00e86257da6820b5314f491e"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" slot0clean 27182818
