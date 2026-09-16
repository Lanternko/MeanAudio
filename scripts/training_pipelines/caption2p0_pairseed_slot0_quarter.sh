#!/bin/bash
# Thin per-arm entry point. accept_guest hashes commands.run[-1].
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/caption2p0_pairseed_action.sh"
EXPECTED="7e0cfd0755a4f6a6693ceb050d892a1de1a45b7f6435d531c6cdd99bf88532ed"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" slot0 quarter 27182818
