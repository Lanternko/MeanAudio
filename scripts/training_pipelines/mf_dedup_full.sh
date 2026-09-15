#!/bin/bash
# Thin per-scale entry point. accept_guest hashes commands.run[-1], so the scale
# cannot ride as a trailing argument; this wrapper carries it and re-verifies the
# shared action's digest before handing over, keeping the hash binding meaningful.
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/mf_dedup_action.sh"
EXPECTED="183a80fc44500d6474553e73fa8e10c964b3de344899f7b2677c2d9a30914705"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" full
