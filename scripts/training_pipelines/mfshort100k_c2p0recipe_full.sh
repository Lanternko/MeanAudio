#!/bin/bash
# Thin per-scale entry point. commands.run[-1] must be the action itself, so the
# scale cannot be passed as a trailing argument; this wrapper carries it and
# re-verifies the shared action's digest before handing over, keeping the
# hash binding meaningful despite the indirection.
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/mfshort100k_c2p0recipe_action.sh"
EXPECTED="b7299d975fb61246372fca15836c272747a5d7984781e8b4bbddc5b8f7986270"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" full
