#!/bin/bash
# Thin per-scale entry point. accept_guest hashes commands.run[-1].
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/caption2p0_slot4_action.sh"
EXPECTED="4e15b289a20586d62d02507b5d283bc7d65e566aed1fdace6e81986c57a858a7"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" quarter
