#!/bin/bash
# Thin per-scale entry point. accept_guest hashes commands.run[-1], so the scale
# cannot ride as a trailing argument; this wrapper carries it and re-verifies the
# shared action's digest before handing over, keeping the hash binding meaningful.
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/mf_fullcov_action.sh"
EXPECTED="fe04f68e1eb12f495e17f8989d77d09adda5f42718dd138182bd97d7e0c5de30"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" quarter
