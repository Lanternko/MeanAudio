#!/bin/bash
# Thin per-scale entry point. accept_guest hashes commands.run[-1], so the scale
# cannot ride as a trailing argument; this wrapper carries it and re-verifies the
# shared action's digest before handing over, keeping the hash binding meaningful.
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/mixcap_01m_random_action.sh"
EXPECTED="$(printf %s 48e08857b596713417ca27e08168e5a39d171234bcf58689c1aed1d6f949151e)"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" full
