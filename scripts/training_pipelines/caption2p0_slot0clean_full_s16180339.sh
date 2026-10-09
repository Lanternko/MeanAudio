#!/bin/bash
# Thin entry point for p2 117. accept_guest hashes commands.run[-1].
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/caption2p0_slot0clean_full_seed_action.sh"
EXPECTED="797e225b38ef7609ac614acbcfaa920507f84524b9541c8605f69ae3b052c0f3"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" 16180339
