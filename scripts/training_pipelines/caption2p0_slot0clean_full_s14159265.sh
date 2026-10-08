#!/bin/bash
# Thin entry point for p2 114. accept_guest hashes commands.run[-1].
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/caption2p0_slot0clean_full_action.sh"
EXPECTED="581ff62357d9b99fc85a3f1b09d40eed3df2c0056dce01c463f49bb4956fb12a"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED"
