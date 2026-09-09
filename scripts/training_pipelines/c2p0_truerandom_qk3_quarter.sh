#!/bin/bash
# Thin per-arm entry point. accept_guest hashes commands.run[-1], so the arm
# cannot ride as a trailing argument; this wrapper carries it and re-verifies the
# shared action's digest before handing over, keeping the hash binding meaningful.
set -eo pipefail
SHARED="/home/kojiek/MeanAudio/scripts/training_pipelines/c2p0_truerandom_q_action.sh"
EXPECTED="$(printf %s 1fb4017ac6332b505792cfd41c75f07e0fb8ab8324c44950998d0283ddd73c57)"
ACTUAL=$(sha256sum "$SHARED" | cut -d' ' -f1)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "[FAIL] shared action digest mismatch: $ACTUAL != $EXPECTED"; exit 2
fi
exec /bin/bash "$SHARED" k3
