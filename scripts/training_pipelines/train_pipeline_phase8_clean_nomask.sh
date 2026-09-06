#!/usr/bin/env bash
# Clean Phase 8 baseline control.
#
# Keeps the corrected NoQ null-token and CFG training paths, the canonical
# TSV-to-audio mapping, and the freshly verified LP-MC text embeddings. The
# sole rollback is text padding behavior: no attention mask is passed during
# either training or evaluation, reproducing the historical all-77-token path.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export EXP_PREFIX=phase8_clean_nomask
export USE_TEXT_ATTENTION_MASK=false

exec bash "$SCRIPT_DIR/train_pipeline_phase8_bugfix_rerun.sh"
