#!/usr/bin/env bash
# ============================================================
# Phase 8 — catalog-matched true NoQ control
# train_pipeline_phase8_catalog_matched_noq.sh
#
# Purpose (P1, paper Table-1 Q contribution):
#   Same audio/text pairing as phase8_legacy_repro (original extraction
#   catalog), same legacy NoMask path, but true NoQ with bug-fixed
#   q=None→10 null token.
#
# Controlled variable vs the completed phase8_legacy_repro full-Q control:
#   - full-Q control: S1=true, S2=true, eval=quality_level 9 (CLAP 0.1684)
#   - this run     : S1=false, S2=false, eval=no_q
#
# This is the clean paper baseline.  It is not an exact emulation of the old
# buggy April routing (S1 effectively q=10, S2 uncond effectively q=9).
#
# Data / gates reuse the provenance-backed legacy cache; do not rebuild.
# Estimated wall clock: ~19–20 h (S1 400k + S2 200k + MusicCaps eval).
#
# Usage:
#   tmux new -s p8_catalog_noq
#   cd ~/MeanAudio && source ~/venvs/dac/bin/activate
#   bash scripts/training_pipelines/train_pipeline_phase8_catalog_matched_noq.sh
# ============================================================

set -euo pipefail

WORK_DIR=/home/kojiek/MeanAudio
DATA_DIR=/mnt/HDD/kojiek/phase4_jamendo_data
LEGACY_NPZ=/mnt/HDD/kojiek/phase8_legacy_matched_npz
LEGACY_TSV="$DATA_DIR/phase8_legacy_catalog_train.tsv"
LEGACY_MANIFEST="$LEGACY_NPZ/MANIFEST.tsv"
LEGACY_VALIDATION="$LEGACY_NPZ/FULL_VALIDATION.json"
LEGACY_GATE="$LEGACY_NPZ/FULL_GATE_PASSED.json"

if [ ! -f "$LEGACY_TSV" ] || [ ! -f "$LEGACY_MANIFEST" ] || \
   [ ! -f "$LEGACY_VALIDATION" ] || [ ! -f "$LEGACY_GATE" ]; then
    echo "[STOP] Provenance-backed legacy cache missing." >&2
    echo "  Need: $LEGACY_TSV + $LEGACY_NPZ/{MANIFEST.tsv,FULL_VALIDATION.json,FULL_GATE_PASSED.json}" >&2
    exit 2
fi

python - "$LEGACY_VALIDATION" "$LEGACY_GATE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

validation_path, gate_path = map(Path, sys.argv[1:])
validation = json.loads(validation_path.read_text())
gate = json.loads(gate_path.read_text())
if validation.get("status") != "passed" or validation.get("expected_rows") != 251599:
    raise SystemExit("[STOP] full structural validation sentinel is invalid")
if gate.get("status") != "passed" or gate.get("decoded_samples") != 512:
    raise SystemExit("[STOP] full semantic cache gate is invalid")
if gate.get("decoded_cache_clap", 0.0) < gate.get("minimum_clap", 1.0):
    raise SystemExit("[STOP] decoded-cache CLAP gate did not pass")
actual = hashlib.sha256(validation_path.read_bytes()).hexdigest()
if gate.get("validation_report_sha256") != actual:
    raise SystemExit("[STOP] validation report changed after semantic gate")
print(
    f"[OK] full structural + semantic gate: "
    f"rows={validation['expected_rows']:,}, "
    f"decoded CLAP={gate['decoded_cache_clap']:.4f}"
)
PY

export EXP_PREFIX="${EXP_PREFIX:-phase8_catalog_matched_noq}"
export TRAIN_TSV="$LEGACY_TSV"
export GT_CACHE="$DATA_DIR/npz_cache_train.txt"
export SINGLECAP_NPZ="$LEGACY_NPZ"
export EXPECTED_ROWS=251599
export EXPERIMENT_REGIME=clean_noq
export EXPERIMENT_RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
export S1_USE_Q_CONDITIONING="${S1_USE_Q_CONDITIONING:-false}"
export S2_USE_Q_CONDITIONING="${S2_USE_Q_CONDITIONING:-false}"
export EVAL_Q_MODE="${EVAL_Q_MODE:-no_q}"
export USE_TEXT_ATTENTION_MASK=false
export RUN_PRIMARY_EVAL=true
export RUN_JAMENDO_EVAL="${RUN_JAMENDO_EVAL:-false}"

echo "======================================================"
echo "  Phase 8 catalog-matched true NoQ control"
echo "  EXP_PREFIX : $EXP_PREFIX"
echo "  run mode   : $EXPERIMENT_RUN_MODE"
echo "  NPZ        : $SINGLECAP_NPZ"
echo "  TSV        : $TRAIN_TSV"
echo "  S1 Q cond  : $S1_USE_Q_CONDITIONING"
echo "  S2 Q cond  : $S2_USE_Q_CONDITIONING"
echo "  Eval Q     : $EVAL_Q_MODE (bug-fixed null token q=10)"
echo "  text mask  : false (legacy NoMask path)"
echo "  vs control : phase8_legacy_repro full-Q q9 CLAP=0.1684"
echo "======================================================"

exec bash "$WORK_DIR/scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh"
