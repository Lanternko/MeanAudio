#!/usr/bin/env bash
# Reproduce the dataset/configuration actually consumed by historical Phase 8.
#
# This is a legacy-result reproduction, not a clean NoQ baseline:
#   * audio/text pairs come from the original extraction catalog for each NPZ;
#   * q_level comes from the historical Phase-7 row paired to that filename;
#   * q conditioning is enabled because the April runner ignored
#     use_q_conditioning=false whenever q_level existed;
#   * the legacy 77-token NoMask path is used.
#
# The full training pipeline is deliberately gated on a provenance-backed cache.

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
    echo "[STOP] Full provenance-backed legacy cache has not been built."
    echo "Use the guarded build/validation/launch runner:"
    echo "  bash $WORK_DIR/scripts/training_pipelines/run_phase8_legacy_guarded.sh"
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

export EXP_PREFIX="${EXP_PREFIX:-phase8_legacy_repro}"
export TRAIN_TSV="$LEGACY_TSV"
export GT_CACHE="$DATA_DIR/npz_cache_train.txt"
export SINGLECAP_NPZ="$LEGACY_NPZ"
export EXPECTED_ROWS=251599
export USE_Q_CONDITIONING=true
export USE_TEXT_ATTENTION_MASK=false
export RUN_PRIMARY_EVAL=true
export RUN_JAMENDO_EVAL="${RUN_JAMENDO_EVAL:-false}"

exec bash "$WORK_DIR/scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh"
