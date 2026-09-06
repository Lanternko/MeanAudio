#!/bin/bash
# Caption 2.0 true-random rotation over slots 0/1/2, quarter scale (S1 100k + S2 50k), NoQ.
#
# Why 012 and not the existing 013 pool: 012 is the only slot set whose every
# constituent has a budget-matched quarter number already on the board
# (slot0 0.2029 / slot1 0.2047 / slot2 0.2017, plus best-of-3 0.2129 and
# worst-of-3 0.1957). This arm therefore answers the one question the 013
# rotation arms cannot -- does per-epoch rotation beat each caption it rotates
# over, at equal budget -- because slot3 was never trained on its own.
#
# No new overlay is encoded. slot0/slot1 are indices 0/1 of the existing 013
# stack and slot2 is its own single-caption overlay; both carry text encoder
# fingerprint 27e88fac..., so the pool is assembled at load time. A duplicate
# 225 GB stack would not fit anyway (142 GB free).
#
# Caveat inherited from the 013 quarter pair: S1 100k is 3.18 epochs, so caption
# rotation reaches only 2.19/3 coverage and a regulariser is being measured in an
# undertrained regime. The comparison is still internally budget-matched.
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env

CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_true012_random_quarter_cfg0_contract.json
PREFIX=phase8_qwen_caption2p0_k3_true012_random_noq_quarter
TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_k3_true_fake_random/k3_true_random_train.tsv
TSV_SHA=5ec90b0f8d963df50546730384446bdca1b185ee4b2e21a4094cf60398b39999
POOL=/home/kojiek/MeanAudio/docs/experiments/caption2p0_true012_caption_pool.json
POOL_SHA=1fbe04b96af8fe2c627a734e87322aade485b3601b909cc82674e59b64a0c96c
OVERLAY=/home/kojiek/text_overlays/true_random          # pool position 0 and 1
SLOT2_OVERLAY=/home/kojiek/text_overlays/slot2          # pool position 2
BASELINE_NOTE="budget-matched quarter arms: slot0 0.2029 / slot1 0.2047 / slot2 0.2017 / best-of-3 0.2129 / worst-of-3 0.1957"

# --- preflight -------------------------------------------------------------
[ -f "$TSV" ] || { echo "FAIL missing train tsv $TSV" >&2; exit 2; }
actual=$(sha256sum "$TSV" | cut -d' ' -f1)
[ "$actual" = "$TSV_SHA" ] || { echo "FAIL train tsv drift: $actual" >&2; exit 2; }
actual=$(sha256sum "$POOL" | cut -d' ' -f1)
[ "$actual" = "$POOL_SHA" ] || { echo "FAIL caption pool drift: $actual" >&2; exit 2; }
for d in "$OVERLAY" "$SLOT2_OVERLAY"; do
  [ -f "$d/DONE.json" ] || { echo "FAIL overlay not complete: $d" >&2; exit 2; }
  python - "$d" <<'PY'
import json, sys
done = json.load(open(f"{sys.argv[1]}/DONE.json"))
assert done["status"] == "passed", done
assert done["text_encoder_fingerprint"] == "27e88fac68d94a8a10e44d2db930a8f79db8ca0454ce996b82e448c48c40ab4c", done
PY
done
echo "[OK] preflight: tsv sha bound, both overlays complete and same-encoder"

# Byte-level check that pool position k really holds slot k's caption for every
# sampled row. This is the guard that the Phase 9 multi-cap runs did not have.
python scripts/preprocess/validate_composed_text_overlay.py \
  --tsv "$TSV" --gt-cache "$CACHE" \
  --source "$OVERLAY:0" --source "$OVERLAY:1" --source "$SLOT2_OVERLAY" \
  --slot-tsv 1:/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/phase8_caption2p0_slot1_train.tsv \
  --slot-tsv 2:/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/phase8_caption2p0_slot2_train.tsv \
  --samples 2000 --epochs 4 \
  --report "/home/kojiek/logs/${PREFIX}_pool_validation.json"

# post_k5_train sets ++require_text_overlay=true, so extracted_audio.py re-checks
# clip_id and caption membership on every single row it loads during training.
post_k5_train "$PREFIX" "$TSV" "$OVERLAY" "null" 100000 50000 false true "sources:$POOL"

CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_noq "$EVAL" \
  "${PREFIX}_musiccaps_mf25_cfg0_noq" "$POST_K5_EMA" --no_q
