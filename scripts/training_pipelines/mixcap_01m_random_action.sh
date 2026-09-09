#!/bin/bash
# Cross-captioner true-random rotation: c2p0 Qwen slot0 / slot1 / Music Flamingo.
#
# What this asks. Every rotation arm so far rotated over captions from ONE
# captioner: 013 (slot0/1/3, full, CFG0 CLAP 0.2221) and 012 (slot0/1/2,
# quarter, 0.2053) are both all-Qwen. 034 then showed that WHICH Qwen slots you
# rotate over is seed noise on all five metrics, i.e. the three Qwen captions of
# a clip are near-interchangeable. So the open question is whether rotation gains
# anything when one of the three captions is written by a different captioner
# with a different failure profile. MF alone loses to Qwen alone (full 0.2078 vs
# slot0 0.2149; paired59k put the captioner-only delta at CLAP +0.0073 for Qwen
# with all four AES inside the seed floor), so a naive read predicts the mixed
# pool lands below the all-Qwen pool. If it does not, the rotation is buying
# captioner diversity rather than caption count.
#
# The comparison is a SINGLE SUBSTITUTION at both scales:
#   quarter  arm slot0/slot1/MF  vs  control 012 slot0/slot1/slot2   0.2053
#   full     arm slot0/slot1/MF  vs  control 013 slot0/slot1/slot3   0.2221
# The third slot differs between the two controls (slot2 vs slot3); 034 licenses
# treating that as noise, and it is recorded as a caveat rather than assumed away.
#
# Nothing is encoded. Positions 0/1 are indices 0/1 of the existing 013 stack and
# position 2 is the mf_dedup single-caption overlay; all three share encoder
# fingerprint 27e88fac... and cover the same 251,599 cache-list rows in the same
# order, so the pool is assembled at load time (0 new bytes).
#
# MF corpus choice: mf_dedup, not mf_fullcov. mf_dedup is the corpus that won the
# 040/042 decision and received the full budget; mf_fullcov's 27,264 duplicate
# captions were regenerated there (unique rate 0.9845 vs slot0's 1.0000).
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env

SCALE="${1:?usage: mixcap_01m_random_action.sh <quarter|full>}"
case "$SCALE" in
  quarter) S1_UPDATES=100000; S2_ADD=50000  ;;
  full)    S1_UPDATES=400000; S2_ADD=200000 ;;
  *) echo "FAIL unknown scale: $SCALE" >&2; exit 2 ;;
esac

PREFIX="mixcap_01m_random_noq_${SCALE}"
CONTRACT="/home/kojiek/MeanAudio/docs/experiments/mixcap_01m_random_${SCALE}_cfg0_contract.json"
TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_k3_true_fake_random/k3_true_random_train.tsv
TSV_SHA=5ec90b0f8d963df50546730384446bdca1b185ee4b2e21a4094cf60398b39999
POOL=/home/kojiek/MeanAudio/docs/experiments/mixcap_01m_caption_pool.json
POOL_SHA=d2802d386f8d5954f34ade711bb71df55101ed9d74b5897b28ccb1f987cfe1e5
OVERLAY=/home/kojiek/text_overlays/true_random   # pool positions 0 and 1
MF_OVERLAY=/home/kojiek/text_overlays/mf_dedup   # pool position 2
MF_TSV=/home/kojiek/exps_nvme/mf_dedup/arm_inputs/mf_dedup_train.tsv
SLOT1_TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/phase8_caption2p0_slot1_train.tsv
QUARTER_REPORT=/home/kojiek/cfg0_eval_runtime/reports/mixcap_01m_random_noq_quarter_musiccaps_mf25_cfg0_noq_REPORT.json
OVERRIDE=/home/kojiek/exps_nvme/mixcap_01m/PROCEED_TO_FULL_ANYWAY
GATE=0.1969   # 012 quarter control 0.2053 minus 2x the CFG0 training-seed floor (0.0042)

# --- Step 0: early kill, full only ------------------------------------------
# The queue has no dependency mechanism (lib_scheduler.py orders by filename
# only), so the gate lives here, not in the contract. This is the failure mode
# 037 hit: its predecessor's number said cancel and it was seated anyway.
if [ "$SCALE" = "full" ] && [ ! -f "$OVERRIDE" ]; then
  [ -f "$QUARTER_REPORT" ] || { echo "FAIL quarter report missing: $QUARTER_REPORT" >&2; exit 5; }
  python - "$QUARTER_REPORT" "$GATE" <<'PY' || exit 5
import json, sys
clap = json.load(open(sys.argv[1]))["metrics"]["clap_score"]
gate = float(sys.argv[2])
print(f"[gate] quarter CFG0 CLAP {clap} vs gate {gate}")
if clap < gate:
    print("[gate] mixing MF into the rotation measurably hurt at quarter; "
          "the full budget is not spent. touch the override file to force.")
    raise SystemExit(1)
PY
fi

# --- preflight ---------------------------------------------------------------
[ -f "$TSV" ] || { echo "FAIL missing train tsv $TSV" >&2; exit 2; }
actual=$(sha256sum "$TSV" | cut -d' ' -f1)
[ "$actual" = "$TSV_SHA" ] || { echo "FAIL train tsv drift: $actual" >&2; exit 2; }
actual=$(sha256sum "$POOL" | cut -d' ' -f1)
[ "$actual" = "$POOL_SHA" ] || { echo "FAIL caption pool drift: $actual" >&2; exit 2; }
[ -f "$MF_TSV" ] || { echo "FAIL missing mf_dedup tsv $MF_TSV" >&2; exit 2; }
for d in "$OVERLAY" "$MF_OVERLAY"; do
  [ -f "$d/DONE.json" ] || { echo "FAIL overlay not complete: $d" >&2; exit 2; }
  python - "$d" <<'PY'
import json, sys
done = json.load(open(f"{sys.argv[1]}/DONE.json"))
assert done["status"] == "passed", done
assert done["rows"] == 251599, done
assert done["text_encoder_fingerprint"] == "27e88fac68d94a8a10e44d2db930a8f79db8ca0454ce996b82e448c48c40ab4c", done
PY
done
echo "[OK] preflight: tsv sha bound, both overlays complete, same encoder, 251599 rows each"

# Byte-level proof that pool position k really holds slot k's caption, against
# each position's own ground-truth TSV. This is the guard Phase 9 lacked; the MF
# position is the one that has never been in a pool before.
python scripts/preprocess/validate_composed_text_overlay.py \
  --tsv "$TSV" --gt-cache "$CACHE" \
  --source "$OVERLAY:0" --source "$OVERLAY:1" --source "$MF_OVERLAY" \
  --slot-tsv "1:$SLOT1_TSV" \
  --slot-tsv "2:$MF_TSV" \
  --samples 2000 --epochs 4 \
  --report "/home/kojiek/logs/${PREFIX}_pool_validation.json"

# require_text_overlay=true, so extracted_audio.py re-checks clip_id and caption
# membership on every row loaded during training, not just the sampled ones.
post_k5_train "$PREFIX" "$TSV" "$OVERLAY" "null" "$S1_UPDATES" "$S2_ADD" false true "sources:$POOL"

CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_noq "$EVAL" \
  "${PREFIX}_musiccaps_mf25_cfg0_noq" "$POST_K5_EMA" --no_q
