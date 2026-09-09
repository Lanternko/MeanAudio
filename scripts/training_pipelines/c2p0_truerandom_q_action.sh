#!/bin/bash
# Q-resolution ablation on top of the 013 true-random caption rotation.
#
# What this asks. Every Q arm this project has trained conditioned on a single
# caption per clip. Every rotation arm trained without Q. The two knobs have
# never been crossed, so we do not know whether a per-clip quality code still
# carries signal when the caption attached to that clip changes every epoch --
# or whether rotation already supplies whatever regularisation Q was providing.
#
# The ablation is the RESOLUTION of the code, not its presence:
#   K=3  balanced -> q codes {0,5,9}, 83,866 / 83,866 / 83,867 rows
#   K=10 balanced -> q codes {0..9},  25,159-25,160 rows each
# Both come from the 2026-07-24 bucket grid, unchanged: same signal (actual-clip
# credibility_analysis.mean_similarity), same q_code_policy
# round-half-up(index*9/(K-1)) with the endpoints pinned at q0/q9, same 251,599
# rows. Only the number of cuts differs, so a difference between the two arms is
# attributable to resolution and nothing else.
#
# Why balanced and not fixed. `fixed` is equal-width on [0,1] and the signal is
# not uniform there: k3_fixed leaves q0 with ZERO rows and k10_fixed occupies
# only 7 of its 10 codes. Its occupancy therefore collapses by a different
# amount at each K, which is exactly the confound a K ablation must not carry.
# `balanced` is equal-frequency, so both arms are fully occupied and K is the
# only thing that moves. (k5_balanced and k5_fixed were run against each other
# on the single-caption corpus for precisely this reason; that stays the
# diagnostic, not part of this claim.)
#
# Comparators, all quarter scale, all MusicCaps 5521 / MeanFlow 25 / CFG 0:
#   013 true-random NoQ quarter    -- produced by Step 0 of the K=3 job below
#   012 true-random NoQ quarter    0.2053  (different third slot; 034 says noise)
#   c2p0 slot0 + Q k3_balanced     0.2126 at FULL, single caption, not rotated
#
# Known caveat, inherited and not fixable inside this arm: S1-Q never trains
# q_embed[10]. Stage 2's MeanFlow CFG target uses q=10 as its unconditional
# code, so both arms carry that same untrained-null contamination. It is a
# shared constant across K=3 and K=10 and so does not confound the comparison,
# but it does mean neither arm's absolute number is comparable to a NoQ arm's
# without that caveat attached.
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env

ARM="${1:?usage: c2p0_truerandom_q_action.sh <k3|k10>}"
case "$ARM" in
  k3)  GRID_ARM=k3_balanced;  TSV_SHA=698fde1a4c37204a2b3a6370fa4a72ef2a2e6534f2efefbeddce9282436acc56  ;;
  k10) GRID_ARM=k10_balanced; TSV_SHA=b126c2a830929b3f622f5ee8f2bce97dc31fbdd0d1f4a8ee4529089700da68ed ;;
  *) echo "FAIL unknown arm: $ARM" >&2; exit 2 ;;
esac

S1_UPDATES=100000
S2_ADD=50000
PREFIX="phase8_qwen_caption2p0_k3_true_random_q${ARM}b_quarter"
CONTRACT="/home/kojiek/MeanAudio/docs/experiments/c2p0_truerandom_q${ARM}_quarter_cfg0_contract.json"
TSV="/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_k3_true_fake_random/k3_true_random_train_q${GRID_ARM}.tsv"
MANIFEST=/home/kojiek/MeanAudio/docs/experiments/c2p0_truerandom_q_tsvs.manifest.json
MANIFEST_SHA=c798f97b5110139fff8d99faf3f0d2bbf6caf3f3f6a831897192207232dcca3e
OVERLAY=/home/kojiek/text_overlays/true_random

# --- Step 0 (K=3 job only): the missing same-scale NoQ control ---------------
# The 013 true-random NoQ arm was trained at quarter scale in August but only
# ever evaluated at full (0.2221). Both Q arms are quarter, so the honest
# comparator is the quarter NoQ number, and it does not exist yet. No training
# here -- the EMA has been on disk since 2026-08-26.
if [ "$ARM" = "k3" ]; then
  CTRL_PREFIX=phase8_qwen_caption2p0_k3_true_random_noq_quarter
  CTRL_EMA="/home/kojiek/MeanAudio/exps/${CTRL_PREFIX}_stage2_50000/${CTRL_PREFIX}_stage2_50000_ema_final.pth"
  [ -f "$CTRL_EMA" ] || { echo "FAIL missing NoQ quarter control EMA: $CTRL_EMA" >&2; exit 2; }
  CFG0_CONTRACT="$CONTRACT" CFG0_ARM=noq_quarter_control "$EVAL" \
    "${CTRL_PREFIX}_musiccaps_mf25_cfg0_noq" "$CTRL_EMA" --no_q
fi

# --- preflight ---------------------------------------------------------------
[ -f "$TSV" ] || { echo "FAIL missing train tsv $TSV" >&2; exit 2; }
actual=$(sha256sum "$TSV" | cut -d' ' -f1)
[ "$actual" = "$TSV_SHA" ] || { echo "FAIL train tsv drift: $actual" >&2; exit 2; }
actual=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
[ "$actual" = "$MANIFEST_SHA" ] || { echo "FAIL q tsv manifest drift: $actual" >&2; exit 2; }
[ -f "$OVERLAY/DONE.json" ] || { echo "FAIL overlay not complete: $OVERLAY" >&2; exit 2; }
python - "$OVERLAY" "$MANIFEST" "$GRID_ARM" "$TSV" <<'PY'
import csv, json, sys
from collections import Counter

overlay, manifest_path, grid_arm, tsv = sys.argv[1:5]
done = json.load(open(f"{overlay}/DONE.json"))
assert done["status"] == "passed", done
assert done["rows"] == 251599, done
assert done["text_encoder_fingerprint"] == \
    "27e88fac68d94a8a10e44d2db930a8f79db8ca0454ce996b82e448c48c40ab4c", done

manifest = json.load(open(manifest_path))
assert manifest["status"] == "passed", manifest["status"]
spec = manifest["outputs"][grid_arm]
assert spec["path"] == tsv, (spec["path"], tsv)

with open(tsv, encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    assert reader.fieldnames == ["id", "caption", "q_level"], reader.fieldnames
    hist = Counter(row["q_level"] for row in reader)
assert sum(hist.values()) == 251599, sum(hist.values())
assert dict(hist) == spec["q_histogram"], (dict(hist), spec["q_histogram"])
# Occupancy is the whole point of choosing `balanced`: every registered code
# must actually carry rows, or K is not what it says it is.
codes = sorted(int(code) for code in hist)
assert codes == spec["occupied_q_codes"], (codes, spec["occupied_q_codes"])
assert min(codes) == 0 and max(codes) == 9, codes
print(f"[OK] {grid_arm}: {len(codes)} occupied q codes {codes}, "
      f"min bucket {min(hist.values())} rows")
PY
echo "[OK] preflight: tsv+manifest sha bound, overlay complete, q occupancy verified"

# use_q=true, multi_cap=true: the per-epoch rotation over the 3-caption stack is
# unchanged from 025/021, and q_level rides along as a per-CLIP code (it is a
# property of the clip's caption-set agreement, not of the rotated caption).
post_k5_train "$PREFIX" "$TSV" "$OVERLAY" "null" "$S1_UPDATES" "$S2_ADD" true true

# The canonical harness requires a Q arm to report at least q0 and q9. q9 is the
# primary cell (it is the code every historical Q number was read at); q0 is the
# response probe -- if q9 and q0 land inside the seed floor of each other, the
# code was ignored regardless of how many cuts it had.
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_q9 "$EVAL" \
  "${PREFIX}_musiccaps_mf25_cfg0_q9" "$POST_K5_EMA" --quality_level 9
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_q0 "$EVAL" \
  "${PREFIX}_musiccaps_mf25_cfg0_q0" "$POST_K5_EMA" --quality_level 0
