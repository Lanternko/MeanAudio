#!/bin/bash
# Caption 2.0 slot3 full (S1 400k + S2 200k), NoQ, CFG0 eval.
#
# slot3 never had a standalone overlay or quarter run. Reuse index 2 of the
# existing 013 stacked overlay (text_overlays/true_random) via cap_index_fixed=2.
# Pairing audit 2026-09-02: 251599/251599 TSV caption sha == overlay idx2.
# A dedicated 76 GB overlay would not fit (NVMe 116 GB free, queue hard-stop
# 112.5 GB; HDD 74 GB).
#
# Cold start: no slot3 quarter checkpoint exists. S1 runs 0->400k.
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env

CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_slot3_full_cfg0_contract.json
PREFIX=phase8_qwen_caption2p0_slot3_noq_full
TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/phase8_caption2p0_slot3_train.tsv
TSV_SHA=f5be85f51367be3bc50aff6d46bc68fff151475690c24363b11afef1e5521547
OVERLAY=/home/kojiek/text_overlays/true_random
ENCODER_FP=27e88fac68d94a8a10e44d2db930a8f79db8ca0454ce996b82e448c48c40ab4c

[ -f "$TSV" ] || { echo "FAIL missing train tsv $TSV" >&2; exit 2; }
actual=$(sha256sum "$TSV" | cut -d' ' -f1)
[ "$actual" = "$TSV_SHA" ] || { echo "FAIL train tsv drift: $actual" >&2; exit 2; }
[ -f "$OVERLAY/DONE.json" ] || { echo "FAIL overlay not complete: $OVERLAY" >&2; exit 2; }
python - "$OVERLAY" "$ENCODER_FP" <<'PY'
import json, sys
done = json.load(open(f"{sys.argv[1]}/DONE.json"))
assert done["status"] == "passed", done
assert done["text_encoder_fingerprint"] == sys.argv[2], done
assert done["n_caps"] == 3, done
assert done["rows"] == 251599, done
PY
echo "[OK] preflight: tsv sha bound, 013 stack overlay complete (n_caps=3)"

# Full-row guard: every TSV caption is overlay index 2 (slot3), not 0/1.
python - "$TSV" "$OVERLAY" <<'PY'
import csv, hashlib, sys
from pathlib import Path
tsv, overlay = Path(sys.argv[1]), Path(sys.argv[2])
want = {}
with tsv.open() as f:
    for row in csv.DictReader(f, delimiter="\t"):
        want[row["id"]] = hashlib.sha256(row["caption"].encode()).hexdigest()
sha_tsv = overlay.parent / "_index" / "true_random.sha.tsv"
n = match = 0
with sha_tsv.open() as f:
    for row in csv.DictReader(f, delimiter="\t"):
        n += 1
        hashes = row["caption_sha256"].split(",")
        cid = row["clip_id"]
        if cid not in want or len(hashes) != 3 or hashes[2] != want[cid]:
            raise SystemExit(f"FAIL idx2 pairing {cid}")
        match += 1
if match != n or n != len(want):
    raise SystemExit(f"FAIL pairing count match={match} overlay={n} tsv={len(want)}")
print(f"[OK] idx2 pairing {match}/{n}")
PY

free_bytes=$(df -B1 --output=avail /home/kojiek | tail -n 1 | tr -d ' ')
[ "$free_bytes" -ge 63687091200 ] || { echo "FAIL free space below 63.687 GB hard stop ($free_bytes)" >&2; exit 2; }

# overlay=true_random, always take stacked caption index 2 (slot3).
# multi_cap=false is required: cap_index_fixed cannot combine with multi_cap.
post_k5_train "$PREFIX" "$TSV" "$OVERLAY" "null" 400000 200000 false false fixed:2

python /home/kojiek/MeanAudio/scripts/experiment_harness/bind_contract_checkpoint_sha.py \
  --contract "$CONTRACT" --arm canonical_noq --checkpoint "$POST_K5_EMA"

CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_noq \
  "$EVAL" "${PREFIX}_musiccaps_mf25_cfg0_noq" "$POST_K5_EMA" --no_q
