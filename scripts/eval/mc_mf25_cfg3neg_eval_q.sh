#!/bin/bash
# MusicCaps 5521 / MeanFlow 25 / CFG 3.0 + fidelity negative prompt, NoQ.
#
# This is the second canonical cell of the caption-arm table. The canonical CFG0
# harness (scripts/caption10s_pipeline/eval_musiccaps_mf25.sh) hard-codes cfg=0
# and refuses any other strength, so the CFG3+neg cell has always been produced
# by each arm's own action script. Every one of those copies is the same block;
# this is that block, factored out, byte-for-byte equivalent in protocol to the
# one in mf_dedup_action.sh Step 6 that produced the comparator numbers.
#
# The negative prompt string is fixed. Changing it silently invalidates every
# comparison against the existing arms, so it lives here and not in an argument.
#
# Q variant (2026-09-13): identical protocol, but for Q-trained checkpoints.
# mc_mf25_cfg3neg_eval.sh hard-codes the NoQ flag and its digest is pinned by the
# mixcap_01m contracts, so it is left untouched; this copy differs only in the
# conditioning flag, the label suffix, and the report's "conditioning" field.
#
# Usage: mc_mf25_cfg3neg_eval_q.sh <exp_name> <s2_ema.pth> <quality_level 0..9>
# Idempotent: an existing metrics.txt is reused, a partial audio dir is topped up.
set -euo pipefail

EXP="${1:?exp_name}"
CKPT="${2:?s2 ema checkpoint}"
QL="${3:?quality_level}"
[[ "$QL" =~ ^[0-9]$ ]] || { echo "FAIL quality_level must be 0..9, got $QL" >&2; exit 2; }

NEG='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
PY=/home/kojiek/venvs/dac/bin/python
MC_TSV=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
LABEL="${EXP}_mc_mf25_cfg3_neg_q${QL}"
OUT="$HOME/eval_output_nvme/$LABEL"
REPORT="$OUT/${LABEL}_REPORT.json"
LOGDIR="$HOME/logs/$LABEL"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd /home/kojiek/MeanAudio
mkdir -p "$OUT/audio" "$LOGDIR"
log(){ echo "[$(date -u +%FT%TZ)] [cfg3neg $EXP q$QL] $*"; }

[ -f "$CKPT" ] || { log "FAIL missing ckpt $CKPT"; exit 2; }
[ -f "$MC_TSV" ] || { log "FAIL missing $MC_TSV"; exit 2; }

if [ ! -f "$OUT/metrics.txt" ] && [ ! -f "$OUT/$LABEL/metrics.txt" ]; then
  HAVE=$(find "$OUT/audio" -name '*.flac' | wc -l)
  if [ "$HAVE" -lt 5400 ]; then
    log "generating (have $HAVE / 5521)"
    "$PY" eval.py --variant meanaudio_s --model_path "$CKPT" \
      --output "$OUT/audio" --tsv "$MC_TSV" --use_meanflow \
      --num_steps 25 --cfg_strength 3.0 --negative_prompt "$NEG" \
      --no_text_attention_mask --encoder_name t5_clap --text_c_dim 512 \
      --seed 42 --full_precision --quality_level "$QL" 2>&1 | tee "$LOGDIR/gen.log"
  fi
  GOT=$(find "$OUT/audio" -name '*.flac' | wc -l)
  log "generated $GOT / 5521"
  [ "$GOT" -ge 5400 ] || { log "FAIL only $GOT clips"; exit 4; }
  "$PY" "$HOME/research/meanaudio_eval/phase4_eval.py" \
    --gen_dir "$OUT/audio" --tsv "$MC_TSV" \
    --exp_name "$LABEL" --out_dir "$OUT" 2>&1 | tee "$LOGDIR/metrics.log"
else
  log "SKIP generation, metrics already present"
fi

METRICS=$(find "$OUT" -name metrics.txt | head -1)
[ -n "$METRICS" ] || { log "FAIL no metrics.txt under $OUT"; exit 2; }

"$PY" - "$REPORT" "$METRICS" "$CKPT" "$MC_TSV" "$LABEL" "$NEG" "$QL" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

report, metrics, ckpt, tsv = map(Path, sys.argv[1:5])
label, neg, ql = sys.argv[5:8]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
vals = {}
for line in metrics.read_text().splitlines():
    if ":" not in line:
        continue
    key, raw = (part.strip() for part in line.split(":", 1))
    if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
        vals[key] = float(raw)
if len(vals) != 5 or not all(math.isfinite(x) for x in vals.values()):
    raise SystemExit(f"FAIL incomplete metrics {vals}")
payload = {
    "schema_version": 1, "status": "passed", "label": label,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol": "MusicCaps 5521; MeanFlow 25; CFG 3.0 + fidelity negative; seed 42; NoMask; full precision",
    "cfg_strength": 3.0, "negative_prompt": neg, "num_steps": 25, "seed": 42,
    "conditioning": f"quality_level_{ql}",
    "checkpoint": str(Path(os.path.realpath(ckpt))), "checkpoint_sha256": sha(ckpt),
    "tsv": str(tsv), "tsv_sha256": sha(tsv),
    "metrics": vals, "metrics_path": str(metrics), "metrics_sha256": sha(metrics),
}
report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload["metrics"], indent=2))
PY
log "report written: $REPORT"
