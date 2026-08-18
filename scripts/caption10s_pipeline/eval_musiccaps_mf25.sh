#!/bin/bash
# Canonical MusicCaps 25-step eval for Caption 2.0 fair compare.
#
# Protocol (do not change without an explicit new baseline):
#   MusicCaps 5521, MeanFlow, num_steps=25, cfg=4.5, seed=42,
#   t5_clap / text_c_dim=512, no_text_attention_mask, full_precision.
#   NoQ arms: pass --no_q
#   Q arms: pass --quality_level N (report at least q0 and q9)
#
# Usage:
#   eval_musiccaps_mf25.sh <label> <s2_ema.pth> [--no_q | --quality_level N]
set -euo pipefail
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

LABEL="${1:?label}"
CKPT="${2:?s2 ema checkpoint}"
shift 2

MEANAUDIO=/home/kojiek/MeanAudio
MUSICCAPS=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
OUT="$MEANAUDIO/eval_output/$LABEL"
METRICS="$MEANAUDIO/eval_output/metrics/$LABEL/metrics.txt"
REPORT="/home/kojiek/logs/${LABEL}_REPORT.json"

[ -f "$CKPT" ] || { echo "FAIL missing ckpt $CKPT" >&2; exit 2; }
[ -f "$MUSICCAPS" ] || { echo "FAIL missing $MUSICCAPS" >&2; exit 2; }

cd "$MEANAUDIO"
if [ -f scripts/runtime/phase8_nvidia_compat_env.sh ]; then
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || true
fi

echo "EVAL_MF25 $LABEL $(date -u +%FT%TZ)"
if [ ! -f "$METRICS" ]; then
  mkdir -p "$OUT/audio"
  python eval.py --variant meanaudio_s --model_path "$CKPT" \
    --output "$OUT/audio" --tsv "$MUSICCAPS" --use_meanflow \
    --num_steps 25 --cfg_strength 4.5 \
    --encoder_name t5_clap --text_c_dim 512 --seed 42 \
    --no_text_attention_mask --full_precision \
    "$@" \
    2>&1 | tee "/home/kojiek/logs/${LABEL}_eval.log"
  python "$EVALUATOR" --gen_dir "$OUT/audio" --tsv "$MUSICCAPS" \
    --out_dir "$MEANAUDIO/eval_output/metrics" --exp_name "$LABEL" \
    --num_samples 5521 \
    2>&1 | tee -a "/home/kojiek/logs/${LABEL}_eval.log"
  rm -rf "$OUT/audio"
else
  echo "SKIP_EVAL $METRICS"
fi
[ -f "$METRICS" ] || { echo "FAIL missing $METRICS" >&2; exit 2; }

python - "$REPORT" "$METRICS" "$CKPT" "$LABEL" <<'PY'
import json, math, sys
from datetime import datetime, timezone
from pathlib import Path

report, metrics, ckpt = map(Path, sys.argv[1:4])
label = sys.argv[4]
vals = {}
for line in metrics.read_text().splitlines():
    if ":" not in line:
        continue
    key, raw = (part.strip() for part in line.split(":", 1))
    if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
        vals[key] = float(raw)
if len(vals) != 5 or not all(math.isfinite(x) for x in vals.values()):
    raise SystemExit(f"incomplete metrics {vals}")
payload = {
    "schema_version": 1,
    "status": "passed",
    "label": label,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "protocol": "MusicCaps 5521; MeanFlow 25; CFG 4.5; seed 42; NoMask; full precision",
    "fair_compare_anchor": "caption2p0_s2_mf25_cfg4p5 CLAP=0.2419 (full) / C2.0 quarter must use this same solver",
    "checkpoint": str(ckpt),
    "metrics": vals,
    "metrics_path": str(metrics),
}
report.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
