#!/bin/bash
# One-step (1-NFE) MusicCaps CFG0 cell for one Stage 2 checkpoint (084, 2026-09-27).
#
# Same protocol as mc_mf25_eval.sh's cfg0 cell except --num_steps 1: MusicCaps 5521 /
# MeanFlow / seed 42 / full precision / NoMask / CLAP batch 1. This is the only setting
# where MeanFlow's CFG-in-training target is the whole story (no inference-time guidance,
# one forward pass), so it reads what 084's training change put into the weights.
# mc_mf25_eval.sh is left untouched because contracts pin its digest.
#
# Usage: mc_nfe1_cfg0_eval.sh <exp_name> <s2_ema.pth> --no_q
# Output: $OUT_ROOT/<exp>_mc_nfe1_cfg0/ audio/ <label>/{metrics.txt,metrics.json,per_clip.tsv} <label>_REPORT.json
# Idempotent the same way as mc_mf25_eval.sh (report -> skip; no report -> from scratch).
# Env (tests only): OUT_ROOT, SMOKE_ROWS=N (first N csv records, label gets _smokeN).
set -euo pipefail
EXP="${1:?exp_name}"
CKPT="${2:?s2 ema checkpoint}"
[ "${3:-}" = --no_q ] || { echo "FAIL only --no_q models are supported" >&2; exit 2; }
PY=/home/kojiek/venvs/dac/bin/python
ROOT=/home/kojiek/MeanAudio
MC_TSV=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
OUT_ROOT="${OUT_ROOT:-$HOME/eval_output_nvme}"
SMOKE_ROWS="${SMOKE_ROWS:-}"
[ -f "$CKPT" ] || { echo "FAIL missing ckpt $CKPT" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

TSV="$MC_TSV" SMOKE_TAG=""
if [ -n "$SMOKE_ROWS" ]; then
  SMOKE_TAG="_smoke$SMOKE_ROWS"
  TMP_TSV="$(mktemp -d)/score.tsv"
  # cut by csv record: MusicCaps has captions that span lines inside quotes
  "$PY" - "$MC_TSV" "$TMP_TSV" "$SMOKE_ROWS" <<'PY'
import csv, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
csv.field_size_limit(10**9)
with open(src, newline='', encoding='utf-8') as f, open(dst, 'w', newline='', encoding='utf-8') as g:
    r, w = csv.reader(f, delimiter='\t'), csv.writer(g, delimiter='\t', lineterminator='\n')
    w.writerow(next(r))
    for i, row in enumerate(r):
        if i == n:
            break
        w.writerow(row)
PY
  TSV="$TMP_TSV"
fi

LABEL="${EXP}_mc_nfe1_cfg0${SMOKE_TAG}"
OUT="$OUT_ROOT/$LABEL"
REPORT="$OUT/${LABEL}_REPORT.json"
log(){ echo "[$(date -u +%FT%TZ)] [nfe1 $LABEL] $*"; }
if [ -f "$REPORT" ]; then log "SKIP report exists: $REPORT"; exit 0; fi
case "$OUT" in "$OUT_ROOT"/*_mc_nfe1_cfg0*) ;; *) log "FAIL refusing output path $OUT"; exit 2 ;; esac
[ -d "$OUT/audio" ] && { log "no report: discarding partial audio"; rm -rf -- "$OUT/audio"; }
mkdir -p "$OUT/audio"
log "generate 1 step cfg 0 --no_q (log: $OUT/gen.log)"
"$PY" eval.py --variant meanaudio_s --model_path "$CKPT" \
  --output "$OUT/audio" --tsv "$TSV" --use_meanflow \
  --num_steps 1 --cfg_strength 0 \
  --no_text_attention_mask --encoder_name t5_clap --text_c_dim 512 \
  --seed 42 --full_precision --no_q > "$OUT/gen.log" 2>&1 \
  || { log "FAIL eval.py exited $? (tail of $OUT/gen.log follows)"; tail -20 "$OUT/gen.log"; exit 3; }
log "generated $(find "$OUT/audio" -name '*.flac' | wc -l) clips"
"$PY" scripts/eval/eval_metrics.py --gen_dir "$OUT/audio" --tsv "$TSV" \
  --exp_name "$LABEL" --out_dir "$OUT" 2>&1 | tee "$OUT/metrics.log"
"$PY" - "$REPORT" "$OUT/$LABEL/metrics.json" "$CKPT" "$LABEL" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
report, metrics, ckpt = map(Path, sys.argv[1:4])
label = sys.argv[4]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
m = json.loads(metrics.read_text())
payload = {
    "schema_version": 2, "status": "passed", "label": label,
    "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "protocol": f"MusicCaps {m['n_rows']}; MeanFlow 1 step; CFG 0; seed 42; NoMask; full precision; CLAP batch 1",
    "cfg_strength": 0.0, "negative_prompt": None, "num_steps": 1, "seed": 42,
    "conditioning": "no_q", "text_attention_mask": False,
    "checkpoint": os.path.realpath(ckpt), "checkpoint_sha256": sha(ckpt),
    "score_tsv": m["tsv"], "score_tsv_sha256": m["tsv_sha256"],
    "metrics": m["metrics"], "metrics_path": str(metrics), "metrics_sha256": sha(metrics),
}
report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload["metrics"], indent=2))
PY
