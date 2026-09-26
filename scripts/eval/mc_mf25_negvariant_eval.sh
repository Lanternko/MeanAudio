#!/bin/bash
# MusicCaps eval with a non-default negative prompt (081 quality-label arm, 2026-09-26).
#
# Same protocol and generation flags as scripts/eval/mc_mf25_eval.sh (MusicCaps 5521 /
# MeanFlow 25 / seed 42 / fp32 / NoMask), one cell: CFG 3.0 with the negative prompt
# given on the command line instead of the fixed fidelity8 string. mc_mf25_eval.sh
# stays the canonical two-cell eval; this wrapper exists so its fixed prompt never
# has to change.
#
# Usage:
#   mc_mf25_negvariant_eval.sh <exp_name> <s2_ema.pth> (--no_q | --quality_level N) --neg TEXT --tag TAG [--mask] [--gen_tsv PATH]
#     TAG names the prompt in the label (lowercase alnum), e.g. lqneg;
#     --gen_tsv generates from another TSV with the same ids; CLAP is always scored
#     against the plain MusicCaps captions.
#
# Output: $OUT_ROOT/<exp>_mc_mf25_cfg3_<TAG>[_qN][_mask]/
# Idempotent the same way as mc_mf25_eval.sh (report -> skip; no report -> regenerate).
set -euo pipefail

EXP="${1:?exp_name}"
CKPT="${2:?s2 ema checkpoint}"
shift 2

NEG="" TAG=""

PY=/home/kojiek/venvs/dac/bin/python
ROOT=/home/kojiek/MeanAudio
MC_TSV=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
OUT_ROOT="${OUT_ROOT:-$HOME/eval_output_nvme}"
SMOKE_ROWS="${SMOKE_ROWS:-}"

COND_ARGS=() COND_TAG="" COND_NAME="" MASK_ARGS=(--no_text_attention_mask) MASK_TAG="" GEN_TSV="$MC_TSV" CELLS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no_q) COND_ARGS=(--no_q); COND_TAG=""; COND_NAME="no_q"; shift ;;
    --quality_level)
      [[ "${2:-}" =~ ^[0-9]$ ]] || { echo "FAIL --quality_level needs 0..9" >&2; exit 2; }
      COND_ARGS=(--quality_level "$2"); COND_TAG="_q$2"; COND_NAME="quality_level_$2"; shift 2 ;;
    --mask) MASK_ARGS=(); MASK_TAG="_mask"; shift ;;
    --gen_tsv) GEN_TSV="${2:?--gen_tsv PATH}"; shift 2 ;;
    --neg) NEG="${2:?--neg TEXT}"; shift 2 ;;
    --tag) TAG="${2:?--tag TAG}"; shift 2 ;;
    *) echo "FAIL unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$COND_NAME" ] || { echo "FAIL pass exactly one of --no_q / --quality_level N" >&2; exit 2; }
[ -n "$NEG" ] || { echo "FAIL --neg TEXT is required" >&2; exit 2; }
[[ "$TAG" =~ ^[a-z0-9]+$ ]] || { echo "FAIL --tag must be lowercase alnum" >&2; exit 2; }
CELLS=(neg)
[ -f "$CKPT" ] || { echo "FAIL missing ckpt $CKPT" >&2; exit 2; }
[ -f "$GEN_TSV" ] || { echo "FAIL missing $GEN_TSV" >&2; exit 2; }
# eval.py takes a per-row q_level column over --quality_level/--no_q; refuse it here
head -1 "$GEN_TSV" | tr '\t' '\n' | grep -qx q_level && { echo "FAIL $GEN_TSV has a q_level column, which would override $COND_NAME" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

SCORE_TSV="$MC_TSV" SMOKE_TAG=""
if [ -n "$SMOKE_ROWS" ]; then
  SMOKE_TAG="_smoke$SMOKE_ROWS"
  TMP_TSV=$(mktemp -d)
  head -n $((SMOKE_ROWS + 1)) "$MC_TSV" > "$TMP_TSV/score.tsv"
  head -n $((SMOKE_ROWS + 1)) "$GEN_TSV" > "$TMP_TSV/gen.tsv"
  SCORE_TSV="$TMP_TSV/score.tsv" GEN_TSV="$TMP_TSV/gen.tsv"
fi

for CELL in "${CELLS[@]}"; do
  CELL_TAG="cfg3_$TAG" CFG=3.0 NEG_ARGS=(--negative_prompt "$NEG") NEG_TEXT="$NEG"
  LABEL="${EXP}_mc_mf25_${CELL_TAG}${COND_TAG}${MASK_TAG}${SMOKE_TAG}"
  OUT="$OUT_ROOT/$LABEL"
  REPORT="$OUT/${LABEL}_REPORT.json"
  log(){ echo "[$(date -u +%FT%TZ)] [mc_mf25 $LABEL] $*"; }

  if [ -f "$REPORT" ]; then
    log "SKIP report exists: $REPORT"
    continue
  fi
  case "$OUT" in "$OUT_ROOT"/*_mc_mf25_*) ;; *) log "FAIL refusing output path $OUT"; exit 2 ;; esac
  if [ -d "$OUT/audio" ]; then
    log "no report: discarding partial audio ($(find "$OUT/audio" -name '*.flac' | wc -l) clips) for a from-scratch run"
    rm -rf -- "$OUT/audio"
  fi
  mkdir -p "$OUT/audio"

  log "generate cfg=$CFG ${COND_ARGS[*]} ${MASK_ARGS[*]:-mask} (log: $OUT/gen.log)"
  "$PY" eval.py --variant meanaudio_s --model_path "$CKPT" \
    --output "$OUT/audio" --tsv "$GEN_TSV" --use_meanflow \
    --num_steps 25 --cfg_strength "$CFG" "${NEG_ARGS[@]}" \
    "${MASK_ARGS[@]}" --encoder_name t5_clap --text_c_dim 512 \
    --seed 42 --full_precision "${COND_ARGS[@]}" > "$OUT/gen.log" 2>&1 \
    || { log "FAIL eval.py exited $? (tail of $OUT/gen.log follows)"; tail -20 "$OUT/gen.log"; exit 3; }
  log "generated $(find "$OUT/audio" -name '*.flac' | wc -l) clips"

  # eval_metrics fails on any missing clip (eval.py skips NaN/unwritable audio)
  "$PY" scripts/eval/eval_metrics.py --gen_dir "$OUT/audio" --tsv "$SCORE_TSV" \
    --exp_name "$LABEL" --out_dir "$OUT" 2>&1 | tee "$OUT/metrics.log"

  "$PY" - "$REPORT" "$OUT/$LABEL/metrics.json" "$CKPT" "$GEN_TSV" "$LABEL" "$CFG" "$NEG_TEXT" \
      "$COND_NAME" "${MASK_TAG:-_nomask}" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
report, metrics, ckpt, gen_tsv = map(Path, sys.argv[1:5])
label, cfg, neg, cond, mask = sys.argv[5:10]
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
    "protocol": f"MusicCaps {m['n_rows']}; MeanFlow 25; CFG {cfg}{' + negative ' + repr(neg) if neg else ''}; "
                f"seed 42; {'Mask' if mask == '_mask' else 'NoMask'}; full precision; CLAP batch 1",
    "cfg_strength": float(cfg), "negative_prompt": neg or None, "num_steps": 25, "seed": 42,
    "conditioning": cond, "text_attention_mask": mask == "_mask",
    "checkpoint": os.path.realpath(ckpt), "checkpoint_sha256": sha(ckpt),
    "gen_tsv": str(gen_tsv), "gen_tsv_sha256": sha(gen_tsv),
    "score_tsv": m["tsv"], "score_tsv_sha256": m["tsv_sha256"],
    "metrics": m["metrics"], "metrics_path": str(metrics), "metrics_sha256": sha(metrics),
}
report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload["metrics"], indent=2))
PY
  log "report written: $REPORT"
done
