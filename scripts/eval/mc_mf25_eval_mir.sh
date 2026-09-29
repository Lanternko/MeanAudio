#!/bin/bash
# Standard MusicCaps eval + MIR readings (2026-09-30).
#
# Runs mc_mf25_eval.sh unchanged (same arguments, same outputs), then scores
# each requested cell with scripts/eval/mir_metrics.py (pulse_clarity, ibi_cv,
# key_cnn_conf, chroma_entropy; CPU only). mc_mf25_eval.sh and eval_metrics.py
# stay byte-identical because queue contracts pin their digests; cells produced
# by those jobs get MIR afterwards with
#   ~/venvs/dac/bin/python scripts/eval/mir_metrics.py --cell <cell dir>
#
# Usage: same as mc_mf25_eval.sh
#   mc_mf25_eval_mir.sh <exp_name> <s2_ema.pth> (--no_q | --quality_level N) [--mask] [--gen_tsv PATH] [cfg0] [cfg3neg]
# Output: mc_mf25_eval.sh's cell dirs, plus <cell>/<label>/mir_metrics.{json,txt} and mir_per_clip.tsv.
# Idempotent: cells with mir_metrics.json are skipped.
set -euo pipefail

HERE=/home/kojiek/MeanAudio/scripts/eval
PY=/home/kojiek/venvs/dac/bin/python
EXP="${1:?exp_name}"
OUT_ROOT="${OUT_ROOT:-$HOME/eval_output_nvme}"

bash "$HERE/mc_mf25_eval.sh" "$@"

# same label rules as mc_mf25_eval.sh
COND_TAG="" MASK_TAG="" CELLS=()
shift 2
while [ "$#" -gt 0 ]; do
  case "$1" in
    --quality_level) COND_TAG="_q$2"; shift 2 ;;
    --gen_tsv) shift 2 ;;
    --mask) MASK_TAG="_mask"; shift ;;
    cfg0) CELLS+=(cfg0); shift ;;
    cfg3neg) CELLS+=(cfg3_neg); shift ;;
    *) shift ;;
  esac
done
[ "${#CELLS[@]}" -gt 0 ] || CELLS=(cfg0 cfg3_neg)
SMOKE_TAG="${SMOKE_ROWS:+_smoke$SMOKE_ROWS}"

for C in "${CELLS[@]}"; do
  CELL="$OUT_ROOT/${EXP}_mc_mf25_${C}${COND_TAG}${MASK_TAG}${SMOKE_TAG}"
  echo "[$(date -u +%FT%TZ)] [mc_mf25_mir] $CELL"
  "$PY" "$HERE/mir_metrics.py" --cell "$CELL" 2>&1 | grep -v pkg_resources | tee -a "$CELL/mir.log"
done
