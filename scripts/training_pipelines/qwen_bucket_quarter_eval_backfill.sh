#!/bin/bash
# Eval-only backfill for the July Phase-8 Qwen bucket quarter grid.
#
# Five S2 EMAs have sat on disk since 2026-07-26..30 without a canonical number:
# NoQ, K=3/5/10 balanced, K=5 fixed. All share one recipe (seed 14159265, batch 8,
# lr 1e-4, S1 100k + S2 50k, NoMask, single caption), so the only thing that moves
# across the grid is how q_level was bucketed. Nothing is trained here.
#
# Cells, in order of how much each answer is worth if the queue is cut short:
#   1. CFG0 canonical: NoQ -> noq; every Q arm -> q9 (primary) then q0 (probe).
#      A Q arm whose q9 and q0 sit inside 2x the seed floor ignored its code.
#   2. CFG3+neg: NoQ -> noq; every Q arm -> q9 only.
# Both helpers are idempotent, so a paused/restarted seat only regenerates the
# label that was in flight.
set -euo pipefail

MA=/home/kojiek/MeanAudio
EXPS="$MA/exps"
CONTRACT="$MA/docs/experiments/qwen_bucket_quarter_eval_backfill_contract.json"
EVAL="$MA/scripts/caption10s_pipeline/eval_musiccaps_mf25.sh"
CFG3NEG="$MA/scripts/eval/mc_mf25_cfg3neg_eval.sh"
CFG3NEG_Q="$MA/scripts/eval/mc_mf25_cfg3neg_eval_q.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$MA"
log(){ echo "[$(date -u +%FT%TZ)] [bucket-backfill] $*"; }

ema(){ echo "$EXPS/phase8_qwen_bucket_quarter_${1}_stage2_50000/phase8_qwen_bucket_quarter_${1}_stage2_50000_ema_final.pth"; }
Q_ARMS=(k10_balanced k3_balanced k5_balanced k5_fixed)

for a in noq "${Q_ARMS[@]}"; do
  [ -f "$(ema "$a")" ] || { log "FAIL missing EMA for $a"; exit 2; }
done

# ---- CFG0 canonical ---------------------------------------------------------
log "CFG0 noq"
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=noq_cfg0 /bin/bash "$EVAL" \
  "phase8_qwen_bucket_quarter_noq_musiccaps_mf25_cfg0_noq" "$(ema noq)" --no_q
for a in "${Q_ARMS[@]}"; do
  for q in 9 0; do
    log "CFG0 $a q$q"
    CFG0_CONTRACT="$CONTRACT" CFG0_ARM="${a}_cfg0_q${q}" /bin/bash "$EVAL" \
      "phase8_qwen_bucket_quarter_${a}_musiccaps_mf25_cfg0_q${q}" "$(ema "$a")" --quality_level "$q"
  done
done

# ---- CFG3+neg ---------------------------------------------------------------
log "CFG3+neg noq"
/bin/bash "$CFG3NEG" phase8_qwen_bucket_quarter_noq "$(ema noq)"
for a in "${Q_ARMS[@]}"; do
  log "CFG3+neg $a q9"
  /bin/bash "$CFG3NEG_Q" "phase8_qwen_bucket_quarter_${a}" "$(ema "$a")" 9
done

log "DONE"
