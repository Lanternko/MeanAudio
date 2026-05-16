#!/bin/bash
# ============================================================
# P8 V4 NoQ p=0.90 baseline — backfill prefixed_ref CLAP
# eval_p8v4_noq_p090_backfill_prefixed_ref.sh
#
# 用途：昨天（2026-04-27 早上）跑的 P8 V4 NoQ baseline 只用 natural_ref
#       (musiccaps_test.tsv / phase4_test_seed42_2048.tsv)。
#       Codex P1 指出這只是 natural-ref CLAP，不是 prompt-following CLAP。
#       這裡用同一份既有 audio dir，加跑 prefixed_ref pass，補齊雙 ref 對照。
#
# 前提：audio 已經存在
#   ~/MeanAudio/eval_output/phase8_v4_stage2_200000_no_q_musiccaps/audio/
#   ~/MeanAudio/eval_output/phase8_v4_stage2_200000_no_q_jamendo_seed42_2048/audio/
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S2="phase8_v4_stage2_200000"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"

echo "======================================================"
echo "  P8 V4 NoQ p=0.90 baseline — backfill prefixed_ref"
echo "======================================================"

# ── MusicCaps prefixed_ref backfill ─────────────────────────
AUDIO_MC="$WORK_DIR/eval_output/${EXP_S2}_no_q_musiccaps/audio"
LOG_MC="$LOG_DIR/${EXP_S2}_no_q_musiccaps_backfill_prefixed_ref.log"
if [ ! -d "$AUDIO_MC" ]; then
    echo "❌ MC audio dir not found：$AUDIO_MC"
    exit 1
fi

echo "[MC backfill prefixed_ref]"
python "$EVAL_SCRIPT" \
    --gen_dir "$AUDIO_MC" \
    --tsv "$DATA_DIR/phase8_v4_musiccaps_test.tsv" \
    --exp_name "${EXP_S2}_no_q_musiccaps_prefixed_ref" \
    --num_samples 5521 \
    2>&1 | tee "$LOG_MC"

# ── Jamendo seed42 prefixed_ref backfill ────────────────────
AUDIO_JM="$WORK_DIR/eval_output/${EXP_S2}_no_q_jamendo_seed42_2048/audio"
LOG_JM="$LOG_DIR/${EXP_S2}_no_q_jamendo_backfill_prefixed_ref.log"
if [ ! -d "$AUDIO_JM" ]; then
    echo "❌ JM audio dir not found：$AUDIO_JM"
    exit 1
fi

echo "[JM backfill prefixed_ref]"
python "$EVAL_SCRIPT" \
    --gen_dir "$AUDIO_JM" \
    --tsv "$DATA_DIR/phase8_v4_jamendo_seed42_2048.tsv" \
    --exp_name "${EXP_S2}_no_q_jamendo_seed42_2048_prefixed_ref" \
    --num_samples 2048 \
    2>&1 | tee "$LOG_JM"

echo "======================================================"
echo "  Backfill 完成"
echo "  既有 natural_ref 結果保留在原 metrics dir"
echo "  新 prefixed_ref 在 metrics/${EXP_S2}_no_q_*_prefixed_ref/"
echo "======================================================"
