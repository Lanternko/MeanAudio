#!/bin/bash
# ============================================================
# P8 V4 Q (S2-only Q variant) eval — q sweep on both benchmarks
# eval_p8v4_q.sh
#
# 用途：拿 train_pipeline_phase8v4_q.sh 訓出來的 ema_final，
#       跑 q=6 + q=9 × MusicCaps + Jamendo seed42 (4 跑)
#
# 用法：
#   bash ~/MeanAudio/eval_p8v4_q.sh
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S2="phase8_v4_q_stage2_200000"
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

if [ ! -f "$S2_EMA" ]; then
    echo "❌ EMA 不存在：$S2_EMA"
    exit 1
fi

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  P8 V4 Q eval (q=6 + q=9 × MusicCaps + Jamendo seed42)"
echo "  EMA: $S2_EMA"
echo "======================================================"

for Q in 6 9; do
    # ── MusicCaps (primary) ───────────────────────────
    EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_q${Q}_musiccaps"
    if [ -f "$WORK_DIR/eval_output/metrics/${EXP_S2}_q${Q}_musiccaps/metrics.txt" ]; then
        echo "[MC q=$Q] ✅ skip (already done)"
    else
        echo "[Eval / MusicCaps q=${Q}] gen → $EVAL_OUT_MC"
        python eval.py \
            --variant "meanaudio_s" \
            --model_path "$S2_EMA" \
            --output "$EVAL_OUT_MC/audio" \
            --tsv "$DATA_DIR/phase8_v4_musiccaps_test.tsv" \
            --use_meanflow --num_steps 1 \
            --encoder_name t5_clap --text_c_dim 512 \
            --cfg_strength 0.5 --quality_level $Q \
            --full_precision \
            2>&1 | tee "$LOG_DIR/${EXP_S2}_q${Q}_musiccaps_eval.log"

        python "$EVAL_SCRIPT" \
            --gen_dir "$EVAL_OUT_MC/audio" \
            --tsv "$DATA_DIR/musiccaps_test.tsv" \
            --exp_name "${EXP_S2}_q${Q}_musiccaps" \
            --num_samples 5521 \
            2>&1 | tee -a "$LOG_DIR/${EXP_S2}_q${Q}_musiccaps_eval.log"
    fi

    # ── Jamendo seed=42 2048 (secondary) ──────────────
    EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_q${Q}_jamendo_seed42_2048"
    if [ -f "$WORK_DIR/eval_output/metrics/${EXP_S2}_q${Q}_jamendo_seed42_2048/metrics.txt" ]; then
        echo "[JM q=$Q] ✅ skip (already done)"
    else
        echo "[Eval / Jamendo seed42 q=${Q}] gen → $EVAL_OUT_JM"
        python eval.py \
            --variant "meanaudio_s" \
            --model_path "$S2_EMA" \
            --output "$EVAL_OUT_JM/audio" \
            --tsv "$DATA_DIR/phase8_v4_jamendo_seed42_2048.tsv" \
            --use_meanflow --num_steps 1 \
            --encoder_name t5_clap --text_c_dim 512 \
            --cfg_strength 0.5 --quality_level $Q \
            --full_precision \
            2>&1 | tee "$LOG_DIR/${EXP_S2}_q${Q}_jamendo_eval.log"

        python "$EVAL_SCRIPT" \
            --gen_dir "$EVAL_OUT_JM/audio" \
            --tsv "$DATA_DIR/phase4_test_seed42_2048.tsv" \
            --exp_name "${EXP_S2}_q${Q}_jamendo_seed42_2048" \
            --num_samples 2048 \
            2>&1 | tee -a "$LOG_DIR/${EXP_S2}_q${Q}_jamendo_eval.log"
    fi
done

echo "======================================================"
echo "  P8 V4 Q eval 完成（q={6,9} × {MusicCaps, Jamendo seed42}）"
echo "  Metrics: eval_output/metrics/${EXP_S2}_q{6,9}_{musiccaps,jamendo_seed42_2048}/"
echo "======================================================"
