#!/bin/bash
# ============================================================
# P8 V4 NoQ + [consistency=1.00] prefix eval（補充消融）
# eval_p8v4_noq_p100.sh
#
# 用途：拿既有 P8 V4 NoQ ckpt（2026-04-27 完成）重跑 eval，
#       caption prefix 從 0.90 (in-support) 改成 1.00 (boundary)。
#
# 對比預期結果：
#   - 0.90 baseline：MusicCaps CLAP 0.0571 / Jamendo seed42 0.0591
#   - 1.00 ablation：能否略好？略差？或本質上一樣（prefix 已經主導）？
#
# 用法：
#   tmux new -s p8v4_p100
#   bash ~/MeanAudio/eval_p8v4_noq_p100.sh
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
P100_TSV_DIR="$HOME/eval_tsvs_p100"
LOG_DIR="$HOME/logs"

EXP_S2="phase8_v4_stage2_200000"
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

if [ ! -f "$S2_EMA" ]; then
    echo "❌ EMA ckpt 不存在：$S2_EMA"
    exit 1
fi

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  P8 V4 NoQ + [consistency=1.00] eval"
echo "  EMA   : $S2_EMA"
echo "  TSV   : $P100_TSV_DIR/*.tsv"
echo "======================================================"

# ── Eval 1: MusicCaps (primary) ─────────────────────────────
EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_no_q_musiccaps_p100"
echo "[Eval / MusicCaps p100] gen → $EVAL_OUT_MC"
python eval.py \
    --variant "meanaudio_s" \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT_MC/audio" \
    --tsv "$P100_TSV_DIR/phase8_v4_musiccaps_test_p100.tsv" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_no_q_musiccaps_p100_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_MC/audio" \
    --tsv "$DATA_DIR/musiccaps_test.tsv" \
    --exp_name "${EXP_S2}_no_q_musiccaps_p100" \
    --num_samples 5521 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_no_q_musiccaps_p100_eval.log"

# ── Eval 2: Jamendo seed=42 2048 (secondary) ─────────────────
EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_no_q_jamendo_seed42_2048_p100"
echo "[Eval / Jamendo seed42_2048 p100] gen → $EVAL_OUT_JM"
python eval.py \
    --variant "meanaudio_s" \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT_JM/audio" \
    --tsv "$P100_TSV_DIR/phase8_v4_jamendo_seed42_2048_p100.tsv" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_no_q_jamendo_p100_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_JM/audio" \
    --tsv "$DATA_DIR/phase4_test_seed42_2048.tsv" \
    --exp_name "${EXP_S2}_no_q_jamendo_seed42_2048_p100" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_no_q_jamendo_p100_eval.log"

echo "======================================================"
echo "  P8 V4 NoQ p100 eval 完成"
echo "  Metrics: eval_output/metrics/${EXP_S2}_no_q_{musiccaps,jamendo_seed42_2048}_p100/"
echo "======================================================"
