#!/bin/bash
# ============================================================
# P7 V1 q-sweep on MusicCaps (q=0..9)
# eval_p7v1_qsweep_musiccaps.sh
#
# 用途：補 P7 V1 完整 q-sweep（已知歷史 q=6 / q=9 in-support，
#       q=0/3 OOD support gating；本次補齊 q=0,1,2,4,5,7,8）
# Eval target: MusicCaps only (5521)
# 預估時間：10 q × ~12 min ≈ 2 hours
#
# 用法：
#   bash ~/MeanAudio/eval_p7v1_qsweep_musiccaps.sh
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP="phase7_v1_stage2_200000"
EMA="$WORK_DIR/exps/$EXP/${EXP}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

if [ ! -f "$EMA" ]; then
    echo "❌ EMA ckpt 不存在：$EMA"
    exit 1
fi

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  P7 V1 q-sweep on MusicCaps (q=0..9)"
echo "  EMA   : $EMA"
echo "  TSV   : $DATA_DIR/musiccaps_test.tsv (n=5521, no prefix)"
echo "======================================================"

for Q in 0 1 2 3 4 5 6 7 8 9; do
    EVAL_OUT="$WORK_DIR/eval_output/${EXP}_q${Q}_musiccaps_qsweep"
    LOG="$LOG_DIR/${EXP}_q${Q}_musiccaps_qsweep_eval.log"

    if [ -f "$WORK_DIR/eval_output/metrics/${EXP}_q${Q}_musiccaps_qsweep/metrics.txt" ]; then
        echo "[q=$Q] ✅ metrics already exist, skip"
        continue
    fi

    echo "[q=${Q}] gen → $EVAL_OUT"
    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$EMA" \
        --output "$EVAL_OUT/audio" \
        --tsv "$DATA_DIR/musiccaps_test.tsv" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --quality_level $Q \
        --full_precision \
        2>&1 | tee "$LOG"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT/audio" \
        --tsv "$DATA_DIR/musiccaps_test.tsv" \
        --exp_name "${EXP}_q${Q}_musiccaps_qsweep" \
        --num_samples 5521 \
        2>&1 | tee -a "$LOG"
done

echo "======================================================"
echo "  P7 V1 q-sweep 完成（q=0..9 × MusicCaps n=5521）"
echo "  Metrics: eval_output/metrics/${EXP}_q{0..9}_musiccaps_qsweep/"
echo "======================================================"
