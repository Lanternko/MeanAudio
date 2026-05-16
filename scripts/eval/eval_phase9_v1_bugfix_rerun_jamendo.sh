#!/bin/bash
# ============================================================
# MeanAudio Phase 9 V1 BUG-FIX — Jamendo eval (historical comparison)
# eval_phase9_v1_bugfix_rerun_jamendo.sh
#
# 目的：補跑 P9 V1 bug-fix 模型的 Jamendo eval，對比修前 Jamendo 0.0260
#       （MusicCaps 已有 CLAP 0.0650）
#
# 用途：
#   訓練已完成（見 train_pipeline_phase9_v1_bugfix_rerun.sh），
#   現只需要在同一個 S2 EMA 上跑另一組 eval TSV
#
# Model: ~/MeanAudio/exps/phase9_v1_bugfix_rerun_stage2_200000/*_ema_final.pth
# TSV:   /mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv (90,063 clips)
#
# 時間估算：~3.5 hr
#   - 生成音訊：90,063 × 1/8 it/s ≈ 3.1 hr
#   - Metrics: ~30 min
#
# 使用方式：
#   tmux new -d -s p9v1_bugfix_jamendo 'bash ~/MeanAudio/eval_phase9_v1_bugfix_rerun_jamendo.sh'
# ============================================================

set -e
set -o pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S2="phase9_v1_bugfix_rerun_stage2_200000"
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

TSV_JAMENDO="$DATA_DIR/phase4_test.tsv"
EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_jamendo"

cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  Phase 9 V1 BUG-FIX — Jamendo eval"
echo "  S2 EMA : $S2_EMA"
echo "  TSV    : $TSV_JAMENDO (90,063 clips)"
echo "  Output : $EVAL_OUT_JM"
echo "  ETA    : ~3.5 hr (gen 3.1 hr + metrics 0.4 hr)"
echo "======================================================"

# Pre-flight
[ -f "$S2_EMA" ] || { echo "[FAIL] S2 EMA not found"; exit 1; }
[ -f "$TSV_JAMENDO" ] || { echo "[FAIL] Jamendo TSV not found"; exit 1; }
echo "[OK] S2 EMA & TSV verified"

# ============================================================
# 生成音訊
# ============================================================
echo "[Gen] 生成音訊（90,063 筆）..."

python eval.py \
    --variant "meanaudio_s" \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT_JM/audio" \
    --tsv "$TSV_JAMENDO" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_jamendo_eval.log"

# ============================================================
# Metrics（重要：必須傳 --tsv，不然會 fallback 到 default，這點先前踩過）
# ============================================================
echo "[Metrics] 計算 CLAP + AES..."

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_JM/audio" \
    --tsv "$TSV_JAMENDO" \
    --exp_name "${EXP_S2}_jamendo" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_jamendo_eval.log"

echo "======================================================"
echo "  P9 V1 BUG-FIX Jamendo eval 完成"
echo "  Metrics → eval_output/metrics/${EXP_S2}_jamendo/metrics.txt"
echo ""
echo "  基線對照："
echo "    P9 V1 修前 Jamendo:   CLAP 0.0260"
echo "    P9 V1 修後 MusicCaps: CLAP 0.0650（已跑）"
echo "    Phase 7 V1 Jamendo:   CLAP 0.1984（歷史最佳）"
echo "======================================================"
