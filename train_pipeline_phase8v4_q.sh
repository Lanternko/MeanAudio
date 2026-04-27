#!/bin/bash
# ============================================================
# Phase 8 V4 + Q (S2-only Q) 訓練腳本
# train_pipeline_phase8v4_q.sh
#
# 用途：
#   複用 phase8_v4 的 Stage 1 (NoQ, 已訓 400k on consistency-prefix data)
#   Stage 2 開啟 use_q_conditioning=true，從零訓 q_embed
#
# 預期時間：S2 ~6.1h + eval ~30 min ≈ 7h
#
# 用法：
#   tmux new -s p8v4q
#   bash ~/MeanAudio/train_pipeline_phase8v4_q.sh
# ============================================================

set -eo pipefail

# ============================================================
# 實驗參數
# ============================================================

# 來源 S1（既有 P8 V4 S1，NoQ）
EXP_S1="phase8_v4_stage1_400000"

# 新 S2 實驗名（加 _q 後綴區分）
EXP_S2="phase8_v4_q_stage2_200000"

BATCH_SIZE=8
ACCUM_STEPS=1
S2_ITERATIONS=200000
LEARNING_RATE=1e-4
USE_Q_CONDITIONING=true   # ← 與 P8 V4 唯一差異：S2 開 Q

# ============================================================
# 路徑
# ============================================================

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

S1_CKPT="$WORK_DIR/exps/$EXP_S1/${EXP_S1}_ckpt_last.pth"
S2_CKPT="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ckpt_last.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"

# 共用訓練參數
COMMON_ARGS=(
    batch_size=$BATCH_SIZE
    +accumulation_steps=$ACCUM_STEPS
    learning_rate=$LEARNING_RATE
    num_workers=4
    save_weights_interval=10000
    save_checkpoint_interval=20000
    +use_rope=False
    +use_wandb=False
    "+use_q_conditioning=$USE_Q_CONDITIONING"
    val_interval=999999
    eval_interval=999999
    save_eval_interval=999999
    "data.AudioCaps_npz.tsv=$DATA_DIR/phase8_v4_train.tsv"
    "data.AudioCaps_val_npz.tsv=$DATA_DIR/phase4_val.tsv"
    "+data.AudioCaps_npz.gt_cache=$DATA_DIR/npz_cache_train.txt"
    "+data.AudioCaps_val_npz.gt_cache=$DATA_DIR/npz_cache_val.txt"
    "++data.AudioCaps_npz.npz_dir=$HOME/research/meanaudio_training/npz_phase8v4"
)

# ============================================================
# Pre-flight
# ============================================================

mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR/exps/$EXP_S2"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

if [ ! -f "$S1_CKPT" ]; then
    echo "❌ S1 ckpt 不存在：$S1_CKPT"
    echo "請先確認 phase8_v4_stage1_400000 訓練已完成。"
    exit 1
fi

echo "======================================================"
echo "  P8 V4 + Q (S2-only Q) 訓練啟動"
echo "  S1 來源 : $EXP_S1（NoQ, 既有）"
echo "  S2 目標 : $EXP_S2（+Q from scratch on S2）"
echo "  S2 iter : $S2_ITERATIONS"
echo "  USE_Q   : $USE_Q_CONDITIONING"
echo "======================================================"

# ============================================================
# Checkpoint 遷移
# ============================================================

echo "[遷移] $S1_CKPT → $S2_CKPT"
python "$MIGRATE_SCRIPT" \
    --s1_ckpt "$S1_CKPT" \
    --s2_out  "$S2_CKPT"
echo "[遷移] 完成"

# ============================================================
# Stage 2
# ============================================================

echo "[Stage 2] 開始訓練：$EXP_S2"
python "$STAGE_SCRIPT" --stage 2

S1_ITER_BASE=400000
S2_MACRO=$(( S2_ITERATIONS / ACCUM_STEPS ))

torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio \
    model=meanaudio_s \
    exp_id="$EXP_S2" \
    num_iterations=$(( S1_ITER_BASE + S2_ITERATIONS )) \
    "lr_schedule_steps=[999999,999999]" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${EXP_S2}.log"

echo "[Stage 2] 訓練完成"

# ============================================================
# Eval（Q 訓練 → 用 --quality_level，q=6 / q=9 sweep）
# ============================================================

S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

# P8 V4 +Q：兩 benchmark × q sweep（與 P7 V1 fullq_control 同層級對照）
for Q in 6 9; do
    # ── MusicCaps (primary) ───────────────────────────
    EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_q${Q}_musiccaps"
    echo "[Eval S2 / MusicCaps q=${Q}] gen → $EVAL_OUT_MC"
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

    # ── Jamendo seed=42 2048 (secondary) ──────────────
    EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_q${Q}_jamendo_seed42_2048"
    echo "[Eval S2 / Jamendo seed42_2048 q=${Q}] gen → $EVAL_OUT_JM"
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
done

echo "======================================================"
echo "  P8 V4 + Q 訓練 + Eval 完成"
echo "  S2 EMA    : exps/$EXP_S2/${EXP_S2}_ema_final.pth"
echo "  Metrics   : eval_output/metrics/${EXP_S2}_q{6,9}_{musiccaps,jamendo_seed42_2048}/"
echo "======================================================"
