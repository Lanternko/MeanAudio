#!/bin/bash
# ============================================================
# MeanAudio 通用兩階段訓練腳本
# train_pipeline.sh
#
# 功能：
#   Stage 1 (FluxAudio) → 自動 Checkpoint 遷移 → Stage 2 (MeanAudio)
#   支援斷點續傳：Stage 1 checkpoint 已存在則自動跳過
#
# 使用方式：
#   tmux new -s train
#   bash ~/MeanAudio/train_pipeline.sh
#
# 修改實驗參數時，只需調整下方「實驗參數設定」區塊。
# ============================================================

set -e  # 任何指令失敗即中止

# ============================================================
# 實驗參數設定（每次新實驗只需修改此區塊）
# ============================================================

EXP_PREFIX="phase6_v2"           # 實驗名稱前綴，自動生成 exp_id

BATCH_SIZE=8                      # 物理 batch size（每張 GPU）
ACCUM_STEPS=1                     # Gradient accumulation 步數（V4 不使用累積）
                                  # 等效 batch size = BATCH_SIZE × ACCUM_STEPS

S1_ITERATIONS=400000              # Stage 1 總 micro-steps
S2_ITERATIONS=200000              # Stage 2 總 micro-steps

LEARNING_RATE=1e-4                # 初始學習率（Stage 1 & 2 共用）

# ── LR 衰減點（Stage 2 專用）────────────────────────────────
# 自動計算：Stage 2 有效 macro-steps = S2_ITERATIONS / ACCUM_STEPS
# 衰減點設在 macro-steps 的 80% 與 90%
S2_MACRO=$(( S2_ITERATIONS / ACCUM_STEPS ))
S2_LR_STEP1=$(( S2_MACRO * 80 / 100 ))
S2_LR_STEP2=$(( S2_MACRO * 90 / 100 ))

# ============================================================
# 固定路徑設定（通常不需修改）
# ============================================================

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S1="${EXP_PREFIX}_stage1_${S1_ITERATIONS}"
EXP_S2="${EXP_PREFIX}_stage2_${S2_ITERATIONS}"

S1_CKPT="$WORK_DIR/exps/$EXP_S1/${EXP_S1}_ckpt_last.pth"
S2_CKPT="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ckpt_last.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"

# ── 共用訓練參數 ─────────────────────────────────────────────
COMMON_ARGS=(
    batch_size=$BATCH_SIZE
    +accumulation_steps=$ACCUM_STEPS
    learning_rate=$LEARNING_RATE
    num_workers=4
    save_weights_interval=10000
    save_checkpoint_interval=20000
    +use_rope=False
    +use_wandb=False
    val_interval=999999
    eval_interval=999999
    save_eval_interval=999999
    "data.AudioCaps_npz.tsv=$DATA_DIR/phase6_train.tsv"
    "data.AudioCaps_val_npz.tsv=$DATA_DIR/phase4_val.tsv"
    "+data.AudioCaps_npz.gt_cache=$DATA_DIR/npz_cache_train.txt"
    "+data.AudioCaps_val_npz.gt_cache=$DATA_DIR/npz_cache_val.txt"
)

# ============================================================
# 初始化
# ============================================================

mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR/exps/$EXP_S2"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  MeanAudio 兩階段訓練啟動"
echo "  Stage 1 exp_id : $EXP_S1"
echo "  Stage 2 exp_id : $EXP_S2"
echo "  等效 batch size: $(( BATCH_SIZE * ACCUM_STEPS ))"
echo "  S2 LR 衰減點   : macro-step $S2_LR_STEP1 / $S2_LR_STEP2"
echo "======================================================"

# ============================================================
# Stage 1（若 checkpoint 已存在則跳過）
# ============================================================

if [ -f "$S1_CKPT" ]; then
    echo "[Stage 1] Checkpoint 已存在，跳過訓練"
    echo "  → $S1_CKPT"
else
    echo "[Stage 1] 開始訓練：$EXP_S1"
    echo "  micro-steps : $S1_ITERATIONS"
    echo "  LR          : $LEARNING_RATE（Stage 1 不衰減）"

    python "$STAGE_SCRIPT" --stage 1

    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio \
        model=fluxaudio_s \
        exp_id="$EXP_S1" \
        num_iterations=$S1_ITERATIONS \
        "lr_schedule_steps=[320000,360000]" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${EXP_S1}.log"

    echo "[Stage 1] 訓練完成"
fi

# ============================================================
# Checkpoint 遷移
# ============================================================

echo "[遷移] Stage 1 → Stage 2 checkpoint"
echo "  來源：$S1_CKPT"
echo "  目標：$S2_CKPT"

python "$MIGRATE_SCRIPT" \
    --s1_ckpt "$S1_CKPT" \
    --s2_out  "$S2_CKPT"

echo "[遷移] 完成"

# ============================================================
# Stage 2
# ============================================================

echo "[Stage 2] 開始訓練：$EXP_S2"
echo "  micro-steps    : $S2_ITERATIONS"
echo "  macro-steps    : $S2_MACRO"
echo "  LR 衰減 macro  : $S2_LR_STEP1 → $S2_LR_STEP2"

python "$STAGE_SCRIPT" --stage 2

torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio \
    model=meanaudio_s \
    exp_id="$EXP_S2" \
    num_iterations=$(( S1_ITERATIONS + S2_ITERATIONS )) \
    "lr_schedule_steps=[999999,999999]" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${EXP_S2}.log"

echo "[Stage 2] 訓練完成"

# ============================================================
# Eval：Stage 1 中間結果（FluxAudio + q_embed）
# ============================================================

S1_EMA="$WORK_DIR/exps/$EXP_S1/${EXP_S1}_ema_final.pth"
EVAL_S1_OUT="$WORK_DIR/eval_output/${EXP_S1}_q9"

echo "[Eval S1] 生成音訊：$EVAL_S1_OUT"
python eval.py \
    --variant "fluxaudio_s" \
    --model_path "$S1_EMA" \
    --output "$EVAL_S1_OUT/audio" \
    --cfg_strength 4.5 \
    --encoder_name t5_clap \
    --duration 10 \
    --text_c_dim 512 \
    --num_steps 25 \
    --quality_level 9 \
    --tsv ./sets/test-audiocaps.tsv \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S1}_eval.log"

echo "[Eval S1] 計算 CLAP + FAD"
python av-benchmark/evaluate.py \
    --gt_audio gt_audio \
    --gt_cache ./data/audiocaps/test-features \
    --pred_audio "$EVAL_S1_OUT/audio" \
    --pred_cache "$EVAL_S1_OUT/cache" \
    --audio_length=10 \
    --recompute_pred_cache \
    --skip_video_related \
    --output_metrics_dir="$EVAL_S1_OUT" \
    2>&1 | tee "$LOG_DIR/${EXP_S1}_metrics.log"

# ============================================================
# Eval：Stage 2 最終結果（MeanAudio + q_embed）
# ============================================================

S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_S2_OUT="$WORK_DIR/eval_output/${EXP_S2}_q9"

echo "[Eval S2] 生成音訊：$EVAL_S2_OUT"
python eval.py \
    --variant "meanaudio_s" \
    --model_path "$S2_EMA" \
    --output "$EVAL_S2_OUT/audio" \
    --cfg_strength 0.9 \
    --encoder_name t5_clap \
    --duration 10 \
    --text_c_dim 512 \
    --num_steps 1 \
    --use_meanflow \
    --quality_level 9 \
    --tsv ./sets/test-audiocaps.tsv \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_eval.log"

echo "[Eval S2] 計算 CLAP + FAD"
python av-benchmark/evaluate.py \
    --gt_audio gt_audio \
    --gt_cache ./data/audiocaps/test-features \
    --pred_audio "$EVAL_S2_OUT/audio" \
    --pred_cache "$EVAL_S2_OUT/cache" \
    --audio_length=10 \
    --recompute_pred_cache \
    --skip_video_related \
    --output_metrics_dir="$EVAL_S2_OUT" \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_metrics.log"

# ============================================================
# 完成
# ============================================================

echo "======================================================"
echo "  Phase 訓練 + Eval 完成"
echo "  S1 EMA    : exps/$EXP_S1/${EXP_S1}_ema_final.pth"
echo "  S2 EMA    : exps/$EXP_S2/${EXP_S2}_ema_final.pth"
echo "  S1 Metrics: eval_output/${EXP_S1}_q9/"
echo "  S2 Metrics: eval_output/${EXP_S2}_q9/"
echo "======================================================"
