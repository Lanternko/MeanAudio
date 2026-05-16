#!/bin/bash
# ============================================================
# MeanAudio Phase 9 V2 — TrueRandom + Q (pairwise MeanSim)
# train_pipeline_phase9_v2.sh
#
# 實驗設計：
#   每個 __getitem__ 從 5 個 LP-MusicCaps human caption 動態隨機抽一個
#   Q signal = pairwise text-text MeanSim of 5 captions（與 Phase 7 V1 相同計算方式）
#   對照：Phase 9 V1（same, no Q）
#
# NPZ：~/phase9_multicap_npz/（SSD，251,599 clips，shape: [5,77,1024]/[5,512]）
# TSV：phase7_v1_train.tsv（同一批 251,599 clips，含 q_level 欄位）
#
# 使用方式：
#   tmux new -s phase9_v2
#   cd ~/MeanAudio && source ~/venvs/dac/bin/activate
#   bash train_pipeline_phase9_v2.sh
# ============================================================

set -e

# ============================================================
# 實驗參數設定
# ============================================================

EXP_PREFIX="phase9_v2"

BATCH_SIZE=8
ACCUM_STEPS=1

S1_ITERATIONS=400000
S2_ITERATIONS=200000

LEARNING_RATE=1e-4

USE_Q_CONDITIONING=true           # V2：with Q

S2_MACRO=$(( S2_ITERATIONS / ACCUM_STEPS ))
S2_LR_STEP1=$(( S2_MACRO * 80 / 100 ))
S2_LR_STEP2=$(( S2_MACRO * 90 / 100 ))

# ============================================================
# 固定路徑設定
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
    "+use_q_conditioning=$USE_Q_CONDITIONING"
    val_interval=999999
    eval_interval=999999
    save_eval_interval=999999
    "data.AudioCaps_npz.tsv=$DATA_DIR/phase7_v1_train.tsv"
    "data.AudioCaps_val_npz.tsv=$DATA_DIR/phase4_val.tsv"
    "++data.AudioCaps_npz.npz_dir=$HOME/phase9_multicap_npz"
    "++multi_cap=True"
)

# ============================================================
# 初始化
# ============================================================

mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR/exps/$EXP_S2"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  Phase 9 V2 — TrueRandom + Q (pairwise MeanSim)"
echo "  Stage 1 exp_id : $EXP_S1"
echo "  Stage 2 exp_id : $EXP_S2"
echo "  等效 batch size: $(( BATCH_SIZE * ACCUM_STEPS ))"
echo "  S2 LR 衰減點   : macro-step $S2_LR_STEP1 / $S2_LR_STEP2"
echo "  NPZ dir        : $HOME/phase9_multicap_npz"
echo "  Train TSV      : $DATA_DIR/phase7_v1_train.tsv"
echo "======================================================"

# ============================================================
# Pre-flight: 驗證 multi-cap NPZ 目錄完整 (防止 iter 6,243-style KeyError)
# ============================================================
echo "[Pre-flight] 驗證 NPZ 目錄..."
python "$HOME/research/meanaudio_training/validate_multicap_npz.py" \
    --tsv "$DATA_DIR/phase7_v1_train.tsv" \
    --npz_dir "$HOME/phase9_multicap_npz" \
    --deep 200 \
    || { echo "[FAIL] NPZ validation failed. Fix the data before training."; exit 1; }

# ============================================================
# Stage 1
# ============================================================

S1_CKPT_COMPLETE=false
if [ -f "$S1_CKPT" ]; then
    CKPT_IT=$(python -c "import torch; c=torch.load('$S1_CKPT', map_location='cpu', weights_only=False); print(c.get('it', 0))" 2>/dev/null)
    if [ -z "$CKPT_IT" ]; then
        CORRUPT_NAME="${S1_CKPT}.corrupted_$(date +%Y%m%d_%H%M%S)"
        echo "[WARN] Stage 1 ckpt 不可讀，已隔離至 $CORRUPT_NAME"
        mv "$S1_CKPT" "$CORRUPT_NAME"
    elif [ "$CKPT_IT" -ge "$S1_ITERATIONS" ]; then
        S1_CKPT_COMPLETE=true
        echo "[Stage 1] ckpt 已完成 (iter $CKPT_IT >= $S1_ITERATIONS)"
    else
        echo "[Stage 1] ckpt 存在但未完成 (iter $CKPT_IT / $S1_ITERATIONS)，將 resume 繼續訓練"
    fi
fi

if [ "$S1_CKPT_COMPLETE" = "true" ]; then
    echo "[Stage 1] 跳過訓練"
    echo "  → $S1_CKPT"
else
    echo "[Stage 1] 開始 / 繼續訓練：$EXP_S1"

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
python "$MIGRATE_SCRIPT" \
    --s1_ckpt "$S1_CKPT" \
    --s2_out  "$S2_CKPT"
echo "[遷移] 完成"

# ============================================================
# Stage 2
# ============================================================

echo "[Stage 2] 開始訓練：$EXP_S2"

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
# Eval：V2 with-Q → q=6、q=9、native_q
# ============================================================

S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
TSV_FIXED="$DATA_DIR/phase4_test.tsv"
TSV_NATIVE="$DATA_DIR/phase6_test.tsv"

for Q in 6 9; do
    EVAL_OUT="$WORK_DIR/eval_output/${EXP_S2}_q${Q}_jamendo"
    echo "[Eval S2] 生成音訊 q=${Q}：$EVAL_OUT"
    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT/audio" \
        --tsv "$TSV_FIXED" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --quality_level $Q \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_q${Q}_eval.log"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT/audio" \
        --exp_name "${EXP_S2}_q${Q}" \
        --num_samples 2048 \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_q${Q}_eval.log"
done

# native_q
EVAL_OUT_NQ="$WORK_DIR/eval_output/${EXP_S2}_native_q_jamendo"
echo "[Eval S2] 生成音訊 native_q：$EVAL_OUT_NQ"
python eval.py \
    --variant "meanaudio_s" \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT_NQ/audio" \
    --tsv "$TSV_NATIVE" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_native_q_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_NQ/audio" \
    --exp_name "${EXP_S2}_native_q" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_native_q_eval.log"

# ============================================================
# 完成
# ============================================================

echo "======================================================"
echo "  Phase 9 V2 訓練 + Eval 完成"
echo "  S1 EMA  : exps/$EXP_S1/${EXP_S1}_ema_final.pth"
echo "  S2 EMA  : exps/$EXP_S2/${EXP_S2}_ema_final.pth"
echo "  Metrics : eval_output/${EXP_S2}_q6/ q9/ native_q/"
echo "======================================================"
