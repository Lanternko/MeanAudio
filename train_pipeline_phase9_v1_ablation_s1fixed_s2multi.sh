#!/bin/bash
# ============================================================
# MeanAudio Phase 9 V1 — ABLATION: S1 fixed-cap + S2 multi-cap
# train_pipeline_phase9_v1_ablation_s1fixed_s2multi.sh
#
# 目的：補完 2×2 ablation matrix 的最後一格
#
#               S2=fixed         S2=multi
#   S1=fixed   Phase 8 ✓        ← THIS EXP
#   S1=multi   Salvage 🔄       Original V1 ✗
#
# 若本實驗 CLAP < 0.15 → S2 multi_cap 是元凶（regardless of S1）
# 若本實驗 CLAP ≥ 0.15 → 與 salvage 結果交叉判讀（見 README）
#
# 設定：
#   - S1: 從頭訓，multi_cap=False, NPZ=phase9_singlecap_slot0_npz, no Q
#   - S2: multi_cap=True, NPZ=phase9_multicap_npz
#   - 其餘超參數與 train_pipeline_phase9_v1.sh 相同
#
# 使用方式：
#   tmux new -d -s p9v1_ablation 'bash ~/MeanAudio/train_pipeline_phase9_v1_ablation_s1fixed_s2multi.sh'
# ============================================================

set -e
set -o pipefail

EXP_PREFIX="phase9_v1_ablation_s1fixed_s2multi"

BATCH_SIZE=8
ACCUM_STEPS=1
S1_ITERATIONS=400000
S2_ITERATIONS=200000
LEARNING_RATE=1e-4
USE_Q_CONDITIONING=false   # V1-style no-Q

S2_MACRO=$(( S2_ITERATIONS / ACCUM_STEPS ))
S2_LR_STEP1=$(( S2_MACRO * 80 / 100 ))
S2_LR_STEP2=$(( S2_MACRO * 90 / 100 ))

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S1="${EXP_PREFIX}_stage1_${S1_ITERATIONS}"
EXP_S2="${EXP_PREFIX}_stage2_${S2_ITERATIONS}"

S1_CKPT="$WORK_DIR/exps/$EXP_S1/${EXP_S1}_ckpt_last.pth"
S2_CKPT="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ckpt_last.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"

SINGLECAP_NPZ="$HOME/phase9_singlecap_slot0_npz"
MULTICAP_NPZ="$HOME/phase9_multicap_npz"

# ── S1 共用訓練參數（multi_cap=False, single-cap NPZ）────────
S1_ARGS=(
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
    "++data.AudioCaps_npz.npz_dir=$SINGLECAP_NPZ"
    "++multi_cap=False"
)

# ── S2 共用訓練參數（multi_cap=True, multi-cap NPZ）─────────
S2_ARGS=(
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
    "++data.AudioCaps_npz.npz_dir=$MULTICAP_NPZ"
    "++multi_cap=True"
)

mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR/exps/$EXP_S1" "$WORK_DIR/exps/$EXP_S2"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  Phase 9 V1 ABLATION — S1 fixed + S2 multi"
echo "  S1 exp_id    : $EXP_S1   (multi_cap=False)"
echo "  S2 exp_id    : $EXP_S2   (multi_cap=True)"
echo "  S1 NPZ       : $SINGLECAP_NPZ"
echo "  S2 NPZ       : $MULTICAP_NPZ"
echo "  Train TSV    : $DATA_DIR/phase7_v1_train.tsv"
echo "  Q            : false (no-Q)"
echo "======================================================"

# ============================================================
# Pre-flight: 兩個 NPZ dir 都驗證
# ============================================================
echo "[Pre-flight] 驗證 single-cap NPZ..."
python -c "
import os, numpy as np
files = sorted(f for f in os.listdir('$SINGLECAP_NPZ') if f.endswith('.npz'))
assert len(files) == 251599, f'expect 251599, got {len(files)}'
s = np.load(f'$SINGLECAP_NPZ/{files[0]}')
assert s['text_features'].shape == (77, 1024)
assert s['text_features_c'].shape == (512,)
print('[OK] single-cap NPZ ✓ (251599 files)')
"

echo "[Pre-flight] 驗證 multi-cap NPZ..."
python "$HOME/research/meanaudio_training/validate_multicap_npz.py" \
    --tsv "$DATA_DIR/phase7_v1_train.tsv" \
    --npz_dir "$MULTICAP_NPZ" \
    --deep 200

# ============================================================
# Stage 1（單 cap）
# ============================================================
S1_CKPT_COMPLETE=false
if [ -f "$S1_CKPT" ]; then
    CKPT_IT=$(python -c "import torch; c=torch.load('$S1_CKPT', map_location='cpu', weights_only=False); print(c.get('it', 0))" 2>/dev/null)
    if [ -z "$CKPT_IT" ]; then
        CORRUPT_NAME="${S1_CKPT}.corrupted_$(date +%Y%m%d_%H%M%S)"
        echo "[WARN] S1 ckpt 不可讀，已隔離至 $CORRUPT_NAME"
        mv "$S1_CKPT" "$CORRUPT_NAME"
    elif [ "$CKPT_IT" -ge "$S1_ITERATIONS" ]; then
        S1_CKPT_COMPLETE=true
        echo "[Stage 1] ckpt 已完成 (iter $CKPT_IT >= $S1_ITERATIONS)"
    else
        echo "[Stage 1] ckpt 存在但未完成 (iter $CKPT_IT / $S1_ITERATIONS)，將 resume"
    fi
fi

if [ "$S1_CKPT_COMPLETE" = "true" ]; then
    echo "[Stage 1] 跳過訓練"
else
    echo "[Stage 1] 開始 / 繼續訓練：$EXP_S1"
    python "$STAGE_SCRIPT" --stage 1
    torchrun --standalone --nproc_per_node=1 train.py \
        data=meanaudio \
        model=fluxaudio_s \
        exp_id="$EXP_S1" \
        num_iterations=$S1_ITERATIONS \
        "lr_schedule_steps=[320000,360000]" \
        "${S1_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${EXP_S1}.log"
    echo "[Stage 1] 訓練完成"
fi

# ============================================================
# Migrate S1 → S2
# ============================================================
echo "[遷移] S1 → S2 checkpoint"
python "$MIGRATE_SCRIPT" --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT"
echo "[遷移] 完成"

# ============================================================
# Stage 2（multi cap）
# ============================================================
echo "[Stage 2] 開始訓練：$EXP_S2 (multi_cap=True)"
python "$STAGE_SCRIPT" --stage 2

torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio \
    model=meanaudio_s \
    exp_id="$EXP_S2" \
    num_iterations=$(( S1_ITERATIONS + S2_ITERATIONS )) \
    "lr_schedule_steps=[999999,999999]" \
    "${S2_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${EXP_S2}.log"

echo "[Stage 2] 訓練完成"

# ============================================================
# Eval (no-Q)
# ============================================================
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
TSV_FIXED="$DATA_DIR/phase4_test.tsv"

EVAL_OUT="$WORK_DIR/eval_output/${EXP_S2}_no_q_jamendo"
echo "[Eval S2] 生成音訊（no_q）：$EVAL_OUT"

python eval.py \
    --variant "meanaudio_s" \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT/audio" \
    --tsv "$TSV_FIXED" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_no_q_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT/audio" \
    --exp_name "${EXP_S2}_no_q" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_no_q_eval.log"

echo "======================================================"
echo "  Phase 9 V1 ABLATION 完成"
echo "  Metrics → eval_output/metrics/${EXP_S2}_no_q/metrics.txt"
echo ""
echo "  2×2 對照（CLAP）："
echo "    S1=fixed × S2=fixed = Phase 8           → 0.1851"
echo "    S1=fixed × S2=multi = THIS              → ?"
echo "    S1=multi × S2=fixed = Salvage           → ?"
echo "    S1=multi × S2=multi = Original V1       → 0.0260"
echo "======================================================"
