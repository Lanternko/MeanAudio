#!/bin/bash
# ============================================================
# MeanAudio Phase 9 V1 — S2 SALVAGE (fixed single-cap)
# train_pipeline_phase9_v1_s2_salvage.sh
#
# 診斷實驗：S1 用 multi_cap=True dynamic random 訓完（已有 ckpt），
# S2 改用 fixed single-cap（slot 0 from LP-MusicCaps 5 caps），
# 驗證「multi_cap=True 是否只傷害 Stage 2」。
#
# 對照：
#   phase9_v1_stage2_200000 (FAILED)：S1 + S2 都 multi_cap=True → CLAP 0.0260
#   phase9_v1_salvage_stage2_200000 (this)：S1 multi_cap=True + S2 multi_cap=False
#
# 輸入：
#   - S1 ckpt: exps/phase9_v1_stage1_400000/phase9_v1_stage1_400000_ckpt_last.pth
#   - Single-cap NPZ: ~/phase9_singlecap_slot0_npz/ (251,599 clips, [77,1024]/[512])
#
# 使用方式：
#   tmux new -d -s p9v1_salvage 'bash ~/MeanAudio/train_pipeline_phase9_v1_s2_salvage.sh'
# ============================================================

set -e
set -o pipefail

EXP_PREFIX="phase9_v1_salvage"
S1_SRC_EXP="phase9_v1_stage1_400000"  # 重用既有 S1 ckpt，不重訓

BATCH_SIZE=8
ACCUM_STEPS=1
S1_ITERATIONS=400000
S2_ITERATIONS=200000
LEARNING_RATE=1e-4
USE_Q_CONDITIONING=false   # V1 is no-Q

S2_MACRO=$(( S2_ITERATIONS / ACCUM_STEPS ))
S2_LR_STEP1=$(( S2_MACRO * 80 / 100 ))
S2_LR_STEP2=$(( S2_MACRO * 90 / 100 ))

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S2="${EXP_PREFIX}_stage2_${S2_ITERATIONS}"

S1_CKPT="$WORK_DIR/exps/$S1_SRC_EXP/${S1_SRC_EXP}_ckpt_last.pth"
S2_CKPT="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ckpt_last.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"

SINGLECAP_NPZ="$HOME/phase9_singlecap_slot0_npz"

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
    "++data.AudioCaps_npz.npz_dir=$SINGLECAP_NPZ"
    "++multi_cap=False"   # ← 關鍵差異：固定 single-cap
)

mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR/exps/$EXP_S2"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  Phase 9 V1 SALVAGE — S1 (multi_cap=True) + S2 (multi_cap=False, slot 0)"
echo "  S1 src ckpt    : $S1_CKPT"
echo "  S2 exp_id      : $EXP_S2"
echo "  等效 batch size: $(( BATCH_SIZE * ACCUM_STEPS ))"
echo "  S2 LR 衰減點   : macro-step $S2_LR_STEP1 / $S2_LR_STEP2"
echo "  NPZ dir        : $SINGLECAP_NPZ"
echo "  Train TSV      : $DATA_DIR/phase7_v1_train.tsv"
echo "  multi_cap      : False (fixed slot 0)"
echo "======================================================"

# ============================================================
# Pre-flight: 驗證 S1 ckpt + single-cap NPZ
# ============================================================
echo "[Pre-flight] 驗證 S1 ckpt..."
if [ ! -f "$S1_CKPT" ]; then
    echo "[FAIL] S1 ckpt 不存在：$S1_CKPT"
    exit 1
fi
CKPT_IT=$(python -c "import torch; c=torch.load('$S1_CKPT', map_location='cpu', weights_only=False); print(c.get('it', 0))")
if [ -z "$CKPT_IT" ] || [ "$CKPT_IT" -lt "$S1_ITERATIONS" ]; then
    echo "[FAIL] S1 ckpt iter $CKPT_IT < $S1_ITERATIONS，未訓完"
    exit 1
fi
echo "[OK] S1 ckpt iter $CKPT_IT ≥ $S1_ITERATIONS ✓"

echo "[Pre-flight] 驗證 single-cap NPZ dir..."
python "$HOME/research/meanaudio_training/validate_multicap_npz.py" \
    --tsv "$DATA_DIR/phase7_v1_train.tsv" \
    --npz_dir "$SINGLECAP_NPZ" \
    --deep 200 \
    2>&1 || {
    # validate_multicap_npz 檢查 [5,77,1024] shape，single-cap 會不過 deep check。
    # 換用簡單檢查：
    echo "[INFO] validator 預期 multi-cap 格式會失敗，改用 single-cap check..."
    python -c "
import os, numpy as np
d = '$SINGLECAP_NPZ'
files = sorted(os.listdir(d), key=lambda x: int(x[:-4]) if x.endswith('.npz') else -1)
files = [f for f in files if f.endswith('.npz')]
print(f'NPZ count: {len(files)}')
assert len(files) == 251599, f'expect 251599, got {len(files)}'
s = np.load(f'{d}/0.npz')
print(f'Sample shape — text_features: {s[\"text_features\"].shape}, text_features_c: {s[\"text_features_c\"].shape}')
assert s['text_features'].shape == (77, 1024), 'expect [77,1024]'
assert s['text_features_c'].shape == (512,), 'expect [512]'
print('[OK] single-cap NPZ 格式正確')
"
}

# ============================================================
# Migrate S1 → S2（重新產生 S2 ckpt @ iter 400K base）
# ============================================================
echo "[遷移] S1 → S2 checkpoint"
python "$MIGRATE_SCRIPT" --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT"
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
echo "  Phase 9 V1 SALVAGE 完成"
echo "  Metrics → eval_output/metrics/${EXP_S2}_no_q/metrics.txt"
echo "  對照："
echo "    phase9_v1_stage2_200000_no_q  — multi_cap S1+S2 → CLAP 0.0260"
echo "    phase8_stage2_200000          — 單 cap 基準       → CLAP 0.1851"
echo "  若本實驗 CLAP ≈ 0.18 → multi_cap 只傷 S2，S1 可 salvage"
echo "  若本實驗 CLAP ≪ 0.15 → S1 也已受損，需控制組 B1 重新驗"
echo "======================================================"
