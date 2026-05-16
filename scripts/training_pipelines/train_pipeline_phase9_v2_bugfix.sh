#!/bin/bash
# ============================================================
# MeanAudio Phase 9 V2 — BUG-FIX (TrueRandom + Q)
# train_pipeline_phase9_v2_bugfix.sh
#
# 目的：驗證 multi_cap true random + Q=pairwise MeanSim 能否救回 CLAP
#       V1 (no Q) bug-fix rerun 顯示 CLAP ~0.06（跨 test set 一致）
#       York 假設：模型忽略 caption、做 unconditional drift
#       V2 加 Q 信號（training-time conditional injection），看是否能強迫
#       模型 attend to prompt；若 CLAP 跳到 ≥ 0.15 代表 Q 是救星。
#
# 關鍵差異 vs V1 bug-fix rerun：
#   - use_q_conditioning=true  → S1 訓 q_embed[0-9]，不能 reuse V1 S1
#   - 必須 S1 + S2 全部重訓
#   - Eval 用 --quality_level 9 (不是 --no_q)
#
# Bug fixes 已套用（2026-04-19）：
#   Bug #1: networks.py MeanAudio q=None 填 10（不是 9）
#   Bug #2: runner_meanflow.py text_f_undrop = text_f.clone()
#
# 時間估算：~20 hr
#   - S1: 400k iter × 0.111 s/iter ≈ 12.3 hr
#   - S2: 200k iter × 0.122 s/iter ≈ 6.7 hr
#   - MusicCaps eval: ~40 min
#
# 對比基線：
#   - Phase 7 V1 (best): Jamendo CLAP 0.1984, MusicCaps CLAP ~0.1975
#   - Phase 8 MusicCaps: CLAP 0.1851
#   - P9 V1 bug-fix MusicCaps: CLAP 0.0650 (AES 四項超 Phase 8)
#   - P9 V1 bug-fix Jamendo:   CLAP 0.0589 (跨 test set 一致)
#
# 判讀：
#   V2 MusicCaps CLAP ≥ 0.15 → Q 是救星，multi_cap + Q 可行
#   V2 MusicCaps CLAP 0.10-0.15 → Q 有貢獻但不足
#   V2 MusicCaps CLAP < 0.10 → Q 也救不了，multi_cap true random 根本不適合
#
# 使用方式：
#   tmux new -d -s p9v2_bugfix 'bash ~/MeanAudio/train_pipeline_phase9_v2_bugfix.sh'
# ============================================================

set -e
set -o pipefail

EXP_PREFIX="phase9_v2_bugfix"

BATCH_SIZE=8
ACCUM_STEPS=1
S1_ITERATIONS=400000
S2_ITERATIONS=200000
LEARNING_RATE=1e-4
USE_Q_CONDITIONING=true          # V2 key difference

# Optional evals (預設只跑 primary MusicCaps q=9 節省時間)
RUN_JAMENDO_EVAL=false           # true → +3.5 hr
RUN_Q6_EVAL=false                # true → +40 min（看 q 差異）
RUN_NATIVE_Q_EVAL=false          # true → +40 min（看 TSV 原始 q_level）

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
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"

MULTICAP_NPZ="$HOME/phase9_multicap_npz"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

TSV_MUSICCAPS="$DATA_DIR/musiccaps_test.tsv"
TSV_JAMENDO="$DATA_DIR/phase4_test.tsv"
TSV_NATIVE="$DATA_DIR/phase6_test.tsv"   # has per-clip q_level

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
    "++data.AudioCaps_npz.npz_dir=$MULTICAP_NPZ"
    "++multi_cap=True"
)

mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR/exps/$EXP_S1" "$WORK_DIR/exps/$EXP_S2"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  Phase 9 V2 BUG-FIX — TrueRandom + Q (pairwise MeanSim)"
echo "  S1 exp_id    : $EXP_S1"
echo "  S2 exp_id    : $EXP_S2"
echo "  NPZ dir      : $MULTICAP_NPZ"
echo "  Train TSV    : $DATA_DIR/phase7_v1_train.tsv (含 q_level)"
echo "  multi_cap    : True"
echo "  use_q_cond   : $USE_Q_CONDITIONING"
echo "  Primary eval : MusicCaps --quality_level 9"
echo "  Extra evals  : Jamendo=$RUN_JAMENDO_EVAL  q6=$RUN_Q6_EVAL  native_q=$RUN_NATIVE_Q_EVAL"
echo "  ETA total    : ~20 hr"
echo "======================================================"

# ============================================================
# Pre-flight: Bug fix 驗證
# ============================================================
echo "[Pre-flight A] Bug fix 驗證..."

python -c "
import re
with open('meanaudio/model/networks.py') as f:
    code = f.read()
mf_start = code.find('class MeanAudio')
mf_code = code[mf_start:]
if re.search(r'q = torch\.full\(\([^,]+,\), 9,', mf_code):
    raise SystemExit('[FAIL] Bug #1 未修')
print('[OK] Bug #1 fix 已套用')
" || exit 1

python -c "
with open('meanaudio/runner_meanflow.py') as f:
    code = f.read()
if 'text_f_undrop = text_f.clone()' not in code or 'text_f_c_undrop = text_f_c.clone()' not in code:
    raise SystemExit('[FAIL] Bug #2 未修')
print('[OK] Bug #2 fix 已套用')
" || exit 1

# ============================================================
# Pre-flight: NPZ 驗證
# ============================================================
echo "[Pre-flight B] Multi-cap NPZ 驗證..."
python "$HOME/research/meanaudio_training/validate_multicap_npz.py" \
    --tsv "$DATA_DIR/phase7_v1_train.tsv" \
    --npz_dir "$MULTICAP_NPZ" \
    --deep 200 \
    || { echo "[FAIL] NPZ validation failed"; exit 1; }

# ============================================================
# Pre-flight: TSV 有 q_level 欄位
# ============================================================
echo "[Pre-flight C] TSV q_level 欄位驗證..."
python -c "
import csv
with open('$DATA_DIR/phase7_v1_train.tsv') as f:
    r = csv.DictReader(f, delimiter='\t')
    first = next(r)
    if 'q_level' not in first:
        raise SystemExit('[FAIL] TSV 缺 q_level 欄位')
    print(f'[OK] q_level 欄位存在，第一筆 q_level={first[\"q_level\"]}')
" || exit 1

# ============================================================
# Stage 1
# ============================================================
S1_CKPT_COMPLETE=false
if [ -f "$S1_CKPT" ]; then
    CKPT_IT=$(python -c "import torch; c=torch.load('$S1_CKPT', map_location='cpu', weights_only=False); print(c.get('it', 0))" 2>/dev/null)
    if [ -z "$CKPT_IT" ]; then
        mv "$S1_CKPT" "${S1_CKPT}.corrupted_$(date +%Y%m%d_%H%M%S)"
    elif [ "$CKPT_IT" -ge "$S1_ITERATIONS" ]; then
        S1_CKPT_COMPLETE=true
        echo "[Stage 1] 已完成 (iter $CKPT_IT)"
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
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "$LOG_DIR/${EXP_S1}.log"

    echo "[Stage 1] 訓練完成"
fi

# ============================================================
# Migrate
# ============================================================
echo "[Migrate] $S1_CKPT → $S2_CKPT"
python "$MIGRATE_SCRIPT" --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT"

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
# Eval #1 (primary): MusicCaps --quality_level 9
# ============================================================
EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_q9_musiccaps"
echo "[Eval MusicCaps q=9] → $EVAL_OUT_MC"

python eval.py \
    --variant "meanaudio_s" \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT_MC/audio" \
    --tsv "$TSV_MUSICCAPS" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --quality_level 9 \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_q9_musiccaps_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_MC/audio" \
    --tsv "$TSV_MUSICCAPS" \
    --exp_name "${EXP_S2}_q9_musiccaps" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_q9_musiccaps_eval.log"

# ============================================================
# Eval #2 (optional): MusicCaps q=6
# ============================================================
if [ "$RUN_Q6_EVAL" = "true" ]; then
    EVAL_OUT_Q6="$WORK_DIR/eval_output/${EXP_S2}_q6_musiccaps"
    echo "[Eval MusicCaps q=6] → $EVAL_OUT_Q6"

    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT_Q6/audio" \
        --tsv "$TSV_MUSICCAPS" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --quality_level 6 \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_q6_musiccaps_eval.log"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT_Q6/audio" \
        --tsv "$TSV_MUSICCAPS" \
        --exp_name "${EXP_S2}_q6_musiccaps" \
        --num_samples 2048 \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_q6_musiccaps_eval.log"
fi

# ============================================================
# Eval #3 (optional): Jamendo q=9
# ============================================================
if [ "$RUN_JAMENDO_EVAL" = "true" ]; then
    EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_q9_jamendo"
    echo "[Eval Jamendo q=9] → $EVAL_OUT_JM"

    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT_JM/audio" \
        --tsv "$TSV_JAMENDO" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --quality_level 9 \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_q9_jamendo_eval.log"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT_JM/audio" \
        --tsv "$TSV_JAMENDO" \
        --exp_name "${EXP_S2}_q9_jamendo" \
        --num_samples 2048 \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_q9_jamendo_eval.log"
fi

# ============================================================
# Eval #4 (optional): native_q (TSV 讀每個 clip 自己的 q_level)
# ============================================================
if [ "$RUN_NATIVE_Q_EVAL" = "true" ]; then
    EVAL_OUT_NATIVE="$WORK_DIR/eval_output/${EXP_S2}_native_q_jamendo"
    echo "[Eval native_q Jamendo] → $EVAL_OUT_NATIVE"

    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT_NATIVE/audio" \
        --tsv "$TSV_NATIVE" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_native_q_eval.log"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT_NATIVE/audio" \
        --tsv "$TSV_NATIVE" \
        --exp_name "${EXP_S2}_native_q" \
        --num_samples 2048 \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_native_q_eval.log"
fi

# ============================================================
# 完成
# ============================================================
echo "======================================================"
echo "  Phase 9 V2 BUG-FIX 完成"
echo "  S1 EMA   : exps/$EXP_S1/${EXP_S1}_ema_final.pth"
echo "  S2 EMA   : $S2_EMA"
echo "  Primary  : eval_output/metrics/${EXP_S2}_q9_musiccaps/metrics.txt"
echo ""
echo "  基線對照:"
echo "    Phase 7 V1 (best): MusicCaps CLAP ~0.1975"
echo "    Phase 8:           MusicCaps CLAP 0.1851"
echo "    P9 V1 bugfix:      MusicCaps CLAP 0.0650"
echo "======================================================"
