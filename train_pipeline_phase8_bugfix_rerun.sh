#!/bin/bash
# ============================================================
# MeanAudio Phase 8 — BUG-FIX RERUN (full S1+S2 retrain)
# train_pipeline_phase8_bugfix_rerun.sh
#
# 目的：單 caption 情境下，驗證 bug fix 對 no-Q 模型的影響
#       用於判讀 P9 V1 bug-fix rerun 的結果是否「multi_cap 特有」
#
# 注意：Phase 8 的原始 S1 ckpt_last.pth 已不存在（只剩 _ema_final.pth），
#       因此必須 S1 + S2 都從頭重訓。整個 pipeline ~19-20 hr。
#       如果 P9 V1 bug-fix rerun 結果已足夠說明問題（CLAP 大幅恢復），
#       此腳本可以跳過，節省 12 hr S1 訓練時間。
#
#   Bug #1 (critical): networks.py MeanAudio.forward/ode_wrapper
#                      q=None → 填 9（應為 10 null token）
#   Bug #2 (minor):    runner_meanflow.py text_f_undrop = text_f 別名
#
# 設計：
#   - S1: 從頭訓練 400k iter（原 ckpt 遺失，code 已修）
#         single-cap NPZ (phase7_v1 random seed=42 已寫入 TSV)
#   - S2: 從頭訓練 200k iter，single-cap NPZ，no-Q
#   - Eval: MusicCaps (primary)、Jamendo (optional)
#
# 基線對照（修前）：
#   Jamendo:   CLAP 0.1851
#   MusicCaps: CLAP 0.1851, CE 5.91, CU 6.75, PC 4.98, PQ 6.54
#
# 判讀標準（與修前對比）：
#   MusicCaps CLAP 提升 → bug fix 對 no-Q 有正向影響
#   MusicCaps CLAP 持平 → bug 在 single-cap 情境影響不顯著
#   MusicCaps CLAP 下降 → 修錯了、或 code 其他變動污染
#
# 使用方式：
#   tmux new -s phase8_bugfix
#   cd ~/MeanAudio && source ~/venvs/dac/bin/activate
#   bash train_pipeline_phase8_bugfix_rerun.sh
# ============================================================

set -e
set -o pipefail

# ============================================================
# 實驗參數
# ============================================================

EXP_PREFIX="phase8_bugfix_rerun"

BATCH_SIZE=8
ACCUM_STEPS=1
S1_ITERATIONS=400000
S2_ITERATIONS=200000

LEARNING_RATE=1e-4
USE_Q_CONDITIONING=false   # no Q

RUN_JAMENDO_EVAL=false     # true 會多跑 ~3 hr

# ============================================================
# 固定路徑
# ============================================================

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

SINGLECAP_NPZ="$HOME/research/meanaudio_training/npz"   # 原 phase7_v1 的 NPZ（每 clip 一個 caption）
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

TSV_MUSICCAPS="$DATA_DIR/musiccaps_test.tsv"
TSV_JAMENDO="$DATA_DIR/phase4_test.tsv"

S2_MACRO=$(( S2_ITERATIONS / ACCUM_STEPS ))

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
    "++multi_cap=False"
)

mkdir -p "$LOG_DIR"
mkdir -p "$WORK_DIR/exps/$EXP_S1" "$WORK_DIR/exps/$EXP_S2"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  Phase 8 — BUG-FIX RERUN (full S1+S2 retrain)"
echo "  S1 exp_id    : $EXP_S1"
echo "  S2 exp_id    : $EXP_S2"
echo "  NPZ dir      : $SINGLECAP_NPZ"
echo "  Train TSV    : $DATA_DIR/phase7_v1_train.tsv"
echo "  multi_cap    : False"
echo "  use_q_cond   : $USE_Q_CONDITIONING"
echo "  Eval MusicCaps: $TSV_MUSICCAPS"
echo "  Eval Jamendo : skip=$([ $RUN_JAMENDO_EVAL = true ] && echo NO || echo YES)"
echo "  Estimated ETA: ~19-20 hr (S1 12.3 + S2 6.7 + eval)"
echo "======================================================"

# ============================================================
# Pre-flight: Bug fix 驗證
# ============================================================
echo "[Pre-flight] 驗證 bug fix 已套用..."

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
# Eval #1: MusicCaps (primary)
# ============================================================
EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_musiccaps"
echo "[Eval MusicCaps] → $EVAL_OUT_MC"

python eval.py \
    --variant "meanaudio_s" \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT_MC/audio" \
    --tsv "$TSV_MUSICCAPS" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_musiccaps_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_MC/audio" \
    --tsv "$TSV_MUSICCAPS" \
    --exp_name "${EXP_S2}_musiccaps" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_musiccaps_eval.log"

# ============================================================
# Eval #2: Jamendo (optional)
# ============================================================
if [ "$RUN_JAMENDO_EVAL" = "true" ]; then
    EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_jamendo"
    echo "[Eval Jamendo] → $EVAL_OUT_JM"

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

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT_JM/audio" \
        --exp_name "${EXP_S2}_jamendo" \
        --num_samples 2048 \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_jamendo_eval.log"
fi

echo "======================================================"
echo "  Phase 8 BUG-FIX RERUN 完成"
echo "  S1 EMA   : exps/$EXP_S1/${EXP_S1}_ema_final.pth"
echo "  S2 EMA   : $S2_EMA"
echo "  MusicCaps: eval_output/metrics/${EXP_S2}_musiccaps/metrics.txt"
echo ""
echo "  基線（修前）:"
echo "    MusicCaps: CLAP 0.1851, CE 5.91, CU 6.75, PC 4.98, PQ 6.54"
echo "    Jamendo:   CLAP 0.1851"
echo "======================================================"
