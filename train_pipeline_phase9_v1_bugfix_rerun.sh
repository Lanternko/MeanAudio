#!/bin/bash
# ============================================================
# MeanAudio Phase 9 V1 — BUG-FIX RERUN
# train_pipeline_phase9_v1_bugfix_rerun.sh
#
# 目的：驗證兩個 S2-only bug 修復是否讓 P9 V1 CLAP 從 0.0260 恢復
#
#   Bug #1 (critical): networks.py MeanAudio.forward/ode_wrapper
#                      q=None → 填 9（應為 10 null token）
#                      造成 use_q_conditioning=False 實驗 train/eval mismatch
#   Bug #2 (minor):    runner_meanflow.py text_f_undrop = text_f 別名
#                      in-place null mask 污染 CFG target
#
# 設計：
#   - S1: 重用 P9 V1 原有 ckpt（S1 不受 bug 影響）
#   - S2: 重新訓練 200k iter，code 已修 + multi_cap=True + no-Q
#   - Eval: MusicCaps (primary, 5527 clips, ~11 min)
#           Jamendo (optional secondary, 90063 clips, ~3 hr)
#
# 基線對照（修前）：
#   Jamendo:   CLAP 0.0260, CE 4.76, CU 5.42, PC 5.37, PQ 5.71
#   MusicCaps: 無歷史數字，本次為首次
#
# Phase 8 MusicCaps 基線（no-Q 同類型，供參考）：
#   CLAP 0.1851, CE 5.91, CU 6.75, PC 4.98, PQ 6.54
#
# 判讀標準：
#   CLAP ≥ 0.18  → bug fix 完全解釋崩盤，experiment design 成立
#   CLAP 0.10-0.18 → bug fix 有大幅貢獻，但可能還有其他問題
#   CLAP < 0.10  → bug 之外還有問題（multi_cap S1 污染？需聽音訊）
#
# 使用方式：
#   tmux new -s p9v1_bugfix
#   cd ~/MeanAudio && source ~/venvs/dac/bin/activate
#   bash train_pipeline_phase9_v1_bugfix_rerun.sh
# ============================================================

set -e
set -o pipefail

# ============================================================
# 實驗參數
# ============================================================

EXP_PREFIX="phase9_v1_bugfix_rerun"

BATCH_SIZE=8
ACCUM_STEPS=1

# S1 重用既有 ckpt，不重訓
S1_ITERATIONS=400000
S2_ITERATIONS=200000

LEARNING_RATE=1e-4
USE_Q_CONDITIONING=false   # no Q

# Eval 範圍：默認只跑 MusicCaps（快速），Jamendo 可開
RUN_JAMENDO_EVAL=false     # 設為 true 會多跑 ~3 hr

# ============================================================
# 固定路徑
# ============================================================

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

# S1: 重用既有 P9 V1 ckpt
S1_SRC_EXP="phase9_v1_stage1_400000"
S1_CKPT="$WORK_DIR/exps/$S1_SRC_EXP/${S1_SRC_EXP}_ckpt_last.pth"

# S2: 新建資料夾
EXP_S2="${EXP_PREFIX}_stage2_${S2_ITERATIONS}"
S2_CKPT="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ckpt_last.pth"
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"

MULTICAP_NPZ="$HOME/phase9_multicap_npz"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

TSV_MUSICCAPS="$DATA_DIR/musiccaps_test.tsv"
TSV_JAMENDO="$DATA_DIR/phase4_test.tsv"

S2_MACRO=$(( S2_ITERATIONS / ACCUM_STEPS ))

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
mkdir -p "$WORK_DIR/exps/$EXP_S2"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  Phase 9 V1 — BUG-FIX RERUN"
echo "  S1 source    : $S1_SRC_EXP (reused)"
echo "  S2 exp_id    : $EXP_S2"
echo "  NPZ dir      : $MULTICAP_NPZ"
echo "  Train TSV    : $DATA_DIR/phase7_v1_train.tsv"
echo "  multi_cap    : True"
echo "  use_q_cond   : $USE_Q_CONDITIONING"
echo "  Eval MusicCaps: $TSV_MUSICCAPS"
echo "  Eval Jamendo : $TSV_JAMENDO (skip=$([ $RUN_JAMENDO_EVAL = true ] && echo NO || echo YES))"
echo "======================================================"

# ============================================================
# Pre-flight A: Bug fix 是否已套用
# ============================================================
echo "[Pre-flight A] 驗證 bug fix 已套用..."

# Bug #1: networks.py MeanAudio.forward/ode_wrapper 必須填 10，不是 9
python -c "
import re
with open('meanaudio/model/networks.py') as f:
    code = f.read()
# 找 MeanAudio 類別（L360 附近開始）
mf_start = code.find('class MeanAudio')
mf_code = code[mf_start:]
# 檢查是否還有 'q = torch.full((latent.shape[0],), 9' 或 'q = torch.full((len(latent),), 9'
if re.search(r'q = torch\.full\(\([^,]+,\), 9,', mf_code):
    raise SystemExit('[FAIL] Bug #1 未修：MeanAudio q=None 仍填 9')
print('[OK] Bug #1 fix 已套用')
" || exit 1

# Bug #2: runner_meanflow.py 必須 .clone()
python -c "
with open('meanaudio/runner_meanflow.py') as f:
    code = f.read()
if 'text_f_undrop = text_f.clone()' not in code or 'text_f_c_undrop = text_f_c.clone()' not in code:
    raise SystemExit('[FAIL] Bug #2 未修：runner_meanflow.py 仍是別名不是 .clone()')
print('[OK] Bug #2 fix 已套用')
" || exit 1

# ============================================================
# Pre-flight B: NPZ 驗證
# ============================================================
echo "[Pre-flight B] 驗證 multi-cap NPZ 目錄..."
python "$HOME/research/meanaudio_training/validate_multicap_npz.py" \
    --tsv "$DATA_DIR/phase7_v1_train.tsv" \
    --npz_dir "$MULTICAP_NPZ" \
    --deep 200 \
    || { echo "[FAIL] NPZ validation failed."; exit 1; }

# ============================================================
# Pre-flight C: S1 ckpt 存在且可讀
# ============================================================
echo "[Pre-flight C] 驗證 S1 ckpt..."
[ -f "$S1_CKPT" ] || { echo "[FAIL] S1 ckpt 不存在：$S1_CKPT"; exit 1; }
CKPT_IT=$(python -c "import torch; c=torch.load('$S1_CKPT', map_location='cpu', weights_only=False); print(c.get('it', 0))")
[ "$CKPT_IT" -ge "$S1_ITERATIONS" ] || { echo "[FAIL] S1 ckpt 未完成 (iter $CKPT_IT / $S1_ITERATIONS)"; exit 1; }
echo "[OK] S1 ckpt 完整，iter $CKPT_IT"

# ============================================================
# Migrate S1 → S2
# ============================================================
echo "[Migrate] $S1_CKPT → $S2_CKPT"
python "$MIGRATE_SCRIPT" --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT"

# ============================================================
# Stage 2 訓練
# ============================================================
echo "[Stage 2] 開始訓練：$EXP_S2"
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
# Eval #1: MusicCaps (primary)
# ============================================================
EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_musiccaps"
echo "[Eval MusicCaps] 生成音訊 → $EVAL_OUT_MC"

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
# Eval #2: Jamendo (optional, for historical comparison)
# ============================================================
if [ "$RUN_JAMENDO_EVAL" = "true" ]; then
    EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_jamendo"
    echo "[Eval Jamendo] 生成音訊 → $EVAL_OUT_JM"

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

# ============================================================
# 完成
# ============================================================
echo "======================================================"
echo "  Phase 9 V1 BUG-FIX RERUN 完成"
echo "  S2 EMA   : $S2_EMA"
echo "  MusicCaps: eval_output/metrics/${EXP_S2}_musiccaps/metrics.txt"
if [ "$RUN_JAMENDO_EVAL" = "true" ]; then
echo "  Jamendo  : eval_output/metrics/${EXP_S2}_jamendo/metrics.txt"
fi
echo ""
echo "  基線（修前）: Jamendo CLAP 0.0260"
echo "  Phase 8 參考: MusicCaps CLAP 0.1851"
echo "======================================================"
