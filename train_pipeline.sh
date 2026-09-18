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

set -eo pipefail  # -e: 任何指令失敗即中止；-o pipefail: pipe exit code 取第一個非 0
                  # 沒有 pipefail 時 `torchrun ... | tee log` 的 exit code = tee 的 0，
                  # 即使 torchrun crash 也會繼續跑後面 && 串起的 eval chain（2026-04-22 踩坑）

# ============================================================
# 實驗參數設定（每次新實驗只需修改此區塊）
# ============================================================

EXP_PREFIX="phase8_v4"                  # 實驗名稱前綴，自動生成 exp_id
                                        # P8 V4 = JamendoFull-Random-PromptConsistency-NoQ
                                        # caption 帶 [consistency=X.XX] prefix（raw float），
                                        # 訊號全走 text encoder，不依賴 q_embed。
                                        # eval inference prefix 固定 [consistency=0.90]（in-support）

BATCH_SIZE=8                      # 物理 batch size（每張 GPU）
ACCUM_STEPS=1                     # Gradient accumulation 步數（V4 不使用累積）
                                  # 等效 batch size = BATCH_SIZE × ACCUM_STEPS

S1_ITERATIONS=400000              # Stage 1 總 micro-steps
S2_ITERATIONS=200000              # Stage 2 總 micro-steps

LEARNING_RATE=1e-4                # 初始學習率（Stage 1 & 2 共用）

USE_Q_CONDITIONING=false          # P8 V4: false → consistency 訊號走 text prefix，不走 q_embed
TEXT_ATTENTION_MASK=false         # 現行協定 NoMask：訓練與 eval 一致（train 預設是 True，必須明寫）

TRAIN_TSV="phase8_v4_train.tsv"   # 相對 DATA_DIR
NPZ_DIR="$HOME/research/meanaudio_training/npz_phase8v4"

# ── Eval（MusicCaps MF25 × {CFG0, CFG3+neg}，見 scripts/eval/mc_mf25_eval.sh）──
EVAL_GEN_TSV="phase8_v4_musiccaps_test.tsv"  # 生成用 TSV（相對 DATA_DIR）；一般實驗用 musiccaps_test.tsv，
                                             # 只有 prefix 訓練（P8 V4）要用 prefixed 版；CLAP 一律對原始 caption 算
EVAL_Q_LEVELS="9 0"               # 只在 USE_Q_CONDITIONING=true 時使用；NoQ 一律 --no_q

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

# Stage 2 patches mean_flow.py in place; contracts hash-pin the Stage 1 (HEAD) form.
# Restore on every exit so a later pinned job is not held. Idempotent.
restore_stage_1() { python "$STAGE_SCRIPT" --stage 1 >/dev/null 2>&1 || true; }
trap restore_stage_1 EXIT

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
    "+use_text_attention_mask=$TEXT_ATTENTION_MASK"
    val_interval=999999
    eval_interval=999999
    save_eval_interval=999999
    "data.AudioCaps_npz.tsv=$DATA_DIR/$TRAIN_TSV"
    "data.AudioCaps_val_npz.tsv=$DATA_DIR/phase4_val.tsv"
    "+data.AudioCaps_npz.gt_cache=$DATA_DIR/npz_cache_train.txt"
    "+data.AudioCaps_val_npz.gt_cache=$DATA_DIR/npz_cache_val.txt"
    "++data.AudioCaps_npz.npz_dir=$NPZ_DIR"
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
# Eval：Stage 2 最終結果 — MusicCaps 5521 / MF25 / seed 42 / fp32，
#       兩格 CFG0 與 CFG3+neg（fidelity8），CLAP batch 1（eval_metrics.py）
#       Jamendo 只在要跟 Phase 4-8 舊數字對照時另外跑
# ============================================================

S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_WRAPPER="$WORK_DIR/scripts/eval/mc_mf25_eval.sh"
EVAL_ARGS=(--gen_tsv "$DATA_DIR/$EVAL_GEN_TSV")
if [ "$TEXT_ATTENTION_MASK" = "true" ]; then EVAL_ARGS+=(--mask); fi

if [ "$USE_Q_CONDITIONING" = "false" ]; then
    echo "[Eval S2] NoQ → --no_q（null token q=10）"
    bash "$EVAL_WRAPPER" "$EXP_S2" "$S2_EMA" --no_q "${EVAL_ARGS[@]}" \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_eval.log"
else
    for Q in $EVAL_Q_LEVELS; do
        echo "[Eval S2] Q-trained → --quality_level $Q"
        bash "$EVAL_WRAPPER" "$EXP_S2" "$S2_EMA" --quality_level "$Q" "${EVAL_ARGS[@]}" \
            2>&1 | tee -a "$LOG_DIR/${EXP_S2}_eval.log"
    done
fi

# ============================================================
# 完成
# ============================================================

echo "======================================================"
echo "  Phase 訓練 + Eval 完成"
echo "  S1 EMA    : exps/$EXP_S1/${EXP_S1}_ema_final.pth"
echo "  S2 EMA    : exps/$EXP_S2/${EXP_S2}_ema_final.pth"
echo "  Metrics   : ~/eval_output_nvme/${EXP_S2}_mc_mf25_{cfg0,cfg3_neg}*/"
echo "======================================================"
