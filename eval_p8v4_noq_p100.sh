#!/bin/bash
# ============================================================
# P8 V4 NoQ + [consistency=1.00] prefix eval（補充消融）
# eval_p8v4_noq_p100.sh
#
# Codex review 2026-04-27 P1：metric ref 必須與 generation prompt 對應。
# 改成跑兩個 ref pass：
#   - prefixed_ref：metric 用 generation 同款 prefixed TSV → 真正測 prompt-following
#   - natural_ref：metric 用原始未 prefix TSV → 測 cross-format alignment to natural caption
# 兩個 metric 都報，不要只報一個（會被誤讀為「prompt-following 崩了」）。
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
P100_TSV_DIR="$HOME/eval_tsvs_p100"
LOG_DIR="$HOME/logs"

EXP_S2="phase8_v4_stage2_200000"
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

if [ ! -f "$S2_EMA" ]; then
    echo "❌ EMA ckpt 不存在：$S2_EMA"
    exit 1
fi

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  P8 V4 NoQ + [consistency=1.00] eval (dual-ref)"
echo "  EMA   : $S2_EMA"
echo "======================================================"

# ── helper: dual-ref metric pass ─────────────────────────────
# args: $1=audio_dir  $2=base_exp_name  $3=prefixed_tsv  $4=natural_tsv  $5=num_samples  $6=log_path
dual_ref_metrics() {
    local audio_dir="$1"
    local base="$2"
    local prefixed_tsv="$3"
    local natural_tsv="$4"
    local n="$5"
    local log="$6"

    echo "[Metric / prefixed_ref] $base" | tee -a "$log"
    python "$EVAL_SCRIPT" \
        --gen_dir "$audio_dir" \
        --tsv "$prefixed_tsv" \
        --exp_name "${base}_prefixed_ref" \
        --num_samples "$n" \
        2>&1 | tee -a "$log"

    echo "[Metric / natural_ref]  $base" | tee -a "$log"
    python "$EVAL_SCRIPT" \
        --gen_dir "$audio_dir" \
        --tsv "$natural_tsv" \
        --exp_name "${base}_natural_ref" \
        --num_samples "$n" \
        2>&1 | tee -a "$log"
}

# ── Eval 1: MusicCaps (primary) ─────────────────────────────
EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_no_q_musiccaps_p100"
LOG_MC="$LOG_DIR/${EXP_S2}_no_q_musiccaps_p100_eval.log"

if [ -d "$EVAL_OUT_MC/audio" ] && [ "$(ls -1 $EVAL_OUT_MC/audio/*.flac 2>/dev/null | wc -l)" -ge 5520 ]; then
    echo "[MC p100] audio 已生成，跳過 gen" | tee "$LOG_MC"
else
    echo "[MC p100] gen → $EVAL_OUT_MC"
    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT_MC/audio" \
        --tsv "$P100_TSV_DIR/phase8_v4_musiccaps_test_p100.tsv" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --no_q \
        --full_precision \
        2>&1 | tee "$LOG_MC"
fi

dual_ref_metrics \
    "$EVAL_OUT_MC/audio" \
    "${EXP_S2}_no_q_musiccaps_p100" \
    "$P100_TSV_DIR/phase8_v4_musiccaps_test_p100.tsv" \
    "$DATA_DIR/musiccaps_test.tsv" \
    5521 \
    "$LOG_MC"

# ── Eval 2: Jamendo seed=42 2048 (secondary) ─────────────────
EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_no_q_jamendo_seed42_2048_p100"
LOG_JM="$LOG_DIR/${EXP_S2}_no_q_jamendo_p100_eval.log"

if [ -d "$EVAL_OUT_JM/audio" ] && [ "$(ls -1 $EVAL_OUT_JM/audio/*.flac 2>/dev/null | wc -l)" -ge 2047 ]; then
    echo "[JM p100] audio 已生成，跳過 gen" | tee "$LOG_JM"
else
    echo "[JM p100] gen → $EVAL_OUT_JM"
    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT_JM/audio" \
        --tsv "$P100_TSV_DIR/phase8_v4_jamendo_seed42_2048_p100.tsv" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --no_q \
        --full_precision \
        2>&1 | tee "$LOG_JM"
fi

dual_ref_metrics \
    "$EVAL_OUT_JM/audio" \
    "${EXP_S2}_no_q_jamendo_seed42_2048_p100" \
    "$P100_TSV_DIR/phase8_v4_jamendo_seed42_2048_p100.tsv" \
    "$DATA_DIR/phase4_test_seed42_2048.tsv" \
    2048 \
    "$LOG_JM"

echo "======================================================"
echo "  P8 V4 NoQ p100 eval 完成（dual-ref）"
echo "  Metrics: eval_output/metrics/${EXP_S2}_no_q_{musiccaps,jamendo_seed42_2048}_p100_{prefixed,natural}_ref/"
echo "======================================================"
