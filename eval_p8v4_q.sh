#!/bin/bash
# ============================================================
# P8 V4 Q (S2-only Q variant) eval — q sweep on both benchmarks
# eval_p8v4_q.sh
#
# Codex review 2026-04-27 P1：dual-ref metric pass。
#   - prefixed_ref：用 generation 同款 prefixed TSV
#   - natural_ref：用原始未 prefix TSV
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S2="phase8_v4_q_stage2_200000"
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

if [ ! -f "$S2_EMA" ]; then
    echo "❌ EMA 不存在：$S2_EMA"
    exit 1
fi

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  P8 V4 Q eval (q=6 + q=9 × MusicCaps + Jamendo seed42, dual-ref)"
echo "  EMA: $S2_EMA"
echo "======================================================"

dual_ref_metrics() {
    local audio_dir="$1"; local base="$2"; local prefixed_tsv="$3"
    local natural_tsv="$4"; local n="$5"; local log="$6"
    echo "[Metric / prefixed_ref] $base" | tee -a "$log"
    python "$EVAL_SCRIPT" --gen_dir "$audio_dir" --tsv "$prefixed_tsv" \
        --exp_name "${base}_prefixed_ref" --num_samples "$n" 2>&1 | tee -a "$log"
    echo "[Metric / natural_ref] $base" | tee -a "$log"
    python "$EVAL_SCRIPT" --gen_dir "$audio_dir" --tsv "$natural_tsv" \
        --exp_name "${base}_natural_ref" --num_samples "$n" 2>&1 | tee -a "$log"
}

for Q in 6 9; do
    # ── MusicCaps (primary) ───────────────────────────
    EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_q${Q}_musiccaps"
    LOG_MC="$LOG_DIR/${EXP_S2}_q${Q}_musiccaps_eval.log"

    if [ -d "$EVAL_OUT_MC/audio" ] && [ "$(ls -1 $EVAL_OUT_MC/audio/*.flac 2>/dev/null | wc -l)" -eq 5521 ]; then
        echo "[MC q=$Q] audio 已生成，跳過 gen" | tee "$LOG_MC"
    else
        echo "[Eval / MusicCaps q=${Q}] gen → $EVAL_OUT_MC"
        python eval.py \
            --variant "meanaudio_s" \
            --model_path "$S2_EMA" \
            --output "$EVAL_OUT_MC/audio" \
            --tsv "$DATA_DIR/phase8_v4_musiccaps_test.tsv" \
            --use_meanflow --num_steps 1 \
            --encoder_name t5_clap --text_c_dim 512 \
            --cfg_strength 0.5 --quality_level $Q \
            --full_precision \
            2>&1 | tee "$LOG_MC"
    fi

    dual_ref_metrics \
        "$EVAL_OUT_MC/audio" \
        "${EXP_S2}_q${Q}_musiccaps" \
        "$DATA_DIR/phase8_v4_musiccaps_test.tsv" \
        "$DATA_DIR/musiccaps_test.tsv" \
        5521 "$LOG_MC"

    # ── Jamendo seed=42 2048 (secondary) ──────────────
    EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_q${Q}_jamendo_seed42_2048"
    LOG_JM="$LOG_DIR/${EXP_S2}_q${Q}_jamendo_eval.log"

    if [ -d "$EVAL_OUT_JM/audio" ] && [ "$(ls -1 $EVAL_OUT_JM/audio/*.flac 2>/dev/null | wc -l)" -eq 2048 ]; then
        echo "[JM q=$Q] audio 已生成，跳過 gen" | tee "$LOG_JM"
    else
        echo "[Eval / Jamendo seed42 q=${Q}] gen → $EVAL_OUT_JM"
        python eval.py \
            --variant "meanaudio_s" \
            --model_path "$S2_EMA" \
            --output "$EVAL_OUT_JM/audio" \
            --tsv "$DATA_DIR/phase8_v4_jamendo_seed42_2048.tsv" \
            --use_meanflow --num_steps 1 \
            --encoder_name t5_clap --text_c_dim 512 \
            --cfg_strength 0.5 --quality_level $Q \
            --full_precision \
            2>&1 | tee "$LOG_JM"
    fi

    dual_ref_metrics \
        "$EVAL_OUT_JM/audio" \
        "${EXP_S2}_q${Q}_jamendo_seed42_2048" \
        "$DATA_DIR/phase8_v4_jamendo_seed42_2048.tsv" \
        "$DATA_DIR/phase4_test_seed42_2048.tsv" \
        2048 "$LOG_JM"
done

echo "======================================================"
echo "  P8 V4 Q eval 完成（dual-ref，q={6,9} × {MC, Jamendo}）"
echo "  Metrics suffix: _prefixed_ref / _natural_ref"
echo "======================================================"
