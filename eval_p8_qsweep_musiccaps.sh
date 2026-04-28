#!/bin/bash
# ============================================================
# P8 NoQ baseline q-sweep on MusicCaps (q=5..9)
# eval_p8_qsweep_musiccaps.sh
#
# 用途：拿 P8 NoQ 既有 ckpt 跑 --quality_level 5..9 eval，
#       作為 Fig.2 的 control。NoQ 訓練只更新 q_embed[10]，
#       q_embed[5..9] 維持 random init。
#
# 三種可能結果：
#   - ≈ baseline (0.1851) → q_embed pathway 本身不重要
#   - > baseline → 連未訓練的 pathway 都加值
#   - < baseline → random q_embed 主動干擾
#
# 注意：違反 reference_eval_q_flag_rule.md（NoQ 模型用 --quality_level）
#       這次是「故意的 control」，記得在 docs 標 caveat。
#
# 預估：5 q × ~12 min = ~1 hr
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP="phase8_stage2_200000"
EMA="$WORK_DIR/exps/$EXP/${EXP}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

if [ ! -f "$EMA" ]; then
    echo "❌ EMA ckpt 不存在：$EMA"
    exit 1
fi

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  P8 NoQ q-sweep on MusicCaps (q=5..9, control)"
echo "  EMA   : $EMA"
echo "  TSV   : $DATA_DIR/musiccaps_test.tsv (n=5521, no prefix)"
echo "  ⚠️  q_embed[5..9] never trained (NoQ regime) — random init"
echo "======================================================"

for Q in 5 6 7 8 9; do
    EVAL_OUT="$WORK_DIR/eval_output/${EXP}_q${Q}_musiccaps_qsweep_control"
    LOG="$LOG_DIR/${EXP}_q${Q}_musiccaps_qsweep_control_eval.log"
    METRICS_FILE="$WORK_DIR/eval_output/metrics/${EXP}_q${Q}_musiccaps_qsweep_control/metrics.txt"
    MARKER_FILE="$WORK_DIR/eval_output/metrics/${EXP}_q${Q}_musiccaps_qsweep_control/.run_manifest"

    if [ -f "$METRICS_FILE" ] && [ -f "$MARKER_FILE" ]; then
        EXPECTED_MARKER="$(printf '%s\n%s\n%s' "$EMA" "$DATA_DIR/musiccaps_test.tsv" 5521)"
        ACTUAL_MARKER="$(cat $MARKER_FILE)"
        if [ "$EXPECTED_MARKER" = "$ACTUAL_MARKER" ]; then
            echo "[q=$Q] ✅ metrics 已存在且 marker 一致，skip"
            continue
        fi
    fi

    echo "[q=${Q}] gen → $EVAL_OUT"
    python eval.py \
        --variant "meanaudio_s" \
        --model_path "$EMA" \
        --output "$EVAL_OUT/audio" \
        --tsv "$DATA_DIR/musiccaps_test.tsv" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --quality_level $Q \
        --full_precision \
        2>&1 | tee "$LOG"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT/audio" \
        --tsv "$DATA_DIR/musiccaps_test.tsv" \
        --exp_name "${EXP}_q${Q}_musiccaps_qsweep_control" \
        --num_samples 5521 \
        2>&1 | tee -a "$LOG"

    mkdir -p "$(dirname $MARKER_FILE)"
    printf '%s\n%s\n%s' "$EMA" "$DATA_DIR/musiccaps_test.tsv" 5521 > "$MARKER_FILE"
done

echo "======================================================"
echo "  P8 NoQ q-sweep control 完成（q=5..9 × MusicCaps n=5521）"
echo "  Metrics: eval_output/metrics/${EXP}_q{5..9}_musiccaps_qsweep_control/"
echo "  Compare to P8 NoQ baseline (--no_q): CLAP 0.1851"
echo "======================================================"
