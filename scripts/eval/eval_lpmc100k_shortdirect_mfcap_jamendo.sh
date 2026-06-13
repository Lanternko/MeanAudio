#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EMA="$WORK_DIR/exps/lpmc100k_noq_stage2_100000/lpmc100k_noq_stage2_100000_ema_final.pth"
SHORTDIRECT_TSV="$DATA_DIR/music_flamingo_slice10_100k_short_direct_mfcap_jamendo_holdout2048.tsv"
EXP_NAME="lpmc100k_noq_stage2_100000_jamendo_holdout2048_shortdirect_mfcap"
EVAL_OUT="$WORK_DIR/eval_output/$EXP_NAME"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
PEAV_SCRIPT="$HOME/research/meanaudio_eval/peav_eval.py"

mkdir -p "$LOG_DIR" "$WORK_DIR/eval_output/metrics"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "======================================================"
echo "  Eval-only: LPMC 100k on short-direct MF Jamendo prompts"
echo "  EMA    : $EMA"
echo "  TSV    : $SHORTDIRECT_TSV"
echo "======================================================"

for path in "$EMA" "$SHORTDIRECT_TSV"; do
    if [ ! -f "$path" ]; then
        echo "[ABORT] missing file: $path"
        exit 2
    fi
done

python eval.py \
    --variant meanaudio_s \
    --model_path "$EMA" \
    --output "$EVAL_OUT/audio" \
    --tsv "$SHORTDIRECT_TSV" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT/audio" \
    --tsv "$SHORTDIRECT_TSV" \
    --exp_name "$EXP_NAME" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_NAME}_eval.log"

if [ -f "$PEAV_SCRIPT" ] && [ -x "$HOME/venvs/peav/bin/python" ]; then
    "$HOME/venvs/peav/bin/python" "$PEAV_SCRIPT" \
        --gen_dir "$EVAL_OUT/audio" \
        --tsv "$SHORTDIRECT_TSV" \
        --out "$WORK_DIR/eval_output/metrics/${EXP_NAME}_peav.json" \
        --batch_size 8 \
        2>&1 | tee "$LOG_DIR/${EXP_NAME}_peav.log"
fi

echo "[done] eval-only complete: $EXP_NAME"
