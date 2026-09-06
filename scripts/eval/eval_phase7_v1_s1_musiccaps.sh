#!/usr/bin/env bash
# Native Stage-1 evaluation for the historical P7V1 checkpoint.
# Matched to eval_phase8_catalog_noq_s1_musiccaps.sh:
# FluxAudio, 25 FM steps, CFG 4.5, NoQ/null q=10, legacy NoMask, MusicCaps 5521.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
EXP=phase7_v1_stage1_400000_musiccaps_fm25_noq_nomask
WEIGHTS="$ROOT/exps/phase7_v1_stage1_400000/phase7_v1_stage1_400000_ema_final.pth"
TSV="$DATA/musiccaps_test.tsv"
OUT="$ROOT/eval_output/$EXP"
METRICS="$ROOT/eval_output/metrics/$EXP/metrics.txt"
LOG="$LOG_ROOT/${EXP}_eval.log"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py

cd "$ROOT"
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

for path in "$WEIGHTS" "$TSV" "$EVALUATOR"; do
    if [ ! -f "$path" ]; then
        echo "[FAIL] missing required file: $path" >&2
        exit 2
    fi
done

if [ -f "$METRICS" ]; then
    echo "[SKIP] metrics already exist: $METRICS"
    cat "$METRICS"
    exit 0
fi

mkdir -p "$OUT/audio" "$(dirname "$METRICS")"

python eval.py \
    --variant fluxaudio_s \
    --model_path "$WEIGHTS" \
    --output "$OUT/audio" \
    --tsv "$TSV" \
    --num_steps 25 \
    --cfg_strength 4.5 \
    --encoder_name t5_clap \
    --text_c_dim 512 \
    --no_q \
    --no_text_attention_mask \
    --full_precision \
    2>&1 | tee "$LOG"

audio_n=$(find "$OUT/audio" -maxdepth 1 -type f -name '*.flac' | wc -l)
if [ "$audio_n" -ne 5521 ]; then
    echo "[FAIL] generated $audio_n/5521 audio files" | tee -a "$LOG" >&2
    exit 2
fi

python "$EVALUATOR" \
    --gen_dir "$OUT/audio" \
    --tsv "$TSV" \
    --exp_name "$EXP" \
    --num_samples 5521 \
    2>&1 | tee -a "$LOG"

if [ ! -f "$METRICS" ]; then
    echo "[FAIL] evaluator did not create $METRICS" | tee -a "$LOG" >&2
    exit 2
fi

echo "[COMPLETE] $EXP" | tee -a "$LOG"
cat "$METRICS"
