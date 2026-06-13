#!/usr/bin/env bash
set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
PREFIX="music_flamingo_slice10_10k_short_rewrite"
CAPTION_OUT="$HOME/eval_output/$PREFIX"

python "$WORK_DIR/scripts/preprocess/rewrite_music_flamingo_short_captions.py" \
    --input-jsonl "$HOME/eval_output/music_flamingo_slice10_10k/caption.jsonl" \
    --output-jsonl "$CAPTION_OUT/caption.jsonl"

python "$WORK_DIR/scripts/preprocess/prepare_music_flamingo_slice10_train.py" \
    --captions-jsonl "$CAPTION_OUT/caption.jsonl" \
    --phase4-test-tsv "$DATA_DIR/phase4_test.tsv" \
    --out-dir "$DATA_DIR" \
    --n 10000 \
    --prefix "$PREFIX"

EXP_PREFIX="${EXP_PREFIX:-mfshort10k_rewrite_noq_fast}" \
EXPECTED_N=10000 \
TRAIN_TSV="$DATA_DIR/${PREFIX}_train.tsv" \
CLIPS_TSV="$DATA_DIR/${PREFIX}_clips.tsv" \
JAMENDO_HOLDOUT_TSV="$DATA_DIR/${PREFIX}_jamendo_holdout2048.tsv" \
NPZ_DIR="${NPZ_DIR:-/mnt/HDD/kojiek/${PREFIX}_npz}" \
LATENT_DIR="${LATENT_DIR:-/mnt/HDD/kojiek/${PREFIX}_latents_tmp}" \
NPZ_TSV="${NPZ_TSV:-/mnt/HDD/kojiek/${PREFIX}_npz.tsv}" \
S1_ITERATIONS="${S1_ITERATIONS:-100000}" \
S2_ITERATIONS="${S2_ITERATIONS:-50000}" \
bash "$WORK_DIR/scripts/training_pipelines/train_pipeline_slice10_from_tsv.sh"
