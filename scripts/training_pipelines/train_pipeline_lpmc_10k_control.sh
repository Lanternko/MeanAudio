#!/usr/bin/env bash
set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
PREFIX="lpmc_slice10_10k_control"

python "$WORK_DIR/scripts/preprocess/prepare_lpmc_slice10_control.py" \
    --ids-jsonl "$HOME/eval_output/music_flamingo_slice10_10k/caption.jsonl" \
    --lpmc-tsv "$DATA_DIR/phase4_test.tsv" \
    --out-dir "$DATA_DIR" \
    --n 10000 \
    --prefix "$PREFIX"

EXP_PREFIX="${EXP_PREFIX:-lpmc10k_noq_fast}" \
EXPECTED_N=10000 \
TRAIN_TSV="$DATA_DIR/${PREFIX}_train.tsv" \
CLIPS_TSV="$DATA_DIR/${PREFIX}_clips.tsv" \
JAMENDO_HOLDOUT_TSV="$DATA_DIR/${PREFIX}_jamendo_holdout2048.tsv" \
NPZ_DIR="${NPZ_DIR:-$HOME/exps_nvme/${PREFIX}_npz}" \
LATENT_DIR="${LATENT_DIR:-$HOME/exps_nvme/${PREFIX}_latents_tmp}" \
NPZ_TSV="${NPZ_TSV:-$HOME/exps_nvme/${PREFIX}_npz.tsv}" \
S1_ITERATIONS="${S1_ITERATIONS:-100000}" \
S2_ITERATIONS="${S2_ITERATIONS:-50000}" \
bash "$WORK_DIR/scripts/training_pipelines/train_pipeline_slice10_from_tsv.sh"
