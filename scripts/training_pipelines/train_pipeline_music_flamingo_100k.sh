#!/usr/bin/env bash
set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
PREFIX="music_flamingo_slice10_100k"
CAPTION_OUT="$HOME/eval_output/$PREFIX"
SOURCE_TSV="${SOURCE_TSV:-$DATA_DIR/_QUARANTINED_phase4_train.tsv}"

N=100000 \
OUT_DIR="$CAPTION_OUT" \
LOG="$HOME/logs/${PREFIX}_caption.log" \
TSV="$SOURCE_TSV" \
LPMC_TSV="$SOURCE_TSV" \
bash "$WORK_DIR/scripts/preprocess/music_flamingo_jamendo_slice10_10k.sh"

python "$WORK_DIR/scripts/preprocess/prepare_music_flamingo_slice10_train.py" \
    --captions-jsonl "$CAPTION_OUT/caption.jsonl" \
    --phase4-test-tsv "$DATA_DIR/phase4_test.tsv" \
    --out-dir "$DATA_DIR" \
    --n 100000 \
    --prefix "$PREFIX"

EXP_PREFIX="${EXP_PREFIX:-mf100k_noq}" \
EXPECTED_N=100000 \
TRAIN_TSV="$DATA_DIR/${PREFIX}_train.tsv" \
CLIPS_TSV="$DATA_DIR/${PREFIX}_clips.tsv" \
JAMENDO_HOLDOUT_TSV="$DATA_DIR/${PREFIX}_jamendo_holdout2048.tsv" \
NPZ_DIR="$HOME/exps_nvme/${PREFIX}_npz" \
LATENT_DIR="$HOME/exps_nvme/${PREFIX}_latents_tmp" \
NPZ_TSV="$HOME/exps_nvme/${PREFIX}_npz.tsv" \
S1_ITERATIONS="${S1_ITERATIONS:-200000}" \
S2_ITERATIONS="${S2_ITERATIONS:-100000}" \
bash "$WORK_DIR/scripts/training_pipelines/train_pipeline_slice10_from_tsv.sh"
