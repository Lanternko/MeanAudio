#!/usr/bin/env bash
set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
PREFIX="music_flamingo_slice10_10k_short_direct"
CAPTION_OUT="$HOME/eval_output/$PREFIX"
SOURCE_TSV="${SOURCE_TSV:-$DATA_DIR/phase4_test.tsv}"

N=10000 \
OUT_DIR="$CAPTION_OUT" \
LOG="$HOME/logs/${PREFIX}_caption.log" \
TSV="$SOURCE_TSV" \
LPMC_TSV="$SOURCE_TSV" \
PROMPT_PRESET="short_direct_v1" \
PROMPT_VERSION="slice10_short_direct_v1" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}" \
bash "$WORK_DIR/scripts/preprocess/music_flamingo_jamendo_slice10_10k.sh"

python "$WORK_DIR/scripts/preprocess/prepare_music_flamingo_slice10_train.py" \
    --captions-jsonl "$CAPTION_OUT/caption.jsonl" \
    --phase4-test-tsv "$DATA_DIR/phase4_test.tsv" \
    --out-dir "$DATA_DIR" \
    --n 10000 \
    --prefix "$PREFIX"

EXP_PREFIX="${EXP_PREFIX:-mfshort10k_direct_noq_fast}" \
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
