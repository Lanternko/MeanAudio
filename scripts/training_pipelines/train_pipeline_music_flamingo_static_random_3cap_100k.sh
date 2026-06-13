#!/usr/bin/env bash
set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
SOURCE_TSV="${SOURCE_TSV:-$DATA_DIR/_QUARANTINED_phase4_train.tsv}"

ORIGINAL_PREFIX="music_flamingo_slice10_100k"
DIRECT_PREFIX="music_flamingo_slice10_100k_short_direct"
AESTHETIC_PREFIX="music_flamingo_slice10_100k_short_aesthetic"
STATIC_PREFIX="music_flamingo_slice10_100k_static_random_3cap"
AESTHETIC_OUT="$HOME/eval_output/$AESTHETIC_PREFIX"

AESTHETIC_PROMPT="${AESTHETIC_PROMPT:-Describe only this 10-second music audio slice as one compact 35-50 word training caption. Start with audible genre or style, instruments or sounds, rhythm and energy, and production texture. Include tasteful aesthetic words such as warm, polished, spacious, gritty, bright, intimate, dreamy, cinematic, energetic, or mellow only when supported by the audio. Do not mention caption length, lyrics, key, BPM, or events outside this clip.}"

caption_count=0
if [ -f "$AESTHETIC_OUT/caption.jsonl" ]; then
    caption_count=$(wc -l < "$AESTHETIC_OUT/caption.jsonl")
fi

if [ "$caption_count" -lt 100000 ]; then
    N=100000 \
    OUT_DIR="$AESTHETIC_OUT" \
    LOG="$HOME/logs/${AESTHETIC_PREFIX}_caption.log" \
    TSV="$SOURCE_TSV" \
    LPMC_TSV="$SOURCE_TSV" \
    PROMPT_PRESET="short_direct_v1" \
    PROMPT_VERSION="slice10_short_aesthetic_v1" \
    PROMPT_TEXT="$AESTHETIC_PROMPT" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}" \
    bash "$WORK_DIR/scripts/preprocess/music_flamingo_jamendo_slice10_10k.sh"
else
    echo "[skip] short-aesthetic captions already exist: $caption_count rows"
fi

python "$WORK_DIR/scripts/preprocess/prepare_music_flamingo_slice10_train.py" \
    --captions-jsonl "$AESTHETIC_OUT/caption.jsonl" \
    --phase4-test-tsv "$DATA_DIR/phase4_test.tsv" \
    --out-dir "$DATA_DIR" \
    --n 100000 \
    --prefix "$AESTHETIC_PREFIX"

python "$WORK_DIR/scripts/preprocess/prepare_music_flamingo_static_random_3cap.py" \
    --original-tsv "$DATA_DIR/${ORIGINAL_PREFIX}_train.tsv" \
    --short-direct-tsv "$DATA_DIR/${DIRECT_PREFIX}_train.tsv" \
    --short-aesthetic-tsv "$DATA_DIR/${AESTHETIC_PREFIX}_train.tsv" \
    --phase4-test-tsv "$DATA_DIR/phase4_test.tsv" \
    --out-dir "$DATA_DIR" \
    --prefix "$STATIC_PREFIX" \
    --seed "${STATIC_RANDOM_SEED:-20260601}"

EXP_PREFIX="${EXP_PREFIX:-mfstatic3cap100k_noq}" \
EXPECTED_N=100000 \
TRAIN_TSV="$DATA_DIR/${STATIC_PREFIX}_train.tsv" \
CLIPS_TSV="$DATA_DIR/${STATIC_PREFIX}_clips.tsv" \
JAMENDO_HOLDOUT_TSV="$DATA_DIR/${STATIC_PREFIX}_jamendo_holdout2048.tsv" \
NPZ_DIR="${NPZ_DIR:-$HOME/exps_nvme/${STATIC_PREFIX}_npz}" \
LATENT_DIR="${LATENT_DIR:-$HOME/exps_nvme/${STATIC_PREFIX}_latents_tmp}" \
NPZ_TSV="${NPZ_TSV:-$HOME/exps_nvme/${STATIC_PREFIX}_npz.tsv}" \
S1_ITERATIONS="${S1_ITERATIONS:-200000}" \
S2_ITERATIONS="${S2_ITERATIONS:-100000}" \
bash "$WORK_DIR/scripts/training_pipelines/train_pipeline_slice10_from_tsv.sh"
