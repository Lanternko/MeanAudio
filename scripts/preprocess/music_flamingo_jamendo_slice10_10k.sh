#!/usr/bin/env bash
set -euo pipefail

N="${N:-10000}"
SEED="${SEED:-42}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT_DIR="${OUT_DIR:-/home/kojiek/eval_output/music_flamingo_slice10_10k}"
LOG="${LOG:-/home/kojiek/logs/music_flamingo_slice10_10k.log}"
TSV="${TSV:-/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test.tsv}"
LPMC_TSV="${LPMC_TSV:-$TSV}"
REVIEW_SEED="${REVIEW_SEED:-20260522}"
PROMPT_PRESET="${PROMPT_PRESET:-slice10_v1}"
PROMPT_VERSION="${PROMPT_VERSION:-$PROMPT_PRESET}"
PROMPT_TEXT="${PROMPT_TEXT:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-220}"

export CUDA_VISIBLE_DEVICES

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

attempt=1
while true; do
  echo "[run] attempt=$attempt n=$N seed=$SEED out=$OUT_DIR cuda=$CUDA_VISIBLE_DEVICES"
  set +e
  extra_args=()
  if [[ -n "$PROMPT_TEXT" ]]; then
    extra_args+=(--prompt-text "$PROMPT_TEXT")
  fi

  /home/kojiek/venvs/music_flamingo/bin/python \
    /home/kojiek/MeanAudio/scripts/preprocess/music_flamingo_jamendo_slice_caption.py \
    --n "$N" \
    --seed "$SEED" \
    --tsv "$TSV" \
    --out_dir "$OUT_DIR" \
    --prompt-preset "$PROMPT_PRESET" \
    --prompt-version "$PROMPT_VERSION" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "${extra_args[@]}" \
    --resume \
    2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -eq 0 ]]; then
    break
  fi
  if [[ "$status" -eq 3 ]]; then
    echo "[resume] CUDA-corrupted slice skipped; restarting after 10s" | tee -a "$LOG"
    sleep 10
    attempt=$((attempt + 1))
    continue
  fi
  echo "[fail] Music Flamingo slice captioning exited with status=$status" | tee -a "$LOG"
  exit "$status"
done

/home/kojiek/venvs/music_flamingo/bin/python \
  /home/kojiek/MeanAudio/scripts/preprocess/build_music_flamingo_lpmc_slice_review.py \
  --flamingo-jsonl "$OUT_DIR/caption.jsonl" \
  --lpmc-tsv "$LPMC_TSV" \
  --out-dir "$OUT_DIR/lpmc_review" \
  --n 20 \
  --seed "$REVIEW_SEED" \
  2>&1 | tee -a "$LOG"

echo "[done] Music Flamingo slice-10 caption + LP-MC review sample complete: $OUT_DIR" | tee -a "$LOG"
