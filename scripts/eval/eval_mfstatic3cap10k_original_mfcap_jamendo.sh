#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EMA="$WORK_DIR/exps/mfstatic3cap10k_noq_fast_stage2_50000/mfstatic3cap10k_noq_fast_stage2_50000_ema_final.pth"
SOURCE_HOLDOUT_TSV="$DATA_DIR/music_flamingo_slice10_10k_static_random_3cap_jamendo_holdout2048.tsv"
CAPTION_OUT="$HOME/eval_output/music_flamingo_slice10_10k_static_random_3cap_original_mfcap_jamendo_holdout2048"
ORIGINAL_TSV="$DATA_DIR/music_flamingo_slice10_10k_static_random_3cap_original_mfcap_jamendo_holdout2048.tsv"
EXP_NAME="mfstatic3cap10k_noq_fast_stage2_50000_jamendo_holdout2048_original_mfcap"
EVAL_OUT="$WORK_DIR/eval_output/$EXP_NAME"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
PEAV_SCRIPT="$HOME/research/meanaudio_eval/peav_eval.py"

mkdir -p "$LOG_DIR" "$WORK_DIR/eval_output/metrics"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "======================================================"
echo "  Eval-only: MF-static-random-3cap 10k on original MF Jamendo prompts"
echo "  EMA       : $EMA"
echo "  source TSV: $SOURCE_HOLDOUT_TSV"
echo "  prompt    : Music Flamingo slice10_v1"
echo "======================================================"

for path in "$EMA" "$SOURCE_HOLDOUT_TSV"; do
    if [ ! -f "$path" ]; then
        echo "[ABORT] missing file: $path"
        exit 2
    fi
done

caption_count=0
if [ -f "$CAPTION_OUT/caption.jsonl" ]; then
    caption_count=$(wc -l < "$CAPTION_OUT/caption.jsonl")
fi

if [ "$caption_count" -lt 2048 ]; then
    N=2048 \
    OUT_DIR="$CAPTION_OUT" \
    LOG="$LOG_DIR/music_flamingo_slice10_10k_static_random_3cap_original_mfcap_jamendo_holdout2048_caption.log" \
    TSV="$SOURCE_HOLDOUT_TSV" \
    LPMC_TSV="$SOURCE_HOLDOUT_TSV" \
    PROMPT_PRESET="slice10_v1" \
    PROMPT_VERSION="slice10_v1_static3cap_holdout2048" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-220}" \
    bash "$WORK_DIR/scripts/preprocess/music_flamingo_jamendo_slice10_10k.sh"
else
    echo "[skip] original MF captions already exist: $caption_count rows"
fi

python - <<PYEOF
import csv
import json
from pathlib import Path

src = Path("$CAPTION_OUT/caption.jsonl")
out = Path("$ORIGINAL_TSV")
rows = []
seen = set()
with src.open() as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("ok"):
            continue
        sid = rec["id"]
        if sid in seen:
            continue
        caption = ((rec.get("output") or {}).get("text") or rec.get("raw_text") or "").strip()
        if not caption:
            continue
        rows.append({"id": sid, "caption": caption})
        seen.add(sid)

if len(rows) != 2048:
    raise SystemExit(f"[FAIL] expected 2048 original MF captions, got {len(rows)}")

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "caption"], delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"[ok] wrote {len(rows)} rows -> {out}")
PYEOF

python eval.py \
    --variant meanaudio_s \
    --model_path "$EMA" \
    --output "$EVAL_OUT/audio" \
    --tsv "$ORIGINAL_TSV" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_NAME}_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT/audio" \
    --tsv "$ORIGINAL_TSV" \
    --exp_name "$EXP_NAME" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_NAME}_eval.log"

if [ -f "$PEAV_SCRIPT" ] && [ -x "$HOME/venvs/peav/bin/python" ]; then
    "$HOME/venvs/peav/bin/python" "$PEAV_SCRIPT" \
        --gen_dir "$EVAL_OUT/audio" \
        --tsv "$ORIGINAL_TSV" \
        --out "$WORK_DIR/eval_output/metrics/${EXP_NAME}_peav.json" \
        --batch_size 8 \
        2>&1 | tee "$LOG_DIR/${EXP_NAME}_peav.log"
fi

echo "[done] eval-only complete: $EXP_NAME"
