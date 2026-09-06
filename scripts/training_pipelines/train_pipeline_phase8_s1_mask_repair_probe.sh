#!/usr/bin/env bash
# Stage-1 counterpart of the Phase 8 mask repair probe.
# Scores are fixed-train-subset diagnostics, not baseline benchmark scores.

set -euo pipefail

source /home/kojiek/venvs/dac/bin/activate

WORK_DIR=/home/kojiek/MeanAudio
LOG_DIR=/home/kojiek/logs
SMOKE_DIR=/home/kojiek/smoke_data/phase8_mask_ab_n128
NPZ_DIR=/home/kojiek/research/meanaudio_training/npz_phase7_clean
TRAIN_TSV="$SMOKE_DIR/train_n128.tsv"
TRAIN_CACHE="$SMOKE_DIR/npz_cache_n128.txt"
EVAL_TSV="$SMOKE_DIR/eval_train_n64.tsv"
EVAL_SCRIPT=/home/kojiek/research/meanaudio_eval/phase4_eval.py
BASE_WEIGHTS="$WORK_DIR/exps/phase8_bugfix_rerun_stage1_400000/phase8_bugfix_rerun_stage1_400000_ema_final.pth"
PREFIX=phase8_s1_mask_repair_probe_n128
ITERATIONS=2000
SEED=42

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

ORIGINAL_STAGE=$(python - <<'PY'
from pathlib import Path
from set_training_stage import detect_current_stage

print(detect_current_stage(Path("meanaudio/model/mean_flow.py").read_text()))
PY
)
restore_stage() {
    python set_training_stage.py --stage "$ORIGINAL_STAGE" >/dev/null
}
trap restore_stage EXIT

echo "[Preflight] Verify fixed inputs and switch to Stage 1"
python - "$TRAIN_TSV" "$TRAIN_CACHE" "$EVAL_TSV" "$NPZ_DIR" "$BASE_WEIGHTS" <<'PY'
import csv
import sys
from pathlib import Path

import numpy as np

train_tsv, cache_path, eval_tsv, npz_dir, weights = map(Path, sys.argv[1:])
for path in (train_tsv, cache_path, eval_tsv, weights):
    if not path.is_file():
        raise SystemExit(f"missing required input: {path}")
with train_tsv.open() as f:
    train_rows = list(csv.DictReader(f, delimiter="\t"))
with eval_tsv.open() as f:
    eval_rows = list(csv.DictReader(f, delimiter="\t"))
with cache_path.open() as f:
    cache = [line.strip() for line in f if line.strip()]
if (len(train_rows), len(cache), len(eval_rows)) != (128, 128, 64):
    raise SystemExit("fixed subset sizes changed")
for name in cache:
    with np.load(npz_dir / name) as data:
        if data["text_features"].shape != (77, 1024):
            raise SystemExit(f"bad clean NPZ: {name}")
print("fixed clean subset: train=128, eval=64; common S1 initialization exists")
PY
python set_training_stage.py --stage 1
python set_training_stage.py --check

COMMON_ARGS=(
    batch_size=8
    +accumulation_steps=1
    learning_rate=1e-4
    seed="$SEED"
    num_workers=4
    save_weights_interval="$ITERATIONS"
    save_checkpoint_interval="$ITERATIONS"
    +use_rope=False
    +use_wandb=False
    +use_q_conditioning=false
    val_interval=999999
    eval_interval=999999
    save_eval_interval=999999
    "data.AudioCaps_npz.tsv=$TRAIN_TSV"
    "+data.AudioCaps_npz.gt_cache=$TRAIN_CACHE"
    "data.AudioCaps_val_npz.tsv=$TRAIN_TSV"
    "data.AudioCaps_val_npz.gt_cache=$TRAIN_CACHE"
    "++data.AudioCaps_npz.npz_dir=$NPZ_DIR"
    "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR"
    "++multi_cap=False"
)

eval_model() {
    local label="$1"
    local weights="$2"
    local use_mask="$3"
    local output="$WORK_DIR/eval_output/${label}_train64"
    local mask_flag=()
    if [ "$use_mask" = false ]; then
        mask_flag+=(--no_text_attention_mask)
    fi

    mkdir -p "$output/audio"
    python eval.py \
        --variant fluxaudio_s \
        --model_path "$weights" \
        --output "$output/audio" \
        --tsv "$EVAL_TSV" \
        --num_steps 25 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 4.5 --no_q \
        "${mask_flag[@]}" \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${label}_train64_eval.log"

    local audio_n
    audio_n=$(find "$output/audio" -maxdepth 1 -name '*.flac' | wc -l)
    if [ "$audio_n" -ne 64 ]; then
        echo "[ABORT] $label generated $audio_n/64 files" >&2
        exit 2
    fi
    python "$EVAL_SCRIPT" \
        --gen_dir "$output/audio" \
        --tsv "$EVAL_TSV" \
        --exp_name "${label}_train64" \
        --skip_aes \
        2>&1 | tee -a "$LOG_DIR/${label}_train64_eval.log"
}

train_arm() {
    local arm="$1"
    local use_mask="$2"
    local exp="${PREFIX}_${arm}_stage1_${ITERATIONS}"
    local exp_dir="$WORK_DIR/exps/$exp"
    local ckpt="$exp_dir/${exp}_ckpt_last.pth"
    local raw_weights="$exp_dir/${exp}_last.pth"
    local curr_it=0

    mkdir -p "$exp_dir"
    if [ -f "$ckpt" ]; then
        curr_it=$(python -c "import torch; print(torch.load('$ckpt', map_location='cpu', weights_only=False)['it'])")
    fi
    if [ "$curr_it" -eq "$ITERATIONS" ] && [ -f "$raw_weights" ]; then
        echo "[$arm] training already complete at $curr_it; use raw weights"
    elif [ "$curr_it" -gt "$ITERATIONS" ]; then
        echo "[ABORT] unexpected checkpoint iteration for $arm: $curr_it" >&2
        exit 2
    else
        echo "[$arm] Stage 1 repair fine-tune: $curr_it -> $ITERATIONS (mask=$use_mask)"
        local init_arg=()
        if [ "$curr_it" -eq 0 ]; then
            init_arg+=("weights=$BASE_WEIGHTS")
        fi
        set +e
        torchrun --standalone --nproc_per_node=1 train.py \
            data=meanaudio model=fluxaudio_s exp_id="$exp" \
            num_iterations="$ITERATIONS" \
            "lr_schedule_steps=[999999,999999]" \
            "+use_text_attention_mask=$use_mask" \
            "${init_arg[@]}" \
            "${COMMON_ARGS[@]}" \
            2>&1 | tee "$LOG_DIR/${exp}.log"
        local train_status=${PIPESTATUS[0]}
        set -e
        curr_it=$(python -c "import torch; print(torch.load('$ckpt', map_location='cpu', weights_only=False)['it'])" 2>/dev/null || echo 0)
        if [ "$curr_it" -ne "$ITERATIONS" ] || [ ! -f "$raw_weights" ]; then
            echo "[ABORT] $arm failed before complete raw weights (status=$train_status, it=$curr_it)" >&2
            exit 2
        fi
        if [ "$train_status" -ne 0 ]; then
            echo "[INFO] accepting complete raw weights; 2k run has no 10k post-hoc EMA snapshot"
        fi
    fi
    eval_model "$exp" "$raw_weights" "$use_mask"
}

echo "[1/3] Same S1 checkpoint, inference-only A/B"
eval_model "${PREFIX}_base_nomask" "$BASE_WEIGHTS" false
eval_model "${PREFIX}_base_mask" "$BASE_WEIGHTS" true

echo "[2/3] Matched 2k-step S1 repair A/B"
train_arm nomask false
train_arm mask true

echo "[3/3] Summary"
python - "$WORK_DIR/eval_output/metrics" "$PREFIX" "$ITERATIONS" <<'PY'
import re
import sys
from pathlib import Path

metrics_root, prefix, iterations = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
labels = {
    "base_nomask": f"{prefix}_base_nomask_train64",
    "base_mask": f"{prefix}_base_mask_train64",
    "repair_nomask": f"{prefix}_nomask_stage1_{iterations}_train64",
    "repair_mask": f"{prefix}_mask_stage1_{iterations}_train64",
}
scores = {}
for key, label in labels.items():
    path = metrics_root / label / "metrics.txt"
    match = re.search(r"^clap_score:\s*([-+0-9.eE]+)$", path.read_text(), re.MULTILINE)
    if not match:
        raise SystemExit(f"missing CLAP score: {path}")
    scores[key] = float(match.group(1))
for key, value in scores.items():
    print(f"{key}: {value:.4f}")
print(f"inference NoMask-Mask: {scores['base_nomask'] - scores['base_mask']:+.4f}")
print(f"repair NoMask-Mask: {scores['repair_nomask'] - scores['repair_mask']:+.4f}")
PY
