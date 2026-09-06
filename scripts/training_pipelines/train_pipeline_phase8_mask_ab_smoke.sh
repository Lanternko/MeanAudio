#!/usr/bin/env bash
# Phase 8 text-mask A/B smoke test.
#
# This is deliberately an overfit test, not a baseline benchmark:
#   * fixed 128-row subset from the canonical clean NPZ alignment
#   * identical seed/config for Mask and legacy NoMask arms
#   * S1 4k + S2 2k, then CLAP on the same fixed 64-row subset
#   * raw final weights for both arms (post-hoc EMA snapshots start at 10k)
#
# A full Phase 8 rerun must not be started from these checkpoints.

set -euo pipefail

if [ -f /home/kojiek/venvs/dac/bin/activate ]; then
    source /home/kojiek/venvs/dac/bin/activate
fi

WORK_DIR=/home/kojiek/MeanAudio
DATA_DIR=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_DIR=/home/kojiek/logs
SMOKE_DIR=/home/kojiek/smoke_data/phase8_mask_ab_n128
NPZ_DIR=/home/kojiek/research/meanaudio_training/npz_phase7_clean
SOURCE_TSV="$DATA_DIR/_QUARANTINED_phase7_v1_train.tsv"
SOURCE_CACHE="$DATA_DIR/npz_cache_train.txt"
MANIFEST="$NPZ_DIR/MANIFEST.tsv"
TRAIN_TSV="$SMOKE_DIR/train_n128.tsv"
TRAIN_CACHE="$SMOKE_DIR/npz_cache_n128.txt"
EVAL_TSV="$SMOKE_DIR/eval_train_n64.tsv"
EVAL_SCRIPT=/home/kojiek/research/meanaudio_eval/phase4_eval.py

TRAIN_N=128
EVAL_N=64
SEED=42
S1_ITERATIONS=4000
S2_ITERATIONS=2000
TOTAL_ITERATIONS=$((S1_ITERATIONS + S2_ITERATIONS))
PREFIX=phase8_mask_ab_smoke_n128

mkdir -p "$SMOKE_DIR" "$LOG_DIR"
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

echo "[1/5] Build and verify the deterministic clean-NPZ subset"
python - "$SOURCE_TSV" "$SOURCE_CACHE" "$MANIFEST" "$NPZ_DIR" \
    "$TRAIN_TSV" "$TRAIN_CACHE" "$EVAL_TSV" "$TRAIN_N" "$EVAL_N" "$SEED" <<'PY'
import csv
import random
import sys
from pathlib import Path

import numpy as np

(
    source_tsv,
    source_cache,
    manifest_path,
    npz_dir,
    train_tsv,
    train_cache,
    eval_tsv,
    train_n,
    eval_n,
    seed,
) = sys.argv[1:]
source_tsv = Path(source_tsv)
source_cache = Path(source_cache)
manifest_path = Path(manifest_path)
npz_dir = Path(npz_dir)
train_tsv = Path(train_tsv)
train_cache = Path(train_cache)
eval_tsv = Path(eval_tsv)
train_n, eval_n, seed = int(train_n), int(eval_n), int(seed)

with source_tsv.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    fieldnames = reader.fieldnames
    rows = list(reader)
with source_cache.open() as f:
    cache = [line.strip() for line in f if line.strip()]
with manifest_path.open() as f:
    manifest = list(csv.DictReader(f, delimiter="\t"))

if not (len(rows) == len(cache) == len(manifest) == 251599):
    raise SystemExit(
        f"canonical alignment mismatch: tsv={len(rows)}, cache={len(cache)}, "
        f"manifest={len(manifest)}"
    )

rng = random.Random(seed)
indices = sorted(rng.sample(range(len(rows)), train_n))
eval_positions = sorted(random.Random(seed + 1).sample(range(train_n), eval_n))
selected_rows = [rows[i] for i in indices]
selected_cache = [cache[i] for i in indices]
eval_rows = [selected_rows[i] for i in eval_positions]

valid_lengths = []
for source_i, row, npz_name in zip(indices, selected_rows, selected_cache):
    item = manifest[source_i]
    if (item["clip_id"], item["npz_fname"]) != (row["id"], npz_name):
        raise SystemExit(f"manifest mismatch at canonical row {source_i}")
    path = npz_dir / npz_name
    with np.load(path) as data:
        if data["text_features"].shape != (77, 1024):
            raise SystemExit(f"bad text feature shape: {path}")
        key = "text_attention_mask" if "text_attention_mask" in data.files else "attention_mask"
        mask = data[key]
        if mask.shape != (77,):
            raise SystemExit(f"bad mask shape: {path}: {mask.shape}")
        valid_lengths.append(int(mask.astype(bool).sum()))

train_tsv.parent.mkdir(parents=True, exist_ok=True)
with train_tsv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected_rows)
with train_cache.open("w") as f:
    f.write("\n".join(selected_cache) + "\n")
with eval_tsv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(eval_rows)

print(f"canonical rows={len(rows):,}; train={train_n}; eval={eval_n}; seed={seed}")
print(
    "valid T5 tokens: "
    f"min={min(valid_lengths)}, mean={sum(valid_lengths)/len(valid_lengths):.1f}, "
    f"max={max(valid_lengths)}"
)
PY

python - "$TRAIN_TSV" "$TRAIN_CACHE" "$NPZ_DIR" <<'PY'
import sys
import torch
from meanaudio.data.extracted_audio import ExtractedAudio

tsv, cache, npz_dir = sys.argv[1:]
common = dict(
    tsv_path=tsv,
    concat_text_fc=False,
    npz_dir=npz_dir,
    data_dim={"latent_seq_len": 312, "text_seq_len": 77, "text_dim": 1024, "text_c_dim": 512},
    repa_npz_dir=None,
    exclude_cls=False,
    repa_version=1,
    gt_cache=cache,
    multi_cap=False,
)
masked = ExtractedAudio(**common, use_text_attention_mask=True)[0]
nomask = ExtractedAudio(**common, use_text_attention_mask=False)[0]
for key in ("a_mean", "a_std", "text_features", "text_features_c"):
    if not torch.equal(masked[key], nomask[key]):
        raise SystemExit(f"Mask/NoMask tensor mismatch for {key}")
if "text_attention_mask" not in masked or "text_attention_mask" in nomask:
    raise SystemExit("Mask/NoMask batch contract is wrong")
print("Mask/NoMask inputs are bit-identical except for mask presence")
PY

if [ "${PREFLIGHT_ONLY:-false}" = true ]; then
    echo "[OK] smoke-test preflight passed; PREFLIGHT_ONLY=true"
    exit 0
fi

COMMON_ARGS=(
    batch_size=8
    +accumulation_steps=1
    learning_rate=1e-4
    seed="$SEED"
    num_workers=4
    save_weights_interval=4000
    save_checkpoint_interval=4000
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

run_arm() {
    local arm="$1"
    local use_mask="$2"
    local eval_mask_flag=()
    local s1_exp="${PREFIX}_${arm}_stage1_${S1_ITERATIONS}"
    local s2_exp="${PREFIX}_${arm}_stage2_${S2_ITERATIONS}"
    local s1_dir="$WORK_DIR/exps/$s1_exp"
    local s2_dir="$WORK_DIR/exps/$s2_exp"
    local s1_ckpt="$s1_dir/${s1_exp}_ckpt_last.pth"
    local s2_ckpt="$s2_dir/${s2_exp}_ckpt_last.pth"
    local s1_weights="$s1_dir/${s1_exp}_last.pth"
    local s2_weights="$s2_dir/${s2_exp}_last.pth"
    local eval_dir="$WORK_DIR/eval_output/${s2_exp}_train64"

    if [ -e "$eval_dir" ]; then
        echo "[ABORT] eval artifact already exists for $arm; refusing to mix generated files" >&2
        exit 2
    fi
    mkdir -p "$s1_dir" "$s2_dir"

    local s1_it=0
    if [ -f "$s1_ckpt" ]; then
        s1_it=$(python -c "import torch; print(torch.load('$s1_ckpt', map_location='cpu', weights_only=False)['it'])")
    fi
    if [ "$s1_it" -eq "$S1_ITERATIONS" ] && [ -f "$s1_weights" ]; then
        echo "[2/5][$arm] S1 already complete at $s1_it; use saved raw weights"
    elif [ "$s1_it" -ne 0 ]; then
        echo "[ABORT] unexpected S1 checkpoint iteration for $arm: $s1_it" >&2
        exit 2
    else
        echo "[2/5][$arm] S1 $S1_ITERATIONS iterations (text mask=$use_mask)"
        python set_training_stage.py --stage 1
        set +e
        torchrun --standalone --nproc_per_node=1 train.py \
            data=meanaudio model=fluxaudio_s exp_id="$s1_exp" \
            num_iterations="$S1_ITERATIONS" \
            "lr_schedule_steps=[999999,999999]" \
            "+use_text_attention_mask=$use_mask" \
            "${COMMON_ARGS[@]}" \
            2>&1 | tee "$LOG_DIR/${s1_exp}.log"
        local train_status=${PIPESTATUS[0]}
        set -e
        s1_it=$(python -c "import torch; print(torch.load('$s1_ckpt', map_location='cpu', weights_only=False)['it'])" 2>/dev/null || echo 0)
        if [ "$s1_it" -ne "$S1_ITERATIONS" ] || [ ! -f "$s1_weights" ]; then
            echo "[ABORT] S1 failed before saving complete raw weights (status=$train_status, it=$s1_it)" >&2
            exit "$train_status"
        fi
        if [ "$train_status" -ne 0 ]; then
            echo "[INFO] accepting complete S1 raw weights; short run has no 10k post-hoc EMA snapshot"
        fi
    fi

    echo "[3/5][$arm] migrate and train S2 $S2_ITERATIONS iterations"
    local s2_it=0
    if [ -f "$s2_ckpt" ]; then
        s2_it=$(python -c "import torch; print(torch.load('$s2_ckpt', map_location='cpu', weights_only=False)['it'])")
    fi
    if [ "$s2_it" -eq "$TOTAL_ITERATIONS" ] && [ -f "$s2_weights" ]; then
        echo "[3/5][$arm] S2 already complete at $s2_it; use saved raw weights"
    else
        if [ "$s2_it" -eq 0 ]; then
            python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$s1_ckpt" --s2_out "$s2_ckpt"
        elif [ "$s2_it" -lt "$S1_ITERATIONS" ] || [ "$s2_it" -gt "$TOTAL_ITERATIONS" ]; then
            echo "[ABORT] unexpected S2 checkpoint iteration for $arm: $s2_it" >&2
            exit 2
        else
            echo "[3/5][$arm] resume S2 from iteration $s2_it"
        fi
        python set_training_stage.py --stage 2
        set +e
        torchrun --standalone --nproc_per_node=1 train.py \
            data=meanaudio model=meanaudio_s exp_id="$s2_exp" \
            num_iterations="$TOTAL_ITERATIONS" \
            "lr_schedule_steps=[999999,999999]" \
            "+use_text_attention_mask=$use_mask" \
            "${COMMON_ARGS[@]}" \
            2>&1 | tee "$LOG_DIR/${s2_exp}.log"
        local train_status=${PIPESTATUS[0]}
        set -e
        s2_it=$(python -c "import torch; print(torch.load('$s2_ckpt', map_location='cpu', weights_only=False)['it'])" 2>/dev/null || echo 0)
        if [ "$s2_it" -ne "$TOTAL_ITERATIONS" ] || [ ! -f "$s2_weights" ]; then
            echo "[ABORT] S2 failed before saving complete raw weights (status=$train_status, it=$s2_it)" >&2
            exit "$train_status"
        fi
        if [ "$train_status" -ne 0 ]; then
            echo "[INFO] accepting complete S2 raw weights; short run has no 10k post-hoc EMA snapshot"
        fi
    fi

    if [ "$use_mask" = false ]; then
        eval_mask_flag+=(--no_text_attention_mask)
    fi
    echo "[4/5][$arm] generate the fixed train-64 subset"
    python eval.py \
        --variant meanaudio_s \
        --model_path "$s2_weights" \
        --output "$eval_dir/audio" \
        --tsv "$EVAL_TSV" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --no_q \
        "${eval_mask_flag[@]}" \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${s2_exp}_train64_eval.log"

    echo "[5/5][$arm] CLAP only (AES/FAD intentionally skipped)"
    python "$EVAL_SCRIPT" \
        --gen_dir "$eval_dir/audio" \
        --tsv "$EVAL_TSV" \
        --exp_name "${s2_exp}_train64" \
        --skip_aes \
        2>&1 | tee -a "$LOG_DIR/${s2_exp}_train64_eval.log"
}

run_arm nomask false
run_arm mask true

echo "======================================================"
echo "Phase 8 Mask A/B smoke test complete"
echo "NoMask: eval_output/metrics/${PREFIX}_nomask_stage2_${S2_ITERATIONS}_train64/metrics.txt"
echo "Mask  : eval_output/metrics/${PREFIX}_mask_stage2_${S2_ITERATIONS}_train64/metrics.txt"
echo "These are raw-weight train-subset overfit scores, not baseline benchmark scores."
echo "======================================================"
