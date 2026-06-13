#!/bin/bash
# ============================================================
# Qwen slot0 masked rerun
#
# Causal test:
#   Keep the original Qwen slot-0 captions fixed and change only the
#   text-conditioning plumbing/cache schema by using text_attention_mask.
#
# This intentionally is not P8-Qwen random single-cap. It is the masked
# rerun of the historical EXP-B / p_qwen_slot0 setup.
#
# Usage:
#   cd ~/MeanAudio
#   bash scripts/training_pipelines/train_pipeline_qwen_slot0_masked.sh
#
# Preflight only:
#   PREFLIGHT_ONLY=1 bash scripts/training_pipelines/train_pipeline_qwen_slot0_masked.sh
#
# Force restart:
#   FORCE_RESTART=1 bash scripts/training_pipelines/train_pipeline_qwen_slot0_masked.sh
# ============================================================

set -eo pipefail

source "$HOME/venvs/dac/bin/activate"

EXP_PREFIX="p_qwen_slot0_masked"

BATCH_SIZE=8
ACCUM_STEPS=1

S1_ITERATIONS=400000
S2_ITERATIONS=200000
TOTAL_ITERATIONS=$((S1_ITERATIONS + S2_ITERATIONS))

LEARNING_RATE=1e-4
USE_Q_CONDITIONING=false

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S1="${EXP_PREFIX}_stage1_${S1_ITERATIONS}"
EXP_S2="${EXP_PREFIX}_stage2_${S2_ITERATIONS}"

S1_DIR="$WORK_DIR/exps/$EXP_S1"
S2_DIR="$WORK_DIR/exps/$EXP_S2"
S1_CKPT="$S1_DIR/${EXP_S1}_ckpt_last.pth"
S2_CKPT="$S2_DIR/${EXP_S2}_ckpt_last.pth"
S2_EMA="$S2_DIR/${EXP_S2}_ema_final.pth"
S1_EMA_FINAL="$S1_DIR/${EXP_S1}_ema_final.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"

TRAIN_TSV="$HOME/eval_tsvs_p100/qwen_slot0_train.tsv"
TRAIN_NPZ="$HOME/exps_nvme/npz_qwen_slot0_masked"
TRAIN_CACHE="$DATA_DIR/npz_cache_train.txt"

# Validation is effectively disabled by interval, but train.py still
# constructs the dataset at startup, so point it at existing lightweight files.
VAL_TSV="$DATA_DIR/phase8_v4_jamendo_seed42_2048.tsv"
VAL_NPZ="$HOME/research/meanaudio_training/npz_phase8v4"

PIPELINE_LOG="$LOG_DIR/${EXP_PREFIX}_pipeline.log"

COMMON_ARGS=(
    batch_size=$BATCH_SIZE
    +accumulation_steps=$ACCUM_STEPS
    learning_rate=$LEARNING_RATE
    num_workers=4
    save_weights_interval=10000
    save_checkpoint_interval=20000
    +use_rope=False
    +use_wandb=False
    "+use_q_conditioning=$USE_Q_CONDITIONING"
    val_interval=999999
    eval_interval=999999
    save_eval_interval=999999
    "data.AudioCaps_npz.tsv=$TRAIN_TSV"
    "data.AudioCaps_val_npz.tsv=$VAL_TSV"
    "++data.AudioCaps_npz.npz_dir=$TRAIN_NPZ"
    "++data.AudioCaps_npz.gt_cache=$TRAIN_CACHE"
    "++data.AudioCaps_val_npz.npz_dir=$VAL_NPZ"
    "++data.AudioCaps_val_npz.gt_cache=null"
)

mkdir -p "$LOG_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "======================================================"
echo "  Qwen slot0 masked rerun"
echo "  S1 exp_id : $EXP_S1"
echo "  S2 exp_id : $EXP_S2"
echo "  Train TSV : $TRAIN_TSV"
echo "  Train NPZ : $TRAIN_NPZ"
echo "  gt_cache  : $TRAIN_CACHE"
echo "  Val TSV   : $VAL_TSV"
echo "  Val NPZ   : $VAL_NPZ"
echo "  CUDA      : $CUDA_VISIBLE_DEVICES"
echo "======================================================"

echo "[Preflight A] verify code paths"
python - <<'PY'
from pathlib import Path

attn_code = Path('meanaudio/model/transformer_layers.py').read_text()
net_code = Path('meanaudio/model/networks.py').read_text()
data_code = Path('meanaudio/data/extracted_audio.py').read_text()
mf_code = Path('meanaudio/runner_meanflow.py').read_text()
fm_code = Path('meanaudio/runner_flowmatching.py').read_text()

checks = {
    'joint attention key_mask': 'key_mask' in attn_code,
    'network text_attention_mask': 'text_attention_mask' in net_code,
    'dataset mask load': 'text_attention_mask' in data_code and 'attention_mask' in data_code,
    'meanflow mask pass': 'text_attention_mask' in mf_code,
    'flowmatching mask pass': 'text_attention_mask' in fm_code,
}
bad = [name for name, ok in checks.items() if not ok]
if bad:
    raise SystemExit('[FAIL] missing mask-aware code path(s): ' + ', '.join(bad))
print('[OK] mask-aware code paths present')
PY

echo "[Preflight B] verify TSV/cache/NPZ alignment"
python - <<PY
from pathlib import Path
import csv
import random
import numpy as np

tsv = Path("$TRAIN_TSV")
npz_dir = Path("$TRAIN_NPZ")
cache = Path("$TRAIN_CACHE")
val_tsv = Path("$VAL_TSV")
val_npz = Path("$VAL_NPZ")

for path in [tsv, npz_dir, cache, val_tsv, val_npz]:
    if not path.exists():
        raise SystemExit(f'[FAIL] missing path: {path}')

with tsv.open() as f:
    rows = list(csv.DictReader(f, delimiter='\\t'))
with cache.open() as f:
    cache_names = [line.strip() for line in f if line.strip()]
npz_files = list(npz_dir.glob('*.npz'))

print(f'  TSV rows : {len(rows):,}')
print(f'  cache rows: {len(cache_names):,}')
print(f'  NPZ files: {len(npz_files):,}')
if not (len(rows) == len(cache_names) == len(npz_files)):
    raise SystemExit('[FAIL] count mismatch')

rng = random.Random(20260520)
sample_names = cache_names[:5] + rng.sample(cache_names, min(200, len(cache_names)))
bad = []
mask_sums = []
for name in sample_names:
    p = npz_dir / name
    if not p.exists():
        bad.append((name, 'missing file'))
        continue
    d = np.load(p)
    required = {'mean', 'std', 'text_features', 'text_features_c', 'text_attention_mask'}
    missing = required - set(d.files)
    if missing:
        bad.append((name, f'missing keys: {sorted(missing)}'))
        continue
    if d['text_features'].shape != (77, 1024):
        bad.append((name, f"text_features shape {d['text_features'].shape}"))
    if d['text_features_c'].shape != (512,):
        bad.append((name, f"text_features_c shape {d['text_features_c'].shape}"))
    if d['text_attention_mask'].shape != (77,):
        bad.append((name, f"text_attention_mask shape {d['text_attention_mask'].shape}"))
    mask_sum = int(d['text_attention_mask'].sum())
    mask_sums.append(mask_sum)
    if mask_sum <= 0 or mask_sum > 77:
        bad.append((name, f'text_attention_mask sum {mask_sum}'))

if bad:
    print('[FAIL] sample errors:')
    for item in bad[:10]:
        print(' ', item)
    raise SystemExit(1)

print(f'  sample checked: {len(sample_names):,}')
print(f'  mask sum min/max: {min(mask_sums)} / {max(mask_sums)}')
print('[OK] masked slot0 NPZ is aligned with gt_cache')
PY

echo "[Preflight C] checkpoint overwrite guard"
if [ "${FORCE_RESTART:-0}" = "1" ]; then
    echo "  FORCE_RESTART=1, removing existing exp dirs"
    rm -rf "$S1_DIR" "$S2_DIR"
elif [ -f "$S2_EMA" ]; then
    echo "[ABORT] S2 already finished: $S2_EMA"
    exit 10
elif [ -f "$S1_EMA_FINAL" ] && [ ! -f "$S1_CKPT" ]; then
    echo "[ABORT] S1 ema_final exists but ckpt_last is missing: $S1_EMA_FINAL"
    exit 10
elif [ -f "$S1_CKPT" ] || [ -f "$S2_CKPT" ]; then
    echo "  existing checkpoint found; training will resume"
fi

mkdir -p "$S1_DIR" "$S2_DIR"

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "[OK] PREFLIGHT_ONLY=1, stopping before training"
    exit 0
fi

echo "[Stage 1] launching $EXP_S1"
python "$STAGE_SCRIPT" --stage 1

torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio \
    model=fluxaudio_s \
    exp_id="$EXP_S1" \
    num_iterations=$S1_ITERATIONS \
    "lr_schedule_steps=[320000,360000]" \
    "${COMMON_ARGS[@]}"

echo "[Stage 1] done"

if [ -f "$S2_CKPT" ] && [ "${FORCE_MIGRATE:-0}" != "1" ]; then
    echo "[Migrate] skip; $S2_CKPT already exists"
else
    echo "[Migrate] $S1_CKPT -> $S2_CKPT"
    python "$MIGRATE_SCRIPT" --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT"
fi

echo "[Stage 2] launching $EXP_S2"
python "$STAGE_SCRIPT" --stage 2

torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio \
    model=meanaudio_s \
    exp_id="$EXP_S2" \
    num_iterations=$TOTAL_ITERATIONS \
    "lr_schedule_steps=[999999,999999]" \
    "${COMMON_ARGS[@]}"

echo "[Stage 2] done"
echo "[Complete] $EXP_PREFIX"
