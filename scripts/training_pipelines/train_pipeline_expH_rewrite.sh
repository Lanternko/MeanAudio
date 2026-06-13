#!/bin/bash
# ============================================================
# EXP-H: Qwen→LP-MC acoustic-style rewrite (NoQ, single-cap)
# train_pipeline_expH_rewrite.sh
#
# Hypothesis: Qwen collapse is caused by caption STYLE, not content.
#   EXP-B   : Qwen slot0 raw → MC CLAP 0.0615 (collapsed)
#   EXP-H   : Qwen slot0 rewritten to LP-MC acoustic style → ?
#   Semantic alignment preserved: rewrite_diag=0.3058 vs qwen_diag=0.3084 (99.2%)
#
# Recipe: identical to P8-Qwen / EXP-B (S1=400K, S2=200K, NoQ, single-cap)
#   - TRAIN_TSV: EXP-H rewrite captions (251,599 rows)
#   - NPZ_DIR:   ~/exps_nvme/npz_expH_rewrite (cache-mapped, needs gt_cache)
#
# Usage:
#   tmux new -s expH_train
#   cd ~/MeanAudio && source ~/venvs/dac/bin/activate
#   bash scripts/training_pipelines/train_pipeline_expH_rewrite.sh
#
# Force restart: FORCE_RESTART=1 bash ...
# ============================================================

set -eo pipefail

# ============================================================
# 実験パラメータ
# ============================================================
EXP_PREFIX="expH_rewrite"

BATCH_SIZE=8
ACCUM_STEPS=1

S1_ITERATIONS=400000
S2_ITERATIONS=200000

LEARNING_RATE=1e-4
USE_Q_CONDITIONING=false

# ============================================================
# パス
# ============================================================
WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP_S1="${EXP_PREFIX}_stage1_${S1_ITERATIONS}"
EXP_S2="${EXP_PREFIX}_stage2_${S2_ITERATIONS}"

S1_CKPT="$WORK_DIR/exps/$EXP_S1/${EXP_S1}_ckpt_last.pth"
S2_CKPT="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ckpt_last.pth"
S2_EMA="$WORK_DIR/exps/$EXP_S2/${EXP_S2}_ema_final.pth"
S1_EMA_FINAL="$WORK_DIR/exps/$EXP_S1/${EXP_S1}_ema_final.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"

# EXP-H NPZ is cache-mapped (filenames from npz_cache_train.txt: 33.npz, 100000.npz, ...)
NPZ_DIR="$HOME/exps_nvme/npz_expH_rewrite"
NPZ_CACHE="$DATA_DIR/npz_cache_train.txt"

TRAIN_TSV="$HOME/eval_tsvs_p100/expH_rewrite_train.tsv"
VAL_TSV="$DATA_DIR/phase4_test.tsv"  # phase4_val.tsv is quarantined; val_interval=999999 so val never runs
TSV_MUSICCAPS="$DATA_DIR/musiccaps_test.tsv"
TSV_JAMENDO_S42="$DATA_DIR/phase4_test_seed42_2048.tsv"

EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
PEAV_SCRIPT="$HOME/research/meanaudio_eval/peav_eval.py"

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
    "data.AudioCaps_val_npz.tsv=$TRAIN_TSV"   # val never runs (val_interval=999999); lpmc_singlecap_npz_archive deleted → point to train
    "++data.AudioCaps_npz.npz_dir=$NPZ_DIR"
    "++data.AudioCaps_npz.gt_cache=$NPZ_CACHE"
    "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR"   # same: val init probe must succeed
    "++data.AudioCaps_val_npz.gt_cache=$NPZ_CACHE" # same cache so npz_files[0] = 33.npz (exists)
    # multi_cap omitted → defaults False (single-cap NPZ schema)
)

# ============================================================
# Pre-flight 0: checkpoint overwrite guard
# ============================================================
if [ "${FORCE_RESTART:-0}" = "1" ]; then
    echo "[Pre-flight 0] FORCE_RESTART=1 — wiping existing exp dirs"
    rm -rf "$WORK_DIR/exps/$EXP_S1" "$WORK_DIR/exps/$EXP_S2"
elif [ -f "$S2_EMA" ]; then
    echo "[Pre-flight 0] [ABORT] already finished (S2 ema_final exists): $S2_EMA"
    exit 10
elif [ -f "$S1_EMA_FINAL" ] && [ ! -f "$S1_CKPT" ]; then
    echo "[Pre-flight 0] [ABORT] S1 ema_final without ckpt_last (cannot resume safely)"
    exit 10
elif [ -f "$S1_CKPT" ] || [ -f "$S2_CKPT" ]; then
    echo "[Pre-flight 0] [INFO] resuming from existing ckpt"
fi

mkdir -p "$LOG_DIR" "$WORK_DIR/exps/$EXP_S1" "$WORK_DIR/exps/$EXP_S2"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================"
echo "  EXP-H — Qwen→LP-MC acoustic-style rewrite (NoQ, single-cap)"
echo "  S1 exp_id  : $EXP_S1"
echo "  S2 exp_id  : $EXP_S2"
echo "  Train TSV  : $TRAIN_TSV"
echo "  NPZ dir    : $NPZ_DIR"
echo "  NPZ cache  : $NPZ_CACHE"
echo "  use_q_cond : $USE_Q_CONDITIONING"
echo "  Eval       : MusicCaps n=5521 + Jamendo seed42 2048"
echo ""
echo "  Hypothesis: Qwen collapse = caption style issue."
echo "    EXP-B Qwen slot0 raw:      MC CLAP 0.0615 (collapsed)"
echo "    EXP-H target:              MC CLAP ≥ 0.15 (style fixes collapse)"
echo "    LP-MC baseline:            MC CLAP 0.1851"
echo "======================================================"

# ============================================================
# Pre-flight A: 4 bug fixes
# ============================================================
echo "[Pre-flight A] verify 4 bug fixes"

python -c "
import re
with open('meanaudio/model/networks.py') as f:
    code = f.read()
mf_start = code.find('class MeanAudio')
mf_code = code[mf_start:]
if re.search(r'q = torch\.full\(\([^,]+,\), 9,', mf_code):
    raise SystemExit('[FAIL] networks.py q=10 fix missing')
print('[OK] networks.py q=10 fix active')
"

python -c "
with open('meanaudio/runner_meanflow.py') as f:
    code = f.read()
if 'text_f_undrop = text_f.clone()' not in code or 'text_f_c_undrop = text_f_c.clone()' not in code:
    raise SystemExit('[FAIL] runner_meanflow clone fix missing')
print('[OK] runner_meanflow.py clone fix active')
"

python -c "
import re
with open('meanaudio/runner_flowmatching.py') as f:
    code = f.read()
if 'q_level' not in code:
    raise SystemExit('[FAIL] runner_flowmatching.py missing q_level reads')
print('[OK] runner_flowmatching.py q passing active')
"

python -c "
with open('meanaudio/model/transformer_layers.py') as f:
    attn_code = f.read()
with open('meanaudio/model/networks.py') as f:
    net_code = f.read()
if 'key_mask' not in attn_code or 'text_attention_mask' not in net_code:
    raise SystemExit('[FAIL] text_attention_mask joint-attention fix missing')
print('[OK] text_attention_mask joint-attention fix active')
"

# ============================================================
# Pre-flight B: TSV + NPZ alignment (cache-mapped)
# ============================================================
echo "[Pre-flight B] verify EXP-H TSV + cache-mapped NPZ"

python - <<PYEOF
import csv, sys
from pathlib import Path
import numpy as np
import random

tsv      = Path('$TRAIN_TSV')
npz_dir  = Path('$NPZ_DIR')
cache    = Path('$NPZ_CACHE')

with open(tsv) as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
n_tsv = len(rows)

with open(cache) as f:
    npz_files = [l.strip() for l in f if l.strip()]
n_cache = len(npz_files)

n_npz = len(list(npz_dir.glob('*.npz')))

print(f'  TSV rows    : {n_tsv:,}')
print(f'  Cache entries: {n_cache:,}')
print(f'  NPZ files   : {n_npz:,}')

if n_tsv != 251599:
    print(f'[FAIL] TSV expected 251599 rows, got {n_tsv}')
    sys.exit(1)
if n_cache != n_tsv:
    print(f'[FAIL] cache vs TSV mismatch: {n_cache} vs {n_tsv}')
    sys.exit(1)
if n_npz != n_tsv:
    print(f'[FAIL] NPZ count mismatch: {n_npz} vs {n_tsv}')
    sys.exit(1)

# Spot-check 200 random NPZs for single-cap shape
rng = random.Random(42)
samples = rng.sample(range(n_tsv), 200)
bad = []
for i in samples:
    fn = npz_files[i]
    p  = npz_dir / fn
    if not p.exists():
        bad.append((i, fn, 'missing'))
        continue
    d   = np.load(p)
    tf  = d['text_features']
    tfc = d['text_features_c']
    if tf.shape != (77, 1024) or tfc.shape != (512,):
        bad.append((i, fn, f'tf={tf.shape} tfc={tfc.shape}'))
        continue
    if 'text_attention_mask' not in d.files:
        bad.append((i, fn, 'missing text_attention_mask'))
        continue
    tam = d['text_attention_mask']
    if tam.shape != (77,) or tam.sum() <= 0 or tam.sum() > 77:
        bad.append((i, fn, f'text_attention_mask shape/sum invalid: {tam.shape}, sum={tam.sum()}'))
if bad:
    print(f'[FAIL] {len(bad)}/200 sampled NPZs bad:')
    for item in bad[:5]: print(f'  {item}')
    sys.exit(1)
print('[OK] 200/200 sampled NPZs are single-cap with text_attention_mask')
PYEOF

# ============================================================
# Stage 1 — FluxAudio
# ============================================================
echo "[Stage 1] launching $EXP_S1"
python "$STAGE_SCRIPT" --stage 1

torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio \
    model=fluxaudio_s \
    exp_id="$EXP_S1" \
    num_iterations=$S1_ITERATIONS \
    "lr_schedule_steps=[320000,360000]" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${EXP_S1}.log"

echo "[Stage 1] done"

# ============================================================
# Migrate S1 → S2
# ============================================================
if [ -f "$S2_CKPT" ] && [ "${FORCE_MIGRATE:-0}" != "1" ]; then
    echo "[Migrate] [SKIP] $S2_CKPT already exists"
else
    echo "[Migrate] $S1_CKPT → $S2_CKPT"
    python "$MIGRATE_SCRIPT" --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT"
fi

# ============================================================
# Stage 2 — MeanAudio
# ============================================================
echo "[Stage 2] launching $EXP_S2"
python "$STAGE_SCRIPT" --stage 2

torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio \
    model=meanaudio_s \
    exp_id="$EXP_S2" \
    num_iterations=$(( S1_ITERATIONS + S2_ITERATIONS )) \
    "lr_schedule_steps=[999999,999999]" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${EXP_S2}.log"

echo "[Stage 2] done"

# ============================================================
# Eval — MusicCaps (n=5521) + Jamendo seed42 (n=2048)
# ============================================================
EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_musiccaps"
EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_jamendo_s42"

# ── MusicCaps ────────────────────────────────────────────────
echo "[Eval MC] gen → $EVAL_OUT_MC"
python eval.py \
    --variant meanaudio_s \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT_MC/audio" \
    --tsv "$TSV_MUSICCAPS" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_musiccaps_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_MC/audio" \
    --tsv "$TSV_MUSICCAPS" \
    --exp_name "${EXP_S2}_musiccaps" \
    --num_samples 5521 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_musiccaps_eval.log"

# ── Jamendo seed42 ───────────────────────────────────────────
echo "[Eval JM s42] gen → $EVAL_OUT_JM"
python eval.py \
    --variant meanaudio_s \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT_JM/audio" \
    --tsv "$TSV_JAMENDO_S42" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_jamendo_s42_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_JM/audio" \
    --tsv "$TSV_JAMENDO_S42" \
    --exp_name "${EXP_S2}_jamendo_s42" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_jamendo_s42_eval.log"

# ── PE-AV ────────────────────────────────────────────────────
if [ -f "$PEAV_SCRIPT" ] && [ -d "$HOME/venvs/peav" ]; then
    echo "[PE-AV] MusicCaps + Jamendo s42"
    deactivate 2>/dev/null || true
    source ~/venvs/peav/bin/activate

    mkdir -p "$WORK_DIR/eval_output/metrics"
    python "$PEAV_SCRIPT" \
        --gen_dir "$EVAL_OUT_MC/audio" \
        --tsv "$TSV_MUSICCAPS" \
        --out "$WORK_DIR/eval_output/metrics/${EXP_S2}_musiccaps_peav.json" \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_musiccaps_peav.log"

    python "$PEAV_SCRIPT" \
        --gen_dir "$EVAL_OUT_JM/audio" \
        --tsv "$TSV_JAMENDO_S42" \
        --out "$WORK_DIR/eval_output/metrics/${EXP_S2}_jamendo_s42_peav.json" \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_jamendo_s42_peav.log"

    deactivate
    source ~/venvs/dac/bin/activate
else
    echo "[PE-AV] [SKIP] script or venv missing"
fi

# ── Steering probe ───────────────────────────────────────────
if [ -f "$WORK_DIR/probe_v1_steering.sh" ]; then
    echo "[Steering probe] $S2_EMA"
    bash "$WORK_DIR/probe_v1_steering.sh" "$S2_EMA" \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_steering.log" || true
fi

# ============================================================
# Done
# ============================================================
echo "======================================================"
echo "  EXP-H done"
echo "  S2 EMA    : $S2_EMA"
echo ""
echo "  Reference baselines:"
echo "    LP-MC NoQ (P8)          : MC CLAP 0.1851"
echo "    Qwen slot0 raw (EXP-B)  : MC CLAP 0.0615 (collapsed)"
echo "    Qwen+prefix (EXP-C)     : MC CLAP 0.0580 (collapsed)"
echo ""
echo "  EXP-H verdict thresholds:"
echo "    MC CLAP ≥ 0.15 → style is bottleneck; LP-MC template fixes Qwen"
echo "    MC CLAP ≥ 0.18 → full recovery: Qwen content + LP-MC style = LP-MC quality"
echo "    MC CLAP ~0.06  → style alone insufficient; deeper failure"
echo "======================================================"
