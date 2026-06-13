#!/bin/bash
# ============================================================
# P7V1-Qwen: JamendoFull-QwenOmni-Random-MeanSim-Q (single-cap)
# train_pipeline_p7v1_qwen.sh
#
# Captioner-only control for P7 V1 (LP-MC random + Q, historical 0.1975):
#   - Same training recipe as historical P7 V1 except caption source
#   - 5 Qwen task-framed captions per clip → static random pick (seed=42)
#   - multi_cap=False (single fixed cap)
#   - Q signal: pairwise mean_sim of 5 Qwen caps, Qwen-LOCAL percentile 0..9
#   - All 4 bug fixes active including runner_flowmatching q-passing
#     and text_attention_mask-aware T5 caches
#     → S1 actually trains q_embed[0..9] (historical P7 V1 didn't, due to bug)
#
# Codex 2026-05-05 caveat: q=N is captioner-LOCAL percentile.
# DON'T compare q=N CLAP across captioners (Qwen ≠ LP-MC).
#
# Pre-flight assumes:
#   - qwen_singlecap_random_q_train.tsv generated with Qwen-local q_level
#   - phase9_5_random_singlecap_npz/ shared with P8-Qwen
#
# Usage:
#   tmux new -s p7v1_qwen
#   bash train_pipeline_p7v1_qwen.sh
# ============================================================

set -eo pipefail

# ============================================================
# 實驗參數
# ============================================================
EXP_PREFIX="p7v1_qwen"

BATCH_SIZE=8
ACCUM_STEPS=1

S1_ITERATIONS=400000
S2_ITERATIONS=200000

LEARNING_RATE=1e-4
USE_Q_CONDITIONING=true   # P7V1-Qwen has Q (Qwen-local percentile bin)

# ============================================================
# 路徑
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

NPZ_DIR="$HOME/phase9_5_random_singlecap_npz"   # shared with P7V1-Qwen
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
PEAV_SCRIPT="$HOME/research/meanaudio_eval/peav_eval.py"

TRAIN_TSV="$DATA_DIR/qwen_singlecap_random_q_train.tsv"   # Q variant of random TSV
VAL_TSV="$DATA_DIR/phase4_val.tsv"
TSV_MUSICCAPS="$DATA_DIR/musiccaps_test.tsv"
TSV_JAMENDO_S42="$DATA_DIR/phase4_test_seed42_2048.tsv"

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
    "++data.AudioCaps_npz.npz_dir=$NPZ_DIR"
    # multi_cap deliberately omitted → defaults False → single-cap NPZ schema
)

# ============================================================
# Pre-flight 0: checkpoint overwrite guard
# ============================================================
if [ "${FORCE_RESTART:-0}" = "1" ]; then
    echo "[Pre-flight 0] FORCE_RESTART=1 — wiping existing exp dirs"
    rm -rf "$WORK_DIR/exps/$EXP_S1" "$WORK_DIR/exps/$EXP_S2"
elif [ -f "$S2_EMA" ]; then
    echo "[Pre-flight 0] [ABORT] V1 already finished (S2 ema_final exists): $S2_EMA"
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
echo "  P8-Qwen — JamendoFull-QwenOmni-Random-NoQ (single-cap)"
echo "  S1 exp_id : $EXP_S1"
echo "  S2 exp_id : $EXP_S2"
echo "  Train TSV : $TRAIN_TSV"
echo "  NPZ dir   : $NPZ_DIR"
echo "  use_q_cond: $USE_Q_CONDITIONING"
echo "  Eval     : MC + JM s42 (q sweep {6,9}, --quality_level) + PE-AV + steering probe"
echo "======================================================"

# ============================================================
# Pre-flight A: bug fixes
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
# Pre-flight B: NPZ + TSV alignment
# ============================================================
echo "[Pre-flight B] verify single-cap NPZ + TSV"

python - <<PYEOF
import csv, sys
from pathlib import Path
import numpy as np

tsv = Path('$TRAIN_TSV')
npz_dir = Path('$NPZ_DIR')

with open(tsv) as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
n_tsv = len(rows)
n_npz = len(list(npz_dir.glob('*.npz')))
print(f'  TSV rows: {n_tsv:,}')
print(f'  NPZ files: {n_npz:,}')
if n_tsv != n_npz:
    print(f'[FAIL] count mismatch')
    sys.exit(1)

# Spot-check 200 random NPZs for single-cap shape
import random
rng = random.Random(0)
samples = rng.sample(range(n_tsv), min(200, n_tsv))
bad = []
for i in samples:
    d = np.load(npz_dir / f'{i}.npz')
    tf  = d['text_features']    # expect (77, 1024) for single-cap
    tfc = d['text_features_c']  # expect (512,) for single-cap
    if tf.shape != (77, 1024) or tfc.shape != (512,):
        bad.append((i, f'tf={tf.shape} tfc={tfc.shape}'))
        continue
    if 'text_attention_mask' not in d.files:
        bad.append((i, 'missing text_attention_mask'))
        continue
    tam = d['text_attention_mask']
    if tam.shape != (77,) or tam.sum() <= 0 or tam.sum() > 77:
        bad.append((i, f'text_attention_mask shape/sum invalid: {tam.shape}, sum={tam.sum()}'))
if bad:
    print(f'[FAIL] {len(bad)}/200 sampled NPZs failed shape/mask checks')
    for i, why in bad[:5]:
        print(f'  idx {i}: {why}')
    sys.exit(1)
print(f'[OK] 200/200 sampled NPZs are single-cap with text_attention_mask')
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
# Migrate
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
# Eval — q sweep {6, 9} × {MusicCaps, Jamendo seed42}
#   q=6: in-support boundary (per P7 V1 LP-MC q-sweep pattern)
#   q=9: high-confidence end
#   Caveat: q=N is captioner-LOCAL percentile (Qwen ≠ LP-MC)
# ============================================================
MC_Q_VALUES="${MC_Q_VALUES:-6 9}"
JM_Q_VALUES="${JM_Q_VALUES:-6 9}"

# ── MusicCaps q sweep ───────────────────────────────────────
for Q in $MC_Q_VALUES; do
    EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_q${Q}_musiccaps"
    echo "[Eval MC q=$Q] gen → $EVAL_OUT_MC"

    python eval.py \
        --variant meanaudio_s \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT_MC/audio" \
        --tsv "$TSV_MUSICCAPS" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --quality_level $Q \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_q${Q}_musiccaps_eval.log"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT_MC/audio" \
        --tsv "$TSV_MUSICCAPS" \
        --exp_name "${EXP_S2}_q${Q}_musiccaps" \
        --num_samples 5521 \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_q${Q}_musiccaps_eval.log"
done

# ── Jamendo seed42 q sweep ──────────────────────────────────
for Q in $JM_Q_VALUES; do
    EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_q${Q}_jamendo_s42"
    echo "[Eval JM s42 q=$Q] gen → $EVAL_OUT_JM"

    python eval.py \
        --variant meanaudio_s \
        --model_path "$S2_EMA" \
        --output "$EVAL_OUT_JM/audio" \
        --tsv "$TSV_JAMENDO_S42" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --quality_level $Q \
        --full_precision \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_q${Q}_jamendo_s42_eval.log"

    python "$EVAL_SCRIPT" \
        --gen_dir "$EVAL_OUT_JM/audio" \
        --tsv "$TSV_JAMENDO_S42" \
        --exp_name "${EXP_S2}_q${Q}_jamendo_s42" \
        --num_samples 2048 \
        2>&1 | tee -a "$LOG_DIR/${EXP_S2}_q${Q}_jamendo_s42_eval.log"
done

# ── PE-AV (q=9 only, primary regime) ────────────────────────
if [ -f "$PEAV_SCRIPT" ] && [ -d "$HOME/venvs/peav" ]; then
    echo "[PE-AV] q=9 MC + JM s42 (~/venvs/peav)"
    deactivate 2>/dev/null || true
    source ~/venvs/peav/bin/activate

    mkdir -p "$WORK_DIR/eval_output/metrics"
    python "$PEAV_SCRIPT" \
        --gen_dir "$WORK_DIR/eval_output/${EXP_S2}_q9_musiccaps/audio" \
        --tsv "$TSV_MUSICCAPS" \
        --out "$WORK_DIR/eval_output/metrics/${EXP_S2}_q9_musiccaps_peav.json" \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_q9_musiccaps_peav.log"

    python "$PEAV_SCRIPT" \
        --gen_dir "$WORK_DIR/eval_output/${EXP_S2}_q9_jamendo_s42/audio" \
        --tsv "$TSV_JAMENDO_S42" \
        --out "$WORK_DIR/eval_output/metrics/${EXP_S2}_q9_jamendo_s42_peav.json" \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_q9_jamendo_s42_peav.log"

    deactivate
    source ~/venvs/dac/bin/activate
else
    echo "[PE-AV] [SKIP] script or venv missing"
fi

# ── steering probe (q=9, Q-trained model) ───────────────────
echo "[Steering probe q=9] $S2_EMA"
PROBE_QUALITY=9 bash "$WORK_DIR/probe_v1_steering.sh" "$S2_EMA" 2>&1 | tee "$LOG_DIR/${EXP_S2}_steering.log" || true

# ============================================================
# Done
# ============================================================
echo "======================================================"
echo "  P7V1-Qwen done"
echo "  S2 EMA   : $S2_EMA"
echo ""
echo "  Reference baselines:"
echo "    P7 V1 LP-MC random + Q  MC CLAP q=9: 0.1975 (historical, S1 q-bug present)"
echo "    P7 V1 fullq_control LP-MC clean q=9: 0.1748 (-11.5%, S1 trains q[0..9])"
echo "    P9 V2 LP-MC multi-cap + Q q=9     : 0.0403 (collapse)"
echo ""
echo "  Question this answers:"
echo "    Does Q conditioning recover prompt-following in Qwen single-cap regime?"
echo "    Compare to P8-Qwen NoQ result + historical P7V1 vs P8 gap (~+6.7%)"
echo "======================================================"
