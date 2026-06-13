#!/usr/bin/env bash
set -eo pipefail

# Music Flamingo slice10 10k MeanAudio quick experiment.
#
# Pipeline:
#   1. Prepare train TSV + clips TSV from Music Flamingo slice10 captions.
#   2. Extract MeanAudio NPZ cache with exact first-10s audio, T5 text features,
#      CLAP pooled text features, and VAE audio latents.
#   3. Train S1 FluxAudio from scratch.
#   4. Migrate S1 checkpoint to S2.
#   5. Train S2 MeanAudio from scratch.
#   6. Evaluate on MusicCaps and a Jamendo heldout subset excluding the 10k train ids.

EXP_PREFIX="${EXP_PREFIX:-mf10k_noq_fast}"

if [ -f "$HOME/venvs/dac/bin/activate" ]; then
    source "$HOME/venvs/dac/bin/activate"
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
S1_ITERATIONS="${S1_ITERATIONS:-100000}"
S2_ITERATIONS="${S2_ITERATIONS:-50000}"
TOTAL_ITERATIONS=$((S1_ITERATIONS + S2_ITERATIONS))
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
USE_Q_CONDITIONING=false

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

CAPTION_JSONL="$HOME/eval_output/music_flamingo_slice10_10k/caption.jsonl"
TRAIN_TSV="$DATA_DIR/music_flamingo_slice10_10k_train.tsv"
CLIPS_TSV="$DATA_DIR/music_flamingo_slice10_10k_clips.tsv"
JAMENDO_HOLDOUT_TSV="$DATA_DIR/music_flamingo_slice10_10k_jamendo_holdout2048.tsv"

NPZ_DIR="/mnt/HDD/kojiek/music_flamingo_slice10_10k_npz"
LATENT_DIR="/mnt/HDD/kojiek/music_flamingo_slice10_10k_latents_tmp"
NPZ_TSV="/mnt/HDD/kojiek/music_flamingo_slice10_10k_npz.tsv"
WAV_DIR="$DATA_DIR/wav_audio"

EXP_S1="${EXP_PREFIX}_stage1_${S1_ITERATIONS}"
EXP_S2="${EXP_PREFIX}_stage2_${S2_ITERATIONS}"
S1_DIR="$WORK_DIR/exps/$EXP_S1"
S2_DIR="$WORK_DIR/exps/$EXP_S2"
S1_CKPT="$S1_DIR/${EXP_S1}_ckpt_last.pth"
S2_CKPT="$S2_DIR/${EXP_S2}_ckpt_last.pth"
S1_EMA_FINAL="$S1_DIR/${EXP_S1}_ema_final.pth"
S2_EMA="$S2_DIR/${EXP_S2}_ema_final.pth"

MIGRATE_SCRIPT="$WORK_DIR/migrate_stage1_to_stage2_ckpt.py"
STAGE_SCRIPT="$WORK_DIR/set_training_stage.py"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
PEAV_SCRIPT="$HOME/research/meanaudio_eval/peav_eval.py"

TSV_MUSICCAPS="$DATA_DIR/musiccaps_test.tsv"
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
    "data.AudioCaps_npz.tsv=$NPZ_TSV"
    "data.AudioCaps_val_npz.tsv=$NPZ_TSV"
    "++data.AudioCaps_npz.npz_dir=$NPZ_DIR"
    "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR"
    "++multi_cap=False"
)

mkdir -p "$LOG_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "======================================================"
echo "  Music Flamingo slice10 10k MeanAudio training"
echo "  EXP_PREFIX : $EXP_PREFIX"
echo "  S1/S2 iters: $S1_ITERATIONS / $S2_ITERATIONS"
echo "  Caption    : $CAPTION_JSONL"
echo "  Train TSV  : $TRAIN_TSV"
echo "  Clips TSV  : $CLIPS_TSV"
echo "  NPZ dir    : $NPZ_DIR"
echo "  Eval       : MusicCaps + Jamendo heldout2048 (train ids excluded)"
echo "======================================================"

echo "[Pre-flight 0] Check checkpoint state"
if [ "${FORCE_RESTART:-0}" = "1" ]; then
    echo "  FORCE_RESTART=1, removing existing experiment dirs"
    rm -rf "$S1_DIR" "$S2_DIR"
elif [ -f "$S2_EMA" ]; then
    echo "[ABORT] S2 already finished: $S2_EMA"
    exit 10
elif [ -f "$S1_EMA_FINAL" ] && [ ! -f "$S1_CKPT" ]; then
    echo "[ABORT] S1 ema_final exists but ckpt_last is missing: $S1_EMA_FINAL"
    exit 10
elif [ -f "$S1_CKPT" ] || [ -f "$S2_CKPT" ]; then
    echo "  existing checkpoint found; train.py should resume from checkpoint"
fi
mkdir -p "$S1_DIR" "$S2_DIR"

echo "[Step 1] Prepare TSVs"
python "$WORK_DIR/scripts/preprocess/prepare_music_flamingo_slice10_train.py" \
    --captions-jsonl "$CAPTION_JSONL" \
    --out-dir "$DATA_DIR" \
    --n 10000 \
    --prefix music_flamingo_slice10_10k

echo "[Step 2] Extract NPZ cache if needed"
NPZ_COUNT=$(find "$NPZ_DIR" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l || true)
if [ "$NPZ_COUNT" -ne 10000 ] || [ ! -f "$NPZ_TSV" ]; then
    echo "  NPZ cache incomplete: count=$NPZ_COUNT; rebuilding"
    rm -rf "$NPZ_DIR" "$LATENT_DIR" "$NPZ_TSV"
    torchrun --standalone --nproc_per_node=1 training/extract_audio_latents.py \
        --data_dir "$WAV_DIR" \
        --captions_tsv "$TRAIN_TSV" \
        --clips_tsv "$CLIPS_TSV" \
        --latent_dir "$LATENT_DIR" \
        --output_dir "$NPZ_DIR" \
        --batch_size 8 \
        --num_workers 4 \
        --text_encoder t5_clap \
        2>&1 | tee "$LOG_DIR/${EXP_PREFIX}_extract_npz.log"
else
    echo "  NPZ cache exists: $NPZ_COUNT files"
fi

echo "[Step 3] Verify NPZ/TSV alignment"
python - <<PYEOF
import csv, sys
from pathlib import Path
import numpy as np

npz_dir = Path("$NPZ_DIR")
npz_tsv = Path("$NPZ_TSV")
rows = list(csv.DictReader(npz_tsv.open(), delimiter="\\t"))
files = list(npz_dir.glob("*.npz"))
print(f"  NPZ TSV rows: {len(rows)}")
print(f"  NPZ files   : {len(files)}")
if len(rows) != 10000 or len(files) != 10000:
    raise SystemExit("[FAIL] expected 10000 NPZ rows/files")
d = np.load(npz_dir / "0.npz")
print(f"  mean={d['mean'].shape} text_features={d['text_features'].shape} text_c={d['text_features_c'].shape}")
if d["mean"].shape != (312, 20) or d["text_features"].shape != (77, 1024) or d["text_features_c"].shape != (512,):
    raise SystemExit("[FAIL] unexpected NPZ tensor shape")
if "text_attention_mask" not in d.files:
    raise SystemExit("[FAIL] text_attention_mask missing")
print("[OK] NPZ cache ready")
PYEOF

echo "[Step 4] Stage 1 training"
python "$STAGE_SCRIPT" --stage 1
torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio \
    model=fluxaudio_s \
    exp_id="$EXP_S1" \
    num_iterations=$S1_ITERATIONS \
    "lr_schedule_steps=[$(( S1_ITERATIONS * 8 / 10 )),$(( S1_ITERATIONS * 9 / 10 ))]" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${EXP_S1}.log"

echo "[Step 5] Migrate S1 -> S2"
python "$MIGRATE_SCRIPT" --s1_ckpt "$S1_CKPT" --s2_out "$S2_CKPT"

echo "[Step 6] Stage 2 training"
python "$STAGE_SCRIPT" --stage 2
torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio \
    model=meanaudio_s \
    exp_id="$EXP_S2" \
    num_iterations=$TOTAL_ITERATIONS \
    "lr_schedule_steps=[999999,999999]" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${EXP_S2}.log"

echo "[Step 7] Eval MusicCaps"
EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_musiccaps"
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
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_musiccaps_eval.log"

echo "[Step 8] Eval Jamendo heldout2048"
EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_jamendo_holdout2048"
python eval.py \
    --variant meanaudio_s \
    --model_path "$S2_EMA" \
    --output "$EVAL_OUT_JM/audio" \
    --tsv "$JAMENDO_HOLDOUT_TSV" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_jamendo_holdout2048_eval.log"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_JM/audio" \
    --tsv "$JAMENDO_HOLDOUT_TSV" \
    --exp_name "${EXP_S2}_jamendo_holdout2048" \
    --num_samples 2048 \
    2>&1 | tee -a "$LOG_DIR/${EXP_S2}_jamendo_holdout2048_eval.log"

if [ -f "$PEAV_SCRIPT" ] && [ -x "$HOME/venvs/peav/bin/python" ]; then
    echo "[Step 9] PE-AV"
    "$HOME/venvs/peav/bin/python" "$PEAV_SCRIPT" \
        --gen_dir "$EVAL_OUT_MC/audio" \
        --tsv "$TSV_MUSICCAPS" \
        --out "$WORK_DIR/eval_output/metrics/${EXP_S2}_musiccaps_peav.json" \
        --batch_size 8 \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_musiccaps_peav.log"

    "$HOME/venvs/peav/bin/python" "$PEAV_SCRIPT" \
        --gen_dir "$EVAL_OUT_JM/audio" \
        --tsv "$JAMENDO_HOLDOUT_TSV" \
        --out "$WORK_DIR/eval_output/metrics/${EXP_S2}_jamendo_holdout2048_peav.json" \
        --batch_size 8 \
        2>&1 | tee "$LOG_DIR/${EXP_S2}_jamendo_holdout2048_peav.log"
fi

echo "======================================================"
echo "  Music Flamingo slice10 10k pipeline complete"
echo "  S2 EMA: $S2_EMA"
echo "  MusicCaps metrics: eval_output/metrics/${EXP_S2}_musiccaps/metrics.txt"
echo "  Jamendo heldout metrics: eval_output/metrics/${EXP_S2}_jamendo_holdout2048/metrics.txt"
echo "======================================================"
