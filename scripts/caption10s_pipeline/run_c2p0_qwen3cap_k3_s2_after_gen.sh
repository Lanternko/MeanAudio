#!/bin/bash
# After 3-slot captions: MeanSim K=3 TSV, then K=3 official quarter (S1 100k + S2 50k).
# Then queue best-of-3 / worst-of-3 NoQ quarters.
# Replaces the previous S2-from-NoQ 200k full run.
set -euo pipefail
source /home/kojiek/venvs/dac/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

PIPE=/home/kojiek/research/meanaudio_training/caption10s_pipeline
MEANAUDIO=/home/kojiek/MeanAudio
OUT=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full
NPZ=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
CACHE=/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt
TSV=$OUT/phase8_caption2p0_qwen3cap_k3_balanced_train.tsv

S1_UPDATES=100000
S2_UPDATES=50000
FINAL_IT=$((S1_UPDATES + S2_UPDATES))
S1_EXP=phase8_qwen_caption2p0_qwen3cap_k3_balanced_quarter_stage1_100000
S2_EXP=phase8_qwen_caption2p0_qwen3cap_k3_balanced_quarter_stage2_50000
S1_DIR=$MEANAUDIO/exps/$S1_EXP
S2_DIR=$MEANAUDIO/exps/$S2_EXP
S1_CKPT=$S1_DIR/${S1_EXP}_ckpt_last.pth
S1_EMA=$S1_DIR/${S1_EXP}_ema_final.pth
S2_CKPT=$S2_DIR/${S2_EXP}_ckpt_last.pth
S2_EMA=$S2_DIR/${S2_EXP}_ema_final.pth

ts() { date -u +%FT%TZ; }

echo "MEANSIM $(ts)"
cd "$PIPE"
python "$PIPE/build_c2p0_qwen3cap_full_k3.py"
[ -f "$TSV" ] || { echo "FAIL missing K=3 TSV $TSV"; exit 2; }

cd "$MEANAUDIO"
if [ -f scripts/runtime/phase8_nvidia_compat_env.sh ]; then
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || true
fi

echo "TRAIN_K3_QUARTER_S1 $(ts)"
if [ ! -f "$S1_EMA" ]; then
  python set_training_stage.py --stage 1
  mkdir -p "$S1_DIR"
  S1_RESUME=()
  if [ -f "$S1_CKPT" ]; then
    S1_RESUME=( "checkpoint=$S1_CKPT" )
    echo "RESUME_S1 $S1_CKPT"
  fi
  torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio model=fluxaudio_s exp_id="$S1_EXP" \
    num_iterations="$S1_UPDATES" "lr_schedule_steps=[999999,999999]" \
    +use_q_conditioning=true batch_size=8 +accumulation_steps=1 \
    learning_rate=1e-4 seed=14159265 linear_warmup_steps=1000 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=10000 \
    ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
    +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
    save_eval_interval=999999 "data.AudioCaps_npz.tsv=$TSV" \
    "+data.AudioCaps_npz.gt_cache=$CACHE" \
    "data.AudioCaps_val_npz.tsv=$TSV" \
    "+data.AudioCaps_val_npz.gt_cache=$CACHE" \
    "++data.AudioCaps_npz.npz_dir=$NPZ" \
    "++data.AudioCaps_val_npz.npz_dir=$NPZ" ++multi_cap=False \
    "${S1_RESUME[@]}" \
    2>&1 | tee /home/kojiek/logs/${S1_EXP}.log
else
  echo "SKIP_S1 $S1_EMA"
fi
[ -f "$S1_CKPT" ] || [ -f "$S1_EMA" ] || { echo "FAIL no K=3 S1"; exit 2; }

echo "TRAIN_K3_QUARTER_S2 $(ts)"
if [ ! -f "$S2_EMA" ]; then
  python set_training_stage.py --stage 2
  mkdir -p "$S2_DIR"
  SRC=$S1_CKPT
  [ -f "$SRC" ] || SRC=$S1_EMA
  if [ ! -f "$S2_CKPT" ]; then
    python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$SRC" --s2_out "$S2_CKPT" --q-init preserve
  fi
  torchrun --standalone --nproc_per_node=1 train.py \
    data=meanaudio model=meanaudio_s exp_id="$S2_EXP" \
    num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
    +use_q_conditioning=true batch_size=8 +accumulation_steps=1 \
    learning_rate=1e-4 seed=14159265 linear_warmup_steps=1000 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=10000 \
    ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
    +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
    save_eval_interval=999999 "data.AudioCaps_npz.tsv=$TSV" \
    "+data.AudioCaps_npz.gt_cache=$CACHE" \
    "data.AudioCaps_val_npz.tsv=$TSV" \
    "+data.AudioCaps_val_npz.gt_cache=$CACHE" \
    "++data.AudioCaps_npz.npz_dir=$NPZ" \
    "++data.AudioCaps_val_npz.npz_dir=$NPZ" ++multi_cap=False \
    2>&1 | tee /home/kojiek/logs/${S2_EXP}.log
else
  echo "SKIP_S2 $S2_EMA"
fi
[ -f "$S2_EMA" ] || { echo "FAIL no K=3 S2 ema"; exit 2; }
echo "TRAIN_K3_QUARTER_DONE $(ts)"

echo "QUEUE_BEST_WORST $(ts)"
"$PIPE/run_c2p0_best_worst_quarter.sh"
echo "AFTER_GEN_DONE $(ts)"
