#!/bin/bash
# Resume slot2 quarter from S1 it=10000 after the AB-test interruption.
# NPZ is still slot2 text. Skip slot1. AB site already published.
set -euo pipefail
source /home/kojiek/venvs/dac/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

PIPE=/home/kojiek/research/meanaudio_training/caption10s_pipeline
MEANAUDIO=/home/kojiek/MeanAudio
OUT=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full
NPZ=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
CACHE=/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_official_matched_npz_cache_train.txt
OFFICIAL_TSV=/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv
REEXTRACT=$PIPE/reextract_text_inplace_caption10s.py
EVAL_MF25=$PIPE/eval_musiccaps_mf25.sh

S1_UPDATES=100000
S2_UPDATES=50000
FINAL_IT=$((S1_UPDATES + S2_UPDATES))
LR=1e-4
BATCH=8
SEED=14159265

cd "$MEANAUDIO"
if [ -f scripts/runtime/phase8_nvidia_compat_env.sh ]; then
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || true
fi

ts() { date -u +%FT%TZ; }
echo "RESUME_SLOT2_AFTER_AB $(ts)"

train_quarter() {
  local prefix="$1"
  local tsv="$2"
  local s1_exp="${prefix}_stage1_${S1_UPDATES}"
  local s2_exp="${prefix}_stage2_${S2_UPDATES}"
  local s1_dir="$MEANAUDIO/exps/$s1_exp"
  local s2_dir="$MEANAUDIO/exps/$s2_exp"
  local s1_ckpt="$s1_dir/${s1_exp}_ckpt_last.pth"
  local s1_ema="$s1_dir/${s1_exp}_ema_final.pth"
  local s2_ckpt="$s2_dir/${s2_exp}_ckpt_last.pth"
  local s2_ema="$s2_dir/${s2_exp}_ema_final.pth"

  echo "TRAIN_QUARTER $prefix $(ts)"
  if [ ! -f "$s1_ema" ]; then
    python set_training_stage.py --stage 1
    mkdir -p "$s1_dir"
    local s1_resume=()
    if [ -f "$s1_ckpt" ]; then
      s1_resume=( "checkpoint=$s1_ckpt" )
      echo "RESUME_S1 $s1_ckpt"
    fi
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=fluxaudio_s exp_id="$s1_exp" \
      num_iterations="$S1_UPDATES" "lr_schedule_steps=[999999,999999]" \
      "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
      learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
      save_weights_interval=10000 save_checkpoint_interval=10000 \
      ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
      +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
      save_eval_interval=999999 "data.AudioCaps_npz.tsv=$tsv" \
      "++data.AudioCaps_npz.npz_dir=$NPZ" \
      "++data.AudioCaps_npz.gt_cache=$CACHE" \
      "data.AudioCaps_val_npz.tsv=$tsv" \
      "+data.AudioCaps_val_npz.gt_cache=$CACHE" \
      "++data.AudioCaps_val_npz.npz_dir=$NPZ" ++multi_cap=False \
      "${s1_resume[@]}" \
      2>&1 | tee -a "/home/kojiek/logs/${s1_exp}.log"
  else
    echo "SKIP_S1 $s1_ema"
  fi
  [ -f "$s1_ckpt" ] || [ -f "$s1_ema" ] || { echo "FAIL no S1 for $prefix"; exit 2; }

  if [ ! -f "$s2_ema" ]; then
    python set_training_stage.py --stage 2
    mkdir -p "$s2_dir"
    local src="$s1_ckpt"
    [ -f "$src" ] || src="$s1_ema"
    if [ ! -f "$s2_ckpt" ]; then
      python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$src" --s2_out "$s2_ckpt" --q-init preserve
    fi
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=meanaudio_s exp_id="$s2_exp" \
      num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
      "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
      learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
      save_weights_interval=10000 save_checkpoint_interval=10000 \
      ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
      +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
      save_eval_interval=999999 "data.AudioCaps_npz.tsv=$tsv" \
      "++data.AudioCaps_npz.npz_dir=$NPZ" \
      "++data.AudioCaps_npz.gt_cache=$CACHE" \
      "data.AudioCaps_val_npz.tsv=$tsv" \
      "+data.AudioCaps_val_npz.gt_cache=$CACHE" \
      "++data.AudioCaps_val_npz.npz_dir=$NPZ" ++multi_cap=False \
      2>&1 | tee -a "/home/kojiek/logs/${s2_exp}.log"
  else
    echo "SKIP_S2 $s2_ema"
  fi
  [ -f "$s2_ema" ] || { echo "FAIL no S2 ema for $prefix"; exit 2; }
}

train_quarter phase8_qwen_caption2p0_slot2_noq_quarter \
  "$OUT/phase8_caption2p0_slot2_train.tsv"
"$EVAL_MF25" phase8_qwen_caption2p0_slot2_noq_quarter_musiccaps_mf25_cfg4p5_noq \
  "$MEANAUDIO/exps/phase8_qwen_caption2p0_slot2_noq_quarter_stage2_50000/phase8_qwen_caption2p0_slot2_noq_quarter_stage2_50000_ema_final.pth" \
  --no_q
echo "SLOT2_QUARTER_DONE $(ts)"

echo "RESTORE_SLOT0_NPZ $(ts)"
python "$REEXTRACT" \
  --train_tsv "$OFFICIAL_TSV" \
  --cache_list "$CACHE" \
  --npz_dir "$NPZ" \
  --progress_json "$OUT/reextract_restore_slot0_after_slot12_progress.json" \
  --done_json "$OUT/reextract_restore_slot0_after_slot12.DONE.json"

echo "RESUME_SLOT2_AFTER_AB_DONE $(ts)"
