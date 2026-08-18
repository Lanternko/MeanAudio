#!/bin/bash
# After 3-slot captions exist: CLAP select best/worst, then NoQ quarter each.
# Sequential in-place NPZ text rewrite (HDD has no room for a second 246G copy).
# Restores C2.0 slot0 text at the end.
set -euo pipefail
source /home/kojiek/venvs/dac/bin/activate
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

PIPE=/home/kojiek/research/meanaudio_training/caption10s_pipeline
MEANAUDIO=/home/kojiek/MeanAudio
OUT=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
NPZ=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
CACHE=$DATA/phase8_qwen_official_matched_npz_cache_train.txt
OFFICIAL_TSV=$DATA/phase8_qwen_caption10s_multisent_train.tsv
SLOT0=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/captions_full_251599_10s_multisent.jsonl
BUILD_TSV=$PIPE/build_train_tsv_from_caption10s.py
REEXTRACT=$PIPE/reextract_text_inplace_caption10s.py
SELECT=$PIPE/select_c2p0_best_worst_clap.py

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
echo "BEST_WORST_START $(ts)"

echo "CLAP_SELECT $(ts)"
python "$SELECT" \
  --ids_jsonl "$OUT/ids.jsonl" \
  --slot0 "$SLOT0" \
  --slot1 "$OUT/slot1_temp115.jsonl" \
  --slot2 "$OUT/slot2_syntax.jsonl" \
  --scores_jsonl "$OUT/clap_scores_full.jsonl" \
  --best_jsonl "$OUT/bestof3.jsonl" \
  --worst_jsonl "$OUT/worstof3.jsonl" \
  --summary_json "$OUT/CLAP_BEST_WORST_SUMMARY.json" \
  --bs 16

echo "BUILD_TSV $(ts)"
python "$BUILD_TSV" \
  --official_tsv "$OFFICIAL_TSV" \
  --caption_jsonl "$OUT/bestof3.jsonl" \
  --out_tsv "$OUT/phase8_caption2p0_bestof3_train.tsv" \
  --out_manifest "$OUT/phase8_caption2p0_bestof3_train.manifest.json"
python "$BUILD_TSV" \
  --official_tsv "$OFFICIAL_TSV" \
  --caption_jsonl "$OUT/worstof3.jsonl" \
  --out_tsv "$OUT/phase8_caption2p0_worstof3_train.tsv" \
  --out_manifest "$OUT/phase8_caption2p0_worstof3_train.manifest.json"

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
      2>&1 | tee "/home/kojiek/logs/${s1_exp}.log"
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
      2>&1 | tee "/home/kojiek/logs/${s2_exp}.log"
  else
    echo "SKIP_S2 $s2_ema"
  fi
  [ -f "$s2_ema" ] || { echo "FAIL no S2 ema for $prefix"; exit 2; }
}

echo "REEXTRACT_BEST $(ts)"
python "$REEXTRACT" \
  --train_tsv "$OUT/phase8_caption2p0_bestof3_train.tsv" \
  --cache_list "$CACHE" \
  --npz_dir "$NPZ" \
  --progress_json "$OUT/reextract_bestof3_progress.json" \
  --done_json "$OUT/reextract_bestof3.DONE.json"

train_quarter phase8_qwen_caption2p0_bestof3_noq_quarter \
  "$OUT/phase8_caption2p0_bestof3_train.tsv"
echo "BEST_QUARTER_DONE $(ts)"

echo "REEXTRACT_WORST $(ts)"
python "$REEXTRACT" \
  --train_tsv "$OUT/phase8_caption2p0_worstof3_train.tsv" \
  --cache_list "$CACHE" \
  --npz_dir "$NPZ" \
  --progress_json "$OUT/reextract_worstof3_progress.json" \
  --done_json "$OUT/reextract_worstof3.DONE.json"

train_quarter phase8_qwen_caption2p0_worstof3_noq_quarter \
  "$OUT/phase8_caption2p0_worstof3_train.tsv"
echo "WORST_QUARTER_DONE $(ts)"

echo "RESTORE_SLOT0_NPZ $(ts)"
python "$REEXTRACT" \
  --train_tsv "$OFFICIAL_TSV" \
  --cache_list "$CACHE" \
  --npz_dir "$NPZ" \
  --progress_json "$OUT/reextract_restore_slot0_progress.json" \
  --done_json "$OUT/reextract_restore_slot0.DONE.json"

echo "BEST_WORST_DONE $(ts)"

# appended 2026-08-18T03:46:33Z: slot1/slot2 + MF25 eval after worst
/home/kojiek/research/meanaudio_training/caption10s_pipeline/run_c2p0_after_worst_slot12_mf25.sh
