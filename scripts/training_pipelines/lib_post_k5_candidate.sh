#!/bin/bash
set -euo pipefail

source /home/kojiek/gpu_queue/lib_common.sh

post_k5_pause_arg() {
  POST_K5_PAUSE_ARGS=()
  if [ -n "${P2_CONTROL_DIR:-}" ]; then
    POST_K5_PAUSE_ARGS=("+pause_request_file=${P2_CONTROL_DIR}/pause.request.json")
  fi
}

post_k5_pause_exit_if_requested() {
  if [ -n "${P2_CONTROL_DIR:-}" ] && { [ -f "${P2_CONTROL_DIR}/pause.ack.json" ] || [ -f "${P2_CONTROL_DIR}/pause.request.json.ack.json" ]; }; then
    exit 75
  fi
}

post_k5_train() {
  local prefix="$1" tsv="$2" overlay="$3" base_ckpt="$4"
  local s1_final="$5" s2_add="$6" use_q="$7" multi="$8"
  # Optional: reuse one slot of an existing stacked overlay instead of re-encoding a copy.
  local cap_spec="${9:-}"
  local cap_args=()
  case "$cap_spec" in
    "") ;;
    fixed:*)  cap_args=( "++cap_index_fixed=${cap_spec#fixed:}" ) ;;
    column:*) cap_args=( "++cap_index_column=${cap_spec#column:}" ) ;;
    sources:*) cap_args=( "++text_npz_sources=${cap_spec#sources:}" ) ;;
    *) echo "FAIL unrecognised cap index spec: $cap_spec" >&2; return 2 ;;
  esac
  local s1_exp="${prefix}_stage1_${s1_final}" s2_exp="${prefix}_stage2_${s2_add}"
  local s1_dir="$MEANAUDIO/exps/$s1_exp" s2_dir="$MEANAUDIO/exps/$s2_exp"
  local s1_ckpt="$s1_dir/${s1_exp}_ckpt_last.pth" s1_ema="$s1_dir/${s1_exp}_ema_final.pth"
  local s2_ckpt="$s2_dir/${s2_exp}_ckpt_last.pth" s2_ema="$s2_dir/${s2_exp}_ema_final.pth"
  local final_it=$((s1_final + s2_add))
  post_k5_pause_arg
  mkdir -p "$s1_dir" "$s2_dir"
  if [ ! -f "$s1_ema" ]; then
    python set_training_stage.py --stage 1
    local resume="$base_ckpt"
    [ -f "$s1_ckpt" ] && resume="$s1_ckpt"
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=fluxaudio_s exp_id="$s1_exp" \
      num_iterations="$s1_final" "lr_schedule_steps=[999999,999999]" \
      "+use_q_conditioning=$use_q" batch_size="$BATCH" +accumulation_steps=1 \
      learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
      save_weights_interval=10000 save_checkpoint_interval=10000 ++ema.checkpoint_every=10000 \
      +use_rope=False +use_wandb=False +use_text_attention_mask=false \
      val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
      "data.AudioCaps_npz.tsv=$tsv" "++data.AudioCaps_npz.npz_dir=$NPZ" \
      "++data.AudioCaps_npz.gt_cache=$CACHE" "++data.AudioCaps_npz.text_npz_dir=$overlay" \
      "data.AudioCaps_val_npz.tsv=$tsv" "+data.AudioCaps_val_npz.gt_cache=$CACHE" \
      "++data.AudioCaps_val_npz.npz_dir=$NPZ" "++data.AudioCaps_val_npz.text_npz_dir=$overlay" \
      "++require_text_overlay=true" "++multi_cap=$multi" "${cap_args[@]}" "checkpoint=$resume" \
      "${POST_K5_PAUSE_ARGS[@]}" 2>&1 | tee -a "/home/kojiek/logs/${s1_exp}.log"
    post_k5_pause_exit_if_requested
  fi
  [ -f "$s1_ckpt" ] || { echo "FAIL missing S1 checkpoint" >&2; return 2; }
  if [ ! -f "$s2_ema" ]; then
    python set_training_stage.py --stage 2
    if [ ! -f "$s2_ckpt" ]; then
      python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$s1_ckpt" --s2_out "$s2_ckpt" --q-init preserve
    fi
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=meanaudio_s exp_id="$s2_exp" \
      num_iterations="$final_it" "lr_schedule_steps=[999999,999999]" \
      "+use_q_conditioning=$use_q" batch_size="$BATCH" +accumulation_steps=1 \
      learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
      save_weights_interval=10000 save_checkpoint_interval=10000 ++ema.checkpoint_every=10000 \
      +use_rope=False +use_wandb=False +use_text_attention_mask=false \
      val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
      "data.AudioCaps_npz.tsv=$tsv" "++data.AudioCaps_npz.npz_dir=$NPZ" \
      "++data.AudioCaps_npz.gt_cache=$CACHE" "++data.AudioCaps_npz.text_npz_dir=$overlay" \
      "data.AudioCaps_val_npz.tsv=$tsv" "+data.AudioCaps_val_npz.gt_cache=$CACHE" \
      "++data.AudioCaps_val_npz.npz_dir=$NPZ" "++data.AudioCaps_val_npz.text_npz_dir=$overlay" \
      "++require_text_overlay=true" "++multi_cap=$multi" "${cap_args[@]}" "checkpoint=$s2_ckpt" \
      "${POST_K5_PAUSE_ARGS[@]}" 2>&1 | tee -a "/home/kojiek/logs/${s2_exp}.log"
    post_k5_pause_exit_if_requested
  fi
  [ -f "$s2_ema" ] || { echo "FAIL missing S2 EMA" >&2; return 2; }
  POST_K5_EMA="$s2_ema"
}
