#!/bin/bash
# Score-aware Beta timestep probe, stage A (arXiv 2606.07387 mechanism check).
#
# Paper claim: at 2k rows / 20k iters the base model overfits (val loss rises after ~7.5k)
# and tilting low-alignment rows toward high-noise t acts as a regulariser. Before spending
# a quarter-scale run on it, check the effect exists in our S1 at all.
#
# 2,000 train / 100 val rows (disjoint tracks) from the c2p0 slot0 corpus; S = PE-AV
# alignment (never LAION-CLAP, which is the eval metric). S1 only, 20k iters, 4 arms x 2
# training seeds: base, lambda 0.2, lambda 1.0, lambda 1.0 with S permuted across rows.
# Design + decision rule: docs/experiments/tscore_beta_schedule_line.md
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
PY="$HOME/venvs/dac/bin/python"
PEAV_PY="$HOME/venvs/peav/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"

restore_stage_1() { "$PY" "$WORK_DIR/set_training_stage.py" --stage 1 >/dev/null 2>&1 || true; }
trap restore_stage_1 EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
SRC_TSV="$DATA/phase8_qwen_caption10s_multisent_train.tsv"
CACHE_LIST="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
NPZ_DIR="/mnt/HDD/kojiek/phase8_qwen_official_matched_npz"
OVERLAY="$HOME/text_overlays/true_random"
MC_TSV="$DATA/musiccaps_test.tsv"

ART="$HOME/nvme_experiment_artifacts/meanaudio/tscore_beta_probe_20260913"
INPUTS="$ART/inputs"
STATE="$HOME/logs/tscore_beta_probe_20260913"
mkdir -p "$INPUTS" "$STATE" "$ART/eval" "$ART/metrics"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

# Pinned at design time (select is deterministic; drift means the source corpus moved).
TRAIN_TSV_SHA=3798080cb30b5e91ebbe0459c87f5a6d7fb09551ddbf0226c6f43389fbec3f89
VAL_TSV_SHA=e182fd2defec21c31150fff0fc8b08c694a9d01bbdba4b489ff0c22e07e8f79c

ITERS=20000
BATCH=8
LR=1e-4
SEEDS=(14159265 27182818)
# name lambda score_column
ARMS=(
  "base 0.0 t_score"
  "lam0p2 0.2 t_score"
  "lam1p0 1.0 t_score"
  "lam1p0shuf 1.0 t_score_shuffled"
)

FREE_NVME=$(df -B1 --output=avail "$HOME" | tail -1)
if [ "$FREE_NVME" -lt 40000000000 ]; then
  log "[FAIL] NVMe free $((FREE_NVME/1000000000))G < 40G"; exit 3
fi

# ---- Step 1: subset ----------------------------------------------------------
if [ ! -f "$INPUTS/select_manifest.json" ]; then
  log "[Step 1] select 2000 train / 100 val"
  "$PY" scripts/preprocess/build_tscore_beta_probe_inputs.py select \
    --src-tsv "$SRC_TSV" --cache-list "$CACHE_LIST" --out-dir "$INPUTS" 2>&1 | tee "$STATE/select.log"
fi
[ "$(sha256sum "$INPUTS/train.tsv" | cut -d' ' -f1)" = "$TRAIN_TSV_SHA" ] || { log "[FAIL] train.tsv drift"; exit 2; }
[ "$(sha256sum "$INPUTS/val.tsv" | cut -d' ' -f1)" = "$VAL_TSV_SHA" ] || { log "[FAIL] val.tsv drift"; exit 2; }

# ---- Step 2: PE-AV alignment scores -----------------------------------------
SCORES="$INPUTS/peav_scores.jsonl"
if [ ! -f "$INPUTS/score_manifest.json" ]; then
  log "[Step 2] PE-AV scoring"
  "$PEAV_PY" scripts/preprocess/score_peav_alignment.py \
    --tsv "$INPUTS/train.tsv" --tsv "$INPUTS/val.tsv" --out "$SCORES" --batch_size 8 2>&1 | tee -a "$STATE/peav_score.log"
  N=$(wc -l < "$SCORES")
  [ "$N" -eq 2100 ] || { log "[FAIL] scored $N / 2100"; exit 2; }
  log "[Step 3] attach S"
  "$PY" scripts/preprocess/build_tscore_beta_probe_inputs.py attach \
    --out-dir "$INPUTS" --scores-jsonl "$SCORES" 2>&1 | tee "$STATE/attach.log"
fi
[ -f "$INPUTS/train_scored.tsv" ] && [ -f "$INPUTS/val_scored.tsv" ] || { log "[FAIL] scored tsv missing"; exit 2; }

if [ ! -f "$INPUTS/musiccaps_head500.tsv" ]; then
  head -n 501 "$MC_TSV" > "$INPUTS/musiccaps_head500.tsv"
fi

"$PY" set_training_stage.py --stage 1

# ---- Step 4: arms ------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
  for SPEC in "${ARMS[@]}"; do
    read -r NAME LAMBDA COLUMN <<< "$SPEC"
    EXP="tscore_beta_probe_${NAME}_seed${SEED}_s1_${ITERS}"
    DIR="$WORK_DIR/exps/$EXP"
    CKPT="$DIR/${EXP}_ckpt_last.pth"
    EMA="$DIR/${EXP}_ema_final.pth"
    mkdir -p "$DIR"

    if [ ! -f "$EMA" ]; then
      log "[Step 4] train $EXP (lambda=$LAMBDA column=$COLUMN)"
      RESUME=(); [ -f "$CKPT" ] && RESUME=( "checkpoint=$CKPT" )
      "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
        model=fluxaudio_s exp_id="$EXP" num_iterations="$ITERS" \
        data=meanaudio "lr_schedule_steps=[999999,999999]" \
        "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
        learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
        save_weights_interval=5000 save_checkpoint_interval=5000 \
        ++ema.checkpoint_every=5000 +use_rope=False +use_wandb=False \
        +use_text_attention_mask=false val_interval=500 eval_interval=999999 \
        save_eval_interval=999999 "++multi_cap=false" "++cap_index_fixed=0" \
        "data.AudioCaps_npz.tsv=$INPUTS/train_scored.tsv" \
        "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
        "++data.AudioCaps_npz.gt_cache=$INPUTS/train_cache.txt" \
        "++data.AudioCaps_npz.text_npz_dir=$OVERLAY" \
        "++data.AudioCaps_npz.require_text_overlay=true" \
        "++data.AudioCaps_npz.t_score_column=$COLUMN" \
        "data.AudioCaps_val_npz.tsv=$INPUTS/val_scored.tsv" \
        "++data.AudioCaps_val_npz.npz_dir=$NPZ_DIR" \
        "++data.AudioCaps_val_npz.gt_cache=$INPUTS/val_cache.txt" \
        "++data.AudioCaps_val_npz.text_npz_dir=$OVERLAY" \
        "++data.AudioCaps_val_npz.require_text_overlay=true" \
        +t_score_beta_lambda="$LAMBDA" +val_fm_mse=true \
        "${RESUME[@]}" 2>&1 | tee -a "$STATE/train_${EXP}.log"
    fi
    [ -f "$EMA" ] || { log "[FAIL] no EMA for $EXP"; exit 2; }

    for SPLIT in mc500 val100; do
      case "$SPLIT" in
        mc500)  TSV="$INPUTS/musiccaps_head500.tsv"; WANT=500 ;;
        val100) TSV="$INPUTS/val_scored.tsv";        WANT=100 ;;
      esac
      OUT="$ART/eval/$EXP/$SPLIT/audio"
      METRICS="$ART/metrics/${EXP}_${SPLIT}/metrics.txt"
      if [ -f "$METRICS" ]; then continue; fi
      mkdir -p "$OUT"
      HAVE=$(find "$OUT" -name '*.flac' | wc -l)
      if [ "$HAVE" -lt "$WANT" ]; then
        log "[Step 5] generate $EXP $SPLIT (have $HAVE)"
        "$PY" eval.py --variant fluxaudio_s --model_path "$EMA" --output "$OUT" --tsv "$TSV" \
          --num_steps 25 --cfg_strength 0 --encoder_name t5_clap --text_c_dim 512 \
          --no_q --no_text_attention_mask --seed 42 --full_precision 2>&1 | tee "$STATE/gen_${EXP}_${SPLIT}.log"
      fi
      GOT=$(find "$OUT" -name '*.flac' | wc -l)
      [ "$GOT" -eq "$WANT" ] || { log "[FAIL] $EXP $SPLIT generated $GOT / $WANT"; exit 4; }
      "$PY" "$HOME/research/meanaudio_eval/phase4_eval.py" --gen_dir "$OUT" --tsv "$TSV" \
        --exp_name "${EXP}_${SPLIT}" --out_dir "$ART/metrics" 2>&1 | tee "$STATE/metrics_${EXP}_${SPLIT}.log"
      [ -f "$METRICS" ] || { log "[FAIL] no metrics for $EXP $SPLIT"; exit 4; }
    done

    # Probe checkpoints have no continuation value: keep ema_final + logs, drop the rest.
    rm -rf "$DIR/ema_ckpts"
    rm -f "$CKPT" "$DIR/${EXP}_last.pth" "$DIR"/${EXP}_ckpt_*.pth "$DIR"/${EXP}_[0-9]*.pth 2>/dev/null || true
  done
done

# ---- Step 6: summary + preregistered decision --------------------------------
log "[Step 6] summary"
"$PY" scripts/analysis/summarize_tscore_beta_probe.py \
  --state-dir "$STATE" --metrics-dir "$ART/metrics" --inputs-dir "$INPUTS" \
  --out "$ART/summary.json" 2>&1 | tee "$STATE/summary.log"
[ -f "$ART/summary.json" ] || { log "[FAIL] no summary"; exit 5; }
log "[DONE] tscore_beta_probe_20260913"
