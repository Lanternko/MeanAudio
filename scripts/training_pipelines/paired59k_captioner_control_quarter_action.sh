#!/bin/bash
# paired59k captioner-only control, quarter budget (S1 100k / S2 50k per arm).
#
# Why this exists: the 036/037 comparison moved three variables at once --
# captioner, corpus size (100,000 vs 251,599) and clip set (59.6% overlap).
# This runs the two captioners over the intersection: identical audio latents,
# identical 59,614 rows in identical order, identical recipe and budget. The
# only thing that differs between the arms is the caption text.
#
# Neither arm re-extracts audio. Both read the MF 100k NPZ cache through an
# explicit gt_cache list; the caption text arrives as a text_npz_dir overlay.
# The Qwen arm's overlay is a symlink farm over slot 0 of text_overlays/
# true_random, so it uses byte-identical features to the c2p0 arms.
#
# require_text_overlay stays off: the MF audio NPZs carry no clip_id (they
# predate that field) so the loader's check cannot pass. build_paired59k_arm_
# inputs.py performs the equivalent audit offline and records it in
# bindings.json -- do not launch without a fresh PREP OK from that script.
set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
PY="$HOME/venvs/dac/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ARM="${1:?usage: $0 <mf or qwen>}"
INPUTS="$HOME/exps_nvme/paired59k_mf_qwen/arm_inputs"
NPZ="$HOME/exps_nvme/mfshort100k_direct_noq_c2p0recipe_npz"
CACHE="$INPUTS/cache_train.txt"
EXPECTED_N=59614
S1_UPDATES=100000
S2_ADD=50000
FINAL_IT=$((S1_UPDATES + S2_ADD))
LR=1e-4
BATCH=8
SEED=14159265

case "$ARM" in
  mf)
    TSV="$INPUTS/mf_recaption_train.tsv"
    OVERLAY="$HOME/text_overlays/paired59k_mf_recaption"
    CAP_ARGS=()          # 1-cap overlay: take the array as stored
    CAP_INDEX_AUDIT=None
    ;;
  qwen)
    TSV="$HOME/exps_nvme/paired59k_mf_qwen/paired59k_qwen_slot0_train.tsv"
    OVERLAY="$INPUTS/qwen_text_overlay"
    CAP_ARGS=( "++cap_index_fixed=0" )   # slot 0 of the 013 stack == c2p0 slot0
    CAP_INDEX_AUDIT=0
    ;;
  *) echo "[FAIL] unknown arm: $ARM"; exit 2 ;;
esac

EXP_PREFIX="paired59k_${ARM}_noq_quarter"
S1_EXP="${EXP_PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${EXP_PREFIX}_stage2_${S2_ADD}"
S1_DIR="$WORK_DIR/exps/$S1_EXP"; S2_DIR="$WORK_DIR/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"; S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"; S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
STATE="$HOME/logs/${EXP_PREFIX}"; mkdir -p "$STATE" "$S1_DIR" "$S2_DIR"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

# ---- Step 0: preflight ------------------------------------------------------
log "[Step 0] preflight arm=$ARM"
[ -f "$INPUTS/bindings.json" ] || { log "[FAIL] no bindings.json; run build_paired59k_arm_inputs.py"; exit 2; }
[ -f "$TSV" ] || { log "[FAIL] missing tsv $TSV"; exit 2; }
[ -d "$OVERLAY" ] || { log "[FAIL] missing overlay $OVERLAY"; exit 2; }
[ -f "$OVERLAY/DONE.json" ] || [ "$ARM" = "qwen" ] || { log "[FAIL] overlay not finished"; exit 2; }

FREE_NVME=$(df -B1 --output=avail "$HOME" | tail -1)
if [ "$FREE_NVME" -lt 40000000000 ]; then
  log "[FAIL] NVMe free $((FREE_NVME/1000000000))G < 40G needed for checkpoints"; exit 3
fi

# ---- Step 1: row + overlay binding audit -----------------------------------
# The loader cannot self-check here (no clip_id in the audio NPZs), so verify
# the three lists line up and that the overlay really holds this TSV's captions.
log "[Step 1] binding audit"
"$PY" - <<PYEOF
import csv, hashlib, random
from pathlib import Path
import numpy as np
csv.field_size_limit(10**9)
rows = list(csv.DictReader(open("$TSV"), delimiter="\t"))
names = [l.strip() for l in open("$CACHE") if l.strip()]
assert len(rows) == $EXPECTED_N, f"tsv rows {len(rows)}"
assert len(names) == len(rows), f"cache {len(names)} vs tsv {len(rows)}"
assert len(set(names)) == len(names), "duplicate cache names"
sha = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()
cap_fixed = ${CAP_INDEX_AUDIT}
bad = 0
for i in random.Random(0).sample(range(len(rows)), 300):
    audio = np.load(Path("$NPZ") / names[i], allow_pickle=False)
    assert audio["mean"].shape == (312, 20), audio["mean"].shape
    text = np.load(Path("$OVERLAY") / names[i], allow_pickle=False)
    stored = str(text["caption_sha256"].item()).split(",")
    want = sha(rows[i]["caption"])
    got = stored[cap_fixed] if cap_fixed is not None else stored[0]
    if got != want:
        bad += 1
        print(f"  [MISMATCH] row {i} id={rows[i]['id']}")
    tf = text["text_features"]
    expect = (77, 1024) if cap_fixed is None else (3, 77, 1024)
    assert tf.shape == expect, f"text_features {tf.shape} != {expect}"
print(f"  binding audit: {300-bad}/300 ok")
if bad:
    raise SystemExit("[FAIL] overlay/TSV binding audit failed")
PYEOF

# ---- Step 2: Stage 1 --------------------------------------------------------
if [ ! -f "$S1_EMA" ]; then
  log "[Step 2] Stage 1 $S1_EXP"
  "$PY" set_training_stage.py --stage 1
  S1_RESUME=(); [ -f "$S1_CKPT" ] && S1_RESUME=( "checkpoint=$S1_CKPT" )
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    data=meanaudio model=fluxaudio_s exp_id="$S1_EXP" \
    num_iterations="$S1_UPDATES" "lr_schedule_steps=[999999,999999]" \
    "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
    learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=10000 \
    ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
    +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
    save_eval_interval=999999 \
    "data.AudioCaps_npz.tsv=$TSV" "++data.AudioCaps_npz.npz_dir=$NPZ" \
    "++data.AudioCaps_npz.gt_cache=$CACHE" "++data.AudioCaps_npz.text_npz_dir=$OVERLAY" \
    "data.AudioCaps_val_npz.tsv=$TSV" "+data.AudioCaps_val_npz.gt_cache=$CACHE" \
    "++data.AudioCaps_val_npz.npz_dir=$NPZ" "++data.AudioCaps_val_npz.text_npz_dir=$OVERLAY" \
    "++require_text_overlay=false" "++multi_cap=false" "${CAP_ARGS[@]}" \
    "${S1_RESUME[@]}" 2>&1 | tee -a "$STATE/train_s1.log"
else
  log "[Step 2] S1 already complete"
fi
[ -f "$S1_CKPT" ] || [ -f "$S1_EMA" ] || { log "[FAIL] no S1"; exit 2; }

# ---- Step 3: migrate + Stage 2 ---------------------------------------------
if [ ! -f "$S2_EMA" ]; then
  log "[Step 3] Stage 2 $S2_EXP"
  "$PY" set_training_stage.py --stage 2
  if [ ! -f "$S2_CKPT" ]; then
    SRC="$S1_CKPT"; [ -f "$SRC" ] || SRC="$S1_EMA"
    "$PY" migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$SRC" --s2_out "$S2_CKPT" \
      --q-init preserve 2>&1 | tee "$STATE/migrate.log"
  fi
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    data=meanaudio model=meanaudio_s exp_id="$S2_EXP" \
    num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
    "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
    learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=10000 \
    ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
    +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
    save_eval_interval=999999 \
    "data.AudioCaps_npz.tsv=$TSV" "++data.AudioCaps_npz.npz_dir=$NPZ" \
    "++data.AudioCaps_npz.gt_cache=$CACHE" "++data.AudioCaps_npz.text_npz_dir=$OVERLAY" \
    "data.AudioCaps_val_npz.tsv=$TSV" "+data.AudioCaps_val_npz.gt_cache=$CACHE" \
    "++data.AudioCaps_val_npz.npz_dir=$NPZ" "++data.AudioCaps_val_npz.text_npz_dir=$OVERLAY" \
    "++require_text_overlay=false" "++multi_cap=false" "${CAP_ARGS[@]}" \
    "checkpoint=$S2_CKPT" 2>&1 | tee -a "$STATE/train_s2.log"
else
  log "[Step 3] S2 already complete"
fi
[ -f "$S2_EMA" ] || { log "[FAIL] no S2 EMA"; exit 2; }

# ---- Step 4: eval, arm-comparison protocol ---------------------------------
# Same protocol as the slot0-vs-fulltrack table: MusicCaps 5521 / MeanFlow 25 /
# CFG 3.0 / fidelity negative / NoMask / seed 42 / --no_q. Comparable across the
# two arms here, and to the other cfg3+neg numbers -- NOT to the CFG 0 table.
NEG='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
OUT="$HOME/eval_output_nvme/${EXP_PREFIX}_mc_mf25_cfg3_neg"
mkdir -p "$OUT/audio"
HAVE=$(find "$OUT/audio" -name '*.flac' | wc -l)
if [ "$HAVE" -lt 5400 ]; then
  log "[Step 4] generating (have $HAVE)"
  "$PY" eval.py --variant meanaudio_s --model_path "$S2_EMA" \
    --output "$OUT/audio" --tsv "$DATA/musiccaps_test.tsv" --use_meanflow \
    --num_steps 25 --cfg_strength 3.0 --negative_prompt "$NEG" \
    --no_text_attention_mask --encoder_name t5_clap --text_c_dim 512 \
    --seed 42 --full_precision --no_q 2>&1 | tee "$STATE/eval_gen.log"
fi
GOT=$(find "$OUT/audio" -name '*.flac' | wc -l)
log "generated $GOT / 5521"
[ "$GOT" -ge 5400 ] || { log "[FAIL] only $GOT clips"; exit 4; }

"$PY" "$HOME/research/meanaudio_eval/phase4_eval.py" \
  --gen_dir "$OUT/audio" --tsv "$DATA/musiccaps_test.tsv" \
  --exp_name "${EXP_PREFIX}_mc_mf25_cfg3_neg" \
  --out_dir "$OUT" 2>&1 | tee "$STATE/eval_metrics.log"

log "[DONE] $EXP_PREFIX"
