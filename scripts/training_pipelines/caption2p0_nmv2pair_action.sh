#!/bin/bash
# Multi-seed paired quarter ablation: slot0clean vs slot0nmv2.
#
# slot0nmv2 = slot0clean with only measurement phrases removed (BPM, meter, key/mode/chord
# quality, Hz/dB, durations; decades/808/8-bit/12-bar kept word-exact). The control is
# slot0clean itself, so the pair isolates measurement removal. Both arms train on the same
# 251,596 ids in the same order (3 rows slot0nmv2 could not rewrite are excluded from both),
# at the same training seed, same recipe as 057/058/059. Operator 2026-09-20: 3 seeds.
#
# Inputs: scripts/preprocess/build_slot0nmv2_pair_arm_inputs.py
# Eval:   scripts/eval/mc_mf25_eval.sh (canonical 2026-09-18: CFG0 + CFG3+neg, CLAP batch 1)
#
# Usage: caption2p0_nmv2pair_action.sh <slot0clean|slot0nmv2> <seed>
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
PY="$HOME/venvs/dac/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"

# set_training_stage.py --stage 2 rewrites meanaudio/model/mean_flow.py in place and
# nothing switches it back, so hash-pinned jobs die in preflight on a dirty tree.
restore_stage_1() { "$PY" "$WORK_DIR/set_training_stage.py" --stage 1 >/dev/null 2>&1 || true; }
trap restore_stage_1 EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ARM="${1:?arm}"
SEED="${2:?seed}"
S1_UPDATES=100000; S2_ADD=50000; CKPT_NEED=30000000000
case "$ARM" in
  slot0clean)
    INPUTS="$HOME/exps_nvme/slot0clean_nmv2matched/arm_inputs"
    TRAIN_TSV="$INPUTS/phase8_caption2p0_slot0clean_nmv2matched_train.tsv"
    OVERLAY="$HOME/text_overlays/slot0clean" ;;
  slot0nmv2)
    INPUTS="$HOME/exps_nvme/slot0nmv2/arm_inputs"
    TRAIN_TSV="$INPUTS/phase8_caption2p0_slot0nmv2_train.tsv"
    OVERLAY="$HOME/text_overlays/slot0nmv2" ;;
  *) echo "[FAIL] unknown arm: $ARM"; exit 2 ;;
esac
FINAL_IT=$((S1_UPDATES + S2_ADD))
EXP_PREFIX="phase8_qwen_caption2p0_${ARM}_nmv2pair_noq_quarter_s${SEED}"
LR=1e-4
BATCH=8

CACHE_LIST="$INPUTS/cache_train.txt"
MANIFEST="$INPUTS/manifest.json"
PAIR_INPUTS="$HOME/exps_nvme/slot0nmv2/arm_inputs"
NPZ_DIR="/mnt/HDD/kojiek/phase8_qwen_official_matched_npz"

S1_EXP="${EXP_PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${EXP_PREFIX}_stage2_${S2_ADD}"
S1_DIR="$WORK_DIR/exps/$S1_EXP"; S2_DIR="$WORK_DIR/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"; S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"; S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
STATE="$HOME/logs/${EXP_PREFIX}"; mkdir -p "$STATE" "$S1_DIR" "$S2_DIR"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

FREE_NVME=$(df -B1 --output=avail "$HOME" | tail -1)
if [ ! -f "$S2_EMA" ] && [ "$FREE_NVME" -lt "$CKPT_NEED" ]; then
  log "[FAIL] NVMe free $((FREE_NVME/1000000000))G < $((CKPT_NEED/1000000000))G needed for S1/S2 checkpoints"
  exit 3
fi

# ---- Step 1/2: verify inputs and the overlay binding ----------------------
log "[Step 1] verify arm inputs ($ARM, seed $SEED)"
[ -f "$MANIFEST" ] || { log "[FAIL] no manifest at $MANIFEST"; exit 2; }
[ -f "$OVERLAY/DONE.json" ] || { log "[FAIL] overlay $OVERLAY not built"; exit 2; }
ARM="$ARM" TRAIN_TSV="$TRAIN_TSV" CACHE_LIST="$CACHE_LIST" MANIFEST="$MANIFEST" \
OVERLAY="$OVERLAY" PAIR_INPUTS="$PAIR_INPUTS" "$PY" - <<'PYEOF'
import csv, hashlib, json, os, random, sys
import numpy as np
import pandas as pd
csv.field_size_limit(10**9)
arm, tsv, cache = os.environ["ARM"], os.environ["TRAIN_TSV"], os.environ["CACHE_LIST"]
overlay, pair_inputs = os.environ["OVERLAY"], os.environ["PAIR_INPUTS"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = json.load(open(os.environ["MANIFEST"]))
assert m["status"] == "arm_inputs_ready", "[FAIL] manifest not ready"
assert sha(tsv) == m["train_tsv_sha256"], "[FAIL] train tsv drift"
assert sha(cache) == m["cache_list_sha256"], "[FAIL] cache list drift"
rows = list(csv.DictReader(open(tsv, newline=""), delimiter="\t"))
names = [l.strip() for l in open(cache) if l.strip()]
assert len(rows) == len(names) == m["rows"], f"[FAIL] rows {len(rows)} cache {len(names)} manifest {m['rows']}"

# The pair is only a pair if both arms train on the same ids in the same order.
pm = json.load(open(f"{pair_inputs}/manifest.json"))
pair_ids = [r["id"] for r in csv.DictReader(open(pm["train_tsv"], newline=""), delimiter="\t")]
assert [r["id"] for r in rows] == pair_ids, "[FAIL] id set/order differs from slot0nmv2; not a paired comparison"
assert sha(cache) == pm["cache_list_sha256"], "[FAIL] cache list differs from slot0nmv2"

sys.path.insert(0, "scripts/preprocess")
import rewrite_slot0nmv2_measurements as R
assert sha(R.__file__) == m["rewrite_script_sha256"], "[FAIL] rewrite script drift (MEASURE gate changed)"
src = {r["id"]: r["caption"] for r in csv.DictReader(open(m["source_tsv"], newline=""), delimiter="\t")}
assert sha(m["source_tsv"]) == m["source_tsv_sha256"], "[FAIL] slot0clean source drift"
if arm == "slot0nmv2":
    left = [r["id"] for r in rows if R.MEASURE.search(r["caption"])]
    assert not left, f"[FAIL] {len(left)} captions still hold a measurement, e.g. {left[:3]}"
    changed = sum(1 for r in rows if src[r["id"]] != r["caption"])
    assert changed == m["changed_vs_source"], f"[FAIL] changed {changed} != manifest {m['changed_vs_source']}"
else:  # the control must be byte-identical to slot0clean
    bad = [r["id"] for r in rows if src[r["id"]] != r["caption"]]
    assert not bad, f"[FAIL] {len(bad)} control captions differ from slot0clean"

df = pd.read_csv(tsv, sep="\t").to_dict("records")  # exactly as extracted_audio.py reads it
bad = sum(1 for d, r in zip(df, rows) if str(d["caption"]) != r["caption"] or str(d["id"]) != r["id"])
assert len(df) == len(rows) and not bad, f"[FAIL] pandas/csv parity broken ({bad})"

# require_text_overlay only fires row by row at train time; check the binding up front:
# a random sample plus (for slot0nmv2) a sample of the rewritten rows, which are the ones
# the overlay patch had to re-encode.
random.seed(20260920)
idx = random.sample(range(len(rows)), 64)
if arm == "slot0nmv2":
    ch = [i for i, r in enumerate(rows) if src[r["id"]] != r["caption"]]
    idx += random.sample(ch, 64)
for i in idx:
    d = np.load(f"{overlay}/{names[i]}", allow_pickle=True)
    assert str(d["clip_id"].item()) == rows[i]["id"], f"[FAIL] overlay clip_id mismatch at {i}"
    stored = str(d["caption_sha256"].item()).split(",")
    want = hashlib.sha256(str(rows[i]["caption"]).encode("utf-8")).hexdigest()
    assert stored[0] == want, f"[FAIL] overlay slot-0 caption mismatch at {i} ({rows[i]['id']})"
print(f"  arm={arm} rows={len(rows)} changed_vs_slot0clean={m['changed_vs_source']} overlay={overlay} "
      f"binding ok ({len(idx)} sampled)")
PYEOF
log "[Step 2] inputs verified"

COMMON=(
  data=meanaudio "lr_schedule_steps=[999999,999999]"
  "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1
  learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4
  save_weights_interval=10000 save_checkpoint_interval=10000
  ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False
  +use_text_attention_mask=false val_interval=999999 eval_interval=999999
  save_eval_interval=999999
  "++multi_cap=false" "++cap_index_fixed=0"
  "data.AudioCaps_npz.tsv=$TRAIN_TSV"
  "++data.AudioCaps_npz.npz_dir=$NPZ_DIR"
  "++data.AudioCaps_npz.gt_cache=$CACHE_LIST"
  "++data.AudioCaps_npz.text_npz_dir=$OVERLAY"
  "++data.AudioCaps_npz.require_text_overlay=true"
  "data.AudioCaps_val_npz.tsv=$DATA/_QUARANTINED_phase4_val.tsv"
  "++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4"
  "++data.AudioCaps_val_npz.gt_cache=null"
)

if [ ! -f "$S1_EMA" ]; then
  log "[Step 3] Stage 1 $S1_EXP"
  "$PY" set_training_stage.py --stage 1
  S1_RESUME=(); [ -f "$S1_CKPT" ] && S1_RESUME=( "checkpoint=$S1_CKPT" )
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    model=fluxaudio_s exp_id="$S1_EXP" num_iterations="$S1_UPDATES" \
    "${COMMON[@]}" "${S1_RESUME[@]}" 2>&1 | tee -a "$STATE/train_s1.log"
else
  log "[Step 3] S1 already complete"
fi
[ -f "$S1_CKPT" ] || [ -f "$S1_EMA" ] || { log "[FAIL] no S1"; exit 2; }

if [ ! -f "$S2_EMA" ]; then
  log "[Step 4] Stage 2 $S2_EXP"
  "$PY" set_training_stage.py --stage 2
  if [ ! -f "$S2_CKPT" ]; then
    SRC="$S1_CKPT"; [ -f "$SRC" ] || SRC="$S1_EMA"
    "$PY" migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$SRC" --s2_out "$S2_CKPT" \
      --q-init preserve 2>&1 | tee "$STATE/migrate.log"
  fi
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    model=meanaudio_s exp_id="$S2_EXP" num_iterations="$FINAL_IT" \
    "${COMMON[@]}" "checkpoint=$S2_CKPT" 2>&1 | tee -a "$STATE/train_s2.log"
else
  log "[Step 4] S2 already complete"
fi
[ -f "$S2_EMA" ] || { log "[FAIL] no S2 EMA"; exit 2; }
restore_stage_1   # eval.py reads the model code; put the tree back before scoring

# ---- Step 5: canonical eval (CFG0 + CFG3+neg, CLAP batch 1) ---------------
log "[Step 5] mc_mf25_eval cfg0 + cfg3neg"
bash scripts/eval/mc_mf25_eval.sh "$EXP_PREFIX" "$S2_EMA" --no_q 2>&1 | tee -a "$STATE/eval.log"
for CELL in cfg0 cfg3_neg; do
  R="$HOME/eval_output_nvme/${EXP_PREFIX}_mc_mf25_${CELL}/${EXP_PREFIX}_mc_mf25_${CELL}_REPORT.json"
  [ -f "$R" ] || { log "[FAIL] missing report $R"; exit 4; }
done
log "[DONE] $EXP_PREFIX"
