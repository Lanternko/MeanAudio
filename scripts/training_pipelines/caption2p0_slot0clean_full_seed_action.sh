#!/bin/bash
# Seed replicates of the slot0clean full-budget checkpoint (TTM comparison, plan
# docs/experiments/ttm_quality_comparison_plan_20261007.md section 9.4). Identical to
# caption2p0_slot0clean_full_action.sh (114, seed 14159265; that file is sha-bound in the 114
# contract and stays unchanged) except that the seed is an argument. Stage 1 resumes the
# matching nmv2pair-control quarter S1 (068 = s27182818, 070 = s16180339) at it 100,000 and
# copies its EMA snapshots <= 100k, exactly as 114 did with 066.
#
# Usage: caption2p0_slot0clean_full_seed_action.sh <seed>
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
PY="$HOME/venvs/dac/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"

restore_stage_1() { "$PY" "$WORK_DIR/set_training_stage.py" --stage 1 >/dev/null 2>&1 || true; }
trap restore_stage_1 EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SEED="${1:?usage: $0 <seed>}"
case "$SEED" in 27182818|16180339) ;; *) echo "unregistered seed $SEED"; exit 2;; esac
S1_UPDATES=400000; S2_ADD=200000; CKPT_NEED=90000000000
FINAL_IT=$((S1_UPDATES + S2_ADD))
EXP_PREFIX="phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full_s${SEED}"
LR=1e-4
BATCH=8

INPUTS="$HOME/exps_nvme/slot0clean_nmv2matched/arm_inputs"
TRAIN_TSV="$INPUTS/phase8_caption2p0_slot0clean_nmv2matched_train.tsv"
CACHE_LIST="$INPUTS/cache_train.txt"
MANIFEST="$INPUTS/manifest.json"
OVERLAY="$HOME/text_overlays/slot0clean"
NPZ_DIR="/mnt/HDD/kojiek/phase8_qwen_official_matched_npz"

SRC_S1_EXP="phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s${SEED}_stage1_100000"
SRC_S1_DIR="$WORK_DIR/exps/$SRC_S1_EXP"
SRC_S1_CKPT="$SRC_S1_DIR/${SRC_S1_EXP}_ckpt_last.pth"
SRC_S1_IT=100000

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
log "[Step 1] verify arm inputs (slot0clean nmv2matched, seed $SEED)"
[ -f "$MANIFEST" ] || { log "[FAIL] no manifest at $MANIFEST"; exit 2; }
[ -f "$OVERLAY/DONE.json" ] || { log "[FAIL] overlay $OVERLAY not built"; exit 2; }
TRAIN_TSV="$TRAIN_TSV" CACHE_LIST="$CACHE_LIST" MANIFEST="$MANIFEST" OVERLAY="$OVERLAY" "$PY" - <<'PYEOF'
import csv, hashlib, json, os, random
import numpy as np
import pandas as pd
csv.field_size_limit(10**9)
tsv, cache, overlay = os.environ["TRAIN_TSV"], os.environ["CACHE_LIST"], os.environ["OVERLAY"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = json.load(open(os.environ["MANIFEST"]))
assert m["status"] == "arm_inputs_ready", "[FAIL] manifest not ready"
assert sha(tsv) == m["train_tsv_sha256"], "[FAIL] train tsv drift"
assert sha(cache) == m["cache_list_sha256"], "[FAIL] cache list drift"
rows = list(csv.DictReader(open(tsv, newline=""), delimiter="\t"))
names = [l.strip() for l in open(cache) if l.strip()]
assert len(rows) == len(names) == m["rows"], f"[FAIL] rows {len(rows)} cache {len(names)} manifest {m['rows']}"
assert sha(m["source_tsv"]) == m["source_tsv_sha256"], "[FAIL] slot0clean source drift"
src = {r["id"]: r["caption"] for r in csv.DictReader(open(m["source_tsv"], newline=""), delimiter="\t")}
bad = [r["id"] for r in rows if src[r["id"]] != r["caption"]]
assert not bad, f"[FAIL] {len(bad)} captions differ from slot0clean"
df = pd.read_csv(tsv, sep="\t").to_dict("records")
bad = sum(1 for d, r in zip(df, rows) if str(d["caption"]) != r["caption"] or str(d["id"]) != r["id"])
assert len(df) == len(rows) and not bad, f"[FAIL] pandas/csv parity broken ({bad})"
random.seed(20261008)
for i in random.sample(range(len(rows)), 64):
    d = np.load(f"{overlay}/{names[i]}", allow_pickle=True)
    assert str(d["clip_id"].item()) == rows[i]["id"], f"[FAIL] overlay clip_id mismatch at {i}"
    stored = str(d["caption_sha256"].item()).split(",")
    want = hashlib.sha256(str(rows[i]["caption"]).encode("utf-8")).hexdigest()
    assert stored[0] == want, f"[FAIL] overlay slot-0 caption mismatch at {i} ({rows[i]['id']})"
print(f"  rows={len(rows)} overlay={overlay} binding ok (64 sampled)")
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
  if [ -f "$S1_CKPT" ]; then
    S1_FROM="$S1_CKPT"   # this run's own checkpoint (pause / crash resume)
  else
    # First seat: seed from the matching quarter control's S1. The queue already verified the
    # ckpt sha (contract resume.checkpoint_sha256); check the iteration and copy the
    # EMA snapshots the post-hoc synthesis needs.
    [ -f "$SRC_S1_CKPT" ] || { log "[FAIL] source S1 ckpt missing: $SRC_S1_CKPT"; exit 2; }
    SRC_S1_CKPT="$SRC_S1_CKPT" SRC_S1_IT="$SRC_S1_IT" "$PY" - <<'PYEOF'
import os, torch
c = torch.load(os.environ["SRC_S1_CKPT"], map_location="cpu", weights_only=True)
want = int(os.environ["SRC_S1_IT"])
assert int(c["it"]) == want, f"[FAIL] source ckpt it={c['it']} != {want}"
assert c.get("ema") is not None and c.get("optimizer") is not None, "[FAIL] source ckpt lacks ema/optimizer state"
print(f"  source S1 ckpt it={c['it']} with optimizer + EMA state")
PYEOF
    mkdir -p "$S1_DIR/ema_ckpts"
    n=0
    for f in "$SRC_S1_DIR"/ema_ckpts/*.pt; do
      step=$(basename "$f" .pt); step=${step#*.}
      if [ "$step" -le "$SRC_S1_IT" ]; then
        cmp -s "$f" "$S1_DIR/ema_ckpts/$(basename "$f")" 2>/dev/null || cp "$f" "$S1_DIR/ema_ckpts/"
        n=$((n + 1))
      fi
    done
    [ "$n" -eq $((2 * SRC_S1_IT / 10000)) ] || { log "[FAIL] expected $((2 * SRC_S1_IT / 10000)) EMA snapshots, copied $n"; exit 2; }
    log "  copied $n EMA snapshots (<= it $SRC_S1_IT) from $SRC_S1_EXP"
    S1_FROM="$SRC_S1_CKPT"
  fi
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    model=fluxaudio_s exp_id="$S1_EXP" num_iterations="$S1_UPDATES" \
    "${COMMON[@]}" "checkpoint=$S1_FROM" 2>&1 | tee -a "$STATE/train_s1.log"
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
restore_stage_1

# ---- Step 5: canonical eval (CFG0 + CFG3+neg, CLAP batch 1) + MIR ---------
log "[Step 5] mc_mf25_eval_mir cfg0 + cfg3neg"
bash scripts/eval/mc_mf25_eval_mir.sh "$EXP_PREFIX" "$S2_EMA" --no_q 2>&1 | tee -a "$STATE/eval.log"
for CELL in cfg0 cfg3_neg; do
  R="$HOME/eval_output_nvme/${EXP_PREFIX}_mc_mf25_${CELL}/${EXP_PREFIX}_mc_mf25_${CELL}_REPORT.json"
  [ -f "$R" ] || { log "[FAIL] missing report $R"; exit 4; }
done
log "[DONE] $EXP_PREFIX"
