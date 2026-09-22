#!/bin/bash
# 075 D2: defect negative-sample quarter arms (prereg docs/experiments/d2_defect_negsample_075_20260923.md).
#
# slot0clean_nmv2matched corpus (251,596 rows, the 066 control's exact inputs) + 25,000
# degraded copies of existing clips (noise / clip / lowpass / bitcrush / crackle, each LUFS-
# matched to its clean window). The two arms share the extra audio and differ only in the
# extra rows' captions:
#   defectlab    "<defect sentence> <original caption>"
#   defectunlab  "<original caption>"
# Recipe = 066 (caption2p0_nmv2pair_action.sh slot0clean 14159265): S1 100k + S2 50k,
# NoQ, NoMask, LR 1e-4, BS 8, seed 14159265, cap_index_fixed=0, require_text_overlay.
#
# Inputs: scripts/preprocess/build_defect_negsample_arm_inputs.py
# Eval:   scripts/eval/mc_mf25_eval.sh (CFG0 + CFG3+neg, CLAP batch 1)
#
# Usage: d2_defect_negsample_action.sh <defectlab|defectunlab>
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

ARM="${1:?arm}"
SEED=14159265
S1_UPDATES=100000; S2_ADD=50000; CKPT_NEED=30000000000
ROOT="${D2_ROOT:-$HOME/exps_nvme/defect_negsample}"
case "$ARM" in
  defectlab)   A=lab ;;
  defectunlab) A=unlab ;;
  *) echo "[FAIL] unknown arm: $ARM"; exit 2 ;;
esac
INPUTS="$ROOT/arm_$A"
TRAIN_TSV="$INPUTS/train.tsv"; CACHE_LIST="$INPUTS/cache_train.txt"; MANIFEST="$INPUTS/manifest.json"
NPZ_DIR="$ROOT/npz_farm"; OVERLAY="$ROOT/overlay_farm_$A"
FINAL_IT=$((S1_UPDATES + S2_ADD))
EXP_PREFIX="phase8_qwen_caption2p0_slot0clean_${ARM}_noq_quarter_s${SEED}"
LR=1e-4; BATCH=8

S1_EXP="${EXP_PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${EXP_PREFIX}_stage2_${S2_ADD}"
S1_DIR="$WORK_DIR/exps/$S1_EXP"; S2_DIR="$WORK_DIR/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"; S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"; S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
STATE="$HOME/logs/${EXP_PREFIX}"; mkdir -p "$STATE" "$S1_DIR" "$S2_DIR"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

FREE_NVME=$(df -B1 --output=avail "$HOME" | tail -1)
if [ ! -f "$S2_EMA" ] && [ "$FREE_NVME" -lt "$CKPT_NEED" ]; then
  log "[FAIL] NVMe free $((FREE_NVME/1000000000))G < $((CKPT_NEED/1000000000))G"; exit 3
fi

log "[Step 1] verify arm inputs ($ARM)"
TRAIN_TSV="$TRAIN_TSV" CACHE_LIST="$CACHE_LIST" MANIFEST="$MANIFEST" OVERLAY="$OVERLAY" NPZ_DIR="$NPZ_DIR" \
A="$A" ROOT="$ROOT" "$PY" - <<'PYEOF'
import csv, hashlib, json, os, random
import numpy as np, pandas as pd
csv.field_size_limit(10**9)
E = os.environ
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = json.load(open(E["MANIFEST"]))
assert m["status"] == "arm_inputs_ready"
assert sha(E["TRAIN_TSV"]) == m["train_tsv_sha256"], "[FAIL] train tsv drift"
assert sha(E["CACHE_LIST"]) == m["cache_list_sha256"], "[FAIL] cache list drift"
rows = list(csv.DictReader(open(E["TRAIN_TSV"], newline=""), delimiter="\t"))
names = [l.strip() for l in open(E["CACHE_LIST"]) if l.strip()]
assert len(rows) == len(names) == m["rows"] == m["clean_rows"] + m["extra_rows"]
# the clean block must be the 066 control's inputs, byte for byte
src = json.load(open(m["source_manifest"]))
src_rows = list(csv.DictReader(open(src["train_tsv"], newline=""), delimiter="\t"))
src_names = [l.strip() for l in open(src["cache_list"]) if l.strip()]
n = m["clean_rows"]
assert rows[:n] == src_rows and names[:n] == src_names, "[FAIL] clean block differs from 066 control"
# both arms share the extra audio; the unlabeled arm's extra captions equal their source captions
other = "unlab" if E["A"] == "lab" else "lab"
o_rows = list(csv.DictReader(open(f"{E['ROOT']}/arm_{other}/train.tsv", newline=""), delimiter="\t"))
assert [r["id"] for r in o_rows] == [r["id"] for r in rows], "[FAIL] arms differ in ids/order"
by_id = {r["id"]: r["caption"] for r in src_rows}
extra = rows[n:]
if E["A"] == "unlab":
    assert all(r["caption"] == by_id[r["id"].split("__dgr_")[0]] for r in extra), "[FAIL] unlab caption edited"
else:
    assert all(r["caption"].endswith(" " + by_id[r["id"].split("__dgr_")[0]]) and
               r["caption"] != by_id[r["id"].split("__dgr_")[0]] for r in extra), "[FAIL] lab caption not prefix+source"
df = pd.read_csv(E["TRAIN_TSV"], sep="\t").to_dict("records")
assert len(df) == len(rows) and all(str(d["id"]) == r["id"] and str(d["caption"]) == r["caption"]
                                    for d, r in zip(df, rows)), "[FAIL] pandas/csv parity"
random.seed(20260923)
idx = random.sample(range(n), 32) + random.sample(range(n, len(rows)), min(96, len(rows) - n))
for i in idx:
    a = np.load(f"{E['NPZ_DIR']}/{names[i]}"); t = np.load(f"{E['OVERLAY']}/{names[i]}")
    assert str(a["clip_id"].item()) == rows[i]["id"] == str(t["clip_id"].item()), f"[FAIL] clip_id at {i}"
    want = hashlib.sha256(rows[i]["caption"].encode()).hexdigest()
    assert str(t["caption_sha256"].item()).split(",")[0] == want, f"[FAIL] overlay caption at {i}"
    assert a["mean"].shape == (312, 20) and t["text_features"].ndim == 3
print(f"  rows={len(rows)} clean={n} extra={m['extra_rows']} kinds={m['kind_counts']} binding ok ({len(idx)} sampled)")
PYEOF
log "[Step 2] inputs verified"
[ "${D2_VERIFY_ONLY:-0}" = 1 ] && exit 0

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
restore_stage_1

log "[Step 5] mc_mf25_eval cfg0 + cfg3neg"
bash scripts/eval/mc_mf25_eval.sh "$EXP_PREFIX" "$S2_EMA" --no_q 2>&1 | tee -a "$STATE/eval.log"
for CELL in cfg0 cfg3_neg; do
  R="$HOME/eval_output_nvme/${EXP_PREFIX}_mc_mf25_${CELL}/${EXP_PREFIX}_mc_mf25_${CELL}_REPORT.json"
  [ -f "$R" ] || { log "[FAIL] missing report $R"; exit 4; }
done
log "[DONE] $EXP_PREFIX"
