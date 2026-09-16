#!/bin/bash
# Paired-seed comparison: slot0 (original captions) vs slot0nm (contamination cleaned +
# guessed measurements removed), same training seed, same 251,598 ids, same recipe.
#
# 057 gave slot0nm at seed 14159265 against a historic slot0 number trained on a slightly
# different row set. CFG0 was a tie and CFG3+neg was positive but inside the (poorly
# measured) seed floor, so neither can be claimed. This runs one matched pair at a NEW
# seed so a per-seed delta can be computed on identical ids.
#
# Usage: caption2p0_pairseed_action.sh <slot0|slot0nm> <quarter|full> <seed>
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
SCALE="${2:-quarter}"
SEED="${3:?seed}"
case "$SCALE" in
  quarter) S1_UPDATES=100000; S2_ADD=50000;  CKPT_NEED=30000000000 ;;
  full)    S1_UPDATES=400000; S2_ADD=200000; CKPT_NEED=80000000000 ;;
  *) echo "[FAIL] unknown scale: $SCALE"; exit 2 ;;
esac
case "$ARM" in
  slot0)
    INPUTS="$HOME/exps_nvme/slot0_rowmatched/arm_inputs"
    TRAIN_TSV="$INPUTS/phase8_caption2p0_slot0_rowmatched_train.tsv"
    OVERLAY="$HOME/text_overlays/true_random"   # original captions, already encoded
    PATCH_OVERLAY=0 ;;
  slot0nm)
    INPUTS="$HOME/exps_nvme/slot0nm/arm_inputs"
    TRAIN_TSV="$INPUTS/phase8_caption2p0_slot0nm_train.tsv"
    OVERLAY="$HOME/text_overlays/slot0nm"
    PATCH_OVERLAY=1 ;;
  *) echo "[FAIL] unknown arm: $ARM"; exit 2 ;;
esac
FINAL_IT=$((S1_UPDATES + S2_ADD))
EXP_PREFIX="phase8_qwen_caption2p0_${ARM}_noq_${SCALE}_s${SEED}"
LR=1e-4
BATCH=8

CACHE_LIST="$INPUTS/cache_train.txt"
MANIFEST="$INPUTS/manifest.json"
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
OVERLAY="$OVERLAY" NM_INPUTS="$HOME/exps_nvme/slot0nm/arm_inputs" "$PY" - <<'PYEOF'
import csv, hashlib, json, os, sys
import numpy as np
import pandas as pd
csv.field_size_limit(10**9)
arm, tsv, cache = os.environ["ARM"], os.environ["TRAIN_TSV"], os.environ["CACHE_LIST"]
overlay, nm_inputs = os.environ["OVERLAY"], os.environ["NM_INPUTS"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
m = json.load(open(os.environ["MANIFEST"]))
assert m["status"] == "arm_inputs_ready", "[FAIL] manifest not ready"
assert sha(tsv) == m["train_tsv_sha256"], "[FAIL] train tsv drift"
assert sha(cache) == m["cache_list_sha256"], "[FAIL] cache list drift"
rows = list(csv.DictReader(open(tsv, newline=""), delimiter="\t"))
names = [l.strip() for l in open(cache) if l.strip()]
assert len(rows) == len(names) == m["rows"], f"[FAIL] rows {len(rows)} cache {len(names)} manifest {m['rows']}"

# The pair is only a pair if both arms train on the same ids in the same order.
nm = json.load(open(f"{nm_inputs}/manifest.json"))
nm_ids = [r["id"] for r in csv.DictReader(open(nm["train_tsv"], newline=""), delimiter="\t")]
assert [r["id"] for r in rows] == nm_ids, "[FAIL] id set/order differs from slot0nm; not a paired comparison"
assert sha(cache) == nm["cache_list_sha256"], "[FAIL] cache list differs from slot0nm"

if arm == "slot0nm":
    sys.path.insert(0, "scripts/preprocess")
    import rewrite_slot0nm_no_measurements as R
    assert sha(R.__file__) == m["rewrite_script_sha256"], "[FAIL] rewrite script drift (MEASURE gate changed)"
    left = [r["id"] for r in rows if R.MEASURE.search(r["caption"])]
    assert not left, f"[FAIL] {len(left)} captions still hold a measurement, e.g. {left[:3]}"
else:  # the control must be byte-identical to the source captions
    src = {r["id"]: r["caption"] for r in csv.DictReader(open(m["source_tsv"], newline=""), delimiter="\t")}
    bad = [r["id"] for r in rows if src[r["id"]] != r["caption"]]
    assert not bad, f"[FAIL] {len(bad)} control captions differ from the source"

df = pd.read_csv(tsv, sep="\t").to_dict("records")  # exactly as extracted_audio.py reads it
bad = sum(1 for d, r in zip(df, rows) if str(d["caption"]) != r["caption"] or str(d["id"]) != r["id"])
assert len(df) == len(rows) and not bad, f"[FAIL] pandas/csv parity broken ({bad})"

# require_text_overlay only fires row by row at train time; sample the binding up front.
import random
random.seed(20260916)
for i in random.sample(range(len(rows)), 64):
    d = np.load(f"{overlay}/{names[i]}", allow_pickle=True)
    assert str(d["clip_id"].item()) == rows[i]["id"], f"[FAIL] overlay clip_id mismatch at {i}"
    stored = str(d["caption_sha256"].item()).split(",")
    want = hashlib.sha256(str(rows[i]["caption"]).encode("utf-8")).hexdigest()
    assert stored[0] == want, f"[FAIL] overlay slot-0 caption mismatch at {i} ({rows[i]['id']})"
print(f"  arm={arm} rows={len(rows)} changed_vs_source={m['changed_vs_source']} overlay={overlay} binding ok (64 sampled)")
PYEOF
log "[Step 2] inputs verified"

if [ "$PATCH_OVERLAY" = "1" ]; then
  [ -f "$OVERLAY/DONE.json" ] || { log "[FAIL] slot0nm overlay missing"; exit 2; }
fi

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

NEG='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
MC_TSV="$DATA/musiccaps_test.tsv"
OUT="$HOME/eval_output_nvme/${EXP_PREFIX}_mc_mf25_cfg3_neg"
mkdir -p "$OUT/audio"
HAVE=$(find "$OUT/audio" -name '*.flac' | wc -l)
if [ "$HAVE" -lt 5400 ]; then
  log "[Step 5] CFG3+neg generation (have $HAVE)"
  "$PY" eval.py --variant meanaudio_s --model_path "$S2_EMA" \
    --output "$OUT/audio" --tsv "$MC_TSV" --use_meanflow \
    --num_steps 25 --cfg_strength 3.0 --negative_prompt "$NEG" \
    --no_text_attention_mask --encoder_name t5_clap --text_c_dim 512 \
    --seed 42 --full_precision --no_q 2>&1 | tee "$STATE/eval_gen.log"
fi
GOT=$(find "$OUT/audio" -name '*.flac' | wc -l)
log "generated $GOT / 5521"
[ "$GOT" -ge 5400 ] || { log "[FAIL] only $GOT clips"; exit 4; }
"$PY" "$HOME/research/meanaudio_eval/phase4_eval.py" \
  --gen_dir "$OUT/audio" --tsv "$MC_TSV" \
  --exp_name "${EXP_PREFIX}_mc_mf25_cfg3_neg" \
  --out_dir "$OUT" 2>&1 | tee "$STATE/eval_metrics.log"
# CFG3+neg comparators are batch-32 CLAP; phase4_eval is per-file.
"$PY" scripts/eval/rescore_clap_batch32.py "$OUT/audio" 2>&1 | tee "$STATE/eval_clap_b32.log"

LABEL="${EXP_PREFIX}_musiccaps_mf25_cfg0_noq"
C0_OUT="$HOME/cfg0_eval_runtime/output/$LABEL"
C0_METRICS="$HOME/cfg0_eval_runtime/metrics/$LABEL"
C0_REPORT="$HOME/cfg0_eval_runtime/reports/${LABEL}_REPORT.json"
mkdir -p "$C0_OUT/audio" "$C0_METRICS" "$(dirname "$C0_REPORT")"
HAVE0=$(find "$C0_OUT/audio" -name '*.flac' | wc -l)
if [ "$HAVE0" -lt 5400 ]; then
  log "[Step 6] CFG0 generation (have $HAVE0)"
  "$PY" eval.py --variant meanaudio_s --model_path "$S2_EMA" \
    --output "$C0_OUT/audio" --tsv "$MC_TSV" --use_meanflow \
    --num_steps 25 --cfg_strength 0 \
    --no_text_attention_mask --encoder_name t5_clap --text_c_dim 512 \
    --seed 42 --full_precision --no_q 2>&1 | tee "$STATE/eval_cfg0_gen.log"
fi
GOT0=$(find "$C0_OUT/audio" -name '*.flac' | wc -l)
[ "$GOT0" -ge 5400 ] || { log "[FAIL] cfg0 only $GOT0 clips"; exit 4; }
"$PY" "$HOME/research/meanaudio_eval/phase4_eval.py" \
  --gen_dir "$C0_OUT/audio" --tsv "$MC_TSV" --exp_name "$LABEL" \
  --out_dir "$HOME/cfg0_eval_runtime/metrics" 2>&1 | tee "$STATE/eval_cfg0_metrics.log"

LABEL="$LABEL" C0_METRICS="$C0_METRICS" C0_OUT="$C0_OUT" C0_REPORT="$C0_REPORT" \
S2_EMA="$S2_EMA" SEED="$SEED" ARM="$ARM" "$PY" - <<'PYEOF'
import hashlib, json, datetime, os, soundfile as sf
from pathlib import Path
E = os.environ
m = {}
for line in Path(f"{E['C0_METRICS']}/metrics.txt").read_text().splitlines():
    if ":" in line:
        k, v = line.split(":", 1)
        try: m[k.strip()] = float(v.strip())
        except ValueError: pass
need = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
missing = need - set(m)
if missing:
    raise SystemExit(f"[FAIL] cfg0 metrics missing {missing}")
files = sorted(Path(f"{E['C0_OUT']}/audio").glob("*.flac"))
info = sf.info(str(files[0]))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
json.dump({
    "status": "passed",
    "label": E["LABEL"], "arm": E["ARM"], "training_seed": int(E["SEED"]),
    "conditioning": "no_q", "cfg_strength": 0, "num_steps": 25,
    "protocol": "MusicCaps 5521; MeanFlow 25; CFG 0; NoMask; seed 42; full precision",
    "checkpoint": E["S2_EMA"], "checkpoint_sha256": sha(E["S2_EMA"]),
    "metrics": {k: m[k] for k in sorted(need)},
    "metrics_path": f"{E['C0_METRICS']}/metrics.txt",
    "metrics_sha256": sha(f"{E['C0_METRICS']}/metrics.txt"),
    "audio_validation": {"rows": len(files), "unique_ids": len({f.stem for f in files}),
                         "sample_rate": info.samplerate, "channels": info.channels},
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, open(E["C0_REPORT"], "w"), indent=1, sort_keys=True)
print(f"wrote {E['C0_REPORT']}")
PYEOF

log "[DONE] $EXP_PREFIX"
