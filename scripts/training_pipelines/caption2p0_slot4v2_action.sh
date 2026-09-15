#!/bin/bash
# c2p0 slot4v2: slot0 captions with every digit removed, clean rewrite.
#
# Same as caption2p0_slot4_action.sh (050/052) except the corpus. slot4's rewrite had
# no EOS (chat-turn junk), a fallback that stripped the LLM output instead of the
# original, and whole-caption rewriting that dropped digit-free sentences: ~8.8k rows
# (3.5% of the corpus) were polluted. rewrite_slot4v2_no_digits.py edits only digit
# sentences, gates every LLM output, and asserts that digit-free sentences are
# byte-identical and that no injection / new dangling fragment appears.
# Overlay reuses true_random via hardlink and re-encodes only the rewritten
# slot-0 features, then trains with cap_index_fixed=0.
set -euo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA="/mnt/HDD/kojiek/phase4_jamendo_data"
PY="$HOME/venvs/dac/bin/python"
TORCHRUN="$HOME/venvs/dac/bin/torchrun"
export PATH="$HOME/venvs/dac/bin:$PATH"
cd "$WORK_DIR"

# set_training_stage.py --stage 2 rewrites meanaudio/model/mean_flow.py in place and
# nothing switched it back, so the tree was left on Stage 2 after every training run.
# Contracts hash-pin the Stage 1 (git HEAD) form, so the next such job died in preflight
# with an unlogged "input drift". Restore on every exit, including kill/preempt.
# set_training_stage.py is a no-op when already on the target stage.
restore_stage_1() { "$PY" "$WORK_DIR/set_training_stage.py" --stage 1 >/dev/null 2>&1 || true; }
trap restore_stage_1 EXIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCALE="${1:-quarter}"
case "$SCALE" in
  # CKPT_NEED: ema_ckpts is 2 sigma series x (iters/10000) snapshots at ~460M each,
  # so the full arm holds ~54G of snapshots on top of ~11G of ckpt/last/shadow.
  quarter) S1_UPDATES=100000; S2_ADD=50000;  CKPT_NEED=30000000000 ;;
  full)    S1_UPDATES=400000; S2_ADD=200000; CKPT_NEED=80000000000 ;;
  *) echo "[FAIL] unknown scale: $SCALE"; exit 2 ;;
esac
FINAL_IT=$((S1_UPDATES + S2_ADD))
EXP_PREFIX="phase8_qwen_caption2p0_slot4v2_no_digits_noq_${SCALE}"
EXPECTED_N=251599
LR=1e-4
BATCH=8
SEED=14159265

SRC_TSV="$DATA/phase8_qwen_caption10s_multisent_train.tsv"
INPUTS="$HOME/exps_nvme/slot4v2/arm_inputs"
TRAIN_TSV="$INPUTS/phase8_caption2p0_slot4v2_train.tsv"
REWRITE_LOG="$INPUTS/rewrites.jsonl"
CACHE_LIST="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
NPZ_DIR="/mnt/HDD/kojiek/phase8_qwen_official_matched_npz"
OVERLAY="$HOME/text_overlays/slot4v2"
TRUE_RANDOM="$HOME/text_overlays/true_random"

S1_EXP="${EXP_PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${EXP_PREFIX}_stage2_${S2_ADD}"
S1_DIR="$WORK_DIR/exps/$S1_EXP"; S2_DIR="$WORK_DIR/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"; S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"; S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
STATE="$HOME/logs/${EXP_PREFIX}"; mkdir -p "$STATE" "$S1_DIR" "$S2_DIR" "$INPUTS"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }

# Overlay patch is ~19G (21k stacked npz) plus ~28G of checkpoints.
FREE_NVME=$(df -B1 --output=avail "$HOME" | tail -1)
if [ ! -f "$OVERLAY/DONE.json" ] && [ "$FREE_NVME" -lt 25000000000 ]; then
  log "[FAIL] NVMe free $((FREE_NVME/1000000000))G < 25G needed to patch overlay"
  exit 3
fi
if [ ! -f "$S2_EMA" ] && [ "$FREE_NVME" -lt "$CKPT_NEED" ]; then
  log "[FAIL] NVMe free $((FREE_NVME/1000000000))G < $((CKPT_NEED/1000000000))G needed for S1/S2 checkpoints"
  exit 3
fi

# ---- Step 1: rewrite digit captions -----------------------------------------
log "[Step 1] rewrite slot0 digit sentences -> slot4v2 tsv"
if [ ! -f "$TRAIN_TSV" ]; then
  "$PY" scripts/preprocess/rewrite_slot4v2_no_digits.py \
    --src-tsv "$SRC_TSV" --out-tsv "$TRAIN_TSV" --log-jsonl "$REWRITE_LOG" --resume \
    --pass3-model Qwen/Qwen2.5-7B-Instruct \
    2>&1 | tee "$STATE/rewrite.log"
else
  log "[Step 1] tsv already built"
fi
[ -f "$TRAIN_TSV" ] || { log "[FAIL] no train tsv"; exit 2; }

log "[Step 2] digit-free gate"
"$PY" - <<PYEOF
import csv, re
csv.field_size_limit(10**9)
rows = list(csv.DictReader(open("$TRAIN_TSV", newline=""), delimiter="\t"))
assert len(rows) == $EXPECTED_N, f"[FAIL] tsv rows {len(rows)} != $EXPECTED_N"
digit = [r["id"] for r in rows if re.search(r"\d", r["caption"])]
empty = sum(1 for r in rows if not r["caption"].strip())
print(f"  rows={len(rows)} digit_rows={len(digit)} empty={empty}")
if digit:
    raise SystemExit(f"[FAIL] {len(digit)} captions still contain digits e.g. {digit[:3]}")
if empty:
    raise SystemExit(f"[FAIL] {empty} empty captions")
src = list(csv.DictReader(open("$SRC_TSV", newline=""), delimiter="\t"))
assert [r["id"] for r in src] == [r["id"] for r in rows], "[FAIL] id order drifted from slot0"
same = sum(1 for a, b in zip(src, rows) if not re.search(r"\d", a["caption"]) and a["caption"] == b["caption"])
clean = sum(1 for a in src if not re.search(r"\d", a["caption"]))
assert same == clean, f"[FAIL] digit-free rows changed: {clean - same}"
import json, sys
sys.path.insert(0, "scripts/preprocess")
import rewrite_slot4v2_no_digits as R
R.audit(src, rows)  # raises on lost sentence / injection / new dangling fragment
m = json.load(open("${TRAIN_TSV%.tsv}.manifest.json"))
print("  manifest", json.dumps(m["sentence_methods"]), json.dumps(m["audit"]))
PYEOF
"$PY" scripts/preprocess/qa_slot4v2_corpus.py --src-tsv "$SRC_TSV" --dst-tsv "$TRAIN_TSV" \
  --report "$INPUTS/qa_report.json" --examples 3 --strict > "$STATE/qa.log" 2>&1 \
  || { log "[FAIL] corpus QA hard failures, see $STATE/qa.log"; exit 2; }
log "[Step 2] corpus QA passed"

# ---- Step 3: overlay (hardlink true_random, patch rewritten slot 0) ---------
log "[Step 3] text overlay"
[ -f "$TRUE_RANDOM/DONE.json" ] || { log "[FAIL] true_random overlay missing"; exit 2; }
if [ ! -f "$OVERLAY/DONE.json" ]; then
  "$PY" scripts/preprocess/patch_stacked_overlay_slot0.py \
    --train-tsv "$TRAIN_TSV" --cache-list "$CACHE_LIST" \
    --output-dir "$OVERLAY" --batch-size 48 2>&1 | tee -a "$STATE/overlay.log"
else
  log "[Step 3] overlay already complete"
fi
[ -f "$OVERLAY/DONE.json" ] || { log "[FAIL] overlay not finished"; exit 2; }

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
  log "[Step 4] Stage 1 $S1_EXP"
  "$PY" set_training_stage.py --stage 1
  S1_RESUME=(); [ -f "$S1_CKPT" ] && S1_RESUME=( "checkpoint=$S1_CKPT" )
  "$TORCHRUN" --standalone --nproc_per_node=1 train.py \
    model=fluxaudio_s exp_id="$S1_EXP" num_iterations="$S1_UPDATES" \
    "${COMMON[@]}" "${S1_RESUME[@]}" 2>&1 | tee -a "$STATE/train_s1.log"
else
  log "[Step 4] S1 already complete"
fi
[ -f "$S1_CKPT" ] || [ -f "$S1_EMA" ] || { log "[FAIL] no S1"; exit 2; }

if [ ! -f "$S2_EMA" ]; then
  log "[Step 5] Stage 2 $S2_EXP"
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
  log "[Step 5] S2 already complete"
fi
[ -f "$S2_EMA" ] || { log "[FAIL] no S2 EMA"; exit 2; }

NEG='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
MC_TSV="$DATA/musiccaps_test.tsv"
OUT="$HOME/eval_output_nvme/${EXP_PREFIX}_mc_mf25_cfg3_neg"
mkdir -p "$OUT/audio"
HAVE=$(find "$OUT/audio" -name '*.flac' | wc -l)
if [ "$HAVE" -lt 5400 ]; then
  log "[Step 6] CFG3+neg generation (have $HAVE)"
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
# CFG3+neg comparators (slot0 / slot4) are batch-32 CLAP; phase4_eval is per-file.
"$PY" scripts/eval/rescore_clap_batch32.py "$OUT/audio" 2>&1 | tee "$STATE/eval_clap_b32.log"

LABEL="${EXP_PREFIX}_musiccaps_mf25_cfg0_noq"
C0_OUT="$HOME/cfg0_eval_runtime/output/$LABEL"
C0_METRICS="$HOME/cfg0_eval_runtime/metrics/$LABEL"
C0_REPORT="$HOME/cfg0_eval_runtime/reports/${LABEL}_REPORT.json"
mkdir -p "$C0_OUT/audio" "$C0_METRICS" "$(dirname "$C0_REPORT")"
HAVE0=$(find "$C0_OUT/audio" -name '*.flac' | wc -l)
if [ "$HAVE0" -lt 5400 ]; then
  log "[Step 7] CFG0 generation (have $HAVE0)"
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

"$PY" - <<PYEOF
import hashlib, json, datetime, soundfile as sf
from pathlib import Path
m = {}
for line in Path("$C0_METRICS/metrics.txt").read_text().splitlines():
    if ":" in line:
        k, v = line.split(":", 1)
        try: m[k.strip()] = float(v.strip())
        except ValueError: pass
need = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
missing = need - set(m)
if missing:
    raise SystemExit(f"[FAIL] cfg0 metrics missing {missing}")
files = sorted(Path("$C0_OUT/audio").glob("*.flac"))
info = sf.info(str(files[0]))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
json.dump({
    "status": "passed",
    "label": "$LABEL",
    "conditioning": "no_q",
    "cfg_strength": 0,
    "num_steps": 25,
    "protocol": "MusicCaps 5521; MeanFlow 25; CFG 0; NoMask; seed 42; full precision",
    "checkpoint": "$S2_EMA",
    "checkpoint_sha256": sha("$S2_EMA"),
    "metrics": {k: m[k] for k in sorted(need)},
    "metrics_path": "$C0_METRICS/metrics.txt",
    "metrics_sha256": sha("$C0_METRICS/metrics.txt"),
    "audio_validation": {"rows": len(files), "unique_ids": len({f.stem for f in files}),
                         "sample_rate": info.samplerate, "channels": info.channels},
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, open("$C0_REPORT", "w"), indent=1, sort_keys=True)
print("wrote $C0_REPORT")
PYEOF

log "[DONE] $EXP_PREFIX"
