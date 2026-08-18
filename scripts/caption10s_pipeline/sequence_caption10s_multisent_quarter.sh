#!/usr/bin/env bash
# Multisent 10s captions (fixed stop/clean) → NPZ text reextract → NoQ quarter train + MusicCaps.
# Fair compare twin of caption10s onesent quarter:
#   same ids / 10s crop / Qwen2.5-Omni-3B / NPZ audio / train hyperparams / eval protocol
#   differ only: caption style (multisent max160+clean vs onesent max80+first_sentence)
# Does NOT overwrite onesent TSV or onesent experiment dirs.
set -euo pipefail

MEANAUDIO=/home/kojiek/MeanAudio
ROOT_CODE=/home/kojiek/research/meanaudio_training
PIPE="$ROOT_CODE/caption10s_pipeline"
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/caption10s_multisent_noq_quarter"
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
NOTIFY="$MEANAUDIO/scripts/notify_experiment_webhook.py"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
MUSICCAPS="$DATA/musiccaps_test.tsv"
OFFICIAL_TSV="$DATA/phase8_qwen_official_matched.tsv"
# KEEP onesent TSV intact; new multisent TSV:
NEW_TSV="$DATA/phase8_qwen_caption10s_multisent_train.tsv"
ONESENT_TSV="$DATA/phase8_qwen_caption10s_train.tsv"
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
CACHE_LIST="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
OUT_DIR="$ROOT_CODE/outputs/caption10s_pipeline"
CAP_JSONL="$OUT_DIR/captions_full_251599_10s_multisent.jsonl"
TSV_MANIFEST="$OUT_DIR/caption10s_multisent_train_tsv.manifest.json"
REEXTRACT_PROGRESS="$OUT_DIR/reextract_multisent_progress.json"
REEXTRACT_DONE="$OUT_DIR/reextract_multisent.DONE.json"
CORPUS_GATE_REPORT="$OUT_DIR/caption10s_multisent_strict_gate.json"
CORPUS_GATE_DEFECTS="$OUT_DIR/caption10s_multisent_strict_gate_defects.tsv"
GEN_SCRIPT="$PIPE/gen_qwen_caption_10s_multisent.py"
BUILD_TSV="$PIPE/build_train_tsv_from_caption10s.py"
REEXTRACT="$PIPE/reextract_text_inplace_caption10s.py"
VALIDATE="$PIPE/validate_multisent_corpus.py"

LR=1e-4
BATCH=8
SEED=14159265
S1_UPDATES=100000
S2_UPDATES=50000
PREFIX=phase8_qwen_caption10s_multisent_noq_quarter
SCALE=quarter

mkdir -p "$STATE" "$LOG_ROOT" "$OUT_DIR"
exec > >(tee -a "$STATE/sequence.log") 2>&1
exec 9>"$STATE/sequence.lock"
flock -n 9 || { echo "[FAIL] sequence already running"; exit 3; }

ts() { date --iso-8601=seconds; }
log() { echo "[$(ts)] $*"; }
notify() {
  local status="$1" summary="$2" code="${3:-0}"
  if [ ! -f "$NOTIFY" ]; then
    log "[WARN] no notify script at $NOTIFY"
    return 0
  fi
  if ! python "$NOTIFY" --status "$status" --experiment "caption10s_multisent_noq_quarter" \
      --exit-code "$code" --summary "$summary"; then
    log "[WARN] notify failed status=$status"
  fi
}
mark() { echo "$(ts)" > "$STATE/${1}.done"; }
is_done() { [ -f "$STATE/${1}.done" ]; }

_on_err() {
  local code=$?
  local line=${1:-?}
  log "[FAIL] sequence abort exit=$code line=$line"
  local free
  free=$(df -h / | awk "NR==2{print \$4}")
  notify failure "caption10s_multisent_noq_quarter FAILED exit=$code line=$line free=$free host=$(hostname)" "$code" || true
  exit "$code"
}
trap "_on_err \$LINENO" ERR
_on_signal() {
  local signal="$1" code="$2"
  trap - HUP INT TERM
  log "[INTERRUPTED] received SIG$signal"
  notify interrupted "caption10s_multisent_noq_quarter interrupted by SIG$signal" "$code" || true
  exit "$code"
}
trap '_on_signal HUP 129' HUP
trap '_on_signal INT 130' INT
trap '_on_signal TERM 143' TERM

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd "$MEANAUDIO"
if [ -f scripts/runtime/phase8_nvidia_compat_env.sh ]; then
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || true
fi

log "===== caption10s MULTISENT fair-compare quarter chain ====="
log "onesent TSV kept: $ONESENT_TSV"
log "multisent TSV: $NEW_TSV"
log "exp prefix: $PREFIX"
log "note: NPZ text will be OVERWRITTEN with multisent (audio features unchanged)"

# --- 1) Full multisent caption gen ---
if ! is_done gen_multisent; then
  log "[GEN] multisent captions → $CAP_JSONL"
  python "$GEN_SCRIPT" \
    --tsv "$OFFICIAL_TSV" \
    --out_jsonl "$CAP_JSONL" \
    --batch_size 8 \
    --max_new_tokens 160 \
    --seed 42 \
    --resume \
    2>&1 | tee -a "$STATE/gen_multisent.log"

else
  log "[SKIP] gen_multisent"
fi

# Never trust a .done marker by itself.  The independent full-corpus gate runs
# on every launch and fails closed before any derived artifact can be used.
log "[PREFLIGHT] strict full-corpus gate"
python "$VALIDATE" \
  --corpus "$CAP_JSONL" --official-tsv "$OFFICIAL_TSV" \
  --report "$CORPUS_GATE_REPORT" --defects-tsv "$CORPUS_GATE_DEFECTS"
mark gen_multisent
log "[DONE] gen_multisent + strict gate"

# --- 2) Build train TSV (does not touch onesent TSV) ---
# Rebuild deterministically on every launch.  This prevents a stale .done marker
# from binding an old corpus to the NPZ cache.
log "[TSV] build and hash-bind $NEW_TSV"
[ -f "$ONESENT_TSV" ] || { log "[FAIL] onesent TSV missing — abort"; exit 2; }
python "$BUILD_TSV" \
  --official_tsv "$OFFICIAL_TSV" \
  --caption_jsonl "$CAP_JSONL" \
  --out_tsv "$NEW_TSV" \
  --out_manifest "$TSV_MANIFEST" \
  2>&1 | tee "$STATE/build_tsv.log"
python - "$TSV_MANIFEST" <<'PY'
import json, os, sys
from pathlib import Path
p=Path(sys.argv[1])
d=json.loads(p.read_text())
d["captioner"]="Qwen2.5-Omni-3B first-10s-crop multisent_max160_stop_clean_v1"
d["fair_compare_baseline_tsv"]="/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_train.tsv"
d["fair_compare_note"]="same 10s crop + model + id set; caption style only differs"
tmp=p.with_name(f".{p.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(d, indent=2)+"\n", encoding="utf-8")
os.replace(tmp, p)
print(json.dumps(d, indent=2))
PY
python "$VALIDATE" \
  --corpus "$CAP_JSONL" --official-tsv "$OFFICIAL_TSV" \
  --train-tsv "$NEW_TSV" --manifest "$TSV_MANIFEST" \
  --report "$CORPUS_GATE_REPORT" --defects-tsv "$CORPUS_GATE_DEFECTS"
mark build_tsv
log "[DONE] build_tsv + provenance gate"

# --- 3) In-place NPZ text reextract (overwrites onesent text features) ---
log "[REEXTRACT] verify/resume inplace text → $NPZ_DIR (audio unchanged)"
free=$(df -h /mnt/HDD | awk "NR==2{print \$4}")
log "HDD free before reextract: $free"
python "$REEXTRACT" \
  --train_tsv "$NEW_TSV" \
  --cache_list "$CACHE_LIST" \
  --npz_dir "$NPZ_DIR" \
  --batch_size 32 \
  --progress_json "$REEXTRACT_PROGRESS" \
  --done_json "$REEXTRACT_DONE" \
  2>&1 | tee "$STATE/reextract.log"
# Recheck the source/TSV contract after the long mutation phase.  Any external
# corpus edit during reextract blocks S1.
python "$VALIDATE" \
  --corpus "$CAP_JSONL" --official-tsv "$OFFICIAL_TSV" \
  --train-tsv "$NEW_TSV" --manifest "$TSV_MANIFEST" \
  --reextract-report "$REEXTRACT_DONE" --cache-list "$CACHE_LIST" \
  --report "$CORPUS_GATE_REPORT" --defects-tsv "$CORPUS_GATE_DEFECTS"
mark reextract
log "[DONE] reextract + postflight provenance gate"
notify success "caption10s_multisent: gen+TSV+reextract verified. Starting NoQ quarter S1=100k S2=50k." 0 || true

# --- 4) Quarter NoQ train ---
FINAL_IT=$((S1_UPDATES + S2_UPDATES))
S1_EXP="${PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${PREFIX}_stage2_${S2_UPDATES}"
S1_DIR="$MEANAUDIO/exps/$S1_EXP"
S2_DIR="$MEANAUDIO/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"
S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"
S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
REPORT="$LOG_ROOT/${PREFIX}_FINAL_METRICS.json"

if ! is_done train_quarter; then
  log "===== train $SCALE S1=${S1_UPDATES} S2=${S2_UPDATES} ====="

  if [ -f "$S1_EMA" ]; then
    log "[SKIP] S1 ema_final exists"
  else
    log "[TRAIN S1] $S1_EXP"
    python set_training_stage.py --stage 1
    mkdir -p "$S1_DIR"
    S1_RESUME=()
    if [ -f "$S1_CKPT" ]; then
      S1_RESUME=( "checkpoint=$S1_CKPT" )
      log "[RESUME] S1 from $S1_CKPT"
    fi
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=fluxaudio_s exp_id="$S1_EXP" \
      num_iterations="$S1_UPDATES" "lr_schedule_steps=[999999,999999]" \
      "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
      learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
      save_weights_interval=10000 save_checkpoint_interval=10000 \
      ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
      +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
      save_eval_interval=999999 "data.AudioCaps_npz.tsv=$NEW_TSV" \
      "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
      "++data.AudioCaps_npz.gt_cache=$CACHE_LIST" \
      "data.AudioCaps_val_npz.tsv=$DATA/_QUARANTINED_phase4_val.tsv" \
      "++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4" \
      "++data.AudioCaps_val_npz.gt_cache=null" \
      "${S1_RESUME[@]}" \
      2>&1 | tee -a "$STATE/train_quarter_s1.log"
  fi
  [ -f "$S1_CKPT" ] || [ -f "$S1_EMA" ] || { log "[FAIL] no S1"; exit 2; }

  if [ ! -f "$S2_EMA" ]; then
    log "[TRAIN S2] $S2_EXP"
    python set_training_stage.py --stage 2
    mkdir -p "$S2_DIR"
    SRC="$S1_CKPT"
    [ -f "$SRC" ] || SRC="$S1_EMA"
    python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$SRC" --s2_out "$S2_CKPT" --q-init preserve \
      2>&1 | tee "$STATE/migrate_quarter.log"
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=meanaudio_s exp_id="$S2_EXP" \
      num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
      "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
      learning_rate="$LR" seed="$SEED" linear_warmup_steps=1000 num_workers=4 \
      save_weights_interval=10000 save_checkpoint_interval=10000 \
      ++ema.checkpoint_every=10000 +use_rope=False +use_wandb=False \
      +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
      save_eval_interval=999999 "data.AudioCaps_npz.tsv=$NEW_TSV" \
      "++data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
      "++data.AudioCaps_npz.gt_cache=$CACHE_LIST" \
      "data.AudioCaps_val_npz.tsv=$DATA/_QUARANTINED_phase4_val.tsv" \
      "++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4" \
      "++data.AudioCaps_val_npz.gt_cache=null" \
      2>&1 | tee -a "$STATE/train_quarter_s2.log"
  else
    log "[SKIP] S2 ema exists"
  fi
  [ -f "$S2_EMA" ] || { log "[FAIL] no S2 ema"; exit 2; }
  mark train_quarter
  log "[DONE] train_quarter"
else
  log "[SKIP] train_quarter"
fi

# --- 5) MusicCaps eval ---
EVAL_EXP="${S2_EXP}_musiccaps_mf1_noq"
METRICS="$MEANAUDIO/eval_output/metrics/${EVAL_EXP}/metrics.txt"
OUT_AUD="$MEANAUDIO/eval_output/${EVAL_EXP}/audio"

if ! is_done eval_quarter; then
  if [ ! -f "$METRICS" ]; then
    log "[EVAL] $EVAL_EXP"
    mkdir -p "$OUT_AUD" "$(dirname "$METRICS")"
    python eval.py --variant meanaudio_s --model_path "$S2_EMA" \
      --output "$OUT_AUD" --tsv "$MUSICCAPS" \
      --use_meanflow --num_steps 1 --cfg_strength 0.5 \
      --encoder_name t5_clap --text_c_dim 512 \
      --no_q --no_text_attention_mask --full_precision \
      2>&1 | tee "$STATE/eval_quarter.log"
    python "$EVALUATOR" --gen_dir "$OUT_AUD" --tsv "$MUSICCAPS" \
      --exp_name "$EVAL_EXP" --num_samples 5521 \
      2>&1 | tee -a "$STATE/eval_quarter.log"
    rm -rf "$OUT_AUD"
  else
    log "[SKIP] metrics exist $METRICS"
  fi

  python - "$REPORT" "$METRICS" "$S2_EMA" <<PY
import json, sys, hashlib
from pathlib import Path
from datetime import datetime, timezone
report, metrics, model = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
text = metrics.read_text(encoding="utf-8", errors="replace")
vals = {}
for line in text.splitlines():
    line=line.strip()
    if ":" in line:
        k,v=line.split(":",1)
        k=k.strip(); v=v.strip()
        try: vals[k]=float(v)
        except ValueError: vals[k]=v
h=hashlib.sha256()
with model.open("rb") as f:
    for chunk in iter(lambda: f.read(8<<20), b""):
        h.update(chunk)
# onesent quarter baseline for fair compare
baseline = {
  "exp": "phase8_qwen_caption10s_noq_quarter",
  "clap_score": 0.1734,
  "aes_CE": 5.7702,
  "aes_CU": 6.4237,
  "aes_PC": 5.0037,
  "aes_PQ": 6.3559,
  "protocol": "MusicCaps 5521; MeanFlow1 CFG0.5",
}
out = {
  "schema_version": 1,
  "status": "passed",
  "experiment": "phase8_qwen_caption10s_multisent_noq_quarter",
  "scale": "quarter",
  "completed_at": datetime.now(timezone.utc).isoformat(),
  "design": "Qwen multisent captions on first-10s crop (max160+stop+clean); NPZ text in-place; NoQ quarter",
  "fair_compare": {
    "baseline_onesent_quarter": baseline,
    "controls": [
      "same id set 251599",
      "same first-10s audio crop",
      "same Qwen2.5-Omni-3B captioner model",
      "same NPZ audio features (text only reextracted)",
      "same NoQ quarter 100k+50k, lr=1e-4, batch=8, seed=14159265",
      "same MusicCaps MF1 CFG0.5 eval",
    ],
    "differs": [
      "caption prompt multisent 2-5 sent vs one-sentence",
      "max_new_tokens 160 vs 80",
      "no first_sentence truncation; leak clean postprocess",
    ],
  },
  "global": {
    "no_q": {
      "clap_score": vals.get("clap_score") or vals.get("CLAP") or vals.get("clap"),
      "aes_CE": vals.get("aes_CE") or vals.get("CE"),
      "aes_CU": vals.get("aes_CU") or vals.get("CU"),
      "aes_PC": vals.get("aes_PC") or vals.get("PC"),
      "aes_PQ": vals.get("aes_PQ") or vals.get("PQ"),
    },
    "raw_metrics": vals,
    "protocol": "MusicCaps 5521; MeanFlow1 CFG0.5",
  },
  "model": {"path": str(model), "sha256": h.hexdigest()},
}
report.write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
PY
  mark eval_quarter
  SUMMARY=$(python - "$REPORT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
g = d["global"]["no_q"]
b = d["fair_compare"]["baseline_onesent_quarter"]
print(
    f"MULTISENT quarter CLAP={g.get('clap_score')} CE={g.get('aes_CE')} | "
    f"onesent quarter CLAP={b['clap_score']} CE={b['aes_CE']} | MF1 CFG0.5"
)
PY
)
  notify success "$SUMMARY" 0 || true
  log "[DONE] eval_quarter $SUMMARY"
else
  log "[SKIP] eval_quarter"
fi

mark all
log "===== ALL DONE multisent quarter fair-compare ====="
