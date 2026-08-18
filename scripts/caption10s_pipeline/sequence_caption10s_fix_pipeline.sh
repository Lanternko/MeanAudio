#!/usr/bin/env bash
# Gated caption-window fix pipeline.
# 1 pilot_n512 → 2 review gate → 3 full caption → 4 TSV → 5 NPZ text reextract
# → 6 NoQ train 400k+200k → 7 MusicCaps eval
set -euo pipefail

ROOT_CODE=/home/kojiek/research/meanaudio_training
MEANAUDIO=/home/kojiek/MeanAudio
LOG_ROOT=/home/kojiek/logs
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
CHAIN=caption10s_fix
STATE="$LOG_ROOT/${CHAIN}_sequence"
LOCK="$STATE/sequence.lock"
OUT="$ROOT_CODE/outputs/caption10s_pipeline"
SCRIPTS="$ROOT_CODE/caption10s_pipeline"
OFFICIAL_TSV="$DATA/phase8_qwen_official_matched.tsv"
CACHE_LIST="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
OLD_NPZ=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
NEW_NPZ=/mnt/HDD/kojiek/phase8_qwen_caption10s_matched_npz
MUSICCAPS="$DATA/musiccaps_test.tsv"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
NOTIFY="$MEANAUDIO/scripts/notify_experiment_webhook.py"
SEED=42
PILOT_N=512
EXPECTED_ROWS=251599
PREFIX=phase8_qwen_caption10s_noq_full
S1_UPDATES=400000
S2_UPDATES=200000
FINAL_IT=$((S1_UPDATES + S2_UPDATES))
LR=1e-4
BATCH=8

mkdir -p "$STATE" "$OUT" "$LOG_ROOT"
exec 9>"$LOCK"
flock -n 9 || { echo "[FAIL] $CHAIN already running" >&2; exit 3; }

ts() { date --iso-8601=seconds; }
log() { echo "[$(ts)] $*" | tee -a "$STATE/sequence.log"; }
mark_done() { echo "$(ts)" > "$STATE/${1}.done"; }
is_done() { [ -f "$STATE/${1}.done" ]; }

notify() {
  local status="$1" summary="$2" code="${3:-0}"
  if [ -f "$NOTIFY" ]; then
    python "$NOTIFY" --status "$status" --experiment "$CHAIN" \
      --exit-code "$code" --summary "$summary" 2>/dev/null || true
  fi
}

on_fail() {
  local code=$?
  log "[FAIL] exit=$code at line (see log)"
  notify failure "caption10s pipeline FAILED exit=$code; see $STATE/sequence.log" "$code"
  exit "$code"
}
trap on_fail ERR

wait_gpu() {
  local active
  while true; do
    active=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)
    if [ -z "${active//[[:space:]]/}" ]; then return 0; fi
    log "[WAIT] GPU busy: ${active//$'\n'/; }"
    sleep 60
  done
}

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT_CODE"

log "===== $CHAIN start ====="

# ── 1 pilot n512 ──
STAGE=pilot_n512
PILOT_JSONL="$OUT/captions_pilot_n${PILOT_N}_seed${SEED}.jsonl"
PILOT_SUM="$OUT/pilot_n${PILOT_N}_seed${SEED}_SUMMARY.json"
if is_done "$STAGE" && [ -f "$PILOT_JSONL" ]; then
  log "[SKIP] $STAGE"
else
  log "[START] $STAGE"
  wait_gpu
  python "$SCRIPTS/gen_qwen_caption_10s_crop.py" \
    --tsv "$OFFICIAL_TSV" --out_jsonl "$PILOT_JSONL" \
    --limit "$PILOT_N" --shuffle_seed "$SEED" --seed "$SEED" \
    --batch_size 8 --resume \
    2>&1 | tee "$STATE/${STAGE}.log"
  python - "$PILOT_JSONL" "$PILOT_N" <<'PY'
import json, sys
from pathlib import Path
p, need = Path(sys.argv[1]), int(sys.argv[2])
rows=[json.loads(l) for l in p.open() if l.strip()]
m={}
for r in rows:
    if r.get("id"): m[r["id"]]=r
ok=sum(1 for r in m.values() if r.get("caption"))
print(f"pilot unique={len(m)} ok={ok}")
if ok < int(need * 0.95):
    raise SystemExit(f"[FAIL] pilot ok {ok} < 95% of {need}")
PY
  mark_done "$STAGE"
  log "[DONE] $STAGE"
fi

# ── 2 review gate ──
STAGE=review_n512
VERDICT="$OUT/pilot_n${PILOT_N}_seed${SEED}_VERDICT.json"
if is_done "$STAGE" && [ -f "$VERDICT" ]; then
  python - "$VERDICT" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]))
assert v.get("status")=="passed", v
print("[OK] cached verdict passed")
PY
  log "[SKIP] $STAGE"
else
  log "[START] $STAGE"
  wait_gpu
  python "$SCRIPTS/compare_caption10s_pilot.py" \
    --new_jsonl "$PILOT_JSONL" --out_dir "$OUT" \
    --tag "pilot_n${PILOT_N}_seed${SEED}" \
    2>&1 | tee "$STATE/${STAGE}_compare.log"
  if ! python "$SCRIPTS/review_caption10s_gate.py" \
      --summary "$PILOT_SUM" --out_verdict "$VERDICT" \
      --min_n 480 --min_delta 0.01 --max_null_rate 0.05 \
      2>&1 | tee "$STATE/${STAGE}_gate.log"; then
    notify failure "caption10s pilot GATE FAILED — stopped before full re-caption. See $VERDICT" 2
    exit 2
  fi
  mark_done "$STAGE"
  notify success "caption10s pilot n=${PILOT_N} GATE PASSED; starting full 251k re-caption" 0
  log "[DONE] $STAGE GATE PASS"
fi

# ── 3 full caption ──
STAGE=full_caption
FULL_JSONL="$OUT/captions_full_251599_10s.jsonl"
need_full=1
if is_done "$STAGE" && [ -f "$FULL_JSONL" ]; then
  ok=$(python - "$FULL_JSONL" <<'PY'
import json,sys
m={}
for l in open(sys.argv[1]):
    if not l.strip(): continue
    r=json.loads(l)
    if r.get("id"): m[r["id"]]=r
print(sum(1 for r in m.values() if r.get("caption")))
PY
)
  if [ "$ok" -ge $((EXPECTED_ROWS * 99 / 100)) ]; then
    log "[SKIP] $STAGE ok=$ok"
    need_full=0
  else
    log "[RESUME] $STAGE ok=$ok"
    rm -f "$STATE/${STAGE}.done"
  fi
fi
if [ "$need_full" -eq 1 ]; then
  log "[START] $STAGE"
  wait_gpu
  python "$SCRIPTS/gen_qwen_caption_10s_crop.py" \
    --tsv "$OFFICIAL_TSV" --out_jsonl "$FULL_JSONL" \
    --seed "$SEED" --batch_size 8 --resume \
    2>&1 | tee -a "$STATE/${STAGE}.log"
  python - "$FULL_JSONL" "$EXPECTED_ROWS" <<'PY'
import json,sys
from pathlib import Path
p, need = Path(sys.argv[1]), int(sys.argv[2])
m={}
for l in p.open():
    if not l.strip(): continue
    r=json.loads(l)
    if r.get("id"): m[r["id"]]=r
ok=sum(1 for r in m.values() if r.get("caption"))
print(f"full unique={len(m)} ok={ok}")
if ok < int(need * 0.99):
    raise SystemExit(f"[FAIL] full ok {ok} < 99% of {need}")
PY
  mark_done "$STAGE"
  notify success "caption10s full re-caption done; building TSV + NPZ" 0
  log "[DONE] $STAGE"
fi

# ── 4 build TSV ──
STAGE=build_tsv
NEW_TSV="$DATA/phase8_qwen_caption10s_train.tsv"
TSV_MANIFEST="$OUT/caption10s_train_tsv.manifest.json"
if is_done "$STAGE" && [ -f "$NEW_TSV" ] && [ -f "$TSV_MANIFEST" ]; then
  log "[SKIP] $STAGE"
else
  log "[START] $STAGE"
  python "$SCRIPTS/build_train_tsv_from_caption10s.py" \
    --official_tsv "$OFFICIAL_TSV" --caption_jsonl "$FULL_JSONL" \
    --out_tsv "$NEW_TSV" --out_manifest "$TSV_MANIFEST" \
    2>&1 | tee "$STATE/${STAGE}.log"
  mark_done "$STAGE"
  log "[DONE] $STAGE"
fi

# ── 5 reextract text NPZ ──
STAGE=reextract_text
REEX_PROGRESS="$STATE/reextract_progress.json"
REEX_MANIFEST="$OUT/caption10s_npz_manifest.json"
if is_done "$STAGE" && [ -f "$REEX_MANIFEST" ]; then
  log "[SKIP] $STAGE"
else
  log "[START] $STAGE"
  python - "$OLD_NPZ" "$NEW_NPZ" <<'PY'
import shutil, sys
from pathlib import Path
old, new = map(Path, sys.argv[1:])
sample = next(old.glob("*.npz"))
n = sum(1 for _ in old.glob("*.npz"))
have = sum(1 for _ in new.glob("*.npz")) if new.exists() else 0
remain = max(0, n - have)
need = int(sample.stat().st_size * remain * 1.08)
parent = new if new.exists() else new.parent
while not parent.exists():
    parent = parent.parent
free = shutil.disk_usage(parent).free
print(f"old_n={n} have={have} remain={remain} need_GB={need/1e9:.1f} free_GB={free/1e9:.1f}")
if free < need:
    raise SystemExit(
        f"[FAIL] disk gate: free {free/1e9:.1f}GB < need {need/1e9:.1f}GB for NEW_NPZ={new}. "
        "Free HDD space then resume sequence."
    )
print("[OK] disk gate")
PY
  mkdir -p "$NEW_NPZ"
  wait_gpu
  python "$SCRIPTS/reextract_text_from_source_caption10s.py" \
    --train_tsv "$NEW_TSV" --cache_list "$CACHE_LIST" \
    --source_npz_dir "$OLD_NPZ" --dest_npz_dir "$NEW_NPZ" \
    --batch_size 64 --progress_json "$REEX_PROGRESS" \
    2>&1 | tee "$STATE/${STAGE}.log"
  python - "$NEW_NPZ" "$CACHE_LIST" "$EXPECTED_ROWS" "$REEX_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone
npz_dir, cache, need, man = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
names=[ln.strip() for ln in cache.open() if ln.strip()]
missing=[n for n in names if not (npz_dir/n).exists()]
if missing:
    raise SystemExit(f"missing {len(missing)} npz")
if len(names)!=need:
    raise SystemExit(f"cache {len(names)} != {need}")
payload={"status":"passed","completed_rows":len(names),"npz_dir":str(npz_dir),
         "cache_list":str(cache),"completed_at":datetime.now(timezone.utc).isoformat(),
         "window_sec":10,"text_only_reextract":True}
man.write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps(payload, indent=2))
PY
  mark_done "$STAGE"
  notify success "caption10s NPZ ready; starting NoQ train 400k+200k" 0
  log "[DONE] $STAGE"
fi

# ── 6 train NoQ ──
STAGE=train_noq
S1_EXP="${PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${PREFIX}_stage2_${S2_UPDATES}"
S1_DIR="$MEANAUDIO/exps/$S1_EXP"
S2_DIR="$MEANAUDIO/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"
S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"
S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"
if is_done "$STAGE" && [ -f "$S2_EMA" ]; then
  log "[SKIP] $STAGE"
else
  log "[START] $STAGE"
  wait_gpu
  cd "$MEANAUDIO"
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || { log "[FAIL] nvidia compat"; exit 2; }

  if [ ! -f "$S1_EMA" ] && [ ! -f "$S1_CKPT" ]; then
    log "[TRAIN S1] $S1_EXP"
    python set_training_stage.py 1
    mkdir -p "$S1_DIR"
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=fluxaudio_s exp_id="$S1_EXP" \
      num_iterations="$S1_UPDATES" "lr_schedule_steps=[999999,999999]" \
      "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
      learning_rate="$LR" seed=14159265 linear_warmup_steps=1000 num_workers=4 \
      save_weights_interval=10000 save_checkpoint_interval=10000 \
      ++ema.checkpoint_every=5000 +use_rope=False +use_wandb=False \
      +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
      save_eval_interval=999999 "data.AudioCaps_npz.tsv=$NEW_TSV" \
      "+data.AudioCaps_npz.npz_dir=$NEW_NPZ" \
      "+data.AudioCaps_npz.gt_cache=$CACHE_LIST" \
      2>&1 | tee "$STATE/train_s1.log"
  else
    log "[SKIP] S1 artifacts exist"
  fi
  [ -f "$S1_CKPT" ] || [ -f "$S1_EMA" ] || { log "[FAIL] no S1 ckpt"; exit 2; }
  # ensure ema_final
  if [ ! -f "$S1_EMA" ]; then
    log "[WARN] no ema_final; training should have produced it — check log"
  fi

  if [ ! -f "$S2_EMA" ]; then
    log "[TRAIN S2] migrate + $S2_EXP"
    python set_training_stage.py 2
    mkdir -p "$S2_DIR"
    SRC_CKPT="$S1_CKPT"
    [ -f "$SRC_CKPT" ] || SRC_CKPT="$S1_EMA"
    python migrate_stage1_to_stage2_ckpt.py \
      --s1_ckpt "$SRC_CKPT" \
      --s2_out "$S2_CKPT" \
      --q-init preserve \
      2>&1 | tee "$STATE/migrate.log"
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=meanaudio_s exp_id="$S2_EXP" \
      num_iterations="$FINAL_IT" "lr_schedule_steps=[999999,999999]" \
      "+use_q_conditioning=false" batch_size="$BATCH" +accumulation_steps=1 \
      learning_rate="$LR" seed=14159265 linear_warmup_steps=1000 num_workers=4 \
      save_weights_interval=10000 save_checkpoint_interval=10000 \
      ++ema.checkpoint_every=5000 +use_rope=False +use_wandb=False \
      +use_text_attention_mask=false val_interval=999999 eval_interval=999999 \
      save_eval_interval=999999 "data.AudioCaps_npz.tsv=$NEW_TSV" \
      "+data.AudioCaps_npz.npz_dir=$NEW_NPZ" \
      "+data.AudioCaps_npz.gt_cache=$CACHE_LIST" \
      2>&1 | tee "$STATE/train_s2.log"
  fi
  [ -f "$S2_EMA" ] || { log "[FAIL] missing S2 ema_final"; exit 2; }
  mark_done "$STAGE"
  notify success "caption10s NoQ train finished; starting MusicCaps eval" 0
  log "[DONE] $STAGE"
fi

# ── 7 eval ──
STAGE=eval_noq
EVAL_EXP="${S2_EXP}_musiccaps_mf1_noq"
METRICS="$MEANAUDIO/eval_output/metrics/$EVAL_EXP/metrics.txt"
FINAL_REPORT="$LOG_ROOT/${PREFIX}_FINAL_METRICS.json"
if is_done "$STAGE" && [ -f "$FINAL_REPORT" ]; then
  log "[SKIP] $STAGE"
else
  log "[START] $STAGE"
  wait_gpu
  cd "$MEANAUDIO"
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || true
  OUT_AUD="$MEANAUDIO/eval_output/$EVAL_EXP/audio"
  mkdir -p "$OUT_AUD" "$(dirname "$METRICS")"
  if [ ! -f "$METRICS" ]; then
    python eval.py --variant meanaudio_s --model_path "$S2_EMA" \
      --output "$OUT_AUD" --tsv "$MUSICCAPS" \
      --use_meanflow --num_steps 1 --cfg_strength 0.5 \
      --encoder_name t5_clap --text_c_dim 512 \
      --no_q --no_text_attention_mask --full_precision \
      2>&1 | tee "$STATE/eval_gen.log"
    python "$EVALUATOR" --gen_dir "$OUT_AUD" --tsv "$MUSICCAPS" \
      --exp_name "$EVAL_EXP" --num_samples 5521 \
      2>&1 | tee -a "$STATE/eval_gen.log"
  fi
  python - "$FINAL_REPORT" "$METRICS" "$S2_EMA" <<'PY'
import json, math, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path
report, metrics, ckpt = map(Path, sys.argv[1:])
vals={}
for line in metrics.read_text().splitlines():
    if ": " in line:
        k,v=line.split(": ",1)
        if k in {"clap_score","aes_CE","aes_CU","aes_PC","aes_PQ"}:
            vals[k]=float(v)
assert set(vals)=={"clap_score","aes_CE","aes_CU","aes_PC","aes_PQ"}
h=hashlib.sha256()
with ckpt.open("rb") as f:
    for b in iter(lambda: f.read(8<<20), b""): h.update(b)
payload={
  "schema_version":1,"status":"passed",
  "experiment":"phase8_qwen_caption10s_noq_full",
  "completed_at":datetime.now(timezone.utc).isoformat(),
  "design":"Qwen captions on first-10s crop (aligned to training window)",
  "baseline_noq_clap_old_30s_caption":0.1735,
  "global":{"no_q":vals,"protocol":"MusicCaps 5521; MeanFlow1 CFG0.5"},
  "model":{"path":str(ckpt),"sha256":h.hexdigest()},
}
report.write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps(payload, indent=2))
print(f"CLAP={vals['clap_score']:.4f} vs old NoQ 0.1735")
PY
  rm -rf "$OUT_AUD"
  mark_done "$STAGE"
  log "[DONE] $STAGE"
fi

log "===== $CHAIN COMPLETE ====="
notify success "caption10s pipeline COMPLETE (pilot→gate→full→npz→train→eval). GPU free." 0
