#!/usr/bin/env bash
# Wait for inplace reextract DONE -> distributional CLAP gate -> Discord -> NoQ train
set -euo pipefail

ROOT_CODE=/home/kojiek/research/meanaudio_training
MEANAUDIO=/home/kojiek/MeanAudio
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/caption10s_fix_sequence"
SCRIPTS="$ROOT_CODE/caption10s_pipeline"
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
OUT="$ROOT_CODE/outputs/caption10s_pipeline"
NOTIFY="$MEANAUDIO/scripts/notify_experiment_webhook.py"
DONE_MARK="$STATE/reextract_inplace.DONE"
PROGRESS="$STATE/reextract_inplace_progress.json"
VERIFY_JSON="$OUT/caption10s_clap_distribution_n1024_VERDICT.json"
OLD_TSV="$DATA/phase8_qwen_official_matched.tsv"
NEW_TSV="$DATA/phase8_qwen_caption10s_train.tsv"
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
CACHE_LIST="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
PREFIX=phase8_qwen_caption10s_noq_full
S1_UPDATES=400000
S2_UPDATES=200000
FINAL_IT=$((S1_UPDATES + S2_UPDATES))
LR=1e-4
BATCH=8
MUSICCAPS="$DATA/musiccaps_test.tsv"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py

mkdir -p "$STATE" "$OUT" "$LOG_ROOT"
exec > >(tee -a "$STATE/verify_and_noq.log") 2>&1

ts() { date --iso-8601=seconds; }
log() { echo "[$(ts)] $*"; }
notify() {
  local status="$1" summary="$2" code="${3:-0}"
  if [ ! -f "$NOTIFY" ]; then
    log "[WARN] no notify script at $NOTIFY"
    return 0
  fi
  # Never swallow Discord errors: log them so pipefail bugs are visible.
  if ! python "$NOTIFY" --status "$status" --experiment "caption10s_verify_noq" \
      --exit-code "$code" --summary "$summary"; then
    log "[WARN] notify failed status=$status exit_code=$code"
  fi
}

log "===== wait for reextract ====="
while [ ! -f "$DONE_MARK" ]; do
  if [ -f "$PROGRESS" ]; then
    log "reextract progress: $(cat "$PROGRESS")"
  else
    log "waiting for reextract progress..."
  fi
  # if tmux session died without DONE, fail
  if ! tmux has-session -t caption10s_inplace 2>/dev/null; then
    if [ ! -f "$DONE_MARK" ]; then
      # check EXIT in log
      if grep -q "EXIT:0" "$STATE/reextract_inplace.log" 2>/dev/null && grep -q "\\[DONE\\] inplace" "$STATE/reextract_inplace.log" 2>/dev/null; then
        echo '{"status":"passed","note":"recovered from log"}' > "$DONE_MARK"
        break
      fi
      notify failure "reextract tmux ended without DONE marker" 2
      exit 2
    fi
  fi
  sleep 120
done
log "reextract DONE: $(cat "$DONE_MARK")"

# sanity: spot-check caption_sha256 matches new tsv on random rows
log "===== sha sanity sample ====="
source /home/kojiek/venvs/dac/bin/activate
python - "$NEW_TSV" "$CACHE_LIST" "$NPZ_DIR" <<'PY'
import csv, hashlib, random, sys
from pathlib import Path
import numpy as np
tsv, cache, npz = map(Path, sys.argv[1:])
rows=list(csv.DictReader(tsv.open(), delimiter="\t"))
names=[ln.strip() for ln in cache.open() if ln.strip()]
assert len(rows)==len(names)
rng=random.Random(0)
idx=rng.sample(range(len(rows)), 64)
ok=0
for i in idx:
    want=hashlib.sha256(rows[i]["caption"].encode()).hexdigest()
    z=np.load(npz/names[i], allow_pickle=False)
    got=str(z["caption_sha256"].item())
    if got==want: ok+=1
    else: print("mismatch", rows[i]["id"], got, want)
print(f"sha_ok={ok}/64")
if ok < 60:
    raise SystemExit("sha sanity failed")
PY

log "===== distributional CLAP verify n=1024 ====="
# Capture verify RC: with set -euo pipefail a non-zero verify used to kill the
# script BEFORE Discord notify ran. Always notify first, then exit on fail.
set +e
python "$SCRIPTS/verify_caption10s_clap_distribution.py" \
  --old_tsv "$OLD_TSV" \
  --new_tsv "$NEW_TSV" \
  --n 1024 \
  --seed 42 \
  --out_json "$VERIFY_JSON" \
  --min_mean_delta 0.015 \
  --min_median_delta 0.01 \
  --min_frac_positive 0.58 \
  --min_frac_delta_ge_0p02 0.40 \
  2>&1 | tee "$STATE/clap_verify.log"
verify_rc=${PIPESTATUS[0]}
set -e
log "verify_rc=$verify_rc"

# Discord report (always, even if gate failed)
if [ ! -f "$VERIFY_JSON" ]; then
  notify failure "caption10s CLAP verify produced no VERDICT json (rc=$verify_rc)" "$verify_rc"
  log "[FAIL] missing $VERIFY_JSON"
  exit 2
fi
SUMMARY=$(python - "$VERIFY_JSON" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
s=p["stats"]["delta_new_minus_old"]
print(
  f"caption10s CLAP gate {p['status'].upper()}: "
  f"n={p['stats']['n']} meanΔ={s['mean']:.4f} medianΔ={s['median']:.4f} "
  f"frac+={s['frac_positive']:.1%} frac≥0.02={s['frac_delta_ge_0p02']:.1%} "
  f"CI95=[{s['bootstrap_mean_ci95'][0]:.4f},{s['bootstrap_mean_ci95'][1]:.4f}] "
  f"failed={p.get('failed')}"
)
PY
)
GATE_STATUS=$(python -c "import json; print(json.load(open('$VERIFY_JSON'))['status'])")
if [ "$GATE_STATUS" != "passed" ]; then
  notify failure "$SUMMARY — STOP before NoQ train" 2
  log "GATE FAILED: $SUMMARY"
  exit 2
fi
notify success "$SUMMARY — starting NoQ 400k+200k train" 0
log "GATE PASS: $SUMMARY"

# ===== NoQ train =====
S1_EXP="${PREFIX}_stage1_${S1_UPDATES}"
S2_EXP="${PREFIX}_stage2_${S2_UPDATES}"
S1_DIR="$MEANAUDIO/exps/$S1_EXP"
S2_DIR="$MEANAUDIO/exps/$S2_EXP"
S1_CKPT="$S1_DIR/${S1_EXP}_ckpt_last.pth"
S1_EMA="$S1_DIR/${S1_EXP}_ema_final.pth"
S2_CKPT="$S2_DIR/${S2_EXP}_ckpt_last.pth"
S2_EMA="$S2_DIR/${S2_EXP}_ema_final.pth"

cd "$MEANAUDIO"
source /home/kojiek/venvs/dac/bin/activate
if [ -f scripts/runtime/phase8_nvidia_compat_env.sh ]; then
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || true
fi
export CUDA_VISIBLE_DEVICES=0

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
    "+data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
    "+data.AudioCaps_npz.gt_cache=$CACHE_LIST" \
    2>&1 | tee "$STATE/train_s1.log"
else
  log "[SKIP] S1 exists"
fi
[ -f "$S1_CKPT" ] || [ -f "$S1_EMA" ] || { log "missing S1"; exit 2; }

if [ ! -f "$S2_EMA" ]; then
  log "[TRAIN S2] $S2_EXP"
  python set_training_stage.py 2
  mkdir -p "$S2_DIR"
  SRC="$S1_CKPT"
  [ -f "$SRC" ] || SRC="$S1_EMA"
  python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$SRC" --s2_out "$S2_CKPT" --q-init preserve \
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
    "+data.AudioCaps_npz.npz_dir=$NPZ_DIR" \
    "+data.AudioCaps_npz.gt_cache=$CACHE_LIST" \
    2>&1 | tee "$STATE/train_s2.log"
fi
[ -f "$S2_EMA" ] || { log "missing S2 ema"; exit 2; }
log "S2 ema ready: $S2_EMA"
notify success "caption10s NoQ train finished S2 ema ready; starting eval optional later" 0
log "===== complete verify+noq train ====="
