#!/usr/bin/env bash
# Manual gate override: caption10s aligned NPZ is approved for training.
# 1) Discord CLAP report (manual pass)
# 2) Quarter NoQ: S1=100k + S2=50k
# 3) MusicCaps eval
# 4) Discord quarter results
# 5) Full NoQ: S1=400k + S2=200k
# 6) MusicCaps eval + Discord
set -euo pipefail

MEANAUDIO=/home/kojiek/MeanAudio
ROOT_CODE=/home/kojiek/research/meanaudio_training
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/caption10s_noq_quarter_full_sequence"
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
NOTIFY="$MEANAUDIO/scripts/notify_experiment_webhook.py"
EVALUATOR=/home/kojiek/research/meanaudio_eval/phase4_eval.py
MUSICCAPS="$DATA/musiccaps_test.tsv"
NEW_TSV="$DATA/phase8_qwen_caption10s_train.tsv"
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
CACHE_LIST="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
VERIFY_JSON="$ROOT_CODE/outputs/caption10s_pipeline/caption10s_clap_distribution_n1024_VERDICT.json"
LR=1e-4
BATCH=8
SEED=14159265

mkdir -p "$STATE" "$LOG_ROOT"
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
  if ! python "$NOTIFY" --status "$status" --experiment "caption10s_noq_chain" \
      --exit-code "$code" --summary "$summary"; then
    log "[WARN] notify failed status=$status exit_code=$code"
  fi
}
mark() { echo "$(ts)" > "$STATE/${1}.done"; }
is_done() { [ -f "$STATE/${1}.done" ]; }

# Always Discord on non-zero exit (was missing → ENOSPC death silent)
_on_err() {
  local code=$?
  local line=${1:-?}
  log "[FAIL] sequence abort exit=$code line=$line"
  local free
  free=$(df -h / | awk 'NR==2{print $4}')
  notify failure "caption10s_noq_chain FAILED exit=$code line=$line free=$free host=$(hostname)" "$code" || true
  exit "$code"
}
trap '_on_err $LINENO' ERR

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd "$MEANAUDIO"
if [ -f scripts/runtime/phase8_nvidia_compat_env.sh ]; then
  # shellcheck source=/dev/null
  source scripts/runtime/phase8_nvidia_compat_env.sh
  phase8_nvidia_compat_apply || true
fi

# --- 0) Manual gate pass + Discord CLAP report ---
if ! is_done manual_gate_discord; then
  log "[GATE] manual pass: distributional CLAP approved by user"
  python - "$VERIFY_JSON" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone
p=Path(sys.argv[1])
d=json.loads(p.read_text())
d["status"]="passed"
d["manual_override"]=True
d["manual_override_reason"]="user approved option2: broad improvement accepted; only p25 failed by 0.004"
d["manual_override_at"]=datetime.now(timezone.utc).isoformat()
# keep failed list for audit but mark override
d["original_failed"]=d.get("failed",[])
d["failed"]=[]
p.write_text(json.dumps(d, indent=2, ensure_ascii=False)+"\n")
s=d["stats"]["delta_new_minus_old"]
print(
  f"MANUAL PASS caption10s CLAP n={d['stats']['n']}: "
  f"meanΔ={s['mean']:.4f} medianΔ={s['median']:.4f} "
  f"frac+={s['frac_positive']:.1%} frac≥0.02={s['frac_delta_ge_0p02']:.1%} "
  f"CI95=[{s['bootstrap_mean_ci95'][0]:.4f},{s['bootstrap_mean_ci95'][1]:.4f}] "
  f"p25={s['p25']:.4f} (was only fail). NPZ text aligned 251599. Starting 1/4 then full NoQ."
)
PY
  SUMMARY=$(python -c "import json; d=json.load(open('$VERIFY_JSON')); s=d['stats']['delta_new_minus_old']; print(f\"MANUAL PASS caption10s CLAP n={d['stats']['n']}: meanΔ={s['mean']:.4f} medianΔ={s['median']:.4f} frac+={s['frac_positive']:.1%} frac≥0.02={s['frac_delta_ge_0p02']:.1%} CI=[{s['bootstrap_mean_ci95'][0]:.4f},{s['bootstrap_mean_ci95'][1]:.4f}] p25={s['p25']:.4f}. NPZ text 251599 aligned. Start quarter then full NoQ.\")")
  notify success "$SUMMARY" 0
  mark manual_gate_discord
  log "[DONE] Discord gate report sent"
fi

train_stage() {
  local scale="$1" s1_updates="$2" s2_updates="$3"
  local prefix="phase8_qwen_caption10s_noq_${scale}"
  local final_it=$((s1_updates + s2_updates))
  local s1_exp="${prefix}_stage1_${s1_updates}"
  local s2_exp="${prefix}_stage2_${s2_updates}"
  local s1_dir="$MEANAUDIO/exps/$s1_exp"
  local s2_dir="$MEANAUDIO/exps/$s2_exp"
  local s1_ckpt="$s1_dir/${s1_exp}_ckpt_last.pth"
  local s1_ema="$s1_dir/${s1_exp}_ema_final.pth"
  local s2_ckpt="$s2_dir/${s2_exp}_ckpt_last.pth"
  local s2_ema="$s2_dir/${s2_exp}_ema_final.pth"
  local report="$LOG_ROOT/${prefix}_FINAL_METRICS.json"

  log "===== train $scale S1=${s1_updates} S2=${s2_updates} ====="

  if [ -f "$s1_ema" ]; then
    log "[SKIP] S1 ema_final exists for $scale"
  else
    log "[TRAIN S1] $s1_exp (fresh or resume)"
    python set_training_stage.py --stage 1
    mkdir -p "$s1_dir"
    # Resume via checkpoint= if partial run left ckpt_last (do NOT skip!)
    local s1_resume=()
    if [ -f "$s1_ckpt" ]; then
      s1_resume=( "checkpoint=$s1_ckpt" )
      log "[RESUME] S1 from $s1_ckpt"
    fi
    # EMA every 10k to reduce disk (was 5k → ENOSPC risk)
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=fluxaudio_s exp_id="$s1_exp" \
      num_iterations="$s1_updates" "lr_schedule_steps=[999999,999999]" \
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
      "${s1_resume[@]}" \
      2>&1 | tee -a "$STATE/train_${scale}_s1.log"
  fi
  [ -f "$s1_ckpt" ] || [ -f "$s1_ema" ] || { log "[FAIL] no S1 for $scale"; return 2; }

  if [ ! -f "$s2_ema" ]; then
    log "[TRAIN S2] $s2_exp"
    python set_training_stage.py --stage 2
    mkdir -p "$s2_dir"
    local src="$s1_ckpt"
    [ -f "$src" ] || src="$s1_ema"
    python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "$src" --s2_out "$s2_ckpt" --q-init preserve \
      2>&1 | tee "$STATE/migrate_${scale}.log"
    torchrun --standalone --nproc_per_node=1 train.py \
      data=meanaudio model=meanaudio_s exp_id="$s2_exp" \
      num_iterations="$final_it" "lr_schedule_steps=[999999,999999]" \
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
      2>&1 | tee -a "$STATE/train_${scale}_s2.log"
  else
    log "[SKIP] S2 ema exists for $scale"
  fi
  [ -f "$s2_ema" ] || { log "[FAIL] no S2 ema for $scale"; return 2; }

  # eval
  local eval_exp="${s2_exp}_musiccaps_mf1_noq"
  local metrics="$MEANAUDIO/eval_output/metrics/${eval_exp}/metrics.txt"
  local out_aud="$MEANAUDIO/eval_output/${eval_exp}/audio"
  if [ ! -f "$metrics" ]; then
    log "[EVAL] $eval_exp"
    mkdir -p "$out_aud" "$(dirname "$metrics")"
    python eval.py --variant meanaudio_s --model_path "$s2_ema" \
      --output "$out_aud" --tsv "$MUSICCAPS" \
      --use_meanflow --num_steps 1 --cfg_strength 0.5 \
      --encoder_name t5_clap --text_c_dim 512 \
      --no_q --no_text_attention_mask --full_precision \
      2>&1 | tee "$STATE/eval_${scale}.log"
    python "$EVALUATOR" --gen_dir "$out_aud" --tsv "$MUSICCAPS" \
      --exp_name "$eval_exp" --num_samples 5521 \
      2>&1 | tee -a "$STATE/eval_${scale}.log"
    rm -rf "$out_aud"
  else
    log "[SKIP] metrics exist $metrics"
  fi

  python - "$report" "$metrics" "$s2_ema" "$scale" <<'PY'
import json, math, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path
report, metrics, ckpt, scale = sys.argv[1:]
vals={}
for line in Path(metrics).read_text().splitlines():
    if ": " in line:
        k,v=line.split(": ",1)
        if k in {"clap_score","aes_CE","aes_CU","aes_PC","aes_PQ"}:
            vals[k]=float(v)
assert set(vals)=={"clap_score","aes_CE","aes_CU","aes_PC","aes_PQ"}
h=hashlib.sha256()
with open(ckpt,"rb") as f:
    for b in iter(lambda: f.read(8<<20), b""): h.update(b)
payload={
  "schema_version":1,"status":"passed",
  "experiment":f"phase8_qwen_caption10s_noq_{scale}",
  "scale":scale,
  "completed_at":datetime.now(timezone.utc).isoformat(),
  "design":"Qwen captions on first-10s crop; NPZ text in-place aligned; NoQ",
  "baseline_old_caption_noq_full_clap":0.1735,
  "global":{"no_q":vals,"protocol":"MusicCaps 5521; MeanFlow1 CFG0.5"},
  "model":{"path":ckpt,"sha256":h.hexdigest()},
}
Path(report).write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps(payload, indent=2))
print(f"CLAP={vals['clap_score']:.4f}")
PY
  local clap
  clap=$(python -c "import json; print(json.load(open('$report'))['global']['no_q']['clap_score'])")
  notify success "caption10s NoQ ${scale} done: MusicCaps CLAP=${clap} (old full NoQ 0.1735)" 0
  log "[DONE] $scale CLAP=$clap"
}

# Quarter: 1/4 of 400k+200k = 100k+50k
if ! is_done quarter; then
  train_stage quarter 100000 50000
  mark quarter
fi

# Full scale
if ! is_done full; then
  train_stage full 400000 200000
  mark full
fi

log "===== ALL DONE quarter + full ====="
notify success "caption10s NoQ chain COMPLETE (quarter then full). GPU free." 0
