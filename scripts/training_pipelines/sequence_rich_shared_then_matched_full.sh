#!/usr/bin/env bash
# R-Shared quarter control; conditionally promote R-Matched to full scale.
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
PIPE=/home/kojiek/research/meanaudio_training/caption10s_pipeline
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/rich_shared_then_matched_full"
NPZ_DIR=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
CACHE_LIST="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
MATCHED_TSV="$DATA/phase8_qwen_caption10s_multisent_train.tsv"
SHARED_TSV="$DATA/caption_alignment_rich_shared_train.tsv"
SHARED_MAPPING="$DATA/caption_alignment_rich_shared_mapping.tsv"
SHARED_MANIFEST="$DATA/caption_alignment_rich_shared_manifest.json"
CONTRACT="$ROOT/docs/experiments/rich_shared_then_matched_full_contract.json"
REEXTRACT="$PIPE/reextract_text_inplace_caption10s.py"
AUDITOR="$ROOT/scripts/preprocess/audit_caption_npz_binding.py"
BUILDER="$ROOT/scripts/preprocess/build_rich_shared_control.py"
GATE_SCRIPT="$ROOT/scripts/analysis/evaluate_rich_shared_gate.py"
MATCHED_QUARTER_REPORT="$LOG_ROOT/phase8_qwen_caption10s_multisent_noq_quarter_FINAL_METRICS.json"
SHARED_PREFIX=phase8_qwen_rich_shared_noq_quarter
FULL_PREFIX=phase8_qwen_caption10s_multisent_noq_full
SHARED_AUDIT="$STATE/rich_shared_npz_binding_audit.json"
MATCHED_AUDIT="$STATE/rich_matched_npz_binding_audit.json"
GATE_REPORT="$LOG_ROOT/${SHARED_PREFIX}_PROMOTION_GATE.json"
FINAL_REPORT="$LOG_ROOT/rich_shared_then_matched_full_FINAL_METRICS.json"
SHARED_METRICS="$ROOT/eval_output/metrics/${SHARED_PREFIX}_stage2_50000_musiccaps/metrics.txt"
SHARED_MODEL="$ROOT/exps/${SHARED_PREFIX}_stage2_50000/${SHARED_PREFIX}_stage2_50000_ema_final.pth"
FULL_METRICS="$ROOT/eval_output/metrics/${FULL_PREFIX}_stage2_200000_musiccaps/metrics.txt"
FULL_MODEL="$ROOT/exps/${FULL_PREFIX}_stage2_200000/${FULL_PREFIX}_stage2_200000_ema_final.pth"

mkdir -p "$STATE"
exec > >(tee -a "$STATE/sequence.log") 2>&1
exec 9>"$STATE/sequence.lock"
flock -n 9 || { echo "[FAIL] sequence already running"; exit 3; }

ts() { date --iso-8601=seconds; }
log() { echo "[$(ts)] $*"; }
mark() { echo "$(ts)" > "$STATE/$1.done"; }
is_done() { [ -f "$STATE/$1.done" ]; }

source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
source "$ROOT/scripts/runtime/phase8_nvidia_compat_env.sh"
phase8_nvidia_compat_apply || { log "[FAIL] NVIDIA preflight"; exit 2; }

for path in "$MATCHED_TSV" "$CACHE_LIST" "$NPZ_DIR" "$CONTRACT" \
            "$MATCHED_QUARTER_REPORT" "$REEXTRACT"; do
    [ -e "$path" ] || { log "[FAIL] missing required input: $path"; exit 2; }
done

root_free=$(df -Pk / | awk 'NR==2 {print $4}')
hdd_free=$(df -Pk /mnt/HDD | awk 'NR==2 {print $4}')
[ "$root_free" -ge $((150 * 1024 * 1024)) ] || { log "[FAIL] root free space below 150 GiB"; exit 2; }
[ "$hdd_free" -ge $((30 * 1024 * 1024)) ] || { log "[FAIL] HDD free space below 30 GiB"; exit 2; }
log "[SPACE] root_free_kib=$root_free hdd_free_kib=$hdd_free"

# Build deterministically every launch; it is small and this prevents stale binding.
python "$BUILDER" \
  --rich-tsv "$MATCHED_TSV" --cache-list "$CACHE_LIST" \
  --out-tsv "$SHARED_TSV" --out-mapping "$SHARED_MAPPING" \
  --manifest "$SHARED_MANIFEST"

if ! is_done shared_quarter; then
  log "[REEXTRACT] bind mutable cache to R-Shared"
  python "$REEXTRACT" \
    --train_tsv "$SHARED_TSV" --cache_list "$CACHE_LIST" --npz_dir "$NPZ_DIR" \
    --batch_size 32 --progress_json "$STATE/shared_reextract_progress.json" \
    --done_json "$STATE/shared_reextract_done.json"
  python "$AUDITOR" --tsv "$SHARED_TSV" --cache-list "$CACHE_LIST" \
    --npz-dir "$NPZ_DIR" --report "$SHARED_AUDIT" --expected-rows 251599

  log "[TRAIN] R-Shared quarter: S1=100k, S2=50k"
  EXP_PREFIX="$SHARED_PREFIX" TRAIN_TSV="$SHARED_TSV" GT_CACHE="$CACHE_LIST" \
  SINGLECAP_NPZ="$NPZ_DIR" NPZ_MANIFEST="$SHARED_AUDIT" EXPECTED_ROWS=251599 \
  EXPERIMENT_REGIME=rich_shared_noq_quarter EXPERIMENT_RUN_MODE=resume \
  S1_ITERATIONS=100000 S2_ITERATIONS=50000 \
  S1_USE_Q_CONDITIONING=false S2_USE_Q_CONDITIONING=false EVAL_Q_MODE=no_q \
  USE_TEXT_ATTENTION_MASK=false RUN_PRIMARY_EVAL=true RUN_JAMENDO_EVAL=false \
  EVAL_NUM_SAMPLES=5521 EVAL_SKIP_AES=false TRAIN_SEED=14159265 \
  SAVE_WEIGHTS_INTERVAL=10000 SAVE_CHECKPOINT_INTERVAL=10000 EMA_CHECKPOINT_INTERVAL=10000 \
  bash "$ROOT/scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh"
  [ -f "$SHARED_METRICS" ] && [ -f "$SHARED_MODEL" ] || {
    log "[FAIL] R-Shared quarter missing metrics/model"; exit 2;
  }
  rm -rf "$ROOT/eval_output/${SHARED_PREFIX}_stage2_50000_musiccaps/audio"
  mark shared_quarter
fi

python "$GATE_SCRIPT" \
  --matched-report "$MATCHED_QUARTER_REPORT" --shared-metrics "$SHARED_METRICS" \
  --shared-model "$SHARED_MODEL" --shared-audit "$SHARED_AUDIT" \
  --contract "$CONTRACT" --gate-report "$GATE_REPORT" --final-report "$FINAL_REPORT"
promote=$(python - "$GATE_REPORT" <<'PY'
import json, sys
print("true" if json.load(open(sys.argv[1]))["promote_r_matched_full"] else "false")
PY
)
if [ "$promote" != true ]; then
  log "[STOP] preregistered gate did not promote R-Matched full scale"
  mark all
  exit 0
fi

log "[PROMOTE] gate passed; restore exact R-Matched text binding before full scale"
python "$REEXTRACT" \
  --train_tsv "$MATCHED_TSV" --cache_list "$CACHE_LIST" --npz_dir "$NPZ_DIR" \
  --batch_size 32 --progress_json "$STATE/matched_reextract_progress.json" \
  --done_json "$STATE/matched_reextract_done.json"
python "$AUDITOR" --tsv "$MATCHED_TSV" --cache-list "$CACHE_LIST" \
  --npz-dir "$NPZ_DIR" --report "$MATCHED_AUDIT" --expected-rows 251599

log "[TRAIN] R-Matched full: S1=400k, S2=200k"
EXP_PREFIX="$FULL_PREFIX" TRAIN_TSV="$MATCHED_TSV" GT_CACHE="$CACHE_LIST" \
SINGLECAP_NPZ="$NPZ_DIR" NPZ_MANIFEST="$MATCHED_AUDIT" EXPECTED_ROWS=251599 \
EXPERIMENT_REGIME=rich_matched_noq_full EXPERIMENT_RUN_MODE=resume \
S1_ITERATIONS=400000 S2_ITERATIONS=200000 \
S1_USE_Q_CONDITIONING=false S2_USE_Q_CONDITIONING=false EVAL_Q_MODE=no_q \
USE_TEXT_ATTENTION_MASK=false RUN_PRIMARY_EVAL=true RUN_JAMENDO_EVAL=false \
EVAL_NUM_SAMPLES=5521 EVAL_SKIP_AES=false TRAIN_SEED=14159265 \
SAVE_WEIGHTS_INTERVAL=50000 SAVE_CHECKPOINT_INTERVAL=50000 EMA_CHECKPOINT_INTERVAL=50000 \
bash "$ROOT/scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh"
[ -f "$FULL_METRICS" ] && [ -f "$FULL_MODEL" ] || {
  log "[FAIL] R-Matched full missing metrics/model"; exit 2;
}

python - "$FINAL_REPORT" "$FULL_METRICS" "$FULL_MODEL" "$MATCHED_AUDIT" <<'PY'
import hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path

report_path, metrics_path, model_path, audit_path = map(Path, sys.argv[1:])
values = {}
for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if ":" in line:
        key, raw = (part.strip() for part in line.split(":", 1))
        if key in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
            try: values[key] = float(raw)
            except ValueError: pass
if set(values) != {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"} or not all(map(math.isfinite, values.values())):
    raise SystemExit("[FAIL] invalid R-Matched full metrics")
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""): h.update(chunk)
    return h.hexdigest()
report = json.loads(report_path.read_text())
report.update({
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "r_matched_full": {
        "metrics": values,
        "protocol": "MusicCaps 5521; MeanFlow1 CFG0.5; seed 42",
        "model": {"path": str(model_path), "sha256": sha(model_path)},
        "npz_binding_audit": str(audit_path),
    },
})
tmp = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
os.replace(tmp, report_path)
print(json.dumps(report["r_matched_full"], indent=2, sort_keys=True))
PY
rm -rf "$ROOT/eval_output/${FULL_PREFIX}_stage2_200000_musiccaps/audio"
mark all
log "[DONE] R-Shared quarter passed gate and R-Matched full completed"
