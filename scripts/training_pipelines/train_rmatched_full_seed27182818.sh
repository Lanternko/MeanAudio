#!/usr/bin/env bash
# Exact second-seed replication of the promoted R-Matched full NoQ run.
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
STATE=/home/kojiek/logs/rmatched_validation_replication_harn
PYTHON=/home/kojiek/venvs/dac/bin/python
PREFIX=phase8_qwen_caption10s_multisent_noq_full_seed27182818
TSV="$DATA/phase8_qwen_caption10s_multisent_train.tsv"
CACHE="$DATA/phase8_qwen_official_matched_npz_cache_train.txt"
NPZ=/mnt/HDD/kojiek/phase8_qwen_official_matched_npz
AUDIT="$STATE/evidence/seed27182818_rmatched_binding_audit.json"
AUDITOR="$ROOT/scripts/preprocess/audit_caption_npz_binding.py"
PIPELINE="$ROOT/scripts/training_pipelines/train_pipeline_phase8_bugfix_rerun.sh"
LOCK="$STATE/rmatched_cache_owner.lock"

MODE=fresh
PREFLIGHT_ONLY=false
HDD_READ_ONLY=false
RESTORE_SHADOW=false
REUSE_PASSED_AUDIT=false
AUDIT_WORKERS=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2"; shift ;;
        --preflight-only) PREFLIGHT_ONLY=true ;;
        --hdd-read-only) HDD_READ_ONLY=true ;;
        --restore-shadow) RESTORE_SHADOW=true ;;
        --reuse-passed-audit) REUSE_PASSED_AUDIT=true ;;
        --audit-workers) AUDIT_WORKERS="$2"; shift ;;
        *) echo "[FAIL] unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
case "$MODE" in fresh|resume) ;; *) echo "[FAIL] --mode must be fresh or resume" >&2; exit 2;; esac

mkdir -p "$STATE/evidence"
chmod 700 "$STATE"
exec 7>"$LOCK"
flock -s -n 7 || { echo "[FAIL] mutable caption cache has an exclusive owner" >&2; exit 3; }

for path in "$TSV" "$CACHE" "$NPZ" "$AUDITOR" "$PIPELINE"; do
    [ -e "$path" ] || { echo "[FAIL] missing input: $path" >&2; exit 2; }
done
root_free=$(stat -f -c '%a * %S' / | awk '{print $1 * $3}')
hdd_free=$(stat -f -c '%a * %S' /mnt/HDD | awk '{print $1 * $3}')
[ "$root_free" -ge $((150 * 1024 * 1024 * 1024)) ] || { echo "[FAIL] root below 150 GiB" >&2; exit 2; }
if [ "$HDD_READ_ONLY" = false ]; then
    [ "$hdd_free" -ge $((50 * 1024 * 1024 * 1024)) ] || { echo "[FAIL] HDD below 50 GiB" >&2; exit 2; }
else
    echo "[OK] HDD registered as read-only input; free_bytes=$hdd_free"
fi

if [ "$RESTORE_SHADOW" = true ]; then
    [ "$MODE" = resume ] || { echo "[FAIL] --restore-shadow requires --mode resume" >&2; exit 2; }
    S2_DIR="$ROOT/exps/${PREFIX}_stage2_200000"
    LAST="$S2_DIR/${PREFIX}_stage2_200000_ckpt_last.pth"
    SHADOW="$S2_DIR/${PREFIX}_stage2_200000_ckpt_shadow.pth"
    [ -f "$SHADOW" ] || { echo "[FAIL] missing resume shadow: $SHADOW" >&2; exit 2; }
    "$PYTHON" - "$SHADOW" <<'PY'
import sys, torch
path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
required = {"weights", "optimizer", "scheduler", "ema", "it"}
if checkpoint.get("it") != 550000 or not required.issubset(checkpoint):
    raise SystemExit(f"[FAIL] shadow is not the complete iteration-550000 checkpoint: {path}")
print("[OK] complete iteration-550000 shadow verified")
PY
    if ! "$PYTHON" - "$LAST" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
raise SystemExit(0 if checkpoint.get("it") == 550000 else 1)
PY
    then
        if [ -e "$LAST" ]; then
            CORRUPT="$LAST.corrupt_interrupted_20260813"
            [ ! -e "$CORRUPT" ] || { echo "[FAIL] preserved corrupt evidence already exists: $CORRUPT" >&2; exit 2; }
            mv "$LAST" "$CORRUPT"
        fi
        cp --reflink=auto --preserve=mode,timestamps "$SHADOW" "$LAST"
        sync "$LAST"
    fi
    "$PYTHON" - "$LAST" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if checkpoint.get("it") != 550000:
    raise SystemExit("[FAIL] restored ckpt_last iteration mismatch")
print("[OK] ckpt_last restored at iteration 550000")
PY
fi

source /home/kojiek/venvs/dac/bin/activate
cd "$ROOT"
if [ "$REUSE_PASSED_AUDIT" = true ]; then
    python - "$AUDIT" "$TSV" "$CACHE" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

report, tsv, cache = map(Path, sys.argv[1:])
payload = json.loads(report.read_text())
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
completed = datetime.fromisoformat(payload["completed_at"])
age = datetime.now(timezone.utc) - completed
if (
    payload.get("status") != "passed"
    or payload.get("completed_rows") != 251599
    or payload.get("rows_checked") != 251599
    or payload.get("tsv_sha256") != sha(tsv)
    or payload.get("cache_list_sha256") != sha(cache)
    or age.total_seconds() < 0
    or age.total_seconds() > 6 * 3600
):
    raise SystemExit("[FAIL] passed binding audit is stale or input hashes changed")
print(f"[OK] reused full binding audit age_seconds={int(age.total_seconds())}")
PY
else
    python "$AUDITOR" --tsv "$TSV" --cache-list "$CACHE" --npz-dir "$NPZ" \
        --report "$AUDIT" --expected-rows 251599 --workers "$AUDIT_WORKERS"
fi

if [ "$PREFLIGHT_ONLY" = true ]; then
    echo "[PREFLIGHT ONLY] exact R-Matched binding passed; no training started"
    exit 0
fi

EXP_PREFIX="$PREFIX" TRAIN_TSV="$TSV" GT_CACHE="$CACHE" \
SINGLECAP_NPZ="$NPZ" NPZ_MANIFEST="$AUDIT" EXPECTED_ROWS=251599 \
EXPERIMENT_REGIME=rich_matched_noq_full_seed_replication EXPERIMENT_RUN_MODE="$MODE" \
S1_ITERATIONS=400000 S2_ITERATIONS=200000 \
S1_USE_Q_CONDITIONING=false S2_USE_Q_CONDITIONING=false EVAL_Q_MODE=no_q \
USE_TEXT_ATTENTION_MASK=false RUN_PRIMARY_EVAL=false RUN_JAMENDO_EVAL=false \
EVAL_NUM_SAMPLES=5521 EVAL_SKIP_AES=false TRAIN_SEED=27182818 \
BATCH_SIZE=8 LEARNING_RATE=1e-4 LINEAR_WARMUP_STEPS=1000 NUM_WORKERS=4 \
SAVE_WEIGHTS_INTERVAL=50000 SAVE_CHECKPOINT_INTERVAL=50000 EMA_CHECKPOINT_INTERVAL=50000 \
bash "$PIPELINE"

EMA="$ROOT/exps/${PREFIX}_stage2_200000/${PREFIX}_stage2_200000_ema_final.pth"
[ -f "$EMA" ] || { echo "[FAIL] second-seed final EMA missing" >&2; exit 2; }
echo "[DONE] $EMA"
