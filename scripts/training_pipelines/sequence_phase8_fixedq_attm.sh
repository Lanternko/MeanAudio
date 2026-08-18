#!/usr/bin/env bash
# Durable sequential queue for Phase-8 Fixed-Q prior vs matched-NoQ FT.
#
# Order (exactly two training arms, never invents a third):
#   1) fixedq9  (phase8_fixedq9_prior_ft100k)
#   2) noq      (phase8_matched_noq_ft100k)
#   3) paired CLAP bootstrap + FINAL_COMPARISON
#
# Single GPU (CUDA_VISIBLE_DEVICES=0).  Fail-closed on contract drift via
# per-arm audit.  Prevents duplicate launch via flock + process probe.
# No fixed sleeps at phase boundaries: each arm blocks until completion;
# the next arm starts only after the previous audit passes.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/phase8_fixedq_attm_monitor"
SEQ_LOG="$LOG_ROOT/phase8_fixedq_attm_sequence.log"
LOCK="$STATE/sequence.lock"
FIXEDQ9_TSV="$DATA/phase8_legacy_catalog_train_fixedq9.tsv"
FIXEDQ9_MANIFEST="$DATA/phase8_legacy_catalog_train_fixedq9.manifest.json"
CATALOG_TSV="$DATA/phase8_legacy_catalog_train.tsv"
FIXED_PREFIX=phase8_fixedq9_prior_ft100k
NOQ_PREFIX=phase8_matched_noq_ft100k
MUSICCAPS="$DATA/musiccaps_test.tsv"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"

cd "$ROOT"
# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$STATE" "$LOG_ROOT"

log() {
    echo "[SEQUENCE] $*" | tee -a "$SEQ_LOG"
}

# Duplicate-launch guard: exclusive non-blocking flock on lock file.
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[FAIL] sequence already running (lock $LOCK held)" >&2
    exit 3
fi

# Also refuse if matching train/eval processes already exist outside this lock.
if pgrep -af 'phase8_(fixedq9_prior|matched_noq)_ft100k|train_pipeline_phase8_fixedq_attm_ft|sequence_phase8_fixedq_attm' \
    >/dev/null 2>&1; then
    # This sequence process itself may match; filter to other PIDs only.
    others=$(pgrep -af 'phase8_(fixedq9_prior|matched_noq)_ft100k|train_pipeline_phase8_fixedq_attm_ft' || true)
    # shellcheck disable=SC2009
    if [ -n "$others" ]; then
        echo "[FAIL] duplicate phase8 fixedq/noq process already present" >&2
        printf '%s\n' "$others"
        exit 3
    fi
fi

if [ -f "$STATE/FINAL_COMPARISON.json" ] && [ "$RUN_MODE" = fresh ]; then
    echo "[FAIL] FINAL_COMPARISON already exists; refuse fresh re-launch" >&2
    exit 2
fi

# Ensure Fixed-Q=9 TSV exists (deterministic, fail-closed).
if [ ! -f "$FIXEDQ9_TSV" ] || [ ! -f "$FIXEDQ9_MANIFEST" ]; then
    log "building fixedq9 TSV"
    python scripts/preprocess/make_phase8_fixedq9_tsv.py \
        --input "$CATALOG_TSV" \
        --output "$FIXEDQ9_TSV" \
        --manifest "$FIXEDQ9_MANIFEST" \
        --expected-rows 251599 \
        --fixed-q 9
fi

# Verify TSV contract before any training.
python - "$FIXEDQ9_TSV" "$FIXEDQ9_MANIFEST" "$CATALOG_TSV" <<'PY'
import csv
import hashlib
import json
import sys
from pathlib import Path

tsv, man_path, catalog = map(Path, sys.argv[1:])
man = json.loads(man_path.read_text())

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if man.get("output_sha256") != sha(tsv):
    raise SystemExit("[FAIL] fixedq9 TSV hash disagrees with manifest")
if man.get("input_sha256") != sha(catalog):
    raise SystemExit("[FAIL] catalog TSV changed after fixedq9 creation")
if man.get("unique_q_support") != [9] or man.get("rows") != 251599:
    raise SystemExit(f"[FAIL] fixedq9 manifest contract broken: {man}")
rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
if len(rows) != 251599:
    raise SystemExit(f"[FAIL] fixedq9 rows={len(rows)}")
if sorted({int(r["q_level"]) for r in rows}) != [9]:
    raise SystemExit("[FAIL] fixedq9 unique Q support not exactly [9]")
print("[OK] fixedq9 TSV contract")
PY

run_arm() {
    local arm="$1"
    local prefix="$2"
    local tsv="$3"
    local audit="$STATE/${prefix}_FINAL_AUDIT.json"
    local exp_id="${prefix}_stage2_ft100000"
    local ckpt="$ROOT/exps/$exp_id/${exp_id}_ckpt_last.pth"
    local arm_mode=fresh

    if [ -f "$audit" ]; then
        python - "$audit" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("status") != "passed":
    raise SystemExit(f"[FAIL] existing audit not passed: {sys.argv[1]}")
print(f"[OK] arm already complete: {sys.argv[1]}")
PY
        log "skip completed arm arm=$arm prefix=$prefix"
        return 0
    fi

    if [ -f "$ckpt" ]; then
        if [ "$RUN_MODE" != resume ]; then
            echo "[FAIL] partial arm requires explicit EXPERIMENT_RUN_MODE=resume: $ckpt" >&2
            exit 2
        fi
        arm_mode=resume
    fi

    log "start arm=$arm prefix=$prefix mode=$arm_mode $(date --iso-8601=seconds)"
    ARM="$arm" EXP_PREFIX="$prefix" TRAIN_TSV="$tsv" \
        EXPERIMENT_RUN_MODE="$arm_mode" \
        bash scripts/training_pipelines/train_pipeline_phase8_fixedq_attm_ft.sh

    if [ ! -f "$audit" ]; then
        echo "[FAIL] missing final audit after arm $arm: $audit" >&2
        exit 2
    fi
    python - "$audit" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("status") != "passed":
    raise SystemExit(f"[FAIL] arm audit failed: {payload.get('issues')}")
print("[OK] arm audit passed")
PY
    log "complete arm=$arm $(date --iso-8601=seconds)"
}

{
    log "queue starts $(date --iso-8601=seconds) run_mode=$RUN_MODE"

    # Arm 1: Fixed-Q=9 prior FT (first by design).
    run_arm fixedq9 "$FIXED_PREFIX" "$FIXEDQ9_TSV"

    # Arm 2: matched NoQ FT.  No sleep; starts only after arm1 audit.
    run_arm noq "$NOQ_PREFIX" "$CATALOG_TSV"

    FIXED_AUDIO="$ROOT/eval_output/${FIXED_PREFIX}_stage2_ft100000_musiccaps_q9/audio"
    NOQ_AUDIO="$ROOT/eval_output/${NOQ_PREFIX}_stage2_ft100000_musiccaps_noq/audio"
    BOOTSTRAP_JSON="$STATE/PAIRED_CLAP_BOOTSTRAP.json"
    SCORES_CSV="$STATE/PAIRED_CLAP_SCORES.csv"

    if [ ! -f "$BOOTSTRAP_JSON" ]; then
        log "paired CLAP bootstrap starts $(date --iso-8601=seconds)"
        python scripts/eval/paired_clap_bootstrap_phase8_fixedq_attm.py \
            --tsv "$MUSICCAPS" \
            --fixedq-dir "$FIXED_AUDIO" \
            --noq-dir "$NOQ_AUDIO" \
            --output "$BOOTSTRAP_JSON" \
            --scores-csv "$SCORES_CSV"
    fi

    python - "$STATE" "$ROOT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

state = Path(sys.argv[1])
root = Path(sys.argv[2])
metrics_root = root / "eval_output" / "metrics"


def read_metrics(prefix: str, label: str) -> dict:
    path = metrics_root / f"{prefix}_stage2_ft100000_musiccaps_{label}" / "metrics.txt"
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}:
                out[key.strip()] = float(value)
    required = {"clap_score", "aes_CE", "aes_CU", "aes_PC", "aes_PQ"}
    if set(out) != required:
        raise SystemExit(f"[FAIL] incomplete metrics at {path}: {out}")
    return out


fixed = read_metrics("phase8_fixedq9_prior_ft100k", "q9")
noq = read_metrics("phase8_matched_noq_ft100k", "noq")
paired = json.loads((state / "PAIRED_CLAP_BOOTSTRAP.json").read_text())

fixed_clap = fixed["clap_score"]
noq_clap = noq["clap_score"]
payload = {
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "experiment": "phase8_fixedq_attm_chain",
    "baseline_source_noq_clap": 0.1888,
    "restoration_target_clap": 0.1900,
    "primary_checkpoint_iteration": 700000,
    "fixedq9_q9": fixed,
    "matched_noq": noq,
    "fixedq_minus_noq_aggregate": fixed_clap - noq_clap,
    "fixedq_minus_source_noq": fixed_clap - 0.1888,
    "paired_clap": paired,
    "fixedq_benefit_supported": bool(paired.get("fixedq_benefit_supported")),
    "restored_clap_0p19": fixed_clap >= 0.1900,
    "interpretation": {
        "primary_checkpoint": 700000,
        "no_cherrypick": True,
        "fixedq_benefit_requires_ci95_lb_gt_0": True,
        "restoration_target": "fixedq9 MusicCaps CLAP >= 0.1900",
    },
}
payload["primary_objective_met"] = payload["fixedq_benefit_supported"]
payload["fallback_restoration_met"] = payload["restored_clap_0p19"]
payload["program_goal_met"] = bool(
    payload["fixedq_benefit_supported"] or payload["restored_clap_0p19"]
)
tmp = state / "FINAL_COMPARISON.json.tmp"
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(state / "FINAL_COMPARISON.json")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

    log "complete $(date --iso-8601=seconds)"
} 2>&1 | tee -a "$SEQ_LOG"
