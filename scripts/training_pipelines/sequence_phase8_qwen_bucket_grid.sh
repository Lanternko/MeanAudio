#!/usr/bin/env bash
# K=2/3/5/10 x balanced/fixed pilot, two-stage gate, then selected quarter runs.

set -euo pipefail

ROOT=/home/kojiek/MeanAudio
DATA=/mnt/HDD/kojiek/phase4_jamendo_data
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/phase8_qwen_bucket_grid"
LOCK="$STATE/sequence.lock"
LOG="$LOG_ROOT/phase8_qwen_bucket_grid_sequence.log"
GRID="$DATA/phase8_qwen_meansim_bucket_grid.manifest.json"
PILOT="$ROOT/smoke_data/phase8_qwen_bucket_grid_musiccaps_seed14159265_n512.tsv"
HOLDOUT="$ROOT/smoke_data/phase8_qwen_bucket_grid_musiccaps_holdout_n5009.tsv"
GATE="$LOG_ROOT/phase8_qwen_bucket_pilot_GATE.json"
FINAL="$LOG_ROOT/phase8_qwen_bucket_grid_FINAL_METRICS.json"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SEQUENCE_PREFLIGHT_ONLY="${SEQUENCE_PREFLIGHT_ONLY:-false}"
case "$SEQUENCE_PREFLIGHT_ONLY" in true|false) ;; *)
    echo "[FAIL] SEQUENCE_PREFLIGHT_ONLY must be true or false" >&2; exit 2;;
esac

cd "$ROOT"
# shellcheck source=/dev/null
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$STATE" "$LOG_ROOT"
exec 9>"$LOCK"
flock -n 9 || { echo "[FAIL] bucket-grid sequence already running" >&2; exit 3; }

log() {
    echo "[QWEN-BUCKET] $(date --iso-8601=seconds) $*" | tee -a "$LOG"
}

log "rebuild/verify all derived aligned bucket TSVs"
python scripts/preprocess/make_phase8_qwen_bucket_grid.py \
    --qwen-tsv "$DATA/phase8_qwen_official_matched.tsv" \
    --qwen-combined-manifest "$DATA/phase8_qwen_meansim_fullq_halfq.manifest.json" \
    --qwen-npz-manifest "$DATA/phase8_qwen_official_matched_npz_manifest.json" \
    --qwen-cache-audit "$DATA/phase8_qwen_official_matched_qwen_cache_audit.json" \
    --qwen-cache-list "$DATA/phase8_qwen_official_matched_npz_cache_train.txt" \
    --official-json /home/kojiek/reference-repos/ICME26-ATTM-GC-FluxAudio/data/captions/jamendo_qwen.json \
    --aligned-tsv "$DATA/phase8_legacy_catalog_train_meansim_aligned.tsv" \
    --aligned-manifest "$DATA/phase8_legacy_catalog_train_meansim_aligned.manifest.json" \
    --source-jsonl /home/kojiek/research/music_cleaning/results_20260119_043407.jsonl \
    --existing-k2-balanced "$DATA/phase8_qwen_meansim_halfq.tsv" \
    --existing-k10-fixed "$DATA/phase8_qwen_meansim_fullq.tsv" \
    --output-dir "$DATA" --manifest "$GRID" \
    --musiccaps "$DATA/musiccaps_test.tsv" --pilot-prompts "$PILOT"

for k in 2 3 5 10; do
    for strategy in balanced fixed; do
        K="$k" STRATEGY="$strategy" SCALE=pilot PREFLIGHT_ONLY=true \
            bash scripts/training_pipelines/train_phase8_qwen_bucket_arm.sh
    done
done
if [ "$SEQUENCE_PREFLIGHT_ONLY" = true ]; then
    log "preflight complete; no GPU process started"
    exit 0
fi

gpu_blockers() {
    python - <<'PY'
import subprocess
from pathlib import Path
result = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
     "--format=csv,noheader,nounits"],
    check=True, capture_output=True, text=True,
)
blockers = []
for line in result.stdout.splitlines():
    if not line.strip():
        continue
    pid_raw, memory_raw = [part.strip() for part in line.split(",", 1)]
    pid = int(pid_raw)
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        command = command.replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        continue
    if "Irodori-TTS" not in command:
        blockers.append((pid, int(memory_raw), command))
for pid, memory, command in blockers:
    print(f"{pid}\t{memory} MiB\t{command}")
raise SystemExit(0 if not blockers else 1)
PY
}

while ! blockers=$(gpu_blockers); do
    log "waiting for GPU; blockers: ${blockers//$'\n'/; }"
    sleep "$POLL_SECONDS"
done

run_arm_experiment() {
    local scale="$1" k="$2" strategy="$3" reuse="${4:-none}"
    local name="phase8_qwen_bucket_${scale}_k${k}_${strategy}"
    local report="$LOG_ROOT/${name}_FINAL_METRICS.json"
    log "arm starts: $name reuse=$reuse"
    scripts/run_with_experiment_report.sh \
        --experiment "$name" --report "$report" \
        --log "$LOG_ROOT/${name}_wrapper.log" \
        -- env K="$k" STRATEGY="$strategy" SCALE="$scale" REUSE="$reuse" \
            EXPERIMENT_RUN_MODE="$RUN_MODE" \
            bash scripts/training_pipelines/execute_phase8_qwen_bucket_arm_eval.sh \
        2>&1 | tee "$LOG_ROOT/${name}_wrapper.log"
}

for k in 2 3 5 10; do
    for strategy in balanced fixed; do
        report="$LOG_ROOT/phase8_qwen_bucket_pilot_k${k}_${strategy}_FINAL_METRICS.json"
        if [ -f "$report" ]; then
            log "skip completed pilot report=$report"
        else
            run_arm_experiment pilot "$k" "$strategy"
        fi
    done
done

python scripts/analysis/select_phase8_qwen_bucket_promotions.py \
    --reports-dir "$LOG_ROOT" --output "$GATE" \
    --musiccaps "$DATA/musiccaps_test.tsv" --pilot-prompts "$PILOT" \
    --grid-manifest "$GRID" \
    --holdout-prompts "$HOLDOUT" --margin 0.005 --cap 4

while read -r k strategy; do
    reuse=none
    if [ "$k:$strategy" = "2:balanced" ]; then
        reuse=k2_balanced_historical
    elif [ "$k:$strategy" = "10:fixed" ]; then
        reuse=k10_fixed_historical
    fi
    report="$LOG_ROOT/phase8_qwen_bucket_quarter_k${k}_${strategy}_FINAL_METRICS.json"
    if [ -f "$report" ]; then
        log "skip completed quarter report=$report"
    else
        run_arm_experiment quarter "$k" "$strategy" "$reuse"
    fi
done < <(
    python - "$GATE" <<'PY'
import json, sys
for arm in json.load(open(sys.argv[1]))["selected"]:
    print(arm["k"], arm["strategy"])
PY
)

python - "$GATE" "$FINAL" "$LOG_ROOT" "$GRID" "$DATA/musiccaps_test.tsv" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
gate_path, out, logs, grid_path, musiccaps = map(Path, sys.argv[1:])
gate = json.loads(gate_path.read_text())
grid = json.loads(grid_path.read_text())
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
quarter = []
for arm in gate["selected"]:
    path = logs / (
        f"phase8_qwen_bucket_quarter_k{arm['k']}_{arm['strategy']}"
        "_FINAL_METRICS.json"
    )
    if not path.is_file():
        raise SystemExit(f"[FAIL] missing selected quarter report: {path}")
    payload = json.loads(path.read_text())
    audit_path = Path(payload.get("training_audit", ""))
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    contract_path = Path(audit.get("contract", ""))
    contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
    expected_name = f"phase8_qwen_bucket_quarter_k{arm['k']}_{arm['strategy']}"
    if (
        payload.get("status") != "passed"
        or payload.get("experiment") != expected_name
        or payload.get("scale") != "quarter"
        or payload.get("k") != arm["k"]
        or payload.get("strategy") != arm["strategy"]
        or payload.get("prompts", {}).get("rows") != 5521
        or payload.get("prompts", {}).get("path") != str(musiccaps)
        or payload.get("prompts", {}).get("sha256") != sha(musiccaps)
        or payload.get("stage1", {}).get("protocol")
        != "MusicCaps 5521; FluxAudio FM25 CFG4.5"
        or payload.get("global", {}).get("protocol")
        != "MusicCaps 5521; MeanFlow1 CFG0.5"
        or payload.get("stage1", {}).get("holdout5009_high_q9") is None
        or payload.get("global", {}).get("holdout5009_high_q9") is None
        or audit.get("status") != "passed"
        or not contract_path.is_file()
        or contract.get("train_tsv_sha256")
        != grid["outputs"][f"k{arm['k']}_{arm['strategy']}"]["sha256"]
        or set(payload.get("models", {})) != {"stage1", "global"}
        or any(
            not Path(model.get("path", "")).is_file()
            or model.get("sha256") != sha(model["path"])
            for model in payload.get("models", {}).values()
        )
    ):
        raise SystemExit(f"[FAIL] incomplete quarter report: {path}")
    quarter.append(payload)
result = {
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "experiment": "phase8_qwen_bucket_grid",
    "pilot_gate": str(gate_path),
    "promotion_rule": gate["gate"],
    "selected": gate["selected"],
    "quarter": quarter,
    "reporting": {
        "stage1_and_global_separate": True,
        "full_musiccaps_rows": 5521,
        "disjoint_holdout_rows": 5009,
        "pilot_selection_rows": 512,
        "quarter_primary_endpoint": "high q9",
        "supported_low_endpoint_is_diagnostic_only": True,
    },
}
tmp = out.with_suffix(out.suffix + ".tmp")
tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
print(json.dumps(result, indent=2, sort_keys=True))
PY
log "complete final=$FINAL"
