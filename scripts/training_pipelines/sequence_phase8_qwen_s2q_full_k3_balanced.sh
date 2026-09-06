#!/usr/bin/env bash
# Queue the best repaired quarter-scale S2-only arm at full iteration count.
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
LOG_ROOT=/home/kojiek/logs
PREFIX=phase8_qwen_s2q_from_noq_full_k3_balanced
STATE="$LOG_ROOT/${PREFIX}_sequence"
LOCK="$STATE/sequence.lock"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
POLL_SECONDS="${POLL_SECONDS:-60}"
mkdir -p "$STATE"
exec 9>"$LOCK"
flock -n 9 || { echo "[FAIL] $PREFIX sequence is already running" >&2; exit 3; }

case "$RUN_MODE" in fresh|resume) ;; *) echo "[FAIL] invalid EXPERIMENT_RUN_MODE=$RUN_MODE" >&2; exit 2;; esac
wait_for_clear_gpu() {
    local active
    while true; do
        active=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1) || {
            echo "[WAIT] cannot determine GPU ownership: $active" >&2; sleep "$POLL_SECONDS"; continue;
        }
        if [ -z "${active//[[:space:]]/}" ]; then return; fi
        echo "[WAIT] GPU is in use: ${active//$'\n'/; }" >&2
        sleep "$POLL_SECONDS"
    done
}

report="$LOG_ROOT/${PREFIX}_FINAL_METRICS.json"
if [ -f "$report" ]; then
    python - "$report" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p.get('status') == 'passed' and p.get('scale') == 'full' and p.get('k') == 3 and p.get('strategy') == 'balanced'
PY
    echo "[SKIP] validated report: $report"
    exit 0
fi

wait_for_clear_gpu
exec "$ROOT/scripts/run_with_experiment_report.sh" --gpu-released \
    --experiment "$PREFIX" --report "$report" \
    --log "$LOG_ROOT/${PREFIX}_wrapper.log" \
    --summary "full-iteration K3 balanced S2-only Q; reuses official-Qwen full NoQ S1 400k" \
    -- env K=3 STRATEGY=balanced SCALE=full EXPERIMENT_RUN_MODE="$RUN_MODE" \
    bash "$ROOT/scripts/training_pipelines/execute_phase8_qwen_bucket_s2q_from_noq.sh"
