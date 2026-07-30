#!/usr/bin/env bash
# Serial, resumable order for the S2-only Q-resolution study.
set -euo pipefail
ROOT=/home/kojiek/MeanAudio
LOG_ROOT=/home/kojiek/logs
STATE="$LOG_ROOT/phase8_qwen_s2q_from_noq_sequence"
LOCK="$STATE/sequence.lock"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
POLL_SECONDS="${POLL_SECONDS:-60}"
mkdir -p "$STATE"
exec 9>"$LOCK"
flock -n 9 || { echo '[FAIL] S2-only sequence already running' >&2; exit 3; }
wait_for_clear_gpu() {
    local active
    while true; do
        active=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits 2>&1) || {
            echo "[WAIT] cannot determine GPU ownership: $active" >&2
            sleep "$POLL_SECONDS"; continue
        }
        if [ -z "${active//[[:space:]]/}" ]; then
            return
        fi
        echo "[WAIT] GPU is owned by another process: ${active//$'\n'/; }" >&2
        sleep "$POLL_SECONDS"
    done
}
for k in 3 5 2 10; do
    prefix="phase8_qwen_s2q_from_noq_quarter_k${k}_balanced"
    report="$LOG_ROOT/${prefix}_FINAL_METRICS.json"
    if [ -f "$report" ]; then
        python - "$report" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); assert p.get('status') == 'passed' and p.get('design') == 'NoQ_S1_to_Q_S2_only'
PY
        echo "[SKIP] validated report: $report"
        continue
    fi
    wrapper_log="$LOG_ROOT/${prefix}_wrapper.log"
    wait_for_clear_gpu
    echo "[START] K${k} balanced $(date --iso-8601=seconds)" | tee -a "$wrapper_log"
    "$ROOT/scripts/run_with_experiment_report.sh" --experiment "$prefix" --report "$report" --log "$wrapper_log" \
        -- env K="$k" STRATEGY=balanced EXPERIMENT_RUN_MODE="$RUN_MODE" \
        bash "$ROOT/scripts/training_pipelines/execute_phase8_qwen_bucket_s2q_from_noq.sh" \
        2>&1 | tee -a "$wrapper_log"
done
