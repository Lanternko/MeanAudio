#!/usr/bin/env bash
# Serial full-scale K ablation. K3 balanced is already complete; this queue
# runs the remaining balanced K arms from the same official NoQ S1 checkpoint.
set -euo pipefail

ROOT=/home/kojiek/MeanAudio
LOG_ROOT=/home/kojiek/logs
CHAIN=phase8_qwen_s2q_full_k_ablation
STATE="$LOG_ROOT/${CHAIN}_sequence"
LOCK="$STATE/sequence.lock"
RUN_MODE="${EXPERIMENT_RUN_MODE:-fresh}"
POLL_SECONDS="${POLL_SECONDS:-60}"
mkdir -p "$STATE"
exec 9>"$LOCK"
flock -n 9 || { echo "[FAIL] $CHAIN sequence is already running" >&2; exit 3; }

case "$RUN_MODE" in fresh|resume) ;; *) echo "[FAIL] invalid EXPERIMENT_RUN_MODE=$RUN_MODE" >&2; exit 2;; esac

wait_for_clear_gpu() {
    local active
    while true; do
        active=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits 2>&1) || {
            echo "[WAIT] cannot determine GPU ownership: $active" >&2
            sleep "$POLL_SECONDS"
            continue
        }
        if [ -z "${active//[[:space:]]/}" ]; then return; fi
        echo "[WAIT] GPU is in use: ${active//$'\n'/; }" >&2
        sleep "$POLL_SECONDS"
    done
}

for k in 2 5 10; do
    prefix="phase8_qwen_s2q_from_noq_full_k${k}_balanced"
    report="$LOG_ROOT/${prefix}_FINAL_METRICS.json"
    wrapper_log="$LOG_ROOT/${prefix}_wrapper.log"
    if [ -f "$report" ]; then
        python - "$report" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p.get('status') == 'passed'
assert p.get('scale') == 'full'
assert p.get('strategy') == 'balanced'
PY
        echo "[SKIP] validated report: $report"
        continue
    fi

    wait_for_clear_gpu
    echo "[START] K${k} balanced $(date --iso-8601=seconds)" | tee -a "$wrapper_log"
    # Per-arm notifications report failure/success. The final chain-level
    # notification below is the only one that claims the GPU is released.
    "$ROOT/scripts/run_with_experiment_report.sh" \
        --experiment "$prefix" --report "$report" --log "$wrapper_log" \
        --summary "full-iteration K${k} balanced S2-only Q; same NoQ S1@400k source" \
        -- env K="$k" STRATEGY=balanced SCALE=full EXPERIMENT_RUN_MODE="$RUN_MODE" \
        bash "$ROOT/scripts/training_pipelines/execute_phase8_qwen_bucket_s2q_from_noq.sh" \
        2>&1 | tee -a "$wrapper_log"
done

echo "[COMPLETE] $CHAIN"
"$ROOT/scripts/notify_experiment_webhook.py" \
    --status success --experiment "$CHAIN" --exit-code 0 \
    --summary "K2/K5/K10 balanced full ablation chain completed; GPU is available" \
    --gpu-released || echo "[WARN] final chain Discord notification failed" >&2
