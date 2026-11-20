#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/metric_coverage_175_contract.json
set -eo pipefail
free=$(df -B1 --output=avail /home/kojiek | tail -1 | tr -d ' ')
if [ "${free:-0}" -lt 10737418240 ]; then
  echo "[DISK] NVMe avail ${free:-0} B < 10737418240 B, refusing to start" >&2
  exit 3
fi
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/metric_coverage_175_contract.json
export PYTHONUNBUFFERED=1; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export METRIC_COVERAGE_PIN_FULL_HASH=0
cd /home/kojiek/MeanAudio || exit 1
exec /home/kojiek/venvs/dac/bin/python -I /home/kojiek/MeanAudio/scripts/experiment_harness/metric_coverage_20261027_guest.py
