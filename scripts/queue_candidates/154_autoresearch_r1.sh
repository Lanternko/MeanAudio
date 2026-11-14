#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/autoresearch_154_r1_contract.json
set -eo pipefail
free=$(df -B1 --output=avail /home/kojiek | tail -1 | tr -d ' ')
if [ "${free:-0}" -lt 20000000000 ]; then
  echo "[DISK] NVMe avail ${free:-0} B < 20000000000 B, refusing to start" >&2
  exit 3
fi
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/autoresearch_154_r1_contract.json
export PYTHONUNBUFFERED=1; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
cd /home/kojiek/MeanAudio || exit 1
exec /home/kojiek/venvs/dac/bin/python -I /home/kojiek/MeanAudio/scripts/experiment_harness/autoresearch_clap_20261027_guest.py
