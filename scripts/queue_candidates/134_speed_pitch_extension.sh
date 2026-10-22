#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/speed_pitch_extension_20261020_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/speed_pitch_extension_20261020_contract.json
export PYTHONUNBUFFERED=1; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
cd /home/kojiek/MeanAudio || exit 1
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/MeanAudio/scripts/experiment_harness/ttm_metrics_115_guest.py
