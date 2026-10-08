#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/ttm_external_baselines_113_20261008_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/ttm_external_baselines_113_20261008_contract.json
export PYTHONUNBUFFERED=1; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2; export MKL_NUM_THREADS=2
cd /home/kojiek/MeanAudio || exit 1
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/MeanAudio/scripts/experiment_harness/ttm_external_baselines_113_guest.py
