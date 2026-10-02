#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/meva_095_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/meva_095_contract.json
export PYTHONUNBUFFERED=1
export HF_HOME=/home/kojiek/MeanAudio/runtime/meva_20261003/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/MeanAudio/scripts/experiment_harness/meva_095_guest.py
