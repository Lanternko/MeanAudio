#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_slot0clean_full_s27182818_116_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_slot0clean_full_s27182818_116_contract.json
export PYTHONUNBUFFERED=1
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/gpu_queue/harn_guest.py
