#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/harn/negmf_084/phase8_qwen_caption2p0_slot0clean_negmfrev100_noq_quarter_s16180339_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/harn/negmf_084/phase8_qwen_caption2p0_slot0clean_negmfrev100_noq_quarter_s16180339_contract.json
export PYTHONUNBUFFERED=1
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/gpu_queue/harn_guest.py
