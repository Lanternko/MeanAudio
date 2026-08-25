#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_fair013_k3_full_cfg0_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_fair013_k3_full_cfg0_contract.json
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/gpu_queue/harn_guest.py run
