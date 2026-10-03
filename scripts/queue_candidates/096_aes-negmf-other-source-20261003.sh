#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/aes-negmf-other-source-20261003_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/aes-negmf-other-source-20261003_contract.json
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd /home/kojiek/MeanAudio || exit 1
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs/aes-formal-20261003/guest.py
