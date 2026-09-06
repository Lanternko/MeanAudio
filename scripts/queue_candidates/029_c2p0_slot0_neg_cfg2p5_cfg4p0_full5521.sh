#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/c2p0_slot0_neg_cfg2p5_cfg4p0_full5521_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/c2p0_slot0_neg_cfg2p5_cfg4p0_full5521_contract.json
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/MeanAudio/scripts/experiment_harness/secondary_cfg_sweep_queue_guest.py
