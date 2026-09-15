#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/instrument_conflict_cfg3_20260908_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/instrument_conflict_cfg3_20260908_contract.json
export PYTHONUNBUFFERED=1
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/MeanAudio/scripts/experiment_harness/instrument_conflict_cfg3_20260908_guest.py
