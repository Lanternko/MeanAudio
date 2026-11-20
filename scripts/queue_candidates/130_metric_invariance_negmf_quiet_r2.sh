#!/bin/bash
# GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/metric_invariance_128c_20261017_quiet_r2_contract.json
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT=/home/kojiek/MeanAudio/docs/experiments/metric_invariance_128c_20261017_quiet_r2_contract.json
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/home/kojiek/eval_output_nvme/metric_invariance_128_20261017_quiet_r2_tmp/c
export PYTHONPATH=/mnt/seagate/kojiek_data/metric_invariance_128_20261017/python_dependencies
export PYTHONDONTWRITEBYTECODE=1
cd /home/kojiek/MeanAudio || exit 1
exec /home/kojiek/venvs/dac/bin/python /home/kojiek/MeanAudio/scripts/experiment_harness/metric_invariance_128_guest.py
