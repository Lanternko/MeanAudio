#!/bin/bash
set -euo pipefail
cd /home/kojiek/MeanAudio
source /home/kojiek/MeanAudio/scripts/notify_lib.sh
notify_on_exit meva095_smoke /home/kojiek/MeanAudio/runtime/meva_20261003/smoke.log
export PYTHONUNBUFFERED=1
export HF_HOME=/home/kojiek/MeanAudio/runtime/meva_20261003/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
/home/kojiek/MeanAudio/runtime/meva_20261003/venv/bin/python /home/kojiek/MeanAudio/scripts/eval/meva_095_smoke.py
