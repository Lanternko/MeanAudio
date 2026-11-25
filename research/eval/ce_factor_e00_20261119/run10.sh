#!/bin/bash
set -eo pipefail
LOG=$HOME/MeanAudio/research/eval/ce_factor_e00_20261119/run10.log
source "$HOME/MeanAudio/scripts/notify_lib.sh"
notify_on_exit "ce_factor_e00_20261119_r10" "$LOG"
cd $HOME/MeanAudio/research/eval/ce_factor_e00_20261119
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 ~/venvs/dac/bin/python run10.py "$@" >> "$LOG" 2>&1
