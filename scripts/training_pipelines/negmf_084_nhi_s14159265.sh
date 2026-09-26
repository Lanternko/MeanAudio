#!/bin/bash
# 084 NegMF arm nhi, training seed 14159265 (prereg docs/experiments/negprompt_distill_meanflow_084_20260927.md).
set -eo pipefail
export NEGMF_ARM=nhi
export NEGMF_SEED=14159265
bash "$HOME/MeanAudio/scripts/training_pipelines/negmf_084_action.sh"
