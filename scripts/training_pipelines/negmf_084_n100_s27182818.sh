#!/bin/bash
# 084 NegMF arm n100, Stage B training seed 27182818 (prereg docs/experiments/negprompt_distill_meanflow_084_20260927.md).
set -eo pipefail
export NEGMF_ARM=n100
export NEGMF_SEED=27182818
bash "$HOME/MeanAudio/scripts/training_pipelines/negmf_084_action.sh"
