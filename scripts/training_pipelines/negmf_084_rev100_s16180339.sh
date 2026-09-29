#!/bin/bash
# 084 NegMF Stage C arm rev100 (reversed text in the guidance branch), training seed 16180339
# (prereg docs/experiments/negprompt_distill_meanflow_084_20260927.md §6.1).
set -eo pipefail
export NEGMF_ARM=rev100
export NEGMF_SEED=16180339
bash "$HOME/MeanAudio/scripts/training_pipelines/negmf_084_action.sh"
