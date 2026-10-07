#!/bin/bash
# 084 NegMF Stage C arm lq100 ('Low quality recording.' in the guidance branch), training seed 16180339
# (prereg docs/experiments/negprompt_distill_meanflow_084_20260927.md + docs/experiments/negmf_lq100_20261007.md).
set -eo pipefail
export NEGMF_ARM=lq100
export NEGMF_SEED=16180339
bash "$HOME/MeanAudio/scripts/training_pipelines/negmf_084_action.sh"
