#!/bin/bash
# NegMF n100 at the full recipe, training seed 27182818 (prereg docs/experiments/negmf_full_120_20261012.md).
set -eo pipefail
export NEGMF_SEED=27182818
bash "$HOME/MeanAudio/scripts/training_pipelines/negmf_full_action.sh"
