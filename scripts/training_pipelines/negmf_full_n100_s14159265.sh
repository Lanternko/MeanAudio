#!/bin/bash
# NegMF n100 at the full recipe, training seed 14159265 (prereg docs/experiments/negmf_full_120_20261012.md).
set -eo pipefail
export NEGMF_SEED=14159265
bash "$HOME/MeanAudio/scripts/training_pipelines/negmf_full_action.sh"
