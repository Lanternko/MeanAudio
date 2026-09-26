#!/bin/bash
# 081 quality-label prefix arm, training seed 14159265 (prereg docs/experiments/quality_label_prefix_081_20260926.md).
set -eo pipefail
export QL_SEED=14159265
bash "$HOME/MeanAudio/scripts/training_pipelines/quality_label_081_action.sh"
