#!/bin/bash
# 081 quality-label prefix arm, training seed 16180339 (prereg docs/experiments/quality_label_prefix_081_20260926.md).
set -eo pipefail
export QL_SEED=16180339
bash "$HOME/MeanAudio/scripts/training_pipelines/quality_label_081_action.sh"
