#!/bin/bash
# 089 PE-AV quality-label prefix arm, training seed 16180339 (prereg docs/experiments/quality_label_peav_089_20260929.md).
set -eo pipefail
export QL_SEED=16180339
bash "$HOME/MeanAudio/scripts/training_pipelines/quality_label_peav_089_action.sh"
