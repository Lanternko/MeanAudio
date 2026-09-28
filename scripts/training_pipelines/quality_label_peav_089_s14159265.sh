#!/bin/bash
# 089 PE-AV quality-label prefix arm, training seed 14159265 (prereg docs/experiments/quality_label_peav_089_20260929.md).
set -eo pipefail
export QL_SEED=14159265
bash "$HOME/MeanAudio/scripts/training_pipelines/quality_label_peav_089_action.sh"
