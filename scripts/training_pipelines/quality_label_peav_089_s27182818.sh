#!/bin/bash
# 089 PE-AV quality-label prefix arm, training seed 27182818 (prereg docs/experiments/quality_label_peav_089_20260929.md).
set -eo pipefail
export QL_SEED=27182818
bash "$HOME/MeanAudio/scripts/training_pipelines/quality_label_peav_089_action.sh"
# last seed: pooled 3-seed readout; writes docs/experiments/results/quality_label_peav_089_summary.json
"$HOME/venvs/dac/bin/python" "$HOME/MeanAudio/scripts/analysis/quality_label_peav_089_analysis.py" || echo "[WARN] 089 analysis failed; rerun it by hand"
