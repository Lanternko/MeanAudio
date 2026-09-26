#!/bin/bash
# 081 quality-label prefix arm, training seed 27182818 (prereg docs/experiments/quality_label_prefix_081_20260926.md).
set -eo pipefail
export QL_SEED=27182818
bash "$HOME/MeanAudio/scripts/training_pipelines/quality_label_081_action.sh"
# last seed: pooled 3-seed readout (E1-E5); writes docs/experiments/results/quality_label_prefix_081_summary.json
"$HOME/venvs/dac/bin/python" "$HOME/MeanAudio/scripts/analysis/quality_label_081_analysis.py" || echo "[WARN] 081 analysis failed; rerun it by hand"
