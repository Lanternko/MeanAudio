#!/bin/bash
# 075 seed replicate: defectlab, training seed 16180339 (recipe identical to s14159265 except seed).
# Prereg addendum: docs/experiments/d2_defect_negsample_075_seed_replicates_20260924.md
set -eo pipefail
export D2_SEED=16180339
bash "$HOME/MeanAudio/scripts/training_pipelines/d2_defect_negsample_action.sh" defectlab
# NVMe headroom: thin this run's own S1 EMA snapshots once ema_final exists
# (reference_ema_ckpts_prune_policy; keep 30k/50k/80k/100k of both sigma series).
E=phase8_qwen_caption2p0_slot0clean_defectlab_noq_quarter_s16180339_stage1_100000
D="$HOME/exps_nvme/$E/ema_ckpts"
if [ -f "$HOME/exps_nvme/$E/${E}_ema_final.pth" ] && [ -d "$D" ]; then
  for p in "$D"/*.pt; do
    case "$(basename "$p")" in 0.30000.pt|0.50000.pt|0.80000.pt|0.100000.pt|1.30000.pt|1.50000.pt|1.80000.pt|1.100000.pt) ;; *) rm -f -- "$p" ;; esac
  done
fi
