#!/bin/bash
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env
CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_fair013_k3_full_cfg0_contract.json
python scripts/experiment_harness/preflight_post_k5_candidate.py --candidate fair013_k3_full --require-launchable
TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/phase8_caption2p0_fair013_k3_balanced_train.tsv
# fair013 K=3 uses the slot0 caption for every row, which is index 0 of the true_random
# stack: reuse it instead of re-encoding an identical ~76 GiB copy.
OVERLAY=/home/kojiek/text_overlays/true_random
BASE=/home/kojiek/MeanAudio/exps/phase8_qwen_caption2p0_fair013_k3_balanced_quarter_stage1_100000/phase8_qwen_caption2p0_fair013_k3_balanced_quarter_stage1_100000_ckpt_last.pth
post_k5_train phase8_qwen_caption2p0_fair013_k3_balanced_full "$TSV" "$OVERLAY" "$BASE" 400000 200000 true false fixed:0
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_q9 "$EVAL" phase8_qwen_caption2p0_fair013_k3_balanced_full_musiccaps_mf25_cfg0_q9 "$POST_K5_EMA" --quality_level 9
