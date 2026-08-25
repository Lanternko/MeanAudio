#!/bin/bash
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env
CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_fair013_best_full_cfg0_contract.json
python scripts/experiment_harness/preflight_post_k5_candidate.py --candidate fair013_best_full --require-launchable
# bestof3 always picks one of slot0/slot1/slot3, which the true_random stack already holds:
# reuse it via the per-row cap_index column instead of re-encoding an identical ~76 GiB copy.
TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/phase8_caption2p0_fair013_bestof3_capidx_train.tsv
OVERLAY=/home/kojiek/text_overlays/true_random
BASE=/home/kojiek/MeanAudio/exps/phase8_qwen_caption2p0_fair013_bestof3_noq_quarter_stage1_100000/phase8_qwen_caption2p0_fair013_bestof3_noq_quarter_stage1_100000_ckpt_last.pth
post_k5_train phase8_qwen_caption2p0_fair013_bestof3_noq_full "$TSV" "$OVERLAY" "$BASE" 400000 200000 false false column:cap_index
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_noq "$EVAL" phase8_qwen_caption2p0_fair013_bestof3_noq_full_musiccaps_mf25_cfg0_noq "$POST_K5_EMA" --no_q
