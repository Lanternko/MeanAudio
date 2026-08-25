#!/bin/bash
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env
CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_true_random_quarter_cfg0_contract.json
python scripts/experiment_harness/preflight_post_k5_candidate.py --candidate true_random_quarter --require-launchable
EXTRACT=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_k3_true_fake_random/k3_true_random_extraction.tsv
TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_k3_true_fake_random/k3_true_random_train.tsv
OVERLAY=/home/kojiek/text_overlays/true_random
python scripts/preprocess/build_stacked_text_overlay_cache.py --extraction-tsv "$EXTRACT" --train-tsv "$TSV" --cache-list "$CACHE" --output-dir "$OVERLAY" --progress-json "$OVERLAY/progress.json" --done-json "$OVERLAY/DONE.json"
BASE=/home/kojiek/MeanAudio/exps/phase8_qwen_caption2p0_k3_true_random_noq_quarter_stage1_100000/phase8_qwen_caption2p0_k3_true_random_noq_quarter_stage1_100000_ckpt_last.pth
post_k5_train phase8_qwen_caption2p0_k3_true_random_noq_quarter "$TSV" "$OVERLAY" "$BASE" 100000 50000 false true
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_q0 "$EVAL" phase8_qwen_caption2p0_k3_true_random_noq_quarter_musiccaps_mf25_cfg0_noq "$POST_K5_EMA" --no_q
