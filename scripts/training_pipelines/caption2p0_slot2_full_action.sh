#!/bin/bash
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env
CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_slot2_full_cfg0_contract.json
python scripts/experiment_harness/preflight_post_k5_candidate.py --candidate slot2_full --require-launchable
TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full/phase8_caption2p0_slot2_train.tsv
OVERLAY=/home/kojiek/text_overlays/slot2
python "$OVERLAY_BUILD" --train-tsv "$TSV" --cache-list "$CACHE" --audio-npz-dir "$NPZ" --output-dir "$OVERLAY" --progress-json "$OVERLAY/progress.json" --done-json "$OVERLAY/DONE.json"
BASE=/home/kojiek/MeanAudio/exps/phase8_qwen_caption2p0_slot2_noq_quarter_stage1_100000/phase8_qwen_caption2p0_slot2_noq_quarter_stage1_100000_ckpt_last.pth
post_k5_train phase8_qwen_caption2p0_slot2_noq_full "$TSV" "$OVERLAY" "$BASE" 400000 200000 false false
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_noq "$EVAL" phase8_qwen_caption2p0_slot2_noq_full_musiccaps_mf25_cfg0_noq "$POST_K5_EMA" --no_q
