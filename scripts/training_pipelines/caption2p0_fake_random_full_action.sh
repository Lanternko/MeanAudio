#!/bin/bash
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env
CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_fake_random_full_cfg0_contract.json
python scripts/experiment_harness/preflight_post_k5_candidate.py --candidate fake_random_full --require-launchable
# Fixed-random control: one caption per clip for the whole run, chosen by the same
# uniform SHA-256 map (slot0/slot1/slot3 -> cap_index 0/1/2, 83716/83494/84389 rows).
# The dedicated fake_random overlay was deleted on 2026-08-26 to reclaim 152 GB, so
# reuse the identical true_random stack through the per-row cap_index column.
TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_k3_true_fake_random/k3_fake_random_fixed_capidx_train.tsv
OVERLAY=/home/kojiek/text_overlays/true_random
# Same cold-start contract as 025 so the pair differs only in caption rotation.
BASE="null"
post_k5_train phase8_qwen_caption2p0_k3_fake_random_noq_full "$TSV" "$OVERLAY" "$BASE" 400000 200000 false false column:cap_index
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_noq "$EVAL" phase8_qwen_caption2p0_k3_fake_random_noq_full_musiccaps_mf25_cfg0_noq "$POST_K5_EMA" --no_q
