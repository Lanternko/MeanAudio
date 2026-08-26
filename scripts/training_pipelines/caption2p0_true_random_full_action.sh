#!/bin/bash
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env
CONTRACT=/home/kojiek/MeanAudio/docs/experiments/caption2p0_true_random_full_cfg0_contract.json
python scripts/experiment_harness/preflight_post_k5_candidate.py --candidate true_random_full --require-launchable
# Per-epoch caption rotation over the 3-caption stack (multi_cap=true reads the
# epoch-seeded sampler in extracted_audio.py:_true_random_cap_index).
TSV=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_k3_true_fake_random/k3_true_random_train.tsv
OVERLAY=/home/kojiek/text_overlays/true_random
# S1 trains 0 -> 400k in one continuous run: BASE="null" makes lib pass
# checkpoint=null on a cold start, and the existing "[ -f $s1_ckpt ]" branch still
# auto-resumes this arm's own ckpt_last after a crash or a queue pause.
# Deliberately NOT the same_arm_100k_restart_boundary pattern used by 022/023/024:
# the fake_random control cannot reproduce that boundary (its quarter S1 ckpt_last
# no longer exists), and an asymmetric restart between the two arms would sit on
# the same order of magnitude as the effect being measured.
BASE="null"
post_k5_train phase8_qwen_caption2p0_k3_true_random_noq_full "$TSV" "$OVERLAY" "$BASE" 400000 200000 false true
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_noq "$EVAL" phase8_qwen_caption2p0_k3_true_random_noq_full_musiccaps_mf25_cfg0_noq "$POST_K5_EMA" --no_q
