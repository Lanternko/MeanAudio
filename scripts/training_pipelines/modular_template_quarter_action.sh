#!/bin/bash
# Modular-template caption arm, quarter scale (S1 100k + S2 50k), NoQ.
#
# Tests the one mechanism that survived every control in
# docs/experiments/results/phase8/fulltrack_pq_gap_mechanism_2026_08_28.md:
# the WIDTH of the training text distribution. Every row keeps its own Caption 2.0
# content (genre / tempo / instruments / mood / production for its own 10s window);
# only the surface form is narrowed, by re-rendering that content through one of
# eight fixed frames. Audio latents, row order, ids, seed and hyperparameters are
# identical to the c2p0 slot0 quarter baseline, so caption FORM is the only variable.
#
# Calibration caveat (docs/experiments/modular_template_caption_arm_2026_08_28.md):
# the rewriter overshoots the fulltrack target on 3 of 4 text statistics. This arm
# therefore tests "much narrower text", not "fulltrack-matched text".
set -euo pipefail
source /home/kojiek/MeanAudio/scripts/training_pipelines/lib_post_k5_candidate.sh
activate_gpu_env

CONTRACT=/home/kojiek/MeanAudio/docs/experiments/modular_template_quarter_contract.json
PREFIX=phase8_qwen_caption2p0_modular_template_noq_quarter
TSV=/home/kojiek/eval_tsvs_p100/phase8_caption2p0_modular_template_train.tsv
TSV_SHA=d465b93fc9122a94ea0d27a25905da80f27c75be73d9b3bc34161c31082ec920
OVERLAY=/home/kojiek/text_overlays/modular_template
OVERLAY_DONE=/home/kojiek/logs/modular_template_overlay_done.json
BASELINE_NOTE="compare against phase8_qwen_caption10s_multisent_noq_quarter (CE 6.1185 / PQ 6.5364 / CLAP 0.2029)"

# --- preflight -------------------------------------------------------------
[ -f "$TSV" ] || { echo "FAIL missing train tsv $TSV" >&2; exit 2; }
actual=$(sha256sum "$TSV" | cut -d' ' -f1)
[ "$actual" = "$TSV_SHA" ] || { echo "FAIL train tsv drift: $actual" >&2; exit 2; }
[ -f "$OVERLAY_DONE" ] || { echo "FAIL text overlay not finished: $OVERLAY_DONE missing" >&2; exit 2; }
overlay_rows=$(find "$OVERLAY" -maxdepth 1 -name '*.npz' | wc -l)
[ "$overlay_rows" -eq 251599 ] || { echo "FAIL overlay rows=$overlay_rows/251599" >&2; exit 2; }
echo "[OK] preflight: tsv sha bound, overlay complete ($overlay_rows rows)"
echo "[note] $BASELINE_NOTE"

free_bytes=$(df -B1 --output=avail /home/kojiek | tail -n 1 | tr -d ' ')
[ "$free_bytes" -ge 60000000000 ] || { echo "FAIL free space below 60 GB" >&2; exit 2; }

# --- train: quarter, NoQ, single caption per row, overlay binding enforced ---
# post_k5_train sets ++require_text_overlay=true, so extracted_audio.py verifies the
# TSV caption sha against the overlay's caption_sha256 on every __getitem__. That is
# the guard that was OFF for the historical fulltrack and c2p0 runs.
post_k5_train "$PREFIX" "$TSV" "$OVERLAY" "null" 100000 50000 false false

# --- bind the produced checkpoint into the contract -------------------------
# validate_caption2p0_cfg0_report.py compares digest(checkpoint) against
# cells[0].checkpoint_sha256, which cannot be known before training. Fill it in here
# so the canonical eval can validate; every other contract field stays preregistered.
python /home/kojiek/MeanAudio/scripts/experiment_harness/bind_contract_checkpoint_sha.py \
  --contract "$CONTRACT" --arm canonical_noq --checkpoint "$POST_K5_EMA"

# --- canonical evaluation --------------------------------------------------
CFG0_CONTRACT="$CONTRACT" CFG0_ARM=canonical_noq \
  "$EVAL" "${PREFIX}_musiccaps_mf25_cfg0_noq" "$POST_K5_EMA" --no_q

echo "TRAINING AND CANONICAL EVAL COMPLETE"
echo "next: add this arm to scripts/eval/novocal_reeval_full_arms.py ARMS for the vocal/no-vocal split"
