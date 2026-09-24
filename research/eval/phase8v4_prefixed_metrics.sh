#!/bin/bash
# ============================================================
# Phase 8 V4 — Supplement metric pass: prefixed reference TSV
#
# Codex 2026-04-26 review P1：建議 primary metric 用 prefixed TSV
# （eval target ≡ inference prompt），unprefixed 當 secondary。
#
# 主 pipeline 已用 unprefixed TSV (musiccaps_test.tsv / phase4_test_seed42_2048.tsv)
# 跑了一輪 metric — 這個 supplement 用 prefixed TSV 再跑一輪，得到雙報表。
#
# Re-gen 不需要 — gen audio (deterministic) 已存在，只要用 phase4_eval.py
# 對 same gen audio 重算 CLAP + AES，差別只是 reference text 字串。
#
# 用法（pipeline 結束後跑）：
#     bash ~/research/meanaudio_eval/phase8v4_prefixed_metrics.sh
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

EXP_S2="phase8_v4_stage2_200000"
EVAL_OUT_MC="$WORK_DIR/eval_output/${EXP_S2}_no_q_musiccaps"
EVAL_OUT_JM="$WORK_DIR/eval_output/${EXP_S2}_no_q_jamendo_seed42_2048"

# ── Sanity check: gen audio dirs 必須已存在 ────────────────────
if [ ! -d "$EVAL_OUT_MC/audio" ]; then
    echo "❌ Missing MusicCaps gen audio dir: $EVAL_OUT_MC/audio"
    echo "   主 pipeline (tmux phase8_v4) 還沒跑完 eval section？"
    exit 1
fi
if [ ! -d "$EVAL_OUT_JM/audio" ]; then
    echo "❌ Missing Jamendo gen audio dir: $EVAL_OUT_JM/audio"
    exit 1
fi

cd "$WORK_DIR"
source ~/venvs/dac/bin/activate

echo "======================================================"
echo "  P8 V4 supplement: prefixed-reference metrics"
echo "======================================================"

# ── MusicCaps prefixed metric ─────────────────────────────────
echo "[Supplement / MusicCaps] metric on $EVAL_OUT_MC/audio with prefixed TSV"
python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_MC/audio" \
    --tsv "$DATA_DIR/phase8_v4_musiccaps_test.tsv" \
    --exp_name "${EXP_S2}_no_q_musiccaps_prefixed_ref" \
    --num_samples 5521 \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_no_q_musiccaps_prefixed_ref.log"

# ── Jamendo prefixed metric ───────────────────────────────────
echo "[Supplement / Jamendo seed42_2048] metric on $EVAL_OUT_JM/audio with prefixed TSV"
python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT_JM/audio" \
    --tsv "$DATA_DIR/phase8_v4_jamendo_seed42_2048.tsv" \
    --exp_name "${EXP_S2}_no_q_jamendo_seed42_2048_prefixed_ref" \
    --num_samples 2048 \
    2>&1 | tee "$LOG_DIR/${EXP_S2}_no_q_jamendo_seed42_2048_prefixed_ref.log"

echo ""
echo "======================================================"
echo "  Done. Two metric reports per benchmark："
echo "    主 pipeline (unprefixed-ref): "
echo "      eval_output/metrics/${EXP_S2}_no_q_musiccaps/metrics.txt"
echo "      eval_output/metrics/${EXP_S2}_no_q_jamendo_seed42_2048/metrics.txt"
echo "    supplement (prefixed-ref):"
echo "      eval_output/metrics/${EXP_S2}_no_q_musiccaps_prefixed_ref/metrics.txt"
echo "      eval_output/metrics/${EXP_S2}_no_q_jamendo_seed42_2048_prefixed_ref/metrics.txt"
echo "======================================================"
