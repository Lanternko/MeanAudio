#!/bin/bash
# ============================================================
# P8 V4 NoQ q-sweep control on MusicCaps (q=5..9, dual-ref)
# eval_p8v4_noq_qsweep_control_musiccaps.sh
#
# 用途：Wei-Jaw Fig.2 + Codex Round 3 P1 後續：用 bug-free
#       P8 V4 NoQ ckpt 跑乾淨 control，q=5..9 全部都是 random
#       (P8 V4 訓練時 null token 訓在 q[10]，已 diff 驗證)。
#
# 對照 P8 NoQ qsweep（Apr 2，bug 在 → q[9] 是 trained null）
# vs 本實驗（Apr 26，bug 已修 → q[5..9] 全 random）。
#
# Caveat：P8 V4 訓練帶 [consistency=0.90] prefix，CLAP 數量級
#       在 0.06-0.07，不是 P8 的 0.18。**不能跟 P7 V1 同圖比**，
#       只能作為 P8 V4 NoQ 內部「trained null vs random q」對照
#       (paper supplementary)。
#
# Existing baseline (--no_q → trained q[10])：
#   MusicCaps prefixed_ref CLAP 0.0665 / natural_ref 0.0571
#
# 預估：5 q × (~10 min gen + ~3 min dual-ref metric) ≈ 65 min
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP="phase8_v4_stage2_200000"
EMA="$WORK_DIR/exps/$EXP/${EXP}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
GEN_TSV="$DATA_DIR/phase8_v4_musiccaps_test.tsv"      # prefixed gen TSV (s=0.90)
NATURAL_TSV="$DATA_DIR/musiccaps_test.tsv"            # natural ref TSV (no prefix)

if [ ! -f "$EMA" ]; then
    echo "❌ EMA ckpt 不存在：$EMA"
    exit 1
fi

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

# Codex P2: marker 加 ckpt identity
ckpt_mtime=$(stat -c %Y "$EMA")
ckpt_sha=$(sha256sum "$EMA" | awk '{print $1}' | head -c 16)

echo "======================================================"
echo "  P8 V4 NoQ q-sweep control on MusicCaps (q=5..9, dual-ref)"
echo "  EMA   : $EMA (mtime $ckpt_mtime, sha16 $ckpt_sha)"
echo "  GEN TSV (prefixed): $GEN_TSV"
echo "  ⚠️  q_embed[5..9] all untrained random (bug-free P8 V4)"
echo "  ⚠️  trained null is q[10]; baseline --no_q gives the trained null"
echo "======================================================"

# dual_ref helper
dual_ref_metrics() {
    local audio_dir="$1"; local base="$2"; local prefixed_tsv="$3"
    local natural_tsv="$4"; local n="$5"; local log="$6"
    echo "[Metric / prefixed_ref] $base" | tee -a "$log"
    python "$EVAL_SCRIPT" --gen_dir "$audio_dir" --tsv "$prefixed_tsv" \
        --exp_name "${base}_prefixed_ref" --num_samples "$n" 2>&1 | tee -a "$log"
    echo "[Metric / natural_ref] $base" | tee -a "$log"
    python "$EVAL_SCRIPT" --gen_dir "$audio_dir" --tsv "$natural_tsv" \
        --exp_name "${base}_natural_ref" --num_samples "$n" 2>&1 | tee -a "$log"
}

for Q in 5 6 7 8 9; do
    EVAL_OUT="$WORK_DIR/eval_output/${EXP}_q${Q}_musiccaps_qsweep_control"
    LOG="$LOG_DIR/${EXP}_q${Q}_musiccaps_qsweep_control_eval.log"
    BASE="${EXP}_q${Q}_musiccaps_qsweep_control"

    PREFIXED_METRIC="$WORK_DIR/eval_output/metrics/${BASE}_prefixed_ref/metrics.txt"
    NATURAL_METRIC="$WORK_DIR/eval_output/metrics/${BASE}_natural_ref/metrics.txt"
    MARKER="$WORK_DIR/eval_output/metrics/${BASE}_prefixed_ref/.run_manifest"

    if [ -f "$PREFIXED_METRIC" ] && [ -f "$NATURAL_METRIC" ] && [ -f "$MARKER" ]; then
        EXPECTED="$(printf '%s\n%s\n%s\n%s\n%s' "$EMA" "$GEN_TSV" 5521 "$ckpt_mtime" "$ckpt_sha")"
        ACTUAL="$(cat $MARKER)"
        if [ "$EXPECTED" = "$ACTUAL" ]; then
            echo "[q=$Q] ✅ dual-ref metrics 已存在且 marker 一致，skip"
            continue
        fi
    fi

    if [ -d "$EVAL_OUT/audio" ] && [ "$(ls -1 $EVAL_OUT/audio/*.flac 2>/dev/null | wc -l)" -eq 5521 ]; then
        echo "[q=$Q] audio 已生成，跳過 gen"
    else
        echo "[q=${Q}] gen → $EVAL_OUT"
        python eval.py \
            --variant "meanaudio_s" \
            --model_path "$EMA" \
            --output "$EVAL_OUT/audio" \
            --tsv "$GEN_TSV" \
            --use_meanflow --num_steps 1 \
            --encoder_name t5_clap --text_c_dim 512 \
            --cfg_strength 0.5 --quality_level $Q \
            --full_precision \
            2>&1 | tee "$LOG"
    fi

    dual_ref_metrics \
        "$EVAL_OUT/audio" "$BASE" \
        "$GEN_TSV" "$NATURAL_TSV" 5521 "$LOG"

    mkdir -p "$(dirname $MARKER)"
    printf '%s\n%s\n%s\n%s\n%s' "$EMA" "$GEN_TSV" 5521 "$ckpt_mtime" "$ckpt_sha" > "$MARKER"
done

echo "======================================================"
echo "  P8 V4 NoQ q-sweep control 完成（q=5..9 × MusicCaps n=5521 dual-ref）"
echo "  Compare to existing baseline (--no_q, trained q[10]):"
echo "    prefixed_ref CLAP 0.0665 / natural_ref CLAP 0.0571"
echo "======================================================"
