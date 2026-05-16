#!/bin/bash
# ============================================================
# P8 NoQ q=10 sanity check
# eval_p8_q10_sanity.sh
#
# 目的：驗證 --quality_level 10 ≡ --no_q（eval.py 內部 --no_q
#       就是 q_levels.append(10)）。如果兩者 CLAP 完全相同，
#       confirms python pipeline behaves consistently。
#
# 期待：與 p8baseline rerun (phase8_stage2_200000_no_q_musiccaps_
#       qsweep_baseline) 數字相同（同 ckpt, 同 TSV, 同 cfg, 同 q
#       input, 同 seed）。
#
# Codex Round 4 fixes 全套：
#   - audio identity manifest
#   - full marker identity
#   - ckpt sha verification
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP="phase8_stage2_200000"
EMA="$WORK_DIR/exps/$EXP/${EXP}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
GEN_TSV="$DATA_DIR/musiccaps_test.tsv"
NUM_SAMPLES=5521
CFG=0.5
SCRIPT_VERSION="codex_round4_v1"

if [ ! -f "$EMA" ]; then echo "❌ EMA 不存在：$EMA"; exit 1; fi

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

ckpt_mtime=$(stat -c %Y "$EMA")
ckpt_sha=$(sha256sum "$EMA" | awk '{print $1}' | head -c 16)
gen_tsv_sha=$(sha256sum "$GEN_TSV" | awk '{print $1}' | head -c 16)

EVAL_OUT="$WORK_DIR/eval_output/${EXP}_q10_musiccaps_qsweep_control"
LOG="$LOG_DIR/${EXP}_q10_musiccaps_qsweep_control_eval.log"
BASE="${EXP}_q10_musiccaps_qsweep_control"
METRICS_FILE="$WORK_DIR/eval_output/metrics/${BASE}/metrics.txt"
MARKER="$WORK_DIR/eval_output/metrics/${BASE}/.run_manifest"

expected_marker=$(printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s' \
    "$EMA" "$GEN_TSV" "$NUM_SAMPLES" "$CFG" "q=10" \
    "$ckpt_mtime" "$ckpt_sha" "$gen_tsv_sha")

echo "======================================================"
echo "  P8 NoQ q=10 sanity (--quality_level 10)"
echo "  EMA: $EMA (mtime $ckpt_mtime, sha16 $ckpt_sha)"
echo "  TSV: $GEN_TSV (sha16 $gen_tsv_sha)"
echo "  Expect: ≈ p8baseline rerun (--no_q ≡ q=10)"
echo "======================================================"

if [ -f "$METRICS_FILE" ] && [ -f "$MARKER" ]; then
    if [ "$(cat $MARKER)" = "$expected_marker" ]; then
        echo "[q=10] ✅ metrics 已存在且 marker match，skip"
        cat "$METRICS_FILE"
        exit 0
    fi
fi

# Codex P1: audio identity manifest
audio_manifest="$EVAL_OUT/audio/.gen_manifest.json"
expected_audio_id=$(printf '{"ckpt_sha":"%s","gen_tsv_sha":"%s","q":10,"cfg":"%s","num_samples":%s,"script_version":"%s"}' \
    "$ckpt_sha" "$gen_tsv_sha" "$CFG" "$NUM_SAMPLES" "$SCRIPT_VERSION")

if [ -d "$EVAL_OUT/audio" ]; then
    n=$(ls -1 $EVAL_OUT/audio/*.flac 2>/dev/null | wc -l)
    if [ -f "$audio_manifest" ] && [ "$n" -eq "$NUM_SAMPLES" ]; then
        if [ "$(cat $audio_manifest)" = "$expected_audio_id" ]; then
            echo "[q=10] audio 完整且 identity match，skip gen"
        else
            echo "[q=10] ❌ audio identity mismatch, aborting"
            exit 1
        fi
    else
        echo "[q=10] ❌ audio dir 存在但無 manifest 或 count 不對, aborting"
        exit 1
    fi
else
    echo "[q=10] gen → $EVAL_OUT"
    python eval.py \
        --variant "meanaudio_s" --model_path "$EMA" \
        --output "$EVAL_OUT/audio" --tsv "$GEN_TSV" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength $CFG --quality_level 10 --full_precision \
        2>&1 | tee "$LOG"
    echo "$expected_audio_id" > "$audio_manifest"
fi

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT/audio" --tsv "$GEN_TSV" \
    --exp_name "$BASE" --num_samples $NUM_SAMPLES \
    2>&1 | tee -a "$LOG"

mkdir -p "$(dirname $MARKER)"
echo -n "$expected_marker" > "$MARKER"

echo "======================================================"
echo "  P8 q=10 sanity 完成"
echo "  期待 ≈ p8baseline rerun（--no_q ≡ q=10）"
echo "======================================================"
