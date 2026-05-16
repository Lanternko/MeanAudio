#!/bin/bash
# ============================================================
# P8 NoQ baseline rerun (--no_q) on MusicCaps — same-pipeline
# eval_p8_noq_baseline_rerun.sh
#
# Codex Round 3 P2: Fig.2 baseline 不能用歷史 0.1851（不同
# pipeline/version）；要在同 pipeline 重跑 --no_q baseline，
# 才能跟 q=5..9 control 公平對照。
#
# 注意：這次跑出來的 baseline 也是 train/eval mismatch
#   - 訓練時 bug 把 null 訓在 q[9]（已驗證）
#   - --no_q 強制用 q[10]（untrained random）
# 所以這個 baseline 是「pipeline-consistent baseline of P8 NoQ」，
# 不是「真 trained null」。Fig.2 caption 要明標。
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"

EXP="phase8_stage2_200000"
EMA="$WORK_DIR/exps/$EXP/${EXP}_ema_final.pth"
EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES=0

EVAL_OUT="$WORK_DIR/eval_output/${EXP}_no_q_musiccaps_qsweep_baseline"
LOG="$LOG_DIR/${EXP}_no_q_musiccaps_qsweep_baseline_eval.log"
METRICS_FILE="$WORK_DIR/eval_output/metrics/${EXP}_no_q_musiccaps_qsweep_baseline/metrics.txt"
MARKER_FILE="$WORK_DIR/eval_output/metrics/${EXP}_no_q_musiccaps_qsweep_baseline/.run_manifest"

# Codex P2: marker 加 ckpt mtime + sha 前 8 字
ckpt_mtime=$(stat -c %Y "$EMA")
ckpt_sha=$(sha256sum "$EMA" | awk '{print $1}' | head -c 16)

if [ -f "$METRICS_FILE" ] && [ -f "$MARKER_FILE" ]; then
    EXPECTED="$(printf '%s\n%s\n%s\n%s\n%s' "$EMA" "$DATA_DIR/musiccaps_test.tsv" 5521 "$ckpt_mtime" "$ckpt_sha")"
    ACTUAL="$(cat $MARKER_FILE)"
    if [ "$EXPECTED" = "$ACTUAL" ]; then
        echo "[baseline] ✅ metrics 已存在且 marker 一致（含 ckpt mtime/sha），skip"
        cat "$METRICS_FILE"
        exit 0
    fi
fi

echo "======================================================"
echo "  P8 NoQ baseline rerun (--no_q) — same-pipeline for Fig.2"
echo "  EMA   : $EMA (mtime $ckpt_mtime, sha16 $ckpt_sha)"
echo "  TSV   : $DATA_DIR/musiccaps_test.tsv (n=5521)"
echo "  ⚠️  --no_q 走 q[10] (untrained random，bug 把 null 訓在 q[9])"
echo "======================================================"

python eval.py \
    --variant "meanaudio_s" \
    --model_path "$EMA" \
    --output "$EVAL_OUT/audio" \
    --tsv "$DATA_DIR/musiccaps_test.tsv" \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --no_q \
    --full_precision \
    2>&1 | tee "$LOG"

python "$EVAL_SCRIPT" \
    --gen_dir "$EVAL_OUT/audio" \
    --tsv "$DATA_DIR/musiccaps_test.tsv" \
    --exp_name "${EXP}_no_q_musiccaps_qsweep_baseline" \
    --num_samples 5521 \
    2>&1 | tee -a "$LOG"

mkdir -p "$(dirname $MARKER_FILE)"
printf '%s\n%s\n%s\n%s\n%s' "$EMA" "$DATA_DIR/musiccaps_test.tsv" 5521 "$ckpt_mtime" "$ckpt_sha" > "$MARKER_FILE"

echo "======================================================"
echo "  P8 NoQ baseline rerun 完成"
echo "  Compare to historical 0.1851"
echo "======================================================"
