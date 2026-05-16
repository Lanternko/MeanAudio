#!/bin/bash
# ============================================================
# P8 V4 NoQ s=1.0 — PE-AV eval (dual-ref)
# eval_p8v4_noq_p100_peav.sh
#
# 用途：拿既有 P8 V4 NoQ s=1.0 的 audio（MC + JM）跑 PE-AV，
#       dual-ref 與 CLAP 同方法論。
#
# Audio 已存在（priority queue #2 跑完）：
#   ~/MeanAudio/eval_output/phase8_v4_stage2_200000_no_q_musiccaps_p100/audio
#   ~/MeanAudio/eval_output/phase8_v4_stage2_200000_no_q_jamendo_seed42_2048_p100/audio
#
# 注意：PE-AV 用獨立 venv (~/venvs/peav)，跟 CLAP/AES 不同
# ============================================================

set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
P100_TSV_DIR="$HOME/eval_tsvs_p100"
LOG_DIR="$HOME/logs"
PEAV_SCRIPT="$HOME/research/meanaudio_eval/peav_eval.py"
PEAV_PYTHON="$HOME/venvs/peav/bin/python"

mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES=0

if [ ! -x "$PEAV_PYTHON" ]; then
    echo "❌ PE-AV venv not found：$PEAV_PYTHON"
    exit 1
fi

if [ ! -f "$PEAV_SCRIPT" ]; then
    echo "❌ PE-AV script 不存在：$PEAV_SCRIPT"
    exit 1
fi

echo "======================================================"
echo "  P8 V4 NoQ s=1.00 PE-AV eval (dual-ref)"
echo "======================================================"

run_peav() {
    local audio_dir="$1"
    local tsv="$2"
    local out_dir="$3"
    local label="$4"
    local log="$5"

    if [ ! -d "$audio_dir" ]; then
        echo "❌ Audio dir 不存在：$audio_dir" | tee -a "$log"
        return 1
    fi
    n_audio=$(ls -1 $audio_dir/*.flac 2>/dev/null | wc -l)
    echo "[PE-AV / $label] audio=$n_audio  tsv=$(basename $tsv)" | tee -a "$log"

    mkdir -p "$out_dir"
    "$PEAV_PYTHON" "$PEAV_SCRIPT" \
        --gen_dir "$audio_dir" \
        --tsv "$tsv" \
        --out "$out_dir/peav_metrics.json" \
        --batch_size 8 \
        2>&1 | tee -a "$log"
}

# ── MusicCaps p100 (n=5521) ─────────────────────────────────
AUDIO_MC="$WORK_DIR/eval_output/phase8_v4_stage2_200000_no_q_musiccaps_p100/audio"
LOG_MC="$LOG_DIR/phase8_v4_stage2_200000_no_q_musiccaps_p100_peav.log"

run_peav "$AUDIO_MC" \
    "$P100_TSV_DIR/phase8_v4_musiccaps_test_p100.tsv" \
    "$WORK_DIR/eval_output/peav_metrics/phase8_v4_stage2_200000_no_q_musiccaps_p100_prefixed_ref" \
    "MC prefixed_ref" "$LOG_MC"

run_peav "$AUDIO_MC" \
    "$DATA_DIR/musiccaps_test.tsv" \
    "$WORK_DIR/eval_output/peav_metrics/phase8_v4_stage2_200000_no_q_musiccaps_p100_natural_ref" \
    "MC natural_ref" "$LOG_MC"

# ── Jamendo seed42 p100 (n=2048) ─────────────────────────────
AUDIO_JM="$WORK_DIR/eval_output/phase8_v4_stage2_200000_no_q_jamendo_seed42_2048_p100/audio"
LOG_JM="$LOG_DIR/phase8_v4_stage2_200000_no_q_jamendo_p100_peav.log"

run_peav "$AUDIO_JM" \
    "$P100_TSV_DIR/phase8_v4_jamendo_seed42_2048_p100.tsv" \
    "$WORK_DIR/eval_output/peav_metrics/phase8_v4_stage2_200000_no_q_jamendo_seed42_2048_p100_prefixed_ref" \
    "JM prefixed_ref" "$LOG_JM"

run_peav "$AUDIO_JM" \
    "$DATA_DIR/phase4_test_seed42_2048.tsv" \
    "$WORK_DIR/eval_output/peav_metrics/phase8_v4_stage2_200000_no_q_jamendo_seed42_2048_p100_natural_ref" \
    "JM natural_ref" "$LOG_JM"

echo "======================================================"
echo "  P8 V4 NoQ s=1.00 PE-AV 完成（dual-ref）"
echo "  Output: eval_output/peav_metrics/phase8_v4_*_p100_*ref/peav_metrics.json"
echo "======================================================"
