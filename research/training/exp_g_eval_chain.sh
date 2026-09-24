#!/bin/bash
# EXP-G eval chain: LP-MC S1 → Qwen S2 (NoQ)
# /mnt/HDD is 100% full — write all output to NVMe via absolute paths.

set -eo pipefail
LOG=$HOME/logs/exp_g_eval_chain.log
mkdir -p $HOME/logs
exec > >(tee -a "$LOG") 2>&1

echo "=== EXP-G eval chain started at $(date) ==="

source $HOME/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd $HOME/MeanAudio

EXP=p_expg_lpmcs1_qwens2_stage2_200000
EMA=$HOME/MeanAudio/exps/${EXP}/${EXP}_ema_final.pth

# NVMe output root (HDD full)
OUT_ROOT=/home/kojiek/eval_output_nvme
mkdir -p "$OUT_ROOT" "$OUT_ROOT/metrics"

DATA=/mnt/HDD/kojiek/phase4_jamendo_data
TSV_MC=${DATA}/musiccaps_test.tsv
TSV_JM_LP=${DATA}/phase4_test_seed42_2048.tsv
TSV_JM_QWEN=${DATA}/qwen_test_seed42_2048_random.tsv

EVAL_PY=$HOME/research/meanaudio_eval/phase4_eval.py
PEAV_PY=$HOME/research/meanaudio_eval/peav_eval.py

[ -f "$EMA" ] || { echo "[FAIL] EMA missing: $EMA"; exit 1; }
echo "EMA: $EMA ($(du -sh $EMA | cut -f1))"
echo "Audio out root: $OUT_ROOT"
df -h $OUT_ROOT | tail -1

gen_eval() {
    local tag=$1
    local tsv=$2
    local samples=$3
    local outdir=${OUT_ROOT}/${EXP}_${tag}
    mkdir -p "${outdir}/audio"

    echo
    echo "[$(date)] === GEN ${tag} ==="
    python eval.py \
        --variant meanaudio_s \
        --model_path "$EMA" \
        --output "${outdir}/audio" \
        --tsv "$tsv" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --full_precision \
        --no_q

    echo "[$(date)] === METRICS ${tag} ==="
    python "$EVAL_PY" \
        --gen_dir "${outdir}/audio" \
        --tsv "$tsv" \
        --exp_name "${EXP}_${tag}" \
        --out_dir "${OUT_ROOT}/metrics" \
        --num_samples ${samples}
}

# 1. MusicCaps (default benchmark, n=5521)
gen_eval "no_q_musiccaps" "$TSV_MC" 5521

# 2. Jamendo seed42 — LP prompts (legacy comparison)
gen_eval "jamendo_s42" "$TSV_JM_LP" 2048

# 3. Jamendo seed42 — Qwen prompts (in-distribution of S2 training)
gen_eval "qwen_random_jamendo_s42" "$TSV_JM_QWEN" 2048

# 4. PE-AV (peav venv) for all three
deactivate 2>/dev/null || true
source $HOME/venvs/peav/bin/activate 2>/dev/null || { echo "[WARN] peav venv unavailable — skip PE-AV"; exit 0; }

for tag in no_q_musiccaps jamendo_s42 qwen_random_jamendo_s42; do
    tsv_for_peav=""
    case $tag in
        no_q_musiccaps)              tsv_for_peav="$TSV_MC" ;;
        jamendo_s42)                 tsv_for_peav="$TSV_JM_LP" ;;
        qwen_random_jamendo_s42)     tsv_for_peav="$TSV_JM_QWEN" ;;
    esac
    echo
    echo "[$(date)] === PE-AV ${tag} ==="
    python "$PEAV_PY" \
        --gen_dir "${OUT_ROOT}/${EXP}_${tag}/audio" \
        --tsv "$tsv_for_peav" \
        --out "${OUT_ROOT}/metrics/${EXP}_${tag}_peav.json" \
        || echo "[WARN] PE-AV ${tag} failed"
done

echo
echo "[$(date)] === EXP-G EVAL CHAIN COMPLETE ==="
echo "=== Summary ==="
for tag in no_q_musiccaps jamendo_s42 qwen_random_jamendo_s42; do
    f=${OUT_ROOT}/metrics/${EXP}_${tag}/metrics.txt
    echo "[${tag}]"
    [ -f "$f" ] && cat "$f" || echo "  (metrics missing)"
    pf=${OUT_ROOT}/metrics/${EXP}_${tag}_peav.json
    [ -f "$pf" ] && { echo "PE-AV:"; cat "$pf"; }
    echo
done
