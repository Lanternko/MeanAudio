#!/bin/bash
# EXP-G follow-up evals (run after CLAP/AES already completed):
#   1. PE-AV for all 3 eval sets (chain log showed OOM crash mid-way; rerun all 3)
#   2. Steering probe (24 wav, NoQ→quality_level=10 per feedback_infer_no_q_workaround.md)
#
# Launched 2026-05-15 by Claude under user instruction "做完 A後 monitor 自己決定要
# 不要做 B，不要讓 GPU 閒置" — EXP-G CLAP/AES showed NULL result (MC 0.0679);
# completing PE-AV + steering probe so EXP-G is fully measured for paper.

set -eo pipefail
LOG=$HOME/logs/exp_g_followup_evals.log
mkdir -p $HOME/logs
exec > >(tee -a "$LOG") 2>&1

echo "=== EXP-G follow-up evals started at $(date) ==="

EXP=p_expg_lpmcs1_qwens2_stage2_200000
EMA=$HOME/MeanAudio/exps/${EXP}/${EXP}_ema_final.pth
[ -f "$EMA" ] || { echo "[FAIL] EMA missing: $EMA"; exit 1; }

OUT_ROOT=/home/kojiek/eval_output_nvme

DATA=/mnt/HDD/kojiek/phase4_jamendo_data
TSV_MC=${DATA}/musiccaps_test.tsv
TSV_JM_LP=${DATA}/phase4_test_seed42_2048.tsv
TSV_JM_QWEN=${DATA}/qwen_test_seed42_2048_random.tsv

PEAV_PY=$HOME/research/meanaudio_eval/peav_eval.py

# ============================================================
# 1. PE-AV — rerun all 3 (previous chain crashed with OOM, no peav.json exists)
# ============================================================
echo
echo "[$(date)] === STAGE 1: PE-AV (3 eval sets) ==="

deactivate 2>/dev/null || true
source $HOME/venvs/peav/bin/activate
echo "PE-AV venv: $(which python)"

# Help avoid OOM fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for spec in "no_q_musiccaps:$TSV_MC" "jamendo_s42:$TSV_JM_LP" "qwen_random_jamendo_s42:$TSV_JM_QWEN"; do
    tag="${spec%%:*}"
    tsv="${spec#*:}"
    outdir="${OUT_ROOT}/${EXP}_${tag}/audio"
    peav_json="${OUT_ROOT}/metrics/${EXP}_${tag}_peav.json"

    echo
    echo "[$(date)] === PE-AV ${tag} ==="

    if [ -f "$peav_json" ]; then
        echo "  ✅ already exists, skipping: $peav_json"
        continue
    fi

    if [ ! -d "$outdir" ] || [ -z "$(ls "$outdir" 2>/dev/null)" ]; then
        echo "  ❌ missing audio dir: $outdir — skip"
        continue
    fi

    python "$PEAV_PY" \
        --gen_dir "$outdir" \
        --tsv "$tsv" \
        --out "$peav_json" \
        && echo "  ✅ PE-AV ${tag} OK" \
        || echo "  ❌ PE-AV ${tag} failed (continuing)"
done

# ============================================================
# 2. Steering probe (NoQ → quality_level=10 null-token workaround)
# ============================================================
echo
echo "[$(date)] === STAGE 2: STEERING PROBE ==="

deactivate 2>/dev/null || true
source $HOME/venvs/dac/bin/activate
cd $HOME/MeanAudio

# PROBE_QUALITY=10 for NoQ models (canonical per feedback_infer_no_q_workaround.md)
export PROBE_QUALITY=10

# Custom output dir on NVMe (HDD full)
export PROBE_OUT="${OUT_ROOT}/${EXP}_steering_probe"

if [ -d "$PROBE_OUT" ] && [ -n "$(ls "$PROBE_OUT/audio" 2>/dev/null | grep '.wav$' | head -1)" ]; then
    n=$(ls "$PROBE_OUT/audio" 2>/dev/null | grep -c '.wav$')
    echo "  Probe output already has $n wav files in $PROBE_OUT/audio"
    if [ "$n" -ge 24 ]; then
        echo "  ✅ ≥24 wav present, skipping probe generation"
    else
        echo "  ⚠️ only $n/24 wav, re-running"
        bash scripts/legacy/probe_v1_steering.sh "$EMA"
    fi
else
    bash scripts/legacy/probe_v1_steering.sh "$EMA"
fi

# ============================================================
# Summary
# ============================================================
echo
echo "[$(date)] === EXP-G FOLLOW-UP COMPLETE ==="
echo "=== Summary ==="
echo
echo "[CLAP / AES (from previous chain)]"
for tag in no_q_musiccaps jamendo_s42 qwen_random_jamendo_s42; do
    f=${OUT_ROOT}/metrics/${EXP}_${tag}/metrics.txt
    echo "--- ${tag} ---"
    [ -f "$f" ] && cat "$f" || echo "  (metrics missing)"
done

echo
echo "[PE-AV]"
for tag in no_q_musiccaps jamendo_s42 qwen_random_jamendo_s42; do
    pf=${OUT_ROOT}/metrics/${EXP}_${tag}_peav.json
    echo "--- ${tag} ---"
    [ -f "$pf" ] && cat "$pf" || echo "  (peav.json missing)"
done

echo
echo "[STEERING PROBE]"
if [ -f "$PROBE_OUT/ratios.txt" ]; then
    cat "$PROBE_OUT/ratios.txt"
else
    echo "  (ratios.txt not yet written; probe still in progress or script wrote elsewhere)"
fi
echo
echo "DONE at $(date)"
