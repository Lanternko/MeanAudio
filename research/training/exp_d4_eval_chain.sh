#!/bin/bash
# EXP-D4 eval chain: MusicCaps CLAP for 3 P8-projection-transplant models
# Run via:
#   tmux new -d -s exp_d4_eval "bash /home/kojiek/research/meanaudio_training/exp_d4_eval_chain.sh"
set -eo pipefail
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

cd /home/kojiek/MeanAudio

TSV=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv

for EXP in exp_a_p8proj_transplant exp_b_p8proj_transplant exp_c_p8proj_transplant; do
    EMA=/home/kojiek/MeanAudio/exps/${EXP}/${EXP}_ema_final.pth
    LABEL=musiccaps
    LOG=/home/kojiek/logs/${EXP}_${LABEL}_eval.log

    echo "[$(date '+%F %T')] === Eval ${EXP} on ${LABEL} ==="
    python eval.py --variant meanaudio_s \
        --model_path "${EMA}" \
        --output eval_output/${EXP}_no_q_${LABEL}/audio \
        --tsv "${TSV}" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --full_precision --no_q \
        2>&1 | tee -a "${LOG}"

    python /home/kojiek/research/meanaudio_eval/phase4_eval.py \
        --gen_dir eval_output/${EXP}_no_q_${LABEL}/audio \
        --exp_name ${EXP}_no_q_${LABEL} \
        --num_samples 2048 \
        --tsv "${TSV}" \
        2>&1 | tee -a "${LOG}"

    echo "[$(date '+%F %T')] === Done ${EXP} ==="
done

echo "[$(date '+%F %T')] === EXP-D4 eval chain complete ==="
echo
echo "=== Results ==="
for EXP in exp_a_p8proj_transplant exp_b_p8proj_transplant exp_c_p8proj_transplant; do
    echo "--- ${EXP} ---"
    cat /home/kojiek/MeanAudio/eval_output/metrics/${EXP}_no_q_musiccaps/metrics.txt 2>/dev/null || echo "(no metrics file)"
done
