#!/bin/bash
# Chain pipeline:
# 1. Wait for P4V2-Qwen Stage 2 ema_final ckpt
# 2. Launch eval (jamendo s42 + musiccaps + Qwen prompts)
# 3. Launch EXP-A NPZ regen
# (Stage 1 retrain launched manually after NPZ regen verified)
#
# Run via:
#   tmux new -d -s chain "bash /home/kojiek/research/meanaudio_training/p4v2_chain_pipeline.sh"

set -eo pipefail

source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

EXP_TR=p4v2_qwen_stage2_200000
EXP_DIR=/home/kojiek/MeanAudio/exps/${EXP_TR}
EMA_FINAL=${EXP_DIR}/${EXP_TR}_ema_final.pth
LP_TSV=/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test_seed42_2048.tsv
MC_TSV=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
QWEN_TSV=/mnt/HDD/kojiek/phase4_jamendo_data/qwen_test_seed42_2048_random.tsv

echo "[$(date '+%F %T')] Waiting for ${EMA_FINAL}..."
while [ ! -f "${EMA_FINAL}" ]; do
    sleep 60
    if ! tmux has-session -t p4v2_qwen 2>/dev/null; then
        if [ -f "${EMA_FINAL}" ]; then break; fi
        if [ -f "${EXP_DIR}/${EXP_TR}_last.pth" ]; then
            EMA_FINAL=${EXP_DIR}/${EXP_TR}_last.pth
            echo "[$(date '+%F %T')] tmux died; using last.pth"
            break
        fi
        echo "[$(date '+%F %T')] ERROR: training tmux died but no ckpt; abort"
        exit 1
    fi
done
echo "[$(date '+%F %T')] ckpt found: ${EMA_FINAL}"
sleep 30  # let tmux session close cleanly

cd /home/kojiek/MeanAudio

run_eval() {
    local label=$1
    local tsv=$2
    local out=eval_output/${EXP_TR}_${label}/audio
    local logf=/home/kojiek/logs/${EXP_TR}_${label}_eval.log
    echo "[$(date '+%F %T')] === eval ${label} (tsv=$(basename ${tsv})) ==="
    python eval.py --variant meanaudio_s \
        --model_path ${EMA_FINAL} \
        --output ${out} \
        --tsv ${tsv} --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --full_precision --no_q 2>&1 | tee ${logf}
    python /home/kojiek/research/meanaudio_eval/phase4_eval.py \
        --gen_dir ${out} \
        --exp_name ${EXP_TR}_${label} \
        --num_samples 2048 \
        --tsv ${tsv} 2>&1 | tee -a ${logf}
}

run_eval jamendo_s42 "${LP_TSV}"
run_eval musiccaps "${MC_TSV}"
run_eval qwen_random_jamendo_s42 "${QWEN_TSV}"
echo "[$(date '+%F %T')] === all P4V2 evals done ==="

# Launch EXP-A NPZ regen
NPZ_DST=/home/kojiek/exps_nvme/npz_phase7v1_destructured
mkdir -p "${NPZ_DST}"
NPZ_COUNT=$(ls "${NPZ_DST}" 2>/dev/null | wc -l)
if [ "${NPZ_COUNT}" -lt 251000 ]; then
    echo "[$(date '+%F %T')] === launching EXP-A NPZ regen (existing=${NPZ_COUNT}) ==="
    cd /home/kojiek/research/meanaudio_training
    python exp_a_regen_npz.py --resume 2>&1 | tee /home/kojiek/logs/exp_a_npz_regen.log
    echo "[$(date '+%F %T')] === EXP-A NPZ regen done ==="
else
    echo "[$(date '+%F %T')] NPZ already regenerated (${NPZ_COUNT} files); skipping"
fi

echo "[$(date '+%F %T')] === pipeline complete; manually launch EXP-A Stage 1 retraining next ==="
