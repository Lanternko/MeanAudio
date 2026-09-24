#!/bin/bash
# EXP-B Stage 2 launch — run this manually after Stage 1 ema_final.pth appears
# Usage: bash exp_b_stage2_launch.sh
set -eo pipefail
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

EXP=p_qwen_slot0
TSV=/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv
NPZ=/home/kojiek/exps_nvme/npz_qwen_slot0
S1_ITER=400000
S2_ITER=200000
S1_CKPT=/home/kojiek/MeanAudio/exps/${EXP}_stage1_${S1_ITER}/${EXP}_stage1_${S1_ITER}_ckpt_last.pth
S2_INIT=/home/kojiek/MeanAudio/exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ckpt_last.pth

cd /home/kojiek/MeanAudio

# Wait for Stage 1 ckpt_last
echo "[$(date '+%F %T')] Waiting for ${S1_CKPT} ..."
while [ ! -f "${S1_CKPT}" ]; do
    sleep 30
done
echo "[$(date '+%F %T')] Stage 1 ckpt_last found."

# Migrate S1 → S2 (canonical pipeline uses ckpt_last.pth, full checkpoint with 'it'/'weights'/'ema'/optim)
echo "[$(date '+%F %T')] === migrate S1 -> S2 ==="
mkdir -p "$(dirname ${S2_INIT})"
python migrate_stage1_to_stage2_ckpt.py \
    --s1_ckpt "${S1_CKPT}" \
    --s2_out  "${S2_INIT}"

# Stage 2 — trainer auto-resumes from ${EXP}_stage2_${S2_ITER}_ckpt_last.pth (just written by migrate)
# num_iterations = total cumulative (S1 + S2) per canonical train_pipeline.sh
S_TOTAL=$(( S1_ITER + S2_ITER ))
echo "[$(date '+%F %T')] === EXP-B Stage 2 (total iterations target = ${S_TOTAL}) ==="
python set_training_stage.py --stage 2
torchrun --nproc_per_node=1 --master_port=23460 train.py \
    data=meanaudio model=meanaudio_s exp_id=${EXP}_stage2_${S2_ITER} \
    num_iterations=${S_TOTAL} \
    lr_schedule_steps=[999999,999999] \
    batch_size=4 +accumulation_steps=2 learning_rate=1e-4 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=20000 \
    +use_rope=False +use_wandb=False +use_q_conditioning=false \
    val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
    data.AudioCaps_npz.tsv=${TSV} \
    data.AudioCaps_val_npz.tsv=/mnt/HDD/kojiek/phase4_jamendo_data/phase4_val.tsv \
    ++data.AudioCaps_npz.npz_dir=${NPZ} \
    ++data.AudioCaps_npz.gt_cache=/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt \
    ++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4 \
    ++data.AudioCaps_val_npz.gt_cache=null \
    2>&1 | tee /home/kojiek/logs/${EXP}_stage2_${S2_ITER}.log
echo "[$(date '+%F %T')] === Stage 2 done ==="

# Eval
echo "[$(date '+%F %T')] === EXP-B eval ==="
EMA_S2=/home/kojiek/MeanAudio/exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ema_final.pth
for label in jamendo_seed42_2048 musiccaps qwen_jamendo_s42; do
    if [ "$label" = "musiccaps" ]; then
        TSV_E=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
    elif [ "$label" = "qwen_jamendo_s42" ]; then
        TSV_E=/mnt/HDD/kojiek/phase4_jamendo_data/qwen_test_seed42_2048_random.tsv
    else
        TSV_E=/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test_seed42_2048.tsv
    fi
    python eval.py --variant meanaudio_s --model_path ${EMA_S2} \
        --output eval_output/${EXP}_stage2_${S2_ITER}_no_q_${label}/audio \
        --tsv ${TSV_E} --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --full_precision --no_q \
        2>&1 | tee -a /home/kojiek/logs/${EXP}_stage2_${S2_ITER}_no_q_${label}_eval.log
    python /home/kojiek/research/meanaudio_eval/phase4_eval.py \
        --gen_dir eval_output/${EXP}_stage2_${S2_ITER}_no_q_${label}/audio \
        --exp_name ${EXP}_stage2_${S2_ITER}_no_q_${label} \
        --num_samples 2048 \
        --tsv ${TSV_E} \
        2>&1 | tee -a /home/kojiek/logs/${EXP}_stage2_${S2_ITER}_no_q_${label}_eval.log
done

echo "[$(date '+%F %T')] === EXP-B complete ==="
