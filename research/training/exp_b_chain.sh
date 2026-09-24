#!/bin/bash
set -eo pipefail
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

cd /home/kojiek/research/meanaudio_training
echo "[$(date '+%F %T')] === EXP-B NPZ regen ==="
python exp_b_regen_npz.py --resume 2>&1 | tee /home/kojiek/logs/exp_b_npz_regen.log
echo "[$(date '+%F %T')] === NPZ regen done ==="

# Build a customized train_pipeline for EXP-B
EXP=p_qwen_slot0
TSV=/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv
NPZ=/home/kojiek/exps_nvme/npz_qwen_slot0
S1_ITER=400000
S2_ITER=200000

cd /home/kojiek/MeanAudio

# Stage 1
echo "[$(date '+%F %T')] === EXP-B Stage 1 ==="
python set_training_stage.py --stage 1
torchrun --nproc_per_node=1 --master_port=23456 train.py \
    data=meanaudio model=fluxaudio_s exp_id=${EXP}_stage1_${S1_ITER} \
    num_iterations=${S1_ITER} \
    lr_schedule_steps=[320000,360000] \
    batch_size=8 +accumulation_steps=1 learning_rate=1e-4 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=20000 \
    +use_rope=False +use_wandb=False +use_q_conditioning=false \
    val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
    data.AudioCaps_npz.tsv=${TSV} \
    data.AudioCaps_val_npz.tsv=/mnt/HDD/kojiek/phase4_jamendo_data/phase4_val.tsv \
    ++data.AudioCaps_npz.npz_dir=${NPZ} \
    ++data.AudioCaps_npz.gt_cache=/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt 2>&1 | tee /home/kojiek/logs/${EXP}_stage1_${S1_ITER}.log

# Migrate
echo "[$(date '+%F %T')] === migrate S1 -> S2 ==="
python migrate_stage1_to_stage2_ckpt.py \
    --src exps/${EXP}_stage1_${S1_ITER}/${EXP}_stage1_${S1_ITER}_ema_final.pth \
    --dst exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_init.pth

# Stage 2
echo "[$(date '+%F %T')] === EXP-B Stage 2 ==="
python set_training_stage.py --stage 2
torchrun --nproc_per_node=1 --master_port=23457 train.py \
    data=meanaudio model=meanaudio_s exp_id=${EXP}_stage2_${S2_ITER} \
    num_iterations=${S2_ITER} weights=exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_init.pth \
    lr_schedule_steps=[999999,999999] \
    batch_size=8 +accumulation_steps=1 learning_rate=1e-4 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=20000 \
    +use_rope=False +use_wandb=False +use_q_conditioning=false \
    val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
    data.AudioCaps_npz.tsv=${TSV} \
    data.AudioCaps_val_npz.tsv=/mnt/HDD/kojiek/phase4_jamendo_data/phase4_val.tsv \
    ++data.AudioCaps_npz.npz_dir=${NPZ} \
    ++data.AudioCaps_npz.gt_cache=/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt 2>&1 | tee /home/kojiek/logs/${EXP}_stage2_${S2_ITER}.log

# Eval
echo "[$(date '+%F %T')] === EXP-B eval ==="
EMA=exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ema_final.pth
for label in jamendo_seed42_2048 musiccaps; do
    if [ "$label" = "musiccaps" ]; then
        TSV_E=/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
    else
        TSV_E=/mnt/HDD/kojiek/phase4_jamendo_data/phase4_test_seed42_2048.tsv
    fi
    python eval.py --variant meanaudio_s --model_path ${EMA} \
        --output eval_output/${EXP}_stage2_${S2_ITER}_no_q_${label}/audio \
        --tsv ${TSV_E} --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --full_precision --no_q 2>&1 | tee -a /home/kojiek/logs/${EXP}_stage2_${S2_ITER}_no_q_${label}_eval.log
    python /home/kojiek/research/meanaudio_eval/phase4_eval.py \
        --gen_dir eval_output/${EXP}_stage2_${S2_ITER}_no_q_${label}/audio \
        --exp_name ${EXP}_stage2_${S2_ITER}_no_q_${label} \
        --num_samples 2048 \
        --tsv ${TSV_E} 2>&1 | tee -a /home/kojiek/logs/${EXP}_stage2_${S2_ITER}_no_q_${label}_eval.log
done

echo "[$(date '+%F %T')] === EXP-B complete ==="
