#!/bin/bash
# EXP-C: P-Qwen-Boilerplate — Qwen slot 0 caption with LP-MC boilerplate prefix
# Inverse of EXP-A: tests whether anchor template alone is sufficient for healthy training
#
# Run via:
#   tmux new -d -s exp_c "bash /home/kojiek/research/meanaudio_training/exp_c_chain.sh"
set -eo pipefail
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

cd /home/kojiek/research/meanaudio_training
echo "[$(date '+%F %T')] === EXP-C NPZ regen ==="
python exp_c_regen_npz.py --resume 2>&1 | tee /home/kojiek/logs/exp_c_npz_regen.log
echo "[$(date '+%F %T')] === NPZ regen done ==="

EXP=p_qwen_slot0_boilerplate
TSV=/home/kojiek/eval_tsvs_p100/qwen_slot0_boilerplate_train.tsv
NPZ=/home/kojiek/exps_nvme/npz_qwen_slot0_boilerplate
S1_ITER=400000
S2_ITER=200000
S_TOTAL=$(( S1_ITER + S2_ITER ))

cd /home/kojiek/MeanAudio

# Stage 1
echo "[$(date '+%F %T')] === EXP-C Stage 1 ==="
python set_training_stage.py --stage 1
torchrun --nproc_per_node=1 --master_port=23461 train.py \
    data=meanaudio model=fluxaudio_s exp_id=${EXP}_stage1_${S1_ITER} \
    num_iterations=${S1_ITER} \
    lr_schedule_steps=[320000,360000] \
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
    2>&1 | tee /home/kojiek/logs/${EXP}_stage1_${S1_ITER}.log

S1_CKPT=/home/kojiek/MeanAudio/exps/${EXP}_stage1_${S1_ITER}/${EXP}_stage1_${S1_ITER}_ckpt_last.pth
S2_INIT=/home/kojiek/MeanAudio/exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ckpt_last.pth

# Migrate
echo "[$(date '+%F %T')] === migrate S1 -> S2 ==="
mkdir -p "$(dirname ${S2_INIT})"
python migrate_stage1_to_stage2_ckpt.py --s1_ckpt "${S1_CKPT}" --s2_out "${S2_INIT}"

# Stage 2
echo "[$(date '+%F %T')] === EXP-C Stage 2 (target total=${S_TOTAL}) ==="
python set_training_stage.py --stage 2
torchrun --nproc_per_node=1 --master_port=23462 train.py \
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

# Eval — 3 benchmarks: jamendo s42 (LP prompt), musiccaps (LP prompt), qwen_jamendo_s42 (Qwen prompt)
EMA_S2=/home/kojiek/MeanAudio/exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ema_final.pth
echo "[$(date '+%F %T')] === EXP-C eval ==="
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
echo "[$(date '+%F %T')] === EXP-C complete ==="
