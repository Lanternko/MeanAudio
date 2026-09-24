#!/bin/bash
set -eo pipefail
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

echo "[$(date '+%F %T')] === EXP-B NPZ regen (after disk cleanup) ==="
cd /home/kojiek/research/meanaudio_training
python exp_b_regen_npz.py 2>&1 | tee /home/kojiek/logs/exp_b_npz_regen2.log
echo "[$(date '+%F %T')] === NPZ regen done ==="

# Stage 1 resume from ckpt_last.pth (it ~70k)
EXP=p_qwen_slot0
TSV=/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv
NPZ=/home/kojiek/exps_nvme/npz_qwen_slot0
S1_ITER=400000
CKPT=/home/kojiek/MeanAudio/exps/${EXP}_stage1_${S1_ITER}/${EXP}_stage1_${S1_ITER}_ckpt_last.pth

cd /home/kojiek/MeanAudio
echo "[$(date '+%F %T')] === Stage 1 resume from $CKPT ==="
python set_training_stage.py --stage 1
torchrun --nproc_per_node=1 --master_port=23459 train.py \
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
    +checkpoint=${CKPT} 2>&1 | tee -a /home/kojiek/logs/${EXP}_stage1_${S1_ITER}.log

echo "[$(date '+%F %T')] === Stage 1 done; manual launch Stage 2 ==="
