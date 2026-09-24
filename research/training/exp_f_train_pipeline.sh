#!/bin/bash
# EXP-F: LP-MC/Qwen 50-50 anchor mixing — single-cap NoQ
# Hypothesis: 50% LP-MC anchor captions prevent S1 co-adaptation collapse.
# Compare to EXP-B (100% Qwen → CLAP 0.0615) and P8 (100% LP-MC → CLAP 0.1851).
#
# Template: exp_b_chain.sh + exp_b_stage2_launch.sh (confirmed working)
#
# Run via:
#   tmux new -d -s exp_f "bash /home/kojiek/research/meanaudio_training/exp_f_train_pipeline.sh 2>&1 | tee /home/kojiek/logs/exp_f.log"
set -eo pipefail
source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

EXP=p_expf_50mix
TSV=/home/kojiek/eval_tsvs_p100/exp_f_50mix_train.tsv
NPZ=/home/kojiek/exps_nvme/npz_expf_50mix
S1_ITER=400000
S2_ITER=200000
S_TOTAL=$(( S1_ITER + S2_ITER ))
DATA_DIR=/mnt/HDD/kojiek/phase4_jamendo_data

cd /home/kojiek/MeanAudio

echo "[$(date '+%F %T')] === EXP-F START: ${EXP} — 50% LP-MC / 50% Qwen slot-0 anchor mixing ==="
echo "  S1: ${EXP}_stage1_${S1_ITER} (400K iter)"
echo "  S2: ${EXP}_stage2_${S2_ITER} (200K iter, cumulative ${S_TOTAL})"
echo "  TSV: ${TSV}"
echo "  NPZ: ${NPZ}"

# Pre-flight
python3 -c "
import csv
with open('${TSV}') as f:
    caps = [r['caption'] for r in csv.DictReader(f, delimiter='\t')]
n = len(caps); uniq = len(set(caps))
print(f'Caption uniqueness: {uniq}/{n} = {uniq/n*100:.2f}% (>90% required)')
assert uniq/n > 0.90, f'FAIL: uniqueness {uniq/n:.3f} < 0.90'
print('PASS')
"

# ===== Stage 1: FluxAudio =====
echo "[$(date '+%F %T')] === Stage 1: FluxAudio ${S1_ITER} iter ==="
python set_training_stage.py --stage 1
torchrun --nproc_per_node=1 --master_port=23462 train.py \
    data=meanaudio model=fluxaudio_s exp_id=${EXP}_stage1_${S1_ITER} \
    num_iterations=${S1_ITER} \
    lr_schedule_steps=[320000,360000] \
    batch_size=8 +accumulation_steps=1 learning_rate=1e-4 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=20000 \
    +use_rope=False +use_wandb=False +use_q_conditioning=false \
    val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
    data.AudioCaps_npz.tsv=${TSV} \
    data.AudioCaps_val_npz.tsv=${DATA_DIR}/phase4_val.tsv \
    ++data.AudioCaps_npz.npz_dir=${NPZ} \
    ++data.AudioCaps_npz.gt_cache=${DATA_DIR}/npz_cache_train.txt \
    ++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4 \
    ++data.AudioCaps_val_npz.gt_cache=null \
    2>&1 | tee /home/kojiek/logs/${EXP}_stage1_${S1_ITER}.log

echo "[$(date '+%F %T')] === Stage 1 done: synthesizing EMA ==="
python -c "
import sys; sys.path.insert(0,'.')
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('hydra', lambda *a: 'dummy')
from meanaudio.model.ema_model import PostHocEMA
import torch
ema = PostHocEMA.load_from_checkpoint(
    'exps/${EXP}_stage1_${S1_ITER}/ema_ckpts',
    sigma_rels=[0.05, 0.1], checkpoint_every=10000
)
sd = ema.synthesize_ema_model(sigma_rel=0.05)
torch.save(sd, 'exps/${EXP}_stage1_${S1_ITER}/${EXP}_stage1_${S1_ITER}_ema_final.pth')
print('Saved ema_final.pth')
"

# ===== Migrate S1 → S2 =====
echo "[$(date '+%F %T')] === Migrating S1 → S2 (ckpt_last.pth) ==="
S1_CKPT=exps/${EXP}_stage1_${S1_ITER}/${EXP}_stage1_${S1_ITER}_ckpt_last.pth
S2_INIT=exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ckpt_last.pth
mkdir -p "$(dirname ${S2_INIT})"
python migrate_stage1_to_stage2_ckpt.py \
    --s1_ckpt "${S1_CKPT}" \
    --s2_out  "${S2_INIT}"

# ===== Stage 2: MeanAudio =====
echo "[$(date '+%F %T')] === Stage 2: MeanAudio ${S2_ITER} iter (cumul ${S_TOTAL}) ==="
python set_training_stage.py --stage 2
torchrun --nproc_per_node=1 --master_port=23463 train.py \
    data=meanaudio model=meanaudio_s exp_id=${EXP}_stage2_${S2_ITER} \
    num_iterations=${S_TOTAL} \
    lr_schedule_steps=[999999,999999] \
    batch_size=8 +accumulation_steps=1 learning_rate=1e-4 num_workers=4 \
    save_weights_interval=10000 save_checkpoint_interval=20000 \
    +use_rope=False +use_wandb=False +use_q_conditioning=false \
    val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
    data.AudioCaps_npz.tsv=${TSV} \
    data.AudioCaps_val_npz.tsv=${DATA_DIR}/phase4_val.tsv \
    ++data.AudioCaps_npz.npz_dir=${NPZ} \
    ++data.AudioCaps_npz.gt_cache=${DATA_DIR}/npz_cache_train.txt \
    ++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4 \
    ++data.AudioCaps_val_npz.gt_cache=null \
    2>&1 | tee /home/kojiek/logs/${EXP}_stage2_${S2_ITER}.log

echo "[$(date '+%F %T')] === Stage 2 done: synthesizing EMA ==="
python -c "
import sys; sys.path.insert(0,'.')
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('hydra', lambda *a: 'dummy')
from meanaudio.model.ema_model import PostHocEMA
import torch
ema = PostHocEMA.load_from_checkpoint(
    'exps/${EXP}_stage2_${S2_ITER}/ema_ckpts',
    sigma_rels=[0.05, 0.1], checkpoint_every=10000
)
sd = ema.synthesize_ema_model(sigma_rel=0.05)
torch.save(sd, 'exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ema_final.pth')
print('Saved ema_final.pth')
"

# ===== Eval: MusicCaps (LP prompt, --no_q) =====
echo "[$(date '+%F %T')] === Eval: MusicCaps ==="
EMA=exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ema_final.pth
for label in musiccaps; do
    TSV_E=${DATA_DIR}/musiccaps_test.tsv
    python eval.py --variant meanaudio_s --model_path ${EMA} \
        --output eval_output/${EXP}_stage2_${S2_ITER}_no_q_${label}/audio \
        --tsv ${TSV_E} --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --full_precision --no_q \
        2>&1 | tee -a /home/kojiek/logs/${EXP}_stage2_${S2_ITER}_no_q_${label}_eval.log
    python /home/kojiek/research/meanaudio_eval/phase4_eval.py \
        --gen_dir eval_output/${EXP}_stage2_${S2_ITER}_no_q_${label}/audio \
        --exp_name ${EXP}_stage2_${S2_ITER}_no_q_${label} \
        --tsv ${TSV_E} \
        2>&1 | tee -a /home/kojiek/logs/${EXP}_stage2_${S2_ITER}_no_q_${label}_eval.log
done

echo "[$(date '+%F %T')] === EXP-F COMPLETE ==="
echo ""
echo "=== RESULTS ==="
cat eval_output/metrics/${EXP}_stage2_${S2_ITER}_no_q_musiccaps/metrics.txt 2>/dev/null || echo "(metrics not found)"
echo ""
echo "=== INTERPRETATION ==="
echo "  EXP-B (100% Qwen NoQ):  MC CLAP ~ 0.0615  [collapsed]"
echo "  P8     (100% LP-MC NoQ): MC CLAP ~ 0.1851  [healthy]"
echo "  EXP-F  (50/50):          MC CLAP = ??"
echo "  Threshold:"
echo "    > 0.13  → LP-MC anchor at 50% rescues text-conditioning"
echo "    0.09-0.12 → partial recovery"
echo "    < 0.08  → mixing insufficient; Qwen poisons S1 regardless"
