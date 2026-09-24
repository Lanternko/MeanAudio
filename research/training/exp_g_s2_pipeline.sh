#!/bin/bash
# EXP-G: LP-MC S1 → Qwen S2 (stage-localization test)
# S1: reused from P8 LP-MC (ema_final wrapped as synthetic ckpt_last)
# S2: Qwen slot-0 captions, NoQ, 200K iter (cumulative 600K)
set -eo pipefail

source ~/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0

EXP=p_expg_lpmcs1_qwens2
S1_ITER=400000
S2_ITER=200000
S_TOTAL=600000

DATA_DIR=/mnt/HDD/kojiek/phase4_jamendo_data
TSV=/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv
NPZ=/home/kojiek/exps_nvme/npz_qwen_slot0
TSV_EVAL=${DATA_DIR}/musiccaps_test.tsv
EMA=exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ema_final.pth

cd /home/kojiek/MeanAudio

echo "[$(date)] === EXP-G S2 START ==="
echo "Experiment: ${EXP}_stage2_${S2_ITER}"
echo "Caption:    Qwen slot-0 (same as EXP-B, cleanest single-framing)"
echo "S1 anchor:  P8 LP-MC healthy FluxAudio (synthetic ckpt from ema_final)"

# ── Stage 2 training ─────────────────────────────────────────────
python set_training_stage.py --stage 2

torchrun --nproc_per_node=1 --master_port=23466 train.py \
    data=meanaudio model=meanaudio_s \
    exp_id=${EXP}_stage2_${S2_ITER} \
    num_iterations=${S_TOTAL} \
    lr_schedule_steps=[999999,999999] \
    batch_size=8 +accumulation_steps=1 \
    learning_rate=1e-4 \
    num_workers=4 \
    +use_rope=False +use_wandb=False \
    +use_q_conditioning=false \
    val_interval=999999 eval_interval=999999 save_eval_interval=999999 \
    data.AudioCaps_npz.tsv=${TSV} \
    data.AudioCaps_val_npz.tsv=${DATA_DIR}/phase4_val.tsv \
    ++data.AudioCaps_npz.npz_dir=${NPZ} \
    ++data.AudioCaps_npz.gt_cache=${DATA_DIR}/npz_cache_train.txt \
    ++data.AudioCaps_val_npz.npz_dir=/home/kojiek/research/meanaudio_training/npz_phase8v4 \
    ++data.AudioCaps_val_npz.gt_cache=null \
    2>&1 | tee ~/logs/${EXP}_stage2_${S2_ITER}.log

echo "[$(date)] S2 training complete."

# ── EMA synthesis (training runner auto-saves ema_final.pth; skip PostHocEMA) ─
echo "[$(date)] Checking ema_final.pth ..."
if [ -f "${EMA}" ]; then
    echo "ema_final.pth found ($(du -sh ${EMA} | cut -f1))."
else
    echo "WARNING: ema_final.pth not found. Falling back to ckpt_last."
    cp exps/${EXP}_stage2_${S2_ITER}/${EXP}_stage2_${S2_ITER}_ckpt_last.pth \
       ${EMA} 2>/dev/null || true
fi

# ── Eval: MusicCaps ──────────────────────────────────────────────
echo "[$(date)] === EVAL: MusicCaps ==="
EVAL_TAG=${EXP}_stage2_${S2_ITER}_no_q_musiccaps
python eval.py \
    --variant meanaudio_s \
    --model_path ${EMA} \
    --output eval_output/${EVAL_TAG}/audio \
    --tsv ${TSV_EVAL} \
    --use_meanflow --num_steps 1 \
    --encoder_name t5_clap --text_c_dim 512 \
    --cfg_strength 0.5 --full_precision \
    --no_q

python ~/research/meanaudio_eval/phase4_eval.py \
    --gen_dir eval_output/${EVAL_TAG}/audio \
    --exp_name ${EVAL_TAG} \
    --tsv ${TSV_EVAL}

echo "[$(date)] === EXP-G COMPLETE ==="
echo "Experiment: ${EVAL_TAG}"
cat eval_output/metrics/${EVAL_TAG}/metrics.txt 2>/dev/null || echo "(metrics file not found)"
