#!/usr/bin/env bash
set -eo pipefail

WORK_DIR="$HOME/MeanAudio"
DATA_DIR="/mnt/HDD/kojiek/phase4_jamendo_data"
LOG_DIR="$HOME/logs"
TSV="$DATA_DIR/music_flamingo_slice10_10k_train_seed20260528_400.tsv"

MF_MODEL="$WORK_DIR/exps/mf10k_noq_fast_stage2_50000/mf10k_noq_fast_stage2_50000_ema_final.pth"
LPMC_MODEL="$WORK_DIR/exps/lpmc10k_noq_fast_stage2_50000/lpmc10k_noq_fast_stage2_50000_ema_final.pth"

EVAL_SCRIPT="$HOME/research/meanaudio_eval/phase4_eval.py"
PEAV_SCRIPT="$HOME/research/meanaudio_eval/peav_eval.py"

mkdir -p "$LOG_DIR"
cd "$WORK_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_one() {
    local label="$1"
    local model_path="$2"
    local out_dir="$WORK_DIR/eval_output/${label}/audio"
    local log="$LOG_DIR/${label}.log"

    echo "======================================================"
    echo "Reverse control: $label"
    echo "TSV: $TSV"
    echo "model: $model_path"
    echo "out: $out_dir"
    echo "======================================================"

    python eval.py \
        --variant meanaudio_s \
        --model_path "$model_path" \
        --output "$out_dir" \
        --tsv "$TSV" \
        --use_meanflow --num_steps 1 \
        --encoder_name t5_clap --text_c_dim 512 \
        --cfg_strength 0.5 --no_q \
        --full_precision \
        2>&1 | tee "$log"

    python "$EVAL_SCRIPT" \
        --gen_dir "$out_dir" \
        --tsv "$TSV" \
        --exp_name "$label" \
        --num_samples 400 \
        2>&1 | tee -a "$log"

    if [ -f "$PEAV_SCRIPT" ] && [ -x "$HOME/venvs/peav/bin/python" ]; then
        "$HOME/venvs/peav/bin/python" "$PEAV_SCRIPT" \
            --gen_dir "$out_dir" \
            --tsv "$TSV" \
            --out "$WORK_DIR/eval_output/metrics/${label}_peav.json" \
            --batch_size 8 \
            2>&1 | tee "$LOG_DIR/${label}_peav.log"
    fi
}

run_one "reverse_mf400_prompts_mf10k_model" "$MF_MODEL"
run_one "reverse_mf400_prompts_lpmc10k_model" "$LPMC_MODEL"

echo "Reverse control complete"
