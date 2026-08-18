#!/bin/bash
# Resume remaining slot2 with larger batch + prefetch, then the existing after-gen chain.
set -euo pipefail
source /home/kojiek/venvs/dac/bin/activate
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
cd /home/kojiek/research/meanaudio_training/caption10s_pipeline
OUT=/home/kojiek/research/meanaudio_training/outputs/caption10s_pipeline/c2p0_qwen3cap_full
IDS=$OUT/ids.jsonl
PROMPT2="Listen carefully to this music clip. Write a rich, detailed caption in 2-5 sentences covering instruments, arrangement, mood, tempo, genre, and production quality if audible. Use different sentence structures and vocabulary than a generic listing; lead with mix and genre, then instruments and tempo. Output ONLY the caption text. Do not write questions, dialogue, code, math, or any text after the caption."
echo "RESUME_SLOT2_FAST $(date -u +%FT%TZ) bs=16"
python gen_qwen_caption_10s_multisent.py \
  --ids_from_jsonl "$IDS" \
  --out_jsonl "$OUT/slot2_syntax.jsonl" \
  --seed 27182818 \
  --temperature 0.8 \
  --variant c2p0_syntax_leadmix_v1 \
  --prompt "$PROMPT2" \
  --max_new_tokens 160 \
  --batch_size 16 \
  --resume
echo "FULL_GEN_DONE $(date -u +%FT%TZ)"
/home/kojiek/research/meanaudio_training/caption10s_pipeline/run_c2p0_qwen3cap_k3_s2_after_gen.sh
echo "CHAIN_DONE $(date -u +%FT%TZ)"
