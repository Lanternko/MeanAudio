#!/bin/bash

EMA_MODEL="./exps/phase3_stage2_hard/ema_ckpts/0.150000.pt"
OUTPUT_DIR="./phase3_systematic_eval"
mkdir -p $OUTPUT_DIR

# 鋼琴測試 (10 個樣本)
PIANO_PROMPTS=(
    "A solo piano playing a gentle melody"
    "Grand piano performing classical music"
    "Piano solo with emotional expression"
    "Upbeat piano music with clear rhythm"
    "Slow piano ballad with sustained notes"
    "Jazz piano improvisation"
    "Piano playing romantic melody"
    "Fast piano performance with complex patterns"
    "Minimalist piano composition"
    "Piano with rich harmonic progression"
)

# 吉他測試 (10 個樣本)
GUITAR_PROMPTS=(
    "Acoustic guitar playing fingerstyle"
    "Electric guitar with clean tone"
    "Classical guitar solo performance"
    "Flamenco guitar with rapid strumming"
    "Blues guitar with expressive bends"
    "Acoustic guitar playing gentle chords"
    "Spanish guitar with passionate melody"
    "Guitar playing melodic arpeggios"
    "Soft acoustic guitar background music"
    "Guitar solo with clear articulation"
)

# EDM 測試 (10 個樣本)
EDM_PROMPTS=(
    "Electronic dance music with strong beat"
    "Upbeat EDM track with synthesizer"
    "High-energy electronic music"
    "EDM with pulsing bassline"
    "Electronic music with driving rhythm"
    "Dance music with electronic sounds"
    "Energetic EDM with kick drum"
    "Electronic track with synthetic textures"
    "Club music with electronic beats"
    "Fast-paced electronic dance music"
)

echo "生成鋼琴樣本..."
for i in {0..9}; do
    python infer.py \
        --variant meanaudio_s \
        --model_path "$EMA_MODEL" \
        --encoder_name t5_clap \
        --text_c_dim 512 \
        --prompt "${PIANO_PROMPTS[$i]}" \
        --output "${OUTPUT_DIR}/piano_${i}" \
        --duration 10 \
        --num_steps 1 \
        --use_meanflow \
        --cfg_strength 3.0 \
        --seed $((42 + $i))
    echo "鋼琴樣本 $i 完成"
done

echo "生成吉他樣本..."
for i in {0..9}; do
    python infer.py \
        --variant meanaudio_s \
        --model_path "$EMA_MODEL" \
        --encoder_name t5_clap \
        --text_c_dim 512 \
        --prompt "${GUITAR_PROMPTS[$i]}" \
        --output "${OUTPUT_DIR}/guitar_${i}" \
        --duration 10 \
        --num_steps 1 \
        --use_meanflow \
        --cfg_strength 3.0 \
        --seed $((42 + $i))
    echo "吉他樣本 $i 完成"
done

echo "生成 EDM 樣本..."
for i in {0..9}; do
    python infer.py \
        --variant meanaudio_s \
        --model_path "$EMA_MODEL" \
        --encoder_name t5_clap \
        --text_c_dim 512 \
        --prompt "${EDM_PROMPTS[$i]}" \
        --output "${OUTPUT_DIR}/edm_${i}" \
        --duration 10 \
        --num_steps 1 \
        --use_meanflow \
        --cfg_strength 3.0 \
        --seed $((42 + $i))
    echo "EDM 樣本 $i 完成"
done

echo "✅ 全部 30 個樣本生成完成！"
ls -lh $OUTPUT_DIR/*.wav
