#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

PROMPTS=(
    "piano playing classical music"
    "acoustic guitar strumming"
    "electronic dance music with strong beat"
)

# 定義要測試的權重路徑
CKPTS=(
    "./exps/phase1_baseline_test/weights/model_1000.pt"
    "./exps/phase1_baseline_test/weights/model_3000.pt"
    "./exps/phase1_baseline_test/phase1_baseline_test_last.pth"
)

# 定義輸出的子資料夾名稱
TAGS=("step1000" "step3000" "step5000")

for i in "${!CKPTS[@]}"; do
    ckpt="${CKPTS[$i]}"
    tag="${TAGS[$i]}"
    out_dir="./phase1_test_output/${tag}"
    
    # 確保權重檔存在才執行
    if [ ! -f "$ckpt" ]; then
        echo "⚠️ 找不到權重檔 $ckpt，跳過此階段。"
        continue
    fi
    
    echo ""
    echo "========================================"
    echo "🚀 測試 Checkpoint: $tag"
    echo "========================================"
    
    for prompt in "${PROMPTS[@]}"; do
        echo "➡️ Generating: $prompt"
        python infer.py \
            --variant meanaudio_s \
            --model_path "$ckpt" \
            --encoder_name t5_clap \
            --text_c_dim 512 \
            --prompt "$prompt" \
            --output "$out_dir" \
            --duration 10 \
            --num_steps 25 \
            --use_meanflow
    done
done

echo ""
echo "✅ 所有推論測試完成！請到 phase1_test_output/ 下查看結果。"
