#!/bin/bash
# Phase 3 Hard Filtering - Stage 2 重新訓練腳本（修復版）
# 目的: 用正確方法重新訓練 Stage 2（從 100k 繼續到 150k）

set -e

echo "========================================"
echo "Phase 3 Hard Filtering - Stage 2 重訓（修復版）"
echo "========================================"

# 配置路徑
DATA_DIR="/mnt/HDD/kojiek/meanaudio_backup/phase3_hard_filtering"
STAGE1_EXP="phase3_stage1_hard_10k"
STAGE2_EXP="phase3_stage2_hard_FIXED"

# 檢查 Stage 1 checkpoint
STAGE1_CKPT="./exps/$STAGE1_EXP/${STAGE1_EXP}_ckpt_last.pth"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ 錯誤: 找不到 Stage 1 checkpoint"
    echo "   預期位置: $STAGE1_CKPT"
    exit 1
fi

echo "✓ 找到 Stage 1 checkpoint: $STAGE1_CKPT"

# 啟動虛擬環境
source /home/kojiek/venvs/dac/bin/activate

# 切換到 MeanAudio 目錄
cd /home/kojiek/MeanAudio

# ============================================================================
# 修復 checkpoint
# ============================================================================

echo ""
echo "🔧 修復 checkpoint（新增 r_embed 參數）..."

FIXED_CKPT="./exps/$STAGE1_EXP/stage2_checkpoint_fixed.pth"

python << 'PYTHON_EOF'
import torch

stage1_ckpt = "./exps/phase3_stage1_hard_10k/phase3_stage1_hard_10k_ckpt_last.pth"
fixed_ckpt = "./exps/phase3_stage1_hard_10k/stage2_checkpoint_fixed.pth"

print(f"載入 Stage 1 checkpoint: {stage1_ckpt}")
ckpt = torch.load(stage1_ckpt, map_location='cpu')

print(f"\nCheckpoint 資訊:")
print(f"  訓練步數: {ckpt.get('it', 'N/A')}")
print(f"  參數數量: {len(ckpt['weights'])}")

# 複製 t_embed → r_embed
print(f"\n修復 weights...")
weights = ckpt['weights']
added = 0

for key in list(weights.keys()):
    if 't_embed' in key:
        r_key = key.replace('t_embed', 'r_embed')
        if r_key not in weights:
            weights[r_key] = weights[key].clone()
            added += 1

print(f"  ✓ 新增 {added} 個 r_embed 參數")
print(f"  ✓ 修復後總參數: {len(weights)}")

# 保存
torch.save(ckpt, fixed_ckpt)
print(f"\n✅ 已儲存: {fixed_ckpt}")

# 驗證
verify = torch.load(fixed_ckpt, map_location='cpu')
checks = {
    'r_embed': any('r_embed' in k for k in verify['weights'].keys()),
    'optimizer': 'optimizer' in verify,
    'scheduler': 'scheduler' in verify,
    'iteration': 'it' in verify
}

print(f"\n驗證:")
for name, ok in checks.items():
    print(f"  {'✅' if ok else '❌'} {name}")

if not all(checks.values()):
    print(f"\n❌ 驗證失敗")
    exit(1)
PYTHON_EOF

if [ ! -f "$FIXED_CKPT" ]; then
    echo "❌ Checkpoint 修復失敗"
    exit 1
fi

# ============================================================================
# 訓練配置
# ============================================================================

echo ""
echo "訓練配置:"
echo "  模型: meanaudio_s"
echo "  資料: Hard Filtering (29,967 samples)"
echo "  起始步數: 100,000"
echo "  目標步數: 150,000"
echo "  實際訓練: 50,000 步"
echo ""
echo "與原版差異:"
echo "  原版: 從 0 訓練到 150k（共 150k 步）"
echo "  修復版: 從 100k 訓練到 150k（共 50k 步）✅"
echo ""

# ============================================================================
# 執行訓練
# ============================================================================

echo "🚀 開始訓練..."
echo "   預計時間: ~2 小時"
echo ""

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 train.py \
    data=AudioCaps_npz \
    data.AudioCaps_npz.tsv="$DATA_DIR/phase3_hard_filtering.tsv" \
    data.AudioCaps_npz.npz_dir="$DATA_DIR/npz" \
    model=meanaudio_s \
    text_encoder_name=t5_clap \
    exp_id="$STAGE2_EXP" \
    checkpoint="$FIXED_CKPT" \
    num_iterations=150000 \
    batch_size=8 \
    learning_rate=0.0001 \
    lr_schedule=step \
    lr_schedule_steps=[120000,135000] \
    lr_schedule_gamma=0.1 \
    save_weights_interval=10000 \
    save_checkpoint_interval=10000 \
    val_interval=5000 \
    use_meanflow=True \
    ema.enable=True \
    ema.sigma_rels=[0.05,0.1] \
    ema.checkpoint_every=10000

echo ""
echo "========================================"
echo "✅ Hard Filtering Stage 2 重訓完成"
echo "========================================"
echo ""
echo "輸出位置: ./exps/$STAGE2_EXP/"
echo "EMA 模型: ./exps/$STAGE2_EXP/ema_ckpts/0.150000.pt"
echo ""
