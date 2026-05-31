#!/usr/bin/env python3
"""
Phase 3 Stage 2 訓練錯誤診斷與修復腳本
診斷 KeyError: 't_embed.mlp.0.weight' 問題並提供解決方案
"""

import torch
import sys
from pathlib import Path

def diagnose_stage1_weights(weights_path):
    """診斷 Stage 1 權重檔案"""
    print("=" * 80)
    print("🔍 Stage 1 權重檔案診斷")
    print("=" * 80)
    
    if not Path(weights_path).exists():
        print(f"❌ 權重檔案不存在: {weights_path}")
        return False
    
    # 載入權重
    weights = torch.load(weights_path, map_location='cpu')
    
    # 檢查結構
    all_keys = list(weights.keys())
    t_embed_keys = [k for k in all_keys if 't_embed' in k]
    r_embed_keys = [k for k in all_keys if 'r_embed' in k]
    
    print(f"\n📊 權重檔案統計:")
    print(f"   總 keys 數量: {len(all_keys)}")
    print(f"   t_embed keys: {len(t_embed_keys)}")
    print(f"   r_embed keys: {len(r_embed_keys)}")
    
    # 顯示一些 t_embed keys
    print(f"\n✓ t_embed 相關的 keys（前 10 個）:")
    for k in t_embed_keys[:10]:
        print(f"     {k}")
    
    # 檢查 r_embed
    if r_embed_keys:
        print(f"\n✓ r_embed 相關的 keys（前 10 個）:")
        for k in r_embed_keys[:10]:
            print(f"     {k}")
        print(f"\n✅ Stage 1 權重已包含 r_embed，可直接使用")
        return True
    else:
        print(f"\n⚠️ Stage 1 權重缺少 r_embed")
        print(f"   這表示 Stage 1 使用的是標準 Flow Matching")
        print(f"   需要複製 t_embed 權重給 r_embed")
        return False

def create_stage2_compatible_weights(stage1_path, output_path):
    """創建 Stage 2 相容的權重檔案"""
    print("\n" + "=" * 80)
    print("🔧 創建 Stage 2 相容權重")
    print("=" * 80)
    
    # 載入 Stage 1 權重
    weights = torch.load(stage1_path, map_location='cpu')
    
    # 檢查是否已有 r_embed
    has_r_embed = any('r_embed' in k for k in weights.keys())
    
    if has_r_embed:
        print("✅ 權重已包含 r_embed，直接複製")
        torch.save(weights, output_path)
    else:
        print("⚙️ 複製 t_embed 權重給 r_embed...")
        
        # 創建新的權重字典
        new_weights = {}
        
        for key, value in weights.items():
            # 保留原始的 key
            new_weights[key] = value
            
            # 如果是 t_embed，也創建對應的 r_embed
            if 't_embed' in key:
                r_key = key.replace('t_embed', 'r_embed')
                new_weights[r_key] = value.clone()
                print(f"   ✓ {key} → {r_key}")
        
        # 儲存新權重
        torch.save(new_weights, output_path)
        print(f"\n✅ 已儲存相容權重到: {output_path}")
        
        # 驗證
        verify_weights = torch.load(output_path, map_location='cpu')
        t_count = sum(1 for k in verify_weights.keys() if 't_embed' in k)
        r_count = sum(1 for k in verify_weights.keys() if 'r_embed' in k)
        print(f"   驗證: t_embed={t_count}, r_embed={r_count}")

def main():
    """主函數"""
    print("\n" + "🚀 Phase 3 Stage 2 錯誤診斷與修復工具\n")
    
    # 設定路徑
    stage1_weights = "./exps/phase3_stage1_hard_10k/stage1_clean_weights.pth"
    compatible_weights = "./exps/phase3_stage1_hard_10k/stage2_compatible_weights.pth"
    
    # Step 1: 診斷
    has_r_embed = diagnose_stage1_weights(stage1_weights)
    
    # Step 2: 創建相容權重
    if not has_r_embed:
        create_stage2_compatible_weights(stage1_weights, compatible_weights)
        final_weights_path = compatible_weights
    else:
        final_weights_path = stage1_weights
    
    # Step 3: 生成訓練指令
    print("\n" + "=" * 80)
    print("📝 Stage 2 訓練指令")
    print("=" * 80)
    
    training_cmd = f"""
cd ~/MeanAudio
export CUDA_VISIBLE_DEVICES=0

PHASE3_DIR="/home/kojiek/research_dev/meanaudio_training/phase3_hard_filtering"
TSV_PATH="${{PHASE3_DIR}}/phase3_hard_filtering.tsv"
NPZ_DIR="${{PHASE3_DIR}}/npz"

torchrun --standalone --nproc_per_node=1 train.py \\
    --config-name train_config.yaml \\
    exp_id=phase3_stage2_hard \\
    model=meanaudio_s \\
    batch_size=8 \\
    num_iterations=150000 \\
    val_interval=5000 \\
    save_checkpoint_interval=10000 \\
    save_weights_interval=10000 \\
    learning_rate=0.0001 \\
    data.AudioCaps_npz.tsv=$TSV_PATH \\
    data.AudioCaps_npz.npz_dir=$NPZ_DIR \\
    data.AudioCaps_val_npz.tsv=$TSV_PATH \\
    data.AudioCaps_val_npz.npz_dir=$NPZ_DIR \\
    data.AudioCaps_test_npz.tsv=$TSV_PATH \\
    data.AudioCaps_test_npz.npz_dir=$NPZ_DIR \\
    weights={final_weights_path} \\
    ++use_rope=False \\
    ++use_wandb=False \\
    ++ema.enable=True \\
    ++use_meanflow=True
"""
    
    print(training_cmd)
    
    print("\n" + "=" * 80)
    print("✅ 診斷完成！")
    print("=" * 80)
    print("\n📋 下一步操作:")
    print("   1. 在伺服器上執行: python3 phase3_stage2_fix.py")
    print(f"   2. 使用生成的訓練指令啟動 Stage 2")
    print(f"   3. 檢查訓練是否正常開始\n")

if __name__ == "__main__":
    main()