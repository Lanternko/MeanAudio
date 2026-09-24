#!/usr/bin/env python3
"""
RTX 5090 相容性測試腳本
"""
import torch
import sys

print("="*70)
print("RTX 5090 + PyTorch 相容性測試")
print("="*70)

# 1. 基本資訊
print(f"\n【環境資訊】")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 編譯版本: {torch.version.cuda}")
print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("❌ CUDA 不可用")
    sys.exit(1)

print(f"GPU 數量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# 2. RTX 5090 張量測試
print(f"\n【RTX 5090 張量運算測試】")
try:
    device = torch.device('cuda:0')
    
    # 測試基本運算
    x = torch.randn(1000, 1000, device=device)
    y = torch.mm(x, x)
    print(f"✅ 矩陣乘法成功")
    
    # 測試 bfloat16（5090 優化）
    x_bf16 = x.to(torch.bfloat16)
    y_bf16 = torch.mm(x_bf16, x_bf16)
    print(f"✅ bfloat16 運算成功")
    
    # 測試 mixed precision
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        z = torch.mm(x, x)
    print(f"✅ 混合精度運算成功")
    
    print(f"\n🎉 RTX 5090 完全可用！")
    
except RuntimeError as e:
    if "no kernel image is available" in str(e):
        print(f"❌ PyTorch 不支援 RTX 5090 架構")
        print(f"   錯誤: {e}")
        print(f"\n建議：")
        print(f"   1. 升級 PyTorch: pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu124")
        print(f"   2. 或使用 RTX 4090: CUDA_VISIBLE_DEVICES=1 python script.py")
    else:
        print(f"❌ 其他 CUDA 錯誤: {e}")
    sys.exit(1)

# 3. MeanAudio 模擬測試
print(f"\n【MeanAudio 模擬測試】")
try:
    # 模擬 MeanAudio 的典型運算
    batch_size = 1
    seq_len = 312  # MeanAudio 的 token 長度
    hidden_dim = 448
    
    # 模擬 transformer 運算
    q = torch.randn(batch_size, 8, seq_len, hidden_dim // 8, device=device)
    k = torch.randn(batch_size, 8, seq_len, hidden_dim // 8, device=device)
    v = torch.randn(batch_size, 8, seq_len, hidden_dim // 8, device=device)
    
    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    print(f"✅ Transformer attention 成功")
    
    # 模擬 convolution
    audio_latent = torch.randn(batch_size, 20, seq_len, device=device)
    conv = torch.nn.Conv1d(20, 448, kernel_size=7, padding=3).to(device)
    out = conv(audio_latent)
    print(f"✅ 1D Convolution 成功")
    
    print(f"\n🎉 MeanAudio 相關運算全部成功！")
    print(f"   RTX 5090 可以運行 MeanAudio")
    
except Exception as e:
    print(f"❌ MeanAudio 模擬失敗: {e}")
    sys.exit(1)

print(f"\n" + "="*70)
print(f"✅ 測試完成：RTX 5090 完全相容")
print(f"="*70)
