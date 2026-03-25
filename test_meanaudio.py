import torch
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {torch.cuda.get_device_name(0)}")

# 重新載入，並確保 trust_remote_code 開啟
print("🚀 Re-loading MeanAudio model (forcing clean download)...")
try:
    model = AutoModel.from_pretrained(
        "AndreasXi/MeanAudio", 
        trust_remote_code=True,
        force_download=True  # 強制重新下載
    )
    model.to(device)
    model.eval()
    
    # 5090 優化
    torch.set_float32_matmul_precision('high')
    
    dummy_input = torch.randn(1, 1, 44100).to(device)
    print("🧪 Running inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 根據 MeanAudio 的實作，可能需要調用 encode
            output = model.encode(dummy_input)
            print(f"🎉 Success! Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Error during execution: {e}")

print("✨ Test Finished!")
