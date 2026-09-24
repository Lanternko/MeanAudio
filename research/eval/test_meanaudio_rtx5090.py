#!/usr/bin/env python3
"""
MeanAudio RTX 5090 完整測試
遵循 Music Research Coding Principles v1.2
"""
import sys
import os
import torch
import time
from pathlib import Path

# 添加 MeanAudio 到路徑
sys.path.insert(0, '/home/kojiek/MeanAudio')

print("="*70)
print("MeanAudio RTX 5090 完整測試")
print("="*70)

# 1. 環境檢查
print(f"\n【環境檢查】")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 2. 載入 MeanAudio 模型
print(f"\n【載入 MeanAudio 模型】")
from meanaudio.model.networks import MeanAudio
from meanaudio.utils.audio_transform import MelSpectrogram
import hydra
from omegaconf import OmegaConf

device = torch.device('cuda:0')

# 載入配置
cfg_path = "/home/kojiek/MeanAudio/config/mean_s_full.yaml"
cfg = OmegaConf.load(cfg_path)

# 初始化模型
model = MeanAudio(
    dit_type=cfg.model.dit_type,
    condition_type=cfg.model.condition_type,
    num_steps=1  # 單步生成（最快）
).to(device)

# 載入權重
model_path = "/home/kojiek/MeanAudio/weights/meanaudio_s_full.pth"
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_weights(checkpoint)
model.eval()

print(f"✅ 模型載入成功")
print(f"   參數量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# 3. 測試生成
print(f"\n【測試音頻生成】")
test_prompts = [
    "a dog barking",
    "piano playing classical music",
    "ocean waves on beach",
    "thunderstorm with rain",
    "electric guitar rock music"
]

output_dir = Path("rtx5090_test_outputs")
output_dir.mkdir(exist_ok=True)

results = []

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[{i}/{len(test_prompts)}] 生成: {prompt}")
    
    try:
        start_time = time.time()
        
        # 生成音頻（使用模型的 sample 方法）
        with torch.no_grad():
            # 準備 prompt
            # 這裡需要根據 MeanAudio 的實際 API 調整
            audio = model.sample(
                prompt=prompt,
                num_steps=1,
                cfg_scale=3.0
            )
        
        elapsed = time.time() - start_time
        
        # 保存音頻
        import soundfile as sf
        output_path = output_dir / f"test_{i}_{prompt.replace(' ', '_')}.wav"
        sf.write(output_path, audio.cpu().numpy().squeeze(), 44100)
        
        # 計算 RTF
        audio_duration = len(audio.squeeze()) / 44100
        rtf = elapsed / audio_duration
        
        result = {
            'prompt': prompt,
            'success': True,
            'time': elapsed,
            'rtf': rtf,
            'output': str(output_path)
        }
        
        print(f"   ✅ 成功")
        print(f"      時間: {elapsed:.2f}s")
        print(f"      RTF: {rtf:.4f}")
        print(f"      輸出: {output_path}")
        
    except Exception as e:
        print(f"   ❌ 失敗: {e}")
        result = {
            'prompt': prompt,
            'success': False,
            'error': str(e)
        }
    
    results.append(result)

# 4. 統計報告
print(f"\n" + "="*70)
print(f"【測試結果統計】")
print(f"="*70)

success_count = sum(1 for r in results if r.get('success', False))
print(f"成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

if success_count > 0:
    avg_time = sum(r['time'] for r in results if r.get('success')) / success_count
    avg_rtf = sum(r['rtf'] for r in results if r.get('success')) / success_count
    print(f"平均生成時間: {avg_time:.2f}s")
    print(f"平均 RTF: {avg_rtf:.4f}")
    print(f"論文宣稱 RTF: 0.013")
    print(f"性能比較: {'✅ 符合' if avg_rtf < 0.02 else '⚠️ 較慢'}")

# 保存結果
import json
results_file = output_dir / "test_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'test_info': {
            'gpu': torch.cuda.get_device_name(0),
            'pytorch': torch.__version__,
            'cuda': torch.version.cuda
        },
        'results': results
    }, f, indent=2)

print(f"\n結果已保存: {results_file}")
print(f"音頻輸出: {output_dir}/")
print("="*70)
