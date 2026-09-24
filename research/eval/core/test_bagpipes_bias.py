#!/usr/bin/env python3
"""
調查 LP-MusicCaps 的 Bagpipes 偏見
為何多個不同樂器都被判斷為風笛？
"""
import subprocess
import torch
import librosa
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
import time
import sys

sys.path.insert(0, '/home/kojiek/lp-music-caps')
from lpmc.music_captioning.model.bart import BartCaptionModel

print("="*70)
print("調查：LP-MusicCaps Bagpipes 偏見")
print("="*70)

# 載入模型
device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n載入 LP-MusicCaps...")
lpmc_model = BartCaptionModel(bart_type="facebook/bart-base").to(device)
ckpt_path = "/home/kojiek/lp-music-caps/exp/transfer/lp_music_caps/last.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
new_state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in state_dict.items()}
lpmc_model.load_state_dict(new_state_dict)
lpmc_model.eval()

print("✅ 模型載入完成")

# 測試不同樂器
instruments = [
    "flute solo",
    "clarinet melody",
    "oboe performance",
    "trumpet fanfare",
    "trombone solo",
    "violin playing",
    "cello music",
    "accordion melody",
    "harmonica blues",
    "bagpipes playing"  # 真正的風笛作為對照
]

output_dir = Path("bagpipes_bias_test")
output_dir.mkdir(exist_ok=True)

results = []
bagpipes_count = 0

for i, prompt in enumerate(instruments, 1):
    print(f"\n[{i}/{len(instruments)}] 測試: {prompt}")
    
    # 生成音頻
    cmd = [
        "python", "/home/kojiek/MeanAudio/demo.py",
        "--prompt", prompt,
        "--num_steps", "8"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, 
                          cwd="/home/kojiek/MeanAudio", timeout=60)
    
    if result.returncode != 0:
        print(f"   ❌ 生成失敗")
        continue
    
    # 找到檔案
    meanaudio_output = Path("/home/kojiek/MeanAudio/output")
    wav_files = sorted(meanaudio_output.glob("*.wav"), key=lambda x: x.stat().st_mtime)
    
    if not wav_files:
        continue
    
    audio_file = wav_files[-1]
    
    # LP-MusicCaps 分析
    try:
        wav, sr = librosa.load(str(audio_file), sr=44100)
        wav_16k = librosa.resample(wav, orig_sr=44100, target_sr=16000)
        
        target_len = 16000 * 10
        if len(wav_16k) > target_len:
            wav_16k = wav_16k[:target_len]
        else:
            wav_16k = np.pad(wav_16k, (0, max(0, target_len - len(wav_16k))))
        
        input_tensor = torch.from_numpy(wav_16k).unsqueeze(0).to(device)
        
        with torch.no_grad():
            audio_embeds = lpmc_model.audio_encoder(input_tensor)
            output_ids = lpmc_model.bart.generate(
                inputs_embeds=audio_embeds,
                max_length=128,
                num_beams=4
            )
            generated_caption = lpmc_model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 檢查是否包含 "bagpipes"
        contains_bagpipes = "bagpipe" in generated_caption.lower()
        
        if contains_bagpipes:
            bagpipes_count += 1
            print(f"   🎵 Caption: {generated_caption}")
            print(f"   ⚠️ 被誤判為 Bagpipes!")
        else:
            print(f"   ✅ Caption: {generated_caption}")
        
        results.append({
            'prompt': prompt,
            'caption': generated_caption,
            'contains_bagpipes': contains_bagpipes,
            'is_actual_bagpipes': 'bagpipe' in prompt.lower()
        })
        
    except Exception as e:
        print(f"   ❌ 錯誤: {e}")
    
    time.sleep(2)

# 統計分析
print(f"\n{'='*70}")
print(f"Bagpipes 偏見分析結果")
print(f"{'='*70}")

non_bagpipes_prompts = [r for r in results if not r['is_actual_bagpipes']]
bagpipes_misclassified = [r for r in non_bagpipes_prompts if r['contains_bagpipes']]

print(f"非風笛樂器數量: {len(non_bagpipes_prompts)}")
print(f"被誤判為風笛: {len(bagpipes_misclassified)}")
print(f"誤判率: {len(bagpipes_misclassified)/len(non_bagpipes_prompts)*100:.1f}%")

print(f"\n被誤判為風笛的樂器:")
for r in bagpipes_misclassified:
    print(f"  • {r['prompt']}")

# 保存結果
results_file = output_dir / "bagpipes_bias_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'experiment': 'LP-MusicCaps Bagpipes Bias Investigation',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'statistics': {
            'total_instruments': len(non_bagpipes_prompts),
            'misclassified_as_bagpipes': len(bagpipes_misclassified),
            'misclassification_rate': len(bagpipes_misclassified)/len(non_bagpipes_prompts) if non_bagpipes_prompts else 0
        },
        'results': results
    }, f, indent=2)

print(f"\n結果已保存: {results_file}")
print("="*70)
