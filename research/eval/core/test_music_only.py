#!/usr/bin/env python3
"""
MeanAudio 音樂生成測試（僅音樂類 prompt）
避免音效類 prompt，專注評估音樂生成能力
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
print("MeanAudio 音樂生成評估（音樂類 Prompt）")
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

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ 模型載入完成")

# 音樂類 Prompts（避免音效）
music_prompts = [
    "piano playing classical music",
    "acoustic guitar folk song",
    "jazz saxophone melody",
    "electronic dance music with synthesizer",
    "orchestral symphony with strings",
    "rock guitar with drums",
    "blues harmonica solo",
    "reggae rhythm with bass"
]

output_dir = Path("music_only_test")
output_dir.mkdir(exist_ok=True)

results = []

for i, prompt in enumerate(music_prompts, 1):
    print(f"\n[{i}/{len(music_prompts)}] Prompt: {prompt}")
    
    # 生成音頻
    print(f"   步驟 1: 生成音頻...")
    cmd = [
        "python", "/home/kojiek/MeanAudio/demo.py",
        "--prompt", prompt,
        "--num_steps", "8"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, 
                          cwd="/home/kojiek/MeanAudio", timeout=60)
    
    if result.returncode != 0:
        print(f"   ❌ 生成失敗")
        results.append({'prompt': prompt, 'success': False, 'error': 'Generation failed'})
        continue
    
    # 找到檔案
    meanaudio_output = Path("/home/kojiek/MeanAudio/output")
    wav_files = sorted(meanaudio_output.glob("*.wav"), key=lambda x: x.stat().st_mtime)
    
    if not wav_files:
        print(f"   ❌ 找不到檔案")
        results.append({'prompt': prompt, 'success': False, 'error': 'No output'})
        continue
    
    audio_file = wav_files[-1]
    print(f"   ✅ 音頻生成: {audio_file.name}")
    
    # LP-MusicCaps 分析
    print(f"   步驟 2: LP-MusicCaps 分析...")
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
        
        print(f"   ✅ Caption: {generated_caption}")
        
        # 計算相似度
        embeddings = semantic_model.encode([prompt, generated_caption])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        print(f"   📊 語義相似度: {similarity:.4f}")
        
        results.append({
            'prompt': prompt,
            'generated_caption': generated_caption,
            'semantic_similarity': float(similarity),
            'audio_file': audio_file.name,
            'success': True
        })
        
    except Exception as e:
        print(f"   ❌ 分析失敗: {e}")
        results.append({'prompt': prompt, 'success': False, 'error': str(e)})
    
    time.sleep(2)

# 統計
print(f"\n{'='*70}")
print(f"音樂生成評估結果")
print(f"{'='*70}")

successful = [r for r in results if r.get('success')]
if successful:
    sims = [r['semantic_similarity'] for r in successful]
    avg_sim = sum(sims) / len(sims)
    
    print(f"成功數: {len(successful)}/{len(music_prompts)}")
    print(f"平均語義相似度: {avg_sim:.4f}")
    print(f"最高: {max(sims):.4f}")
    print(f"最低: {min(sims):.4f}")
    
    # 分級
    high_quality = len([s for s in sims if s >= 0.6])
    medium_quality = len([s for s in sims if 0.4 <= s < 0.6])
    low_quality = len([s for s in sims if s < 0.4])
    
    print(f"\n品質分布:")
    print(f"  高品質 (≥0.6): {high_quality} ({high_quality/len(sims)*100:.1f}%)")
    print(f"  中品質 (0.4-0.6): {medium_quality} ({medium_quality/len(sims)*100:.1f}%)")
    print(f"  低品質 (<0.4): {low_quality} ({low_quality/len(sims)*100:.1f}%)")
    
    # 排名
    sorted_results = sorted(successful, key=lambda x: x['semantic_similarity'], reverse=True)
    print(f"\n相似度排名:")
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {r['semantic_similarity']:.4f} - {r['prompt']}")

# 保存
results_file = output_dir / "music_generation_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'test_info': {
            'focus': 'Music generation only (no sound effects)',
            'model': 'MeanAudio + LP-MusicCaps',
            'gpu': 'RTX 5090',
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'statistics': {
            'avg_similarity': avg_sim if successful else 0,
            'high_quality_ratio': high_quality/len(sims) if successful else 0,
            'success_count': len(successful),
            'total_tests': len(music_prompts)
        },
        'results': results
    }, f, indent=2)

print(f"\n結果已保存: {results_file}")
print("="*70)
