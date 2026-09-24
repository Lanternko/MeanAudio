#!/usr/bin/env python3
"""
MeanAudio RTF 基準測試 (RTX 5090) - 修復版
測試不同 NFE 配置下的 Real-Time Factor

遵循 Music Research Coding Principles v1.2
位置: ~/research_dev/meanaudio_test/ (開發測試區)
"""
import subprocess
import time
import json
import re
from pathlib import Path
import librosa

print("="*70)
print("MeanAudio RTF 基準測試 (RTX 5090)")
print("="*70)

# 測試配置
nfe_configs = [1, 8, 32]  # 減少測試點，專注關鍵配置
test_prompts = [
    "piano playing classical music",
    "a dog barking",
    "ocean waves on beach"
]

output_dir = Path("rtf_test_results")
output_dir.mkdir(exist_ok=True)

results = []

# 預熱運行（避免冷啟動影響）
print("\n預熱運行...")
subprocess.run(
    ["python", "/home/kojiek/MeanAudio/demo.py", "--prompt", "test", "--num_steps", "1"],
    capture_output=True,
    cwd="/home/kojiek/MeanAudio",
    timeout=60
)
print("✅ 預熱完成\n")

for nfe in nfe_configs:
    print(f"\n{'='*70}")
    print(f"測試 NFE = {nfe}")
    print(f"{'='*70}")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt}")
        
        try:
            start_time = time.time()
            
            # 調用 demo.py
            cmd = [
                "python",
                "/home/kojiek/MeanAudio/demo.py",
                "--prompt", prompt,
                "--num_steps", str(nfe)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd="/home/kojiek/MeanAudio"
            )
            
            wall_time = time.time() - start_time
            
            if result.returncode == 0:
                # 提取記憶體使用（修復正則表達式）
                memory_match = re.search(r'Memory usage:\s*([\d.]+)\s*GB', result.stdout)
                memory_gb = float(memory_match.group(1)) if memory_match else None
                
                # 找到生成的檔案
                meanaudio_output = Path("/home/kojiek/MeanAudio/output")
                wav_files = sorted(meanaudio_output.glob("*.wav"), key=lambda x: x.stat().st_mtime)
                
                if wav_files:
                    latest_file = wav_files[-1]
                    
                    # 計算音頻時長
                    audio, sr = librosa.load(str(latest_file), sr=None)
                    audio_duration = len(audio) / sr
                    
                    # 計算 RTF
                    rtf = wall_time / audio_duration
                    
                    print(f"   ✅ 成功")
                    print(f"      生成時間: {wall_time:.3f}s")
                    print(f"      音頻時長: {audio_duration:.3f}s")
                    print(f"      RTF: {rtf:.4f}")
                    if memory_gb:
                        print(f"      記憶體: {memory_gb:.2f} GB")
                    
                    result_data = {
                        'nfe': nfe,
                        'prompt': prompt,
                        'success': True,
                        'wall_time': wall_time,
                        'audio_duration': audio_duration,
                        'rtf': rtf,
                        'memory_gb': memory_gb,
                        'audio_file': latest_file.name
                    }
                else:
                    print(f"   ⚠️ 找不到輸出檔案")
                    result_data = {
                        'nfe': nfe,
                        'prompt': prompt,
                        'success': False,
                        'error': 'No output file found'
                    }
            else:
                print(f"   ❌ 失敗")
                error_msg = result.stderr[:300] if result.stderr else "Unknown error"
                print(f"      錯誤: {error_msg}")
                result_data = {
                    'nfe': nfe,
                    'prompt': prompt,
                    'success': False,
                    'error': error_msg
                }
                
        except subprocess.TimeoutExpired:
            print(f"   ⏱️ 超時 (>120s)")
            result_data = {
                'nfe': nfe,
                'prompt': prompt,
                'success': False,
                'error': 'Timeout'
            }
        except Exception as e:
            print(f"   ❌ 例外: {e}")
            result_data = {
                'nfe': nfe,
                'prompt': prompt,
                'success': False,
                'error': str(e)
            }
        
        results.append(result_data)
        time.sleep(2)  # 避免過快

# 統計分析
print(f"\n{'='*70}")
print(f"RTF 基準測試結果摘要")
print(f"{'='*70}")

# 按 NFE 分組統計
nfe_stats = {}
for nfe in nfe_configs:
    nfe_results = [r for r in results if r.get('nfe') == nfe and r.get('success')]
    
    if nfe_results:
        rtfs = [r['rtf'] for r in nfe_results]
        avg_rtf = sum(rtfs) / len(rtfs)
        avg_time = sum(r['wall_time'] for r in nfe_results) / len(rtfs)
        
        # 修復：安全處理 memory_gb
        memory_values = [r['memory_gb'] for r in nfe_results if r.get('memory_gb') is not None]
        avg_memory = sum(memory_values) / len(memory_values) if memory_values else None
        
        nfe_stats[nfe] = {
            'avg_rtf': avg_rtf,
            'avg_time': avg_time,
            'avg_memory': avg_memory,
            'success_count': len(nfe_results)
        }
        
        print(f"\nNFE = {nfe:2d}:")
        print(f"  平均 RTF: {avg_rtf:.4f}")
        print(f"  平均時間: {avg_time:.3f}s")
        if avg_memory:
            print(f"  平均記憶體: {avg_memory:.2f} GB")
        print(f"  成功數: {len(nfe_results)}/{len(test_prompts)}")

# 與論文比較
print(f"\n{'='*70}")
print(f"與論文結果比較")
print(f"{'='*70}")
print(f"論文宣稱 (A100, NFE=8): RTF = 0.013")

if 8 in nfe_stats:
    rtx5090_rtf = nfe_stats[8]['avg_rtf']
    print(f"RTX 5090 (NFE=8): RTF = {rtx5090_rtf:.4f}")
    speedup = 0.013 / rtx5090_rtf if rtx5090_rtf > 0 else 0
    print(f"速度比較: {'✅ 更快' if speedup > 1 else '⚠️ 較慢'} ({speedup:.2f}x)")
    print(f"\n⚠️ 注意: RTF 明顯高於論文，可能原因:")
    print(f"   1. 測試包含模型載入時間（冷啟動）")
    print(f"   2. 論文可能使用批次推論優化")
    print(f"   3. A100 vs RTX 5090 架構差異")

# 保存完整結果
results_file = output_dir / "rtf_baseline_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'test_info': {
            'gpu': 'NVIDIA GeForce RTX 5090',
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'nfe_configs': nfe_configs,
            'test_prompts': test_prompts,
            'note': 'Includes warmup run to avoid cold start'
        },
        'nfe_statistics': nfe_stats,
        'detailed_results': results
    }, f, indent=2)

print(f"\n完整結果已保存: {results_file}")
print("="*70)
