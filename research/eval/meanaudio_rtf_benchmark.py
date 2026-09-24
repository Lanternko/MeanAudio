#!/usr/bin/env python3
"""
MeanAudio RTF 基準測試 - 生產級版本
直接調用 MeanAudioDemoInfer（而非 subprocess）實現最佳性能

測試目標：
1. 驗證 RTX 5090 在不同 NFE 配置下的 RTF 性能
2. 與論文基準比較（A100, NFE=8, RTF=0.013）
3. 為大規模音樂生成提供性能基準

遵循 Music Research Coding Principles v1.2
位置: ~/research_dev/meanaudio_test/meanaudio_rtf_benchmark.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# MeanAudio 路徑
sys.path.insert(0, '/home/kojiek/MeanAudio')

import torch
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
from tqdm import tqdm

# MeanAudio imports
from demo import MeanAudioDemoInfer

# 抑制過多的日誌
logging.getLogger().setLevel(logging.WARNING)


class MeanAudioRTFBenchmark:
    """MeanAudio RTF 基準測試類別"""
    
    def __init__(self, output_dir: str = "rtf_benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 測試配置
        self.nfe_configs = [1, 4, 8, 16, 32]  # 論文測試的範圍
        self.duration = 10  # 秒
        self.variant = "meanaudio_s"  # Small variant
        
        # 測試 prompts - 涵蓋音樂和自然聲音
        self.test_prompts = [
            # 音樂 (14 個)
            "piano playing classical music",
            "acoustic guitar folk song",
            "electric guitar rock solo",
            "violin playing romantic melody",
            "saxophone jazz improvisation",
            "drums playing fast rhythm",
            "electronic dance music beat",
            "orchestra playing symphony",
            "flute playing peaceful tune",
            "cello playing deep melody",
            "trumpet playing bright notes",
            "harp playing gentle music",
            "bass guitar playing groove",
            "synthesizer ambient music",
            # 自然聲音 (6 個)
            "a dog barking loudly",
            "ocean waves on beach",
            "thunderstorm with rain",
            "birds chirping in forest",
            "wind blowing through trees",
            "waterfall flowing down"
        ]
        
        # 結果存儲
        self.results = []
        self.model_load_time = 0.0
        
        print("="*80)
        print("MeanAudio RTF 基準測試 - RTX 5090")
        print("="*80)
        print(f"測試配置:")
        print(f"  - NFE 配置: {self.nfe_configs}")
        print(f"  - 音頻長度: {self.duration}s")
        print(f"  - 模型變體: {self.variant}")
        print(f"  - 測試 prompts: {len(self.test_prompts)} 個")
        print(f"  - 總測試數: {len(self.nfe_configs) * len(self.test_prompts)}")
        print(f"  - 輸出目錄: {self.output_dir}")
        print("="*80)
    
    def warmup_model(self):
        """模型預熱 - 第一次調用會載入模型"""
        print("\n🔥 模型預熱中...")
        
        warmup_start = time.time()
        
        try:
            # 執行一次生成來載入模型
            _ = MeanAudioDemoInfer(
                prompt="warmup test",
                num_steps=1,
                duration=1,
                variant=self.variant,
                output=str(self.output_dir / "warmup"),
                seed=42
            )
            
            self.model_load_time = time.time() - warmup_start
            
            print(f"✅ 模型預熱完成")
            print(f"   載入時間: {self.model_load_time:.2f}s")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            
            # 清理 warmup 輸出
            warmup_dir = self.output_dir / "warmup"
            if warmup_dir.exists():
                import shutil
                shutil.rmtree(warmup_dir)
            
            return True
            
        except Exception as e:
            print(f"❌ 模型預熱失敗: {e}")
            return False
    
    def test_single_generation(self, prompt: str, nfe: int, test_index: int) -> Dict:
        """測試單次生成"""
        
        output_path = self.output_dir / f"test_{test_index:04d}_nfe{nfe}"
        
        try:
            # 記錄生成時間（不包含模型載入）
            gen_start = time.time()
            
            result_path = MeanAudioDemoInfer(
                prompt=prompt,
                num_steps=nfe,
                duration=self.duration,
                variant=self.variant,
                output=str(output_path),
                seed=42,
                use_meanflow=False  # 使用 Flow Matching
            )
            
            generation_time = time.time() - gen_start
            
            # 計算 RTF
            rtf = generation_time / self.duration
            
            # 檢查 GPU 記憶體使用
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_allocated = 0
            
            return {
                'success': True,
                'prompt': prompt,
                'nfe': nfe,
                'generation_time': generation_time,
                'audio_duration': self.duration,
                'rtf': rtf,
                'memory_gb': memory_allocated,
                'output_path': str(result_path) if result_path else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'prompt': prompt,
                'nfe': nfe,
                'error': str(e)
            }
    
    def run_benchmark(self):
        """執行完整基準測試"""
        
        # 預熱
        if not self.warmup_model():
            print("❌ 無法繼續測試")
            return False
        
        print(f"\n🚀 開始 RTF 基準測試")
        print(f"{'='*80}\n")
        
        total_tests = len(self.nfe_configs) * len(self.test_prompts)
        test_index = 0
        
        # 按 NFE 分組測試
        for nfe_idx, nfe in enumerate(self.nfe_configs, 1):
            print(f"\n📊 NFE = {nfe} ({nfe_idx}/{len(self.nfe_configs)})")
            print(f"{'-'*80}")
            
            nfe_results = []
            
            # 進度條
            pbar = tqdm(self.test_prompts, desc=f"NFE={nfe}", leave=True)
            
            for prompt in pbar:
                test_index += 1
                
                # 執行測試
                result = self.test_single_generation(prompt, nfe, test_index)
                result['test_index'] = test_index
                result['timestamp'] = datetime.now().isoformat()
                
                nfe_results.append(result)
                self.results.append(result)
                
                # 更新進度條
                if result['success']:
                    pbar.set_postfix({
                        'RTF': f"{result['rtf']:.4f}",
                        'Time': f"{result['generation_time']:.2f}s"
                    })
                else:
                    pbar.set_postfix({'Status': '❌ Failed'})
                
                # 短暫暫停避免過熱
                time.sleep(0.5)
            
            # NFE 組別統計
            self.print_nfe_statistics(nfe, nfe_results)
        
        # 最終統計
        self.print_final_statistics()
        
        # 保存結果
        self.save_results()
        
        return True
    
    def print_nfe_statistics(self, nfe: int, nfe_results: List[Dict]):
        """打印單個 NFE 配置的統計"""
        
        successful = [r for r in nfe_results if r.get('success')]
        
        if not successful:
            print(f"   ❌ NFE={nfe}: 所有測試失敗")
            return
        
        rtfs = [r['rtf'] for r in successful]
        times = [r['generation_time'] for r in successful]
        
        print(f"\n   NFE = {nfe} 統計:")
        print(f"   - 成功率: {len(successful)}/{len(nfe_results)} ({len(successful)/len(nfe_results)*100:.1f}%)")
        print(f"   - 平均 RTF: {np.mean(rtfs):.4f} (±{np.std(rtfs):.4f})")
        print(f"   - RTF 範圍: {np.min(rtfs):.4f} - {np.max(rtfs):.4f}")
        print(f"   - 平均生成時間: {np.mean(times):.2f}s (±{np.std(times):.2f}s)")
    
    def print_final_statistics(self):
        """打印最終統計和與論文的比較"""
        
        print(f"\n{'='*80}")
        print(f"📊 最終統計摘要")
        print(f"{'='*80}")
        
        # 按 NFE 分組統計
        nfe_stats = {}
        for nfe in self.nfe_configs:
            nfe_results = [r for r in self.results if r.get('nfe') == nfe and r.get('success')]
            
            if nfe_results:
                rtfs = [r['rtf'] for r in nfe_results]
                times = [r['generation_time'] for r in nfe_results]
                
                nfe_stats[nfe] = {
                    'avg_rtf': np.mean(rtfs),
                    'std_rtf': np.std(rtfs),
                    'avg_time': np.mean(times),
                    'success_count': len(nfe_results),
                    'total_count': len([r for r in self.results if r.get('nfe') == nfe])
                }
        
        # 打印表格
        print(f"\n{'NFE':<6} {'平均 RTF':<12} {'標準差':<12} {'平均時間':<12} {'成功率':<10}")
        print(f"{'-'*60}")
        
        for nfe in self.nfe_configs:
            if nfe in nfe_stats:
                stats = nfe_stats[nfe]
                success_rate = stats['success_count'] / stats['total_count'] * 100
                
                print(f"{nfe:<6} {stats['avg_rtf']:<12.4f} {stats['std_rtf']:<12.4f} "
                      f"{stats['avg_time']:<12.2f} {success_rate:<10.1f}%")
        
        # 與論文比較
        print(f"\n{'='*80}")
        print(f"📈 與論文基準比較")
        print(f"{'='*80}")
        
        paper_rtf = 0.013  # 論文宣稱的 A100 NFE=8 RTF
        
        if 8 in nfe_stats:
            rtx5090_rtf = nfe_stats[8]['avg_rtf']
            
            print(f"\nNFE = 8 配置:")
            print(f"  論文 (A100):        RTF = {paper_rtf:.4f}")
            print(f"  RTX 5090:          RTF = {rtx5090_rtf:.4f}")
            
            if rtx5090_rtf < paper_rtf:
                speedup = paper_rtf / rtx5090_rtf
                print(f"  結果: ✅ RTX 5090 更快 ({speedup:.2f}x)")
            else:
                slowdown = rtx5090_rtf / paper_rtf
                print(f"  結果: ⚠️ RTX 5090 較慢 ({slowdown:.2f}x)")
            
            # 實時率評估
            print(f"\n  實時性能評估:")
            if rtx5090_rtf < 1.0:
                print(f"  ✅ 快於實時 ({1/rtx5090_rtf:.1f}x 實時速度)")
            else:
                print(f"  ❌ 慢於實時 ({rtx5090_rtf:.1f}x 實時時間)")
        else:
            print(f"  ⚠️ NFE=8 測試數據不可用")
        
        # NFE 效率分析
        print(f"\n{'='*80}")
        print(f"💡 NFE 效率分析")
        print(f"{'='*80}")
        
        if len(nfe_stats) >= 2:
            print(f"\nRTF 隨 NFE 變化:")
            for nfe in sorted(nfe_stats.keys()):
                rtf = nfe_stats[nfe]['avg_rtf']
                time_per_nfe = nfe_stats[nfe]['avg_time'] / nfe
                print(f"  NFE={nfe:2d}: RTF={rtf:.4f}, 每步耗時={time_per_nfe:.3f}s")
    
    def save_results(self):
        """保存詳細結果到 JSON"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"rtf_benchmark_{timestamp}.json"
        
        # 計算統計
        nfe_statistics = {}
        for nfe in self.nfe_configs:
            nfe_results = [r for r in self.results if r.get('nfe') == nfe and r.get('success')]
            
            if nfe_results:
                rtfs = [r['rtf'] for r in nfe_results]
                times = [r['generation_time'] for r in nfe_results]
                
                nfe_statistics[str(nfe)] = {
                    'avg_rtf': float(np.mean(rtfs)),
                    'std_rtf': float(np.std(rtfs)),
                    'min_rtf': float(np.min(rtfs)),
                    'max_rtf': float(np.max(rtfs)),
                    'avg_time': float(np.mean(times)),
                    'success_count': len(nfe_results),
                    'total_count': len([r for r in self.results if r.get('nfe') == nfe])
                }
        
        output_data = {
            'experiment_info': {
                'title': 'MeanAudio RTF 基準測試',
                'timestamp': timestamp,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'model_variant': self.variant,
                'audio_duration': self.duration,
                'nfe_configs': self.nfe_configs,
                'total_prompts': len(self.test_prompts),
                'total_tests': len(self.results),
                'model_load_time': self.model_load_time
            },
            'nfe_statistics': nfe_statistics,
            'paper_comparison': {
                'paper_gpu': 'A100',
                'paper_nfe': 8,
                'paper_rtf': 0.013,
                'our_gpu': 'RTX 5090',
                'our_nfe_8_rtf': nfe_statistics.get('8', {}).get('avg_rtf')
            },
            'detailed_results': self.results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"💾 完整結果已保存: {results_file}")
        print(f"📁 音頻輸出目錄: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    """主函數"""
    
    print("🎯 MeanAudio RTF 基準測試 - 生產級版本")
    print("基於直接調用 MeanAudioDemoInfer（非 subprocess）")
    print()
    
    # 檢查 CUDA
    if not torch.cuda.is_available():
        print("⚠️ 警告: CUDA 不可用，將使用 CPU（速度會很慢）")
        response = input("是否繼續? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # 創建測試器
    benchmark = MeanAudioRTFBenchmark()
    
    # 執行測試
    success = benchmark.run_benchmark()
    
    if success:
        print("✅ RTF 基準測試完成！")
        return 0
    else:
        print("❌ RTF 基準測試失敗")
        return 1


if __name__ == "__main__":
    exit(main())
