#!/usr/bin/env python3
"""
MeanAudio RTF 基準測試 - 修正版
直接使用 MeanAudio 內部 API，實現真正的模型重用

關鍵改進：
1. 模型只載入一次
2. 所有組件重用
3. 避免重複初始化
4. 預期性能提升：10-100 倍

位置: ~/research_dev/meanaudio_test/meanaudio_rtf_benchmark_fixed.py
遵循: Music Research Coding Principles v1.2
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
from typing import List, Dict
import numpy as np
from datetime import datetime
from tqdm import tqdm

# MeanAudio 內部 imports
from meanaudio.eval_utils import (
    ModelConfig, all_model_cfg, generate_mf, generate_fm, setup_eval_logging
)
from meanaudio.model.flow_matching import FlowMatching
from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import get_mean_audio
from meanaudio.model.utils.features_utils import FeaturesUtils
import soundfile as sf

# 設置 PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 抑制日誌
logging.getLogger().setLevel(logging.WARNING)


class OptimizedMeanAudioBenchmark:
    """優化版 MeanAudio RTF 測試 - 真正的模型重用"""
    
    def __init__(self, output_dir: str = "rtf_benchmark_fixed_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 測試配置
        self.nfe_configs = [1, 4, 8, 16, 32]
        self.duration = 10  # 秒
        self.variant = "meanaudio_s"
        self.seed = 42
        
        # 測試 prompts
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
        
        # 模型組件（一次性載入）
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16
        self.net = None
        self.feature_utils = None
        self.model_config = None
        self.seq_cfg = None
        
        # 結果
        self.results = []
        self.model_load_time = 0.0
        
        print("="*80)
        print("MeanAudio RTF 基準測試 - 修正版（真正的模型重用）")
        print("="*80)
        print(f"測試配置:")
        print(f"  - NFE: {self.nfe_configs}")
        print(f"  - 音頻長度: {self.duration}s")
        print(f"  - 模型變體: {self.variant}")
        print(f"  - Prompts: {len(self.test_prompts)} 個")
        print(f"  - 總測試數: {len(self.nfe_configs) * len(self.test_prompts)}")
        print(f"  - 設備: {self.device}")
        print("="*80)
    
    def load_models_once(self):
        """載入所有模型組件（只執行一次）"""
        print("\n模型載入中...")
        load_start = time.time()
        
        try:
            # 1. 載入模型配置
            self.model_config = all_model_cfg[self.variant]
            self.seq_cfg = self.model_config.seq_cfg
            self.seq_cfg.duration = self.duration
            
            print(f"  [1/3] 模型配置載入...")
            
            # 2. 載入主模型
            self.net = get_mean_audio(
                self.model_config.model_name,
                use_rope=True,
                text_c_dim=512
            )
            self.net = self.net.to(self.device, self.dtype).eval()
            self.net.load_weights(
                torch.load(self.model_config.model_path, 
                          map_location=self.device, 
                          weights_only=True)
            )
            self.net.update_seq_lengths(self.seq_cfg.latent_seq_len)
            
            print(f"  [2/3] 主模型載入完成")
            
            # 3. 載入特徵提取工具（包括 T5 text encoder）
            self.feature_utils = FeaturesUtils(
                tod_vae_ckpt=self.model_config.vae_path,
                enable_conditions=True,
                encoder_name='t5_clap',
                mode=self.model_config.mode,
                bigvgan_vocoder_ckpt=self.model_config.bigvgan_16k_path,
                need_vae_encoder=False
            )
            self.feature_utils = self.feature_utils.to(self.device, self.dtype).eval()
            
            print(f"  [3/3] 特徵提取工具載入完成")
            
            self.model_load_time = time.time() - load_start
            
            print(f"\n模型載入成功")
            print(f"  載入時間: {self.model_load_time:.2f}s")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            
            return True
            
        except Exception as e:
            print(f"模型載入失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @torch.inference_mode()
    def generate_audio(self, prompt: str, nfe: int, test_index: int) -> Dict:
        """生成音頻（使用已載入的模型）"""
        
        try:
            # 創建輸出目錄
            output_path = self.output_dir / f"test_{test_index:04d}_nfe{nfe}"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 記錄生成時間
            gen_start = time.time()
            
            # 創建隨機數生成器
            rng = torch.Generator(device=self.device)
            rng.manual_seed(self.seed)
            
            # 創建 generation function（輕量級操作）
            generation_func = MeanFlow(steps=nfe)
            
            # 生成音頻
            audios = generate_mf(
                text=[prompt],
                negative_text=[''],
                feature_utils=self.feature_utils,
                net=self.net,
                rng=rng,
                cfg_strength=0,
                mf=generation_func
            )
            
            # 確保 GPU 完成所有計算
            torch.cuda.synchronize()
            generation_time = time.time() - gen_start
            
            # 計算 RTF
            rtf = generation_time / self.duration
            
            # 保存音頻
            audio = audios.float().cpu()[0]
            safe_filename = prompt.replace(' ', '_').replace('/', '_')
            save_path = output_path / f'{safe_filename}--nfe{nfe}--seed{self.seed}.wav'
            sf.write(save_path, audio.squeeze(0).numpy(), self.seq_cfg.sampling_rate)
            
            # GPU 記憶體
            if torch.cuda.is_available():
                memory_gb = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_gb = 0
            
            return {
                'success': True,
                'prompt': prompt,
                'nfe': nfe,
                'generation_time': generation_time,
                'audio_duration': self.duration,
                'rtf': rtf,
                'memory_gb': memory_gb,
                'output_path': str(save_path)
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
        
        # 載入模型（只執行一次）
        if not self.load_models_once():
            print("無法繼續測試")
            return False
        
        print(f"\n開始 RTF 基準測試")
        print(f"{'='*80}\n")
        
        total_tests = len(self.nfe_configs) * len(self.test_prompts)
        test_index = 0
        
        # 按 NFE 分組測試
        for nfe_idx, nfe in enumerate(self.nfe_configs, 1):
            print(f"\nNFE = {nfe} ({nfe_idx}/{len(self.nfe_configs)})")
            print(f"{'-'*80}")
            
            nfe_results = []
            
            # 進度條
            pbar = tqdm(self.test_prompts, desc=f"NFE={nfe}", leave=True)
            
            for prompt in pbar:
                test_index += 1
                
                # 生成音頻
                result = self.generate_audio(prompt, nfe, test_index)
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
                    pbar.set_postfix({'Status': 'Failed'})
                
                # 短暫暫停
                time.sleep(0.1)
            
            # NFE 組統計
            self.print_nfe_statistics(nfe, nfe_results)
        
        # 最終統計
        self.print_final_statistics()
        
        # 保存結果
        self.save_results()
        
        return True
    
    def print_nfe_statistics(self, nfe: int, nfe_results: List[Dict]):
        """打印單個 NFE 的統計"""
        
        successful = [r for r in nfe_results if r.get('success')]
        
        if not successful:
            print(f"   NFE={nfe}: 所有測試失敗")
            return
        
        rtfs = [r['rtf'] for r in successful]
        times = [r['generation_time'] for r in successful]
        
        print(f"\n   NFE = {nfe} 統計:")
        print(f"   - 成功率: {len(successful)}/{len(nfe_results)}")
        print(f"   - 平均 RTF: {np.mean(rtfs):.4f} (±{np.std(rtfs):.4f})")
        print(f"   - RTF 範圍: {np.min(rtfs):.4f} - {np.max(rtfs):.4f}")
        print(f"   - 平均生成時間: {np.mean(times):.2f}s")
    
    def print_final_statistics(self):
        """打印最終統計"""
        
        print(f"\n{'='*80}")
        print(f"最終統計摘要")
        print(f"{'='*80}")
        
        # 按 NFE 分組
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
        print(f"與論文基準比較")
        print(f"{'='*80}")
        
        paper_rtf = 0.013
        
        if 8 in nfe_stats:
            our_rtf = nfe_stats[8]['avg_rtf']
            
            print(f"\nNFE = 8:")
            print(f"  論文 (A100):     RTF = {paper_rtf:.4f}")
            print(f"  RTX 5090:       RTF = {our_rtf:.4f}")
            
            if our_rtf < paper_rtf:
                speedup = paper_rtf / our_rtf
                print(f"  結果: RTX 5090 更快 ({speedup:.2f}x)")
            else:
                slowdown = our_rtf / paper_rtf
                print(f"  結果: RTX 5090 較慢 ({slowdown:.2f}x)")
    
    def save_results(self):
        """保存結果"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"rtf_benchmark_fixed_{timestamp}.json"
        
        # 統計
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
                    'success_count': len(nfe_results)
                }
        
        output_data = {
            'experiment_info': {
                'title': 'MeanAudio RTF 基準測試 - 修正版（模型重用）',
                'timestamp': timestamp,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'model_variant': self.variant,
                'audio_duration': self.duration,
                'nfe_configs': self.nfe_configs,
                'model_load_time': self.model_load_time,
                'optimization': 'Direct API, model reuse, no repeated initialization'
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
        print(f"完整結果已保存: {results_file}")
        print(f"音頻輸出目錄: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    """主函數"""
    
    print("MeanAudio RTF 基準測試 - 修正版")
    print("關鍵改進: 模型只載入一次，真正實現重用")
    print()
    
    # 檢查 CUDA
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用")
        response = input("是否繼續? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # 執行測試
    benchmark = OptimizedMeanAudioBenchmark()
    success = benchmark.run_benchmark()
    
    if success:
        print("RTF 基準測試完成！")
        return 0
    else:
        print("RTF 基準測試失敗")
        return 1


if __name__ == "__main__":
    exit(main())