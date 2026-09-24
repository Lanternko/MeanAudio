#!/usr/bin/env python3
"""
Bagpipe 誤判分析腳本

任務：從 421K LP-MusicCaps 結果中隨機抽取 10000 個樣本
分析 LP-MusicCaps 將非 bagpipe 樂器誤判為 bagpipe 的比例

位置：~/research_dev/bagpipe_analysis/bagpipe_analysis.py
"""

import json
import random
from pathlib import Path
from typing import Dict, List
import argparse
from datetime import datetime

class BagpipeAnalyzer:
    """Bagpipe 誤判分析器"""
    
    def __init__(self, results_file: str, sample_size: int = 10000):
        self.results_file = Path(results_file)
        self.sample_size = sample_size
        
        # Bagpipe 相關關鍵詞
        self.bagpipe_keywords = [
            'bagpipe', 'bagpipes', 'bag pipe', 'bag pipes',
            'scottish pipe', 'highland pipe', 'uilleann pipe'
        ]
        
        print(f"🔍 Bagpipe 誤判分析")
        print(f"📁 結果檔案: {self.results_file}")
        print(f"🎯 抽樣數量: {self.sample_size:,}")
        print("=" * 70)
    
    def load_all_results(self) -> List[Dict]:
        """載入所有結果"""
        print("\n📂 載入 LP-MusicCaps 結果...")
        
        all_results = []
        
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            result = json.loads(line)
                            if result.get('success') and result.get('generated_caption'):
                                all_results.append(result)
                        except json.JSONDecodeError:
                            continue
                    
                    # 進度報告
                    if line_num % 50000 == 0:
                        print(f"   已讀取 {line_num:,} 行, 有效結果: {len(all_results):,}")
            
            print(f"✅ 載入完成: {len(all_results):,} 個有效結果\n")
            return all_results
            
        except FileNotFoundError:
            print(f"❌ 檔案不存在: {self.results_file}")
            return []
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
            return []
    
    def random_sample(self, all_results: List[Dict]) -> List[Dict]:
        """隨機抽樣"""
        if len(all_results) <= self.sample_size:
            print(f"⚠️  總樣本數 ({len(all_results):,}) 少於目標 ({self.sample_size:,})，使用全部樣本\n")
            return all_results
        
        print(f"🎲 隨機抽取 {self.sample_size:,} 個樣本...")
        sampled = random.sample(all_results, self.sample_size)
        print(f"✅ 抽樣完成\n")
        
        return sampled
    
    def contains_bagpipe(self, caption: str) -> bool:
        """檢查 caption 是否包含 bagpipe 關鍵詞"""
        caption_lower = caption.lower()
        
        for keyword in self.bagpipe_keywords:
            if keyword in caption_lower:
                return True
        
        return False
    
    def analyze_bagpipe_mentions(self, samples: List[Dict]) -> Dict:
        """分析 bagpipe 提及情況"""
        print("🔬 分析 bagpipe 提及...")
        
        bagpipe_samples = []
        
        for sample in samples:
            caption = sample.get('generated_caption', '')
            
            if self.contains_bagpipe(caption):
                bagpipe_samples.append({
                    'index': sample.get('index', 0),
                    'audio_path': sample.get('path', ''),
                    'caption': caption,
                    'similarity': sample.get('similarity', 0.0)
                })
        
        total_count = len(samples)
        bagpipe_count = len(bagpipe_samples)
        bagpipe_percentage = (bagpipe_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"📊 Bagpipe 分析結果")
        print(f"{'='*70}")
        print(f"總樣本數: {total_count:,}")
        print(f"包含 bagpipe: {bagpipe_count:,}")
        print(f"百分比: {bagpipe_percentage:.2f}%")
        print(f"{'='*70}\n")
        
        # 顯示一些範例
        if bagpipe_samples:
            print(f"📝 Bagpipe 提及範例（前 10 個）:")
            print("-" * 70)
            for i, sample in enumerate(bagpipe_samples[:10], 1):
                print(f"\n[{i}] 相似度: {sample['similarity']:.3f}")
                print(f"    Caption: {sample['caption'][:100]}...")
            print()
        
        return {
            'total_samples': total_count,
            'bagpipe_count': bagpipe_count,
            'bagpipe_percentage': bagpipe_percentage,
            'bagpipe_samples': bagpipe_samples
        }
    
    def save_results(self, analysis_results: Dict, output_dir: str = "bagpipe_analysis_results"):
        """保存分析結果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整結果（JSON）
        results_file = output_path / f"bagpipe_analysis_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment_info': {
                    'title': 'LP-MusicCaps Bagpipe 誤判分析',
                    'timestamp': timestamp,
                    'source_file': str(self.results_file),
                    'sample_size': self.sample_size
                },
                'results': analysis_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 完整結果已保存: {results_file}")
        
        # 保存 bagpipe 樣本列表（CSV 格式）
        if analysis_results['bagpipe_samples']:
            import csv
            
            csv_file = output_path / f"bagpipe_samples_{timestamp}.csv"
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['index', 'audio_path', 'caption', 'similarity'])
                writer.writeheader()
                writer.writerows(analysis_results['bagpipe_samples'])
            
            print(f"📋 Bagpipe 樣本列表已保存: {csv_file}")
        
        # 生成摘要報告（Markdown）
        report_file = output_path / f"bagpipe_analysis_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# LP-MusicCaps Bagpipe 誤判分析報告\n\n")
            f.write(f"**分析時間**: {timestamp}\n")
            f.write(f"**來源檔案**: {self.results_file}\n")
            f.write(f"**抽樣數量**: {self.sample_size:,}\n\n")
            
            f.write(f"## 分析結果\n\n")
            f.write(f"- **總樣本數**: {analysis_results['total_samples']:,}\n")
            f.write(f"- **包含 bagpipe**: {analysis_results['bagpipe_count']:,}\n")
            f.write(f"- **百分比**: {analysis_results['bagpipe_percentage']:.2f}%\n\n")
            
            if analysis_results['bagpipe_samples']:
                f.write(f"## Bagpipe 提及範例（前 20 個）\n\n")
                for i, sample in enumerate(analysis_results['bagpipe_samples'][:20], 1):
                    f.write(f"### 範例 {i}\n\n")
                    f.write(f"- **相似度**: {sample['similarity']:.3f}\n")
                    f.write(f"- **檔案**: {sample['audio_path']}\n")
                    f.write(f"- **Caption**: {sample['caption']}\n\n")
        
        print(f"📄 摘要報告已保存: {report_file}\n")
        
        return results_file, csv_file, report_file
    
    def run_analysis(self, output_dir: str = "bagpipe_analysis_results"):
        """執行完整分析"""
        # 1. 載入所有結果
        all_results = self.load_all_results()
        
        if not all_results:
            print("❌ 沒有可分析的數據")
            return None
        
        # 2. 隨機抽樣
        samples = self.random_sample(all_results)
        
        # 3. 分析 bagpipe 提及
        analysis_results = self.analyze_bagpipe_mentions(samples)
        
        # 4. 保存結果
        self.save_results(analysis_results, output_dir)
        
        return analysis_results


def main():
    parser = argparse.ArgumentParser(description="LP-MusicCaps Bagpipe 誤判分析")
    parser.add_argument('--results_file', type=str,
                       default="/mnt/HDD/kojiek/music_semantic_fidelity/research_results/research_results_20251228_144519.jsonl",
                       help='LP-MusicCaps 結果檔案路徑（JSONL 格式）')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='抽樣數量（預設: 10000）')
    parser.add_argument('--output_dir', type=str, default="bagpipe_analysis_results",
                       help='輸出目錄（預設: bagpipe_analysis_results）')
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子（用於可重複性，預設: 42）')
    
    args = parser.parse_args()
    
    # 設置隨機種子
    random.seed(args.seed)
    print(f"🎲 使用隨機種子: {args.seed}\n")
    
    # 執行分析
    analyzer = BagpipeAnalyzer(args.results_file, args.sample_size)
    result = analyzer.run_analysis(args.output_dir)
    
    if result:
        print("✅ Bagpipe 分析完成！")
        return 0
    else:
        print("❌ Bagpipe 分析失敗")
        return 1


if __name__ == "__main__":
    exit(main())
