#!/usr/bin/env python3
"""
資料管線驗證腳本 - MeanAudio 訓練前置檢查

目的：
1. 驗證 JSONL 資料完整性（277,100 首歌曲）
2. 統計 credibility 與 similarity_with_gt 分佈
3. 估算四種模型變體的可用資料量
4. 產出驗證報告

使用方式：
    python data_pipeline_validation.py

輸出：
    - validation_report.txt: 文字報告
    - validation_stats.json: 統計資料（JSON格式）
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import statistics


class DataPipelineValidator:
    """資料管線驗證器"""
    
    def __init__(self, jsonl_path):
        self.jsonl_path = Path(jsonl_path)
        
        # 統計容器
        self.total_records = 0
        self.valid_records = 0
        self.invalid_records = []
        
        # 分數分佈
        self.credibility_scores = []
        self.gt_similarity_scores = []
        self.representative_similarities = []
        
        # 資料完整性統計
        self.missing_fields = defaultdict(int)
        
        # Credibility 分層統計
        self.credibility_high = 0    # ≥ 0.8
        self.credibility_medium = 0  # 0.6 - 0.8
        self.credibility_low = 0     # < 0.6
        
    def validate_record(self, record, line_num):
        """驗證單筆記錄的完整性"""
        issues = []
        
        # 檢查必要欄位
        required_fields = [
            'audio_path',
            'seeds_used',
            'caption_details',
            'representative_caption',
            'credibility_analysis',
            'ground_truth_comparison'
        ]
        
        for field in required_fields:
            if field not in record:
                issues.append(f"Missing field: {field}")
                self.missing_fields[field] += 1
        
        # 檢查 caption_details 數量
        if 'caption_details' in record:
            if len(record['caption_details']) != 5:
                issues.append(f"Expected 5 captions, got {len(record['caption_details'])}")
        
        # 檢查 credibility_analysis
        if 'credibility_analysis' in record:
            if 'credibility_score' not in record['credibility_analysis']:
                issues.append("Missing credibility_score")
                self.missing_fields['credibility_score'] += 1
        
        # 檢查 ground_truth_comparison
        if 'ground_truth_comparison' in record:
            if 'similarity_with_gt' not in record['ground_truth_comparison']:
                issues.append("Missing similarity_with_gt")
                self.missing_fields['similarity_with_gt'] += 1
        
        if issues:
            self.invalid_records.append({
                'line': line_num,
                'audio_path': record.get('audio_path', 'unknown'),
                'issues': issues
            })
            return False
        
        return True
    
    def collect_statistics(self, record):
        """收集統計資料"""
        # Credibility score
        cred_score = record['credibility_analysis'].get('credibility_score')
        if cred_score is not None:
            self.credibility_scores.append(cred_score)
            
            # 分層統計
            if cred_score >= 0.8:
                self.credibility_high += 1
            elif cred_score >= 0.6:
                self.credibility_medium += 1
            else:
                self.credibility_low += 1
        
        # Ground truth similarity
        gt_sim = record['ground_truth_comparison'].get('similarity_with_gt')
        if gt_sim is not None:
            self.gt_similarity_scores.append(gt_sim)
        
        # Representative caption similarity
        if 'avg_similarity_with_others' in record['representative_caption']:
            rep_sim = record['representative_caption'].get('avg_similarity_with_others')
            if rep_sim is not None:
                self.representative_similarities.append(rep_sim)
    
    def process_jsonl(self):
        """逐行處理 JSONL 檔案"""
        print(f"開始驗證資料檔: {self.jsonl_path}")
        print(f"檔案大小: {self.jsonl_path.stat().st_size / 1024 / 1024:.1f} MB")
        print()
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                self.total_records += 1
                
                # 進度顯示
                if line_num % 10000 == 0:
                    print(f"處理中... {line_num:,} 筆", end='\r')
                
                try:
                    record = json.loads(line)
                    
                    # 驗證完整性
                    is_valid = self.validate_record(record, line_num)
                    
                    if is_valid:
                        self.valid_records += 1
                        self.collect_statistics(record)
                    
                except json.JSONDecodeError as e:
                    self.invalid_records.append({
                        'line': line_num,
                        'audio_path': 'parse_error',
                        'issues': [f"JSON parse error: {e}"]
                    })
        
        print(f"處理完成: {self.total_records:,} 筆           ")
        print()
    
    def calculate_stats(self, values, name):
        """計算統計數據"""
        if not values:
            return None
        
        return {
            'name': name,
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'percentiles': {
                '25th': statistics.quantiles(values, n=4)[0],
                '50th': statistics.quantiles(values, n=4)[1],
                '75th': statistics.quantiles(values, n=4)[2]
            }
        }
    
    def generate_report(self):
        """產生驗證報告"""
        report_lines = [
            "=" * 80,
            "資料管線驗證報告 - MeanAudio 訓練資料",
            "=" * 80,
            f"產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"資料來源: {self.jsonl_path}",
            "",
            "=" * 80,
            "一、資料完整性檢查",
            "=" * 80,
            f"總記錄數: {self.total_records:,}",
            f"有效記錄: {self.valid_records:,} ({self.valid_records/self.total_records*100:.2f}%)",
            f"無效記錄: {len(self.invalid_records):,}",
            ""
        ]
        
        # 無效記錄詳情（最多顯示10筆）
        if self.invalid_records:
            report_lines.extend([
                "無效記錄樣本（前10筆）：",
                "-" * 80
            ])
            for inv in self.invalid_records[:10]:
                report_lines.append(f"Line {inv['line']}: {inv['audio_path']}")
                for issue in inv['issues']:
                    report_lines.append(f"  - {issue}")
            report_lines.append("")
        
        # 缺失欄位統計
        if self.missing_fields:
            report_lines.extend([
                "缺失欄位統計：",
                "-" * 80
            ])
            for field, count in sorted(self.missing_fields.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  {field}: {count:,} 筆")
            report_lines.append("")
        
        # Credibility 統計
        report_lines.extend([
            "=" * 80,
            "二、Credibility Score 分佈",
            "=" * 80
        ])
        
        cred_stats = self.calculate_stats(self.credibility_scores, "Credibility Score")
        if cred_stats:
            report_lines.extend([
                f"樣本數: {cred_stats['count']:,}",
                f"平均值: {cred_stats['mean']:.4f}",
                f"中位數: {cred_stats['median']:.4f}",
                f"標準差: {cred_stats['std']:.4f}",
                f"最小值: {cred_stats['min']:.4f}",
                f"最大值: {cred_stats['max']:.4f}",
                "",
                "百分位數:",
                f"  25th: {cred_stats['percentiles']['25th']:.4f}",
                f"  50th: {cred_stats['percentiles']['50th']:.4f}",
                f"  75th: {cred_stats['percentiles']['75th']:.4f}",
                "",
                "分層統計（用於 Hard Filtering 模型）：",
                "-" * 80,
                f"高可信度 (≥ 0.8):  {self.credibility_high:,} 首 ({self.credibility_high/self.valid_records*100:.2f}%)",
                f"中可信度 (0.6-0.8): {self.credibility_medium:,} 首 ({self.credibility_medium/self.valid_records*100:.2f}%)",
                f"低可信度 (< 0.6):  {self.credibility_low:,} 首 ({self.credibility_low/self.valid_records*100:.2f}%)",
                ""
            ])
        else:
            report_lines.append("⚠ 無法計算統計資料（樣本數為 0 或包含無效值）")
            report_lines.append("")
        
        # Ground Truth Similarity 統計
        report_lines.extend([
            "=" * 80,
            "三、Ground Truth Similarity 分佈",
            "=" * 80
        ])
        
        gt_stats = self.calculate_stats(self.gt_similarity_scores, "GT Similarity")
        if gt_stats:
            report_lines.extend([
                f"樣本數: {gt_stats['count']:,}",
                f"平均值: {gt_stats['mean']:.4f}",
                f"中位數: {gt_stats['median']:.4f}",
                f"標準差: {gt_stats['std']:.4f}",
                f"最小值: {gt_stats['min']:.4f}",
                f"最大值: {gt_stats['max']:.4f}",
                "",
                "百分位數:",
                f"  25th: {gt_stats['percentiles']['25th']:.4f}",
                f"  50th: {gt_stats['percentiles']['50th']:.4f}",
                f"  75th: {gt_stats['percentiles']['75th']:.4f}",
                ""
            ])
        else:
            report_lines.append("⚠ 無法計算統計資料（樣本數為 0 或包含無效值）")
            report_lines.append("")
        
        # Representative Caption Similarity 統計
        report_lines.extend([
            "=" * 80,
            "四、Representative Caption 內部相似度",
            "=" * 80
        ])
        
        rep_stats = self.calculate_stats(self.representative_similarities, "Representative Similarity")
        if rep_stats:
            report_lines.extend([
                f"樣本數: {rep_stats['count']:,}",
                f"平均值: {rep_stats['mean']:.4f}",
                f"中位數: {rep_stats['median']:.4f}",
                f"標準差: {rep_stats['std']:.4f}",
                f"最小值: {rep_stats['min']:.4f}",
                f"最大值: {rep_stats['max']:.4f}",
                ""
            ])
        else:
            report_lines.append("⚠ 無法計算統計資料（樣本數為 0 或包含無效值）")
            report_lines.append("")
        
        # 四種模型資料量估算
        report_lines.extend([
            "=" * 80,
            "五、四種模型變體的資料量估算",
            "=" * 80,
            "",
            f"1. Baseline 模型",
            f"   - 使用任意一個 caption",
            f"   - 可用資料量: {self.valid_records:,} 首",
            "",
            f"2. Best-Similarity 模型（模範樣本）",
            f"   - 使用 representative_caption",
            f"   - 可用資料量: {self.valid_records:,} 首",
            "",
            f"3. Hard Filtering 模型（可信度門檻 ≥ 0.8）",
            f"   - 移除低可信度樣本",
            f"   - 可用資料量: {self.credibility_high:,} 首 ({self.credibility_high/self.valid_records*100:.2f}%)",
            f"   - 捨棄資料量: {self.valid_records - self.credibility_high:,} 首",
            "",
            f"4. Credibility Tag 條件模型",
            f"   - 加入可信度標籤作為條件輸入",
            f"   - 可用資料量: {self.valid_records:,} 首",
            ""
        ])
        
        # 資料品質建議
        report_lines.extend([
            "=" * 80,
            "六、資料品質建議",
            "=" * 80,
            ""
        ])
        
        # 根據統計結果給出建議
        if cred_stats:
            if cred_stats['mean'] >= 0.7:
                report_lines.append("✓ Credibility 平均值良好 (≥ 0.7)")
            else:
                report_lines.append("⚠ Credibility 平均值偏低 (< 0.7)，建議提高 Hard Filtering 門檻")
        
        if gt_stats:
            if gt_stats['mean'] >= 0.6:
                report_lines.append("✓ Ground Truth 相似度良好 (≥ 0.6)")
            else:
                report_lines.append("⚠ Ground Truth 相似度偏低 (< 0.6)，LP-MusicCaps 品質可能需要改善")
        
        if self.credibility_high / self.valid_records < 0.3:
            report_lines.append("⚠ 高可信度樣本比例偏低 (< 30%)，Hard Filtering 可能損失過多資料")
        else:
            report_lines.append(f"✓ 高可信度樣本比例充足 ({self.credibility_high/self.valid_records*100:.1f}%)")
        
        report_lines.extend([
            "",
            "=" * 80,
            "驗證完成",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def save_json_stats(self, output_path):
        """儲存統計資料為 JSON"""
        stats = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source_file': str(self.jsonl_path),
                'total_records': self.total_records,
                'valid_records': self.valid_records
            },
            'credibility': self.calculate_stats(self.credibility_scores, "credibility"),
            'gt_similarity': self.calculate_stats(self.gt_similarity_scores, "gt_similarity"),
            'representative_similarity': self.calculate_stats(self.representative_similarities, "representative_similarity"),
            'credibility_tiers': {
                'high': {'count': self.credibility_high, 'percentage': self.credibility_high/self.valid_records*100},
                'medium': {'count': self.credibility_medium, 'percentage': self.credibility_medium/self.valid_records*100},
                'low': {'count': self.credibility_low, 'percentage': self.credibility_low/self.valid_records*100}
            },
            'model_variants_data_size': {
                'baseline': self.valid_records,
                'best_similarity': self.valid_records,
                'hard_filtering': self.credibility_high,
                'credibility_tag': self.valid_records
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"統計資料已儲存: {output_path}")


def main():
    """主程式"""
    # 資料檔路徑
    jsonl_path = Path.home() / "music_cleaning_results" / "results_20260119_043407.jsonl"
    
    # 檢查檔案是否存在
    if not jsonl_path.exists():
        print(f"錯誤: 找不到資料檔 {jsonl_path}")
        print("請確認檔案路徑是否正確")
        sys.exit(1)
    
    # 建立驗證器
    validator = DataPipelineValidator(jsonl_path)
    
    # 執行驗證
    print("開始資料管線驗證...")
    print()
    validator.process_jsonl()
    
    # 產生報告
    print("產生驗證報告...")
    report = validator.generate_report()
    
    # 儲存報告
    output_dir = Path.cwd()
    report_path = output_dir / "validation_report.txt"
    stats_path = output_dir / "validation_stats.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"驗證報告已儲存: {report_path}")
    
    # 儲存 JSON 統計
    validator.save_json_stats(stats_path)
    
    # 在終端顯示報告
    print()
    print("=" * 80)
    print("報告預覽")
    print("=" * 80)
    print(report)


if __name__ == "__main__":
    main()