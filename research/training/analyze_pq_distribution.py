"""
分析 PQ 分數分佈，輸出 percentile 統計與建議的 quantization bins

用法：
  python analyze_pq_distribution.py [--input pq_labels.jsonl]
"""

import json
import argparse
import numpy as np
from pathlib import Path

DEFAULT_IN = Path.home() / 'research/meanaudio_training/pq_labels.jsonl'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=str(DEFAULT_IN))
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    pq_scores = []
    n_none = 0
    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            if d['pq'] is None:
                n_none += 1
            else:
                pq_scores.append(d['pq'])

    pq = np.array(pq_scores)
    print(f'有效樣本：{len(pq):,}（無效/缺失：{n_none:,}）')
    print()
    print('── PQ 分佈統計 ──────────────────────────')
    print(f'  min    = {pq.min():.4f}')
    print(f'  max    = {pq.max():.4f}')
    print(f'  mean   = {pq.mean():.4f}')
    print(f'  median = {np.median(pq):.4f}')
    print(f'  std    = {pq.std():.4f}')
    print()
    print('── Percentile ───────────────────────────')
    for p in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        print(f'  p{p:02d} = {np.percentile(pq, p):.4f}')
    print()

    # 建議 10 個 equal-frequency bins（0~9）
    bin_edges = [np.percentile(pq, p) for p in range(0, 101, 10)]
    print('── 建議 quantization bins（equal-frequency，0~9）────')
    print('  bin_edges =', [round(e, 4) for e in bin_edges])
    print()
    for i in range(10):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        count = np.sum((pq >= lo) & (pq < hi if i < 9 else pq <= hi))
        print(f'  q={i}: [{lo:.4f}, {hi:.4f})  n={count:,}')

    print()
    print('下一步：確認 bin 邊界合理後，執行 gen_phase8v2_tsv.py')


if __name__ == '__main__':
    main()
