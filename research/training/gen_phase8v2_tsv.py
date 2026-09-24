"""
生成 Phase 8 V2 訓練 TSV
- Caption：random（seed=42，沿用 Phase 7 V1）
- q_level：來自 Audiobox PQ 分數，percentile-based equal-frequency mapping（0~9）

用法：
  python gen_phase8v2_tsv.py [--pq_labels PATH] [--output PATH]

前置需求：先跑 gen_pq_labels.py 產生 pq_labels.jsonl
"""

import json
import csv
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

JSONL_PATH  = Path.home() / 'research/music_cleaning/results_20260119_043407.jsonl'
INPUT_TSV   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUTPUT_TSV  = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v2_train.tsv')
DEFAULT_PQ  = Path.home() / 'research/meanaudio_training/pq_labels.jsonl'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pq_labels', type=str, default=str(DEFAULT_PQ))
    parser.add_argument('--output',    type=str, default=str(OUTPUT_TSV))
    return parser.parse_args()


def relative_path_to_id(relative_path: str) -> str:
    return relative_path.replace('.mp3', '').replace('/', '_')


def load_pq_labels(pq_path: Path):
    """載入 PQ labels，回傳 {clip_id: pq_score}（跳過 None）"""
    pq_map = {}
    n_none = 0
    with open(pq_path) as f:
        for line in f:
            d = json.loads(line)
            if d['pq'] is None:
                n_none += 1
                continue
            pq_map[d['id']] = float(d['pq'])
    print(f'  載入 {len(pq_map):,} 筆 PQ labels（{n_none:,} 筆無效跳過）')
    return pq_map


def build_quantizer(pq_map: dict):
    """
    用 equal-frequency binning 把 PQ 映射到 q_level 0~9
    回傳 bin_edges（11 個邊界值）與 quantize function
    """
    scores = np.array(list(pq_map.values()))
    bin_edges = np.percentile(scores, np.linspace(0, 100, 11))
    # 確保邊界單調遞增（避免重複值問題）
    bin_edges = np.unique(bin_edges)

    print(f'\n── PQ 分佈 ───────────────────────────────')
    print(f'  min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}  std={scores.std():.4f}')
    print(f'\n── Percentile-based bin edges（{len(bin_edges)-1} bins）──')
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i+1]
        count = np.sum((scores >= lo) & (scores < hi if i < len(bin_edges)-2 else scores <= hi))
        print(f'  q={i}: [{lo:.4f}, {hi:.4f})  n={count:,}')

    # 若 unique 後 bin 數不足 10，提示
    n_bins = len(bin_edges) - 1
    if n_bins < 10:
        print(f'\n  [WARN] PQ 分佈有重複 percentile，只產生 {n_bins} 個 bin（正常現象）')

    def quantize(pq_score: float) -> int:
        """把 PQ 分數映射到 0~(n_bins-1)"""
        idx = np.searchsorted(bin_edges[1:-1], pq_score, side='right')
        return int(min(idx, n_bins - 1))

    return bin_edges, quantize, n_bins


def load_caption_lookup(jsonl_path: Path):
    """載入 JSONL，回傳 {clip_id: [caption, ...]}"""
    lookup = {}
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            clip_id = relative_path_to_id(d['relative_path'])
            captions = [c['caption'].replace('\n', ' ').replace('\r', ' ').strip()
                        for c in d['caption_details']]
            lookup[clip_id] = captions
    return lookup


def main():
    args = parse_args()
    pq_path    = Path(args.pq_labels)
    output_tsv = Path(args.output)

    print(f'載入 PQ labels: {pq_path}')
    pq_map = load_pq_labels(pq_path)

    bin_edges, quantize, n_bins = build_quantizer(pq_map)

    print(f'\n載入 caption JSONL: {JSONL_PATH}')
    caption_lookup = load_caption_lookup(JSONL_PATH)
    print(f'  載入 {len(caption_lookup):,} 筆 caption lookup')

    rng = random.Random(42)  # 與 Phase 7 V1 相同 seed

    print(f'\n讀取輸入 TSV: {INPUT_TSV}')
    rows_in = []
    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows_in.append(row)
    print(f'  共 {len(rows_in):,} 筆')

    rows_out     = []
    n_pq_found   = 0
    n_pq_missing = 0
    n_cap_found  = 0
    n_cap_missing = 0

    for row in tqdm(rows_in, desc='生成 TSV'):
        cid = row['id']
        new_row = dict(row)

        # Caption：random（seed=42）
        caps = caption_lookup.get(cid)
        if caps:
            new_row['caption'] = rng.choice(caps)
            n_cap_found += 1
        else:
            n_cap_missing += 1  # 保留原 caption

        # q_level：來自 PQ 分數
        pq = pq_map.get(cid)
        if pq is not None:
            new_row['q_level'] = str(quantize(pq))
            n_pq_found += 1
        else:
            # fallback：保留原 mean_similarity-based q_level
            n_pq_missing += 1

        rows_out.append(new_row)

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tsv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'caption', 'q_level'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows_out)

    print(f'\n完成！')
    print(f'  PQ 替換：{n_pq_found:,}（fallback：{n_pq_missing:,}）')
    print(f'  Caption 替換：{n_cap_found:,}（保留原始：{n_cap_missing:,}）')
    print(f'  輸出 TSV → {output_tsv}')
    print(f'\n下一步：更新 train_pipeline.sh EXP_PREFIX="phase8_v2" 後啟動訓練')


if __name__ == '__main__':
    main()
