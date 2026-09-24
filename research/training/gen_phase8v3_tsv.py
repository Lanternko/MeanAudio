"""
Phase 8 V3：分析 CLAP sim 分佈，產生 q_level，生成訓練 TSV

前置：先跑 gen_phase8v3_clap_sim.py 產生 phase8_v3_clap_sim.jsonl

輸入：
  - phase8_v3_clap_sim.jsonl（audio-text CLAP sim）
  - phase7_v1_train.tsv（caption，直接沿用不重抽）

輸出：
  - phase8_v3_train.tsv：id \t caption \t q_level

用法：
  python gen_phase8v3_tsv.py
"""

import json
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm

CLAP_SIM_JSONL = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v3_clap_sim.jsonl')
INPUT_TSV      = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUTPUT_TSV     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v3_train.tsv')


def load_clap_sim(jsonl_path: Path) -> dict:
    """載入 {id: clap_sim}，跳過 None"""
    sim_map = {}
    n_none = 0
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d['clap_sim'] is None:
                n_none += 1
                continue
            sim_map[d['id']] = float(d['clap_sim'])
    print(f'載入 {len(sim_map):,} 筆 CLAP sim（{n_none:,} 筆 None 跳過）')
    return sim_map


def build_quantizer(sim_map: dict):
    """
    Percentile-based equal-frequency binning → q_level 0~9
    回傳 bin_edges 與 quantize function
    """
    scores = np.array(list(sim_map.values()))
    bin_edges = np.percentile(scores, np.linspace(0, 100, 11))
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1

    print(f'\n── CLAP Sim 分佈 ────────────────────────────────')
    print(f'  n={len(scores):,}  min={scores.min():.4f}  max={scores.max():.4f}'
          f'  mean={scores.mean():.4f}  std={scores.std():.4f}')
    print(f'\n── Percentile bin edges（{n_bins} bins）──────────')
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi if i < n_bins - 1 else scores <= hi)
        print(f'  q={i}: [{lo:.4f}, {hi:.4f})  n={mask.sum():,}')

    if n_bins < 10:
        print(f'\n  [WARN] 只有 {n_bins} 個 bin（CLAP sim 分佈有重複 percentile）')

    def quantize(sim: float) -> int:
        idx = np.searchsorted(bin_edges[1:-1], sim, side='right')
        return int(min(idx, n_bins - 1))

    return bin_edges, quantize, n_bins


def main():
    print(f'載入 CLAP sim: {CLAP_SIM_JSONL}')
    sim_map = load_clap_sim(CLAP_SIM_JSONL)

    bin_edges, quantize, n_bins = build_quantizer(sim_map)

    print(f'\n讀取輸入 TSV: {INPUT_TSV}')
    rows_in = []
    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows_in.append(row)
    print(f'  共 {len(rows_in):,} 筆')

    rows_out = []
    n_found = n_fallback = 0

    for row in tqdm(rows_in, desc='生成 TSV'):
        cid = row['id']
        new_row = {'id': cid, 'caption': row['caption']}

        sim = sim_map.get(cid)
        if sim is not None:
            new_row['q_level'] = str(quantize(sim))
            n_found += 1
        else:
            # fallback：保留 phase7_v1 的 q_level（mean_similarity based）
            new_row['q_level'] = row['q_level']
            n_fallback += 1

        rows_out.append(new_row)

    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'caption', 'q_level'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows_out)

    print(f'\n完成！')
    print(f'  CLAP-based q_level：{n_found:,}（fallback mean_sim：{n_fallback:,}）')
    print(f'  輸出 → {OUTPUT_TSV}')
    print(f'\n下一步：更新 train_pipeline.sh EXP_PREFIX="phase8_v3" 後啟動訓練')


if __name__ == '__main__':
    main()
