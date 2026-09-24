"""
生成 Phase 7 V1 訓練 TSV
caption 改為從 5 個候選中隨機選 1 個（seed=42）

用法：
  python3 gen_phase7_v1_tsv.py
"""

import json
import csv
import random
from pathlib import Path

JSONL_PATH   = Path.home() / 'research/music_cleaning/results_20260119_043407.jsonl'
INPUT_TSV    = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase6_train.tsv')
OUTPUT_TSV   = Path.home() / 'research/meanaudio_training/phase7_v1_train.tsv'


def relative_path_to_id(relative_path: str) -> str:
    """00/1002000/segment_0.mp3 → 00_1002000_segment_0"""
    return relative_path.replace('.mp3', '').replace('/', '_')


def load_jsonl(jsonl_path):
    """回傳 id → [caption, ...] 的 lookup dict"""
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
    print(f'讀取 JSONL: {JSONL_PATH}')
    lookup = load_jsonl(JSONL_PATH)
    print(f'  載入 {len(lookup):,} 筆 caption lookup')

    rng = random.Random(42)

    rows_out = []
    n_replaced = 0
    n_missing  = 0

    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        for row in reader:
            clip_id = row['id']
            if clip_id in lookup:
                row['caption'] = rng.choice(lookup[clip_id])
                n_replaced += 1
            else:
                n_missing += 1  # 保留原本 caption
            rows_out.append(row)

    total = len(rows_out)

    # 印前 3 筆確認
    print('\n前 3 筆預覽：')
    for r in rows_out[:3]:
        print(f"  id={r['id']}  q_level={r.get('q_level','-')}")
        print(f"  caption={r['caption'][:100]}...")
        print()

    # 統計
    print(f'統計：')
    print(f'  總筆數：   {total:,}')
    print(f'  成功替換： {n_replaced:,}')
    print(f'  找不到：   {n_missing:,}')

    # 寫出
    with open(OUTPUT_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows_out)

    print(f'\n輸出：{OUTPUT_TSV}')

    # 驗證行數
    in_lines  = sum(1 for _ in open(INPUT_TSV))
    out_lines = sum(1 for _ in open(OUTPUT_TSV))
    print(f'行數驗證：input={in_lines:,}  output={out_lines:,}  {"✅ 一致" if in_lines == out_lines else "❌ 不一致"}')


if __name__ == '__main__':
    main()
