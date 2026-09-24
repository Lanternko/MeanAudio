"""
Phase 8 V4：生成訓練 TSV

Caption：來自 gen_qwen2audio_captions.py 的 Qwen2-Audio 生成結果
Q signal：沿用 phase7_v1_train.tsv 的 q_level（MeanSim-based）

設計原則：只換 captioning model，q signal 不變 → 單一變量實驗
  Phase 7 V1：LP-MusicCaps random caption + MeanSim-Q  ← 目前最佳
  Phase 8 V4：Qwen2-Audio caption    + MeanSim-Q  ← 這次

對外名稱：JamendoFull-Qwen2Audio-MeanSim-Q

輸出：
  /mnt/HDD/kojiek/phase4_jamendo_data/phase8_v4_train.tsv

用法：
  python gen_qwen2audio_tsv.py
"""

import json
import csv
from pathlib import Path
from tqdm import tqdm

CAPTIONS_JSONL = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v4_captions.jsonl')
INPUT_TSV      = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUTPUT_TSV     = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v4_train.tsv')


def load_captions(jsonl_path: Path) -> dict:
    """載入 {id: caption}，跳過 None"""
    cap_map = {}
    n_none = 0
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d['caption'] is None:
                n_none += 1
                continue
            cap_map[d['id']] = d['caption']
    print(f'載入 {len(cap_map):,} 筆 caption（{n_none:,} 筆 None 跳過）')
    return cap_map


def main():
    print(f'載入 Qwen2-Audio captions: {CAPTIONS_JSONL}')
    cap_map = load_captions(CAPTIONS_JSONL)

    print(f'\n讀取 phase7_v1 TSV（q_level 沿用）: {INPUT_TSV}')
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
        cap = cap_map.get(cid)
        if cap:
            caption = cap
            n_found += 1
        else:
            # fallback：保留 LP-MusicCaps random caption
            caption = row['caption']
            n_fallback += 1

        rows_out.append({
            'id':      cid,
            'caption': caption,
            'q_level': row['q_level'],  # MeanSim-based，不動
        })

    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'caption', 'q_level'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows_out)

    print(f'\n完成！')
    print(f'  Qwen2-Audio caption：{n_found:,}（fallback LP-MusicCaps：{n_fallback:,}）')
    print(f'  輸出 → {OUTPUT_TSV}')
    print(f'\n對外名稱：JamendoFull-Qwen2Audio-MeanSim-Q')
    print(f'下一步：更新 train_pipeline.sh EXP_PREFIX="phase8_v4" 後啟動訓練')


if __name__ == '__main__':
    main()
