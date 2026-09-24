"""
生成 Phase 8 V4 訓練 TSV — `JamendoFull-Random-PromptConsistency-NoQ`

把 mean_similarity (raw float, 5-caption pairwise consistency) 直接寫進 caption 前綴：
    [consistency=0.85] <original caption>

設計動機：QA-MDT 風格的 verbalize quality token；不依賴 q_embed lookup，訊號全
走 text encoder。配 use_q_conditioning=False 訓練、eval 用 --no_q。

來源：
  - `phase7_v1_train.tsv`（已有 seed=42 single-caption 選擇）
  - `results_20260119_043407.jsonl`（含 credibility_analysis.mean_similarity）

用法：
  python3 gen_phase8v4_tsv.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

JSONL_PATH = Path.home() / 'research/music_cleaning/results_20260119_043407.jsonl'
INPUT_TSV  = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
OUTPUT_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_v4_train.tsv')


def relative_path_to_id(relative_path: str) -> str:
    """00/1002000/segment_0.mp3 → 00_1002000_segment_0"""
    return relative_path.replace('.mp3', '').replace('/', '_')


def load_mean_sim(jsonl_path: Path) -> dict:
    """回傳 id → mean_similarity (float) 的 lookup dict"""
    lookup = {}
    n_total = 0
    n_skipped = 0
    with open(jsonl_path) as f:
        for line in f:
            n_total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            ca = d.get('credibility_analysis')
            if not isinstance(ca, dict):
                n_skipped += 1
                continue
            ms = ca.get('mean_similarity')
            if ms is None:
                n_skipped += 1
                continue
            clip_id = relative_path_to_id(d['relative_path'])
            lookup[clip_id] = float(ms)
    print(f'  JSONL: {n_total:,} lines, {len(lookup):,} 有 mean_similarity ({n_skipped:,} skipped)')
    return lookup


def histogram_bins(values, n_bins=10, lo=0.0, hi=1.0):
    """簡易直方圖：回傳 [(bin_low, bin_high, count), ...]"""
    bins = [0] * n_bins
    width = (hi - lo) / n_bins
    for v in values:
        if v < lo:
            idx = 0
        elif v >= hi:
            idx = n_bins - 1
        else:
            idx = int((v - lo) / width)
        bins[idx] += 1
    return [(lo + i * width, lo + (i + 1) * width, bins[i]) for i in range(n_bins)]


def main():
    print(f'讀取 JSONL: {JSONL_PATH}')
    sim_lookup = load_mean_sim(JSONL_PATH)

    print(f'\n讀取輸入 TSV: {INPUT_TSV}')
    rows_out = []
    sims_used = []
    n_prefixed = 0
    n_missing = 0
    fieldnames = None

    with open(INPUT_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        for row in reader:
            clip_id = row['id']
            sim = sim_lookup.get(clip_id)
            if sim is None:
                n_missing += 1
                # 不加 prefix，保留原 caption（與 gen_phase7_v1_tsv 對 missing 的處理一致）
            else:
                row['caption'] = f'[consistency={sim:.2f}] ' + row['caption']
                sims_used.append(sim)
                n_prefixed += 1
            rows_out.append(row)

    total = len(rows_out)

    # 預覽前 5 筆
    print(f'\n前 5 筆預覽：')
    for r in rows_out[:5]:
        print(f"  id={r['id']}  q_level={r.get('q_level','-')}")
        print(f"  caption={r['caption'][:120]}")
        print()

    # 統計
    print(f'統計：')
    print(f'  總筆數：       {total:,}')
    print(f'  成功 prefix：  {n_prefixed:,}')
    print(f'  找不到 sim：   {n_missing:,}')

    # Hard fail on missing — 任何 unprefixed row 會混兩種 conditioning regime，破壞 ablation
    # 純警告會讓 automation 在壞資料上繼續跑（Codex 2026-04-26 review P2.1）
    if n_missing > 0:
        raise SystemExit(
            f'❌ {n_missing} rows have no mean_similarity → unprefixed captions would '
            f'mix conditioning regimes. Aborting before downstream NPZ regen.'
        )

    # mean_similarity 分佈（檢查訊號非退化）
    if sims_used:
        sims_sorted = sorted(sims_used)
        n = len(sims_used)
        mean_s = sum(sims_used) / n
        median_s = sims_sorted[n // 2]
        p10 = sims_sorted[int(n * 0.10)]
        p90 = sims_sorted[int(n * 0.90)]
        print(f'\nmean_similarity 分佈（n={n:,}）：')
        print(f'  min={sims_sorted[0]:.4f}  p10={p10:.4f}  median={median_s:.4f}  '
              f'p90={p90:.4f}  max={sims_sorted[-1]:.4f}  mean={mean_s:.4f}')
        print(f'\nHistogram (10 bins, 0.0–1.0)：')
        for lo, hi, cnt in histogram_bins(sims_used, n_bins=10):
            bar = '█' * int(60 * cnt / max(b[2] for b in histogram_bins(sims_used, n_bins=10)))
            print(f'  [{lo:.1f}, {hi:.1f}): {cnt:>7,}  {bar}')

    # 寫出
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows_out)
    print(f'\n輸出 → {OUTPUT_TSV}')

    # 行數驗證
    in_lines = sum(1 for _ in open(INPUT_TSV))
    out_lines = sum(1 for _ in open(OUTPUT_TSV))
    ok = in_lines == out_lines
    print(f'行數驗證：input={in_lines:,}  output={out_lines:,}  '
          f'{"✅ 一致" if ok else "❌ 不一致"}')


if __name__ == '__main__':
    main()
