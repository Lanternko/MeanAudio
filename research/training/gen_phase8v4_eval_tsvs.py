"""
生成 Phase 8 V4 eval TSVs — 把固定 prefix `[consistency=0.90] ` 寫進 caption。

eval pipeline 不走 NPZ cache（`eval.py` 直接呼叫 `feature_utils.encode_text(caption)`
走 T5/CLAP encode），所以 train/eval 對齊只要 caption 帶相同格式 prefix 即可。

主設定固定 0.90（不是 1.00 邊界） — 見
`memory/feedback_inference_value_in_support_2026_04_26.md`：
in-support、靠近訓練分布 p90，避免 OOD-edge 把 ablation 結果做髒。

來源：
  - `phase4_test.tsv`               (Jamendo full, 90,063 rows)
  - `phase4_test_seed42_2048.tsv`  (Jamendo seed=42 random 2048, 10-exp benchmark)
  - `musiccaps_test.tsv`            (MusicCaps full, 5,526 rows — primary benchmark)

輸出（同目錄）：
  - phase8_v4_jamendo_test.tsv
  - phase8_v4_jamendo_seed42_2048.tsv
  - phase8_v4_musiccaps_test.tsv

用法：
  python3 gen_phase8v4_eval_tsvs.py
"""

import csv
from pathlib import Path

PREFIX = '[consistency=0.90] '
DATA_DIR = Path('/mnt/HDD/kojiek/phase4_jamendo_data')

PAIRS = [
    (DATA_DIR / 'phase4_test.tsv',                DATA_DIR / 'phase8_v4_jamendo_test.tsv'),
    (DATA_DIR / 'phase4_test_seed42_2048.tsv',    DATA_DIR / 'phase8_v4_jamendo_seed42_2048.tsv'),
    (DATA_DIR / 'musiccaps_test.tsv',             DATA_DIR / 'phase8_v4_musiccaps_test.tsv'),
]


def prefix_tsv(src: Path, dst: Path) -> tuple[int, int]:
    """Read src TSV, prepend PREFIX to caption, write dst. Returns (n_rows, n_prefixed).

    Caption 內 \\n / \\r normalize 成空格（與 train side gen_phase7_v1_tsv.py:31-32 對齊）—
    eval.py 用 csv.DictReader 能正確 parse 多行 caption，但留 \\n 會讓送進 T5/CLAP 的字串
    與 train 不一致。
    """
    rows = []
    with open(src) as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        if 'caption' not in fieldnames:
            raise ValueError(f'{src}: no `caption` column (cols={fieldnames})')
        n_prefixed = 0
        for row in reader:
            cap = row['caption'].replace('\n', ' ').replace('\r', ' ').strip()
            row['caption'] = PREFIX + cap
            n_prefixed += 1
            rows.append(row)

    with open(dst, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), n_prefixed


def main():
    """Hard-fail on any error — automation 不應在缺檔或部分輸出上繼續跑
    （Codex 2026-04-26 review P2.2）。"""
    print(f'Prefix: {PREFIX!r}\n')
    errors = []
    for src, dst in PAIRS:
        if not src.exists():
            errors.append(f'Missing source: {src}')
            print(f'❌ Missing source: {src}')
            continue
        n_rows, n_prefixed = prefix_tsv(src, dst)
        print(f'✅ {src.name}')
        print(f'   → {dst.name}')
        print(f'   rows: {n_rows:,}  prefixed: {n_prefixed:,}')

        # 預覽前 1 筆 + 行數驗證
        with open(dst) as f:
            f.readline()  # header
            sample = f.readline().rstrip('\n')
        # 取 caption 欄位前 100 字
        preview = sample.split('\t')[1][:100]
        print(f'   preview: {preview}...')

        # Normalize 後：output line count 應該 == row count + 1 (header)
        # 若 ≠，代表 normalize 沒成功（仍有 embedded newline）
        out_lines = sum(1 for _ in open(dst))
        expected = n_prefixed + 1
        ok = out_lines == expected
        status = "✅ no embedded \\n" if ok else "❌ embedded \\n still present"
        print(f'   line count: out={out_lines:,}  expected={expected:,}  {status}\n')
        if not ok:
            errors.append(f'{dst.name}: embedded newline not normalized '
                          f'(out={out_lines:,} vs expected={expected:,})')

    if errors:
        print('\n=== ERRORS ===')
        for e in errors:
            print(f'  ❌ {e}')
        raise SystemExit(f'Failed: {len(errors)} eval TSV(s) had errors')


if __name__ == '__main__':
    main()
