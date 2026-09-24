"""
EXP-F data preparation: 50% LP-MC / 50% Qwen slot-0 anchor mixing

Design:
  - Canonical row set = phase7_v1_train.tsv (251,599 rows), matched 1:1 to npz_cache_train.txt
  - For each audio clip, randomly assign LP-MC or Qwen caption (seed=42, 50/50)
  - Create NPZ dir via symlinks — NO re-encoding of T5/CLAP, takes ~5 min
  - Write mixed TSV for documentation

Output:
  - ~/eval_tsvs_p100/exp_f_50mix_train.tsv   (251,599 rows, mixed captions)
  - ~/exps_nvme/npz_expf_50mix/              (251,599 symlinks)

Hypothesis: LP-MC anchor exposure at 50% prevents the co-adaptation collapse
            that happens when 100% Qwen slot-0 captions are used (EXP-B → CLAP 0.0615).
"""

import os, csv, random, sys
from pathlib import Path

LPMC_TSV  = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
QWEN_TSV  = Path('/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv')
NPZ_CACHE = Path('/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt')

LPMC_NPZ  = Path('/home/kojiek/research/meanaudio_training/npz_phase8v4')
QWEN_NPZ  = Path('/home/kojiek/exps_nvme/npz_qwen_slot0')
DST_NPZ   = Path('/home/kojiek/exps_nvme/npz_expf_50mix')
DST_TSV   = Path('/home/kojiek/eval_tsvs_p100/exp_f_50mix_train.tsv')

SEED      = 42
MIX_RATIO = 0.50   # fraction assigned to LP-MC


def main():
    rng = random.Random(SEED)

    print(f'Loading LP-MC TSV: {LPMC_TSV}')
    lpmc_rows = []
    with open(LPMC_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            lpmc_rows.append(r)
    print(f'  {len(lpmc_rows):,} rows')

    print(f'Loading Qwen TSV: {QWEN_TSV}')
    qwen_by_id = {}
    with open(QWEN_TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            qwen_by_id[r['id']] = r['caption']
    print(f'  {len(qwen_by_id):,} unique ids')

    print(f'Loading NPZ cache: {NPZ_CACHE}')
    with open(NPZ_CACHE) as f:
        npz_fnames = [l.strip() for l in f if l.strip()]
    assert len(npz_fnames) == len(lpmc_rows), \
        f'cache/TSV mismatch: {len(npz_fnames)} vs {len(lpmc_rows)}'
    print(f'  {len(npz_fnames):,} entries')

    DST_NPZ.mkdir(parents=True, exist_ok=True)

    n_lpmc = 0; n_qwen = 0; n_qwen_missing = 0; n_already = 0; n_err = 0

    out_rows = []
    for i, (row, fname) in enumerate(zip(lpmc_rows, npz_fnames)):
        clip_id = row['id']
        q_level = row.get('q_level', '')

        use_lpmc = (rng.random() < MIX_RATIO)

        # Qwen fallback if missing
        if not use_lpmc and clip_id not in qwen_by_id:
            n_qwen_missing += 1
            use_lpmc = True

        if use_lpmc:
            caption = row['caption']
            src_npz = LPMC_NPZ / fname
            n_lpmc += 1
        else:
            caption = qwen_by_id[clip_id]
            src_npz = QWEN_NPZ / fname
            n_qwen += 1

        dst_link = DST_NPZ / fname

        if dst_link.exists():
            n_already += 1
        else:
            if not src_npz.exists():
                print(f'  MISSING src: {src_npz}')
                n_err += 1
                continue
            os.symlink(src_npz, dst_link)

        out_rows.append({'id': clip_id, 'caption': caption, 'q_level': q_level})

        if i % 50000 == 0:
            print(f'  [{i:,}/{len(lpmc_rows):,}] lpmc={n_lpmc} qwen={n_qwen} already={n_already}')

    print(f'\nSymlink summary:')
    print(f'  LP-MC assigned : {n_lpmc:,} ({n_lpmc/len(lpmc_rows)*100:.1f}%)')
    print(f'  Qwen  assigned : {n_qwen:,} ({n_qwen/len(lpmc_rows)*100:.1f}%)')
    print(f'  Qwen fallback→LPMC (missing) : {n_qwen_missing}')
    print(f'  Already existed : {n_already}')
    print(f'  Errors : {n_err}')

    # Verify all links in DST_NPZ
    created = len(list(DST_NPZ.iterdir()))
    print(f'  DST_NPZ total files: {created:,}')

    # Write mixed TSV
    print(f'\nWriting mixed TSV: {DST_TSV}')
    with open(DST_TSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'caption', 'q_level'], delimiter='\t')
        writer.writeheader()
        writer.writerows(out_rows)
    print(f'  Written {len(out_rows):,} rows')

    print('\nDone. Sanity check a few entries:')
    for i in [0, 1000, 50000, 125000, 200000, 251598]:
        if i < len(out_rows):
            src_assigned = 'LPMC' if out_rows[i]['caption'] == lpmc_rows[i]['caption'] else 'Qwen'
            print(f'  row {i}: {src_assigned} — "{out_rows[i]["caption"][:60]}..."')


if __name__ == '__main__':
    main()
