"""
Slice multi-cap NPZ → single-cap NPZ for Qwen rerun configs.

Source: ~/phase9_5_multicap_npz/{idx}.npz
  - text_features:   (5, 77, 1024)
  - text_features_c: (5, 512)
  - text_attention_mask: (5, 77), if produced by the source cache
  - mean:            (312, 20)
  - std:             (312, 20)

Per clip, pick a cap_idx based on selection variant and slice:
  text_features  → (77, 1024)
  text_features_c → (512,)
  text_attention_mask → (77,)
mean/std unchanged.

This is pure I/O (no GPU, no T5/CLAP re-encoding) — much faster
than gen_multicap_npz.py since text features are already encoded.

Usage:
  python slice_qwen_singlecap_npz.py --variant random
  python slice_qwen_singlecap_npz.py --variant bestconsensus

Variants:
  random          → ~/phase9_5_random_singlecap_npz   (P8-Qwen + P7V1-Qwen)
  bestconsensus   → ~/phase9_5_bc_singlecap_npz       (P4V2-Qwen)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

INPUT_TSV   = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')  # canonical id order
SRC_NPZ_DIR = Path.home() / 'phase9_5_multicap_npz'
SEL_JSON    = Path.home() / 'research/meanaudio_training/qwen_singlecap_selections.json'

VARIANT_TO_DST = {
    'random':         Path.home() / 'phase9_5_random_singlecap_npz',
    'bestconsensus':  Path.home() / 'phase9_5_bc_singlecap_npz',
}

VARIANT_TO_KEY = {
    'random':         'random_seed42_idx',
    'bestconsensus':  'bestconsensus_idx',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=list(VARIANT_TO_DST), required=True)
    p.add_argument('--limit', type=int, default=None, help='Test on first N clips')
    p.add_argument('--resume', action='store_true', help='Skip already-existing dst NPZs')
    return p.parse_args()


def main():
    args = parse_args()
    dst_dir = VARIANT_TO_DST[args.variant]
    sel_key = VARIANT_TO_KEY[args.variant]
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f'Variant : {args.variant}')
    print(f'Src     : {SRC_NPZ_DIR}')
    print(f'Dst     : {dst_dir}')
    print(f'TSV     : {INPUT_TSV}')

    # Load id-order TSV (drives idx → id mapping; same order as gen_multicap_npz.py wrote)
    with open(INPUT_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if args.limit:
        rows = rows[:args.limit]
    n = len(rows)
    print(f'  {n:,} rows')

    # Load selections
    print(f'Loading selections: {SEL_JSON}')
    sel = json.loads(SEL_JSON.read_text())
    print(f'  {len(sel):,} clips have selections')

    # Slice loop
    n_done = 0
    n_skipped = 0
    n_fallback_idx0 = 0
    bad_shape = []
    for idx, row in enumerate(tqdm(rows, desc=f'slicing→{args.variant}')):
        cid = row['id']
        dst_path = dst_dir / f'{idx}.npz'
        if args.resume and dst_path.exists():
            n_skipped += 1
            continue

        src_path = SRC_NPZ_DIR / f'{idx}.npz'
        if not src_path.exists():
            print(f'\n[WARN] missing src {src_path}')
            continue

        s = sel.get(cid)
        if s is None or s.get(sel_key) is None:
            cap_idx = 0  # fallback
            n_fallback_idx0 += 1
        else:
            cap_idx = int(s[sel_key])

        d = np.load(src_path)
        tf  = d['text_features']     # (5, 77, 1024)
        tfc = d['text_features_c']   # (5, 512)
        tam = d['text_attention_mask'] if 'text_attention_mask' in d.files \
            else np.ones(tf.shape[:2], dtype=bool)
        if tf.shape != (5, 77, 1024) or tfc.shape != (5, 512):
            bad_shape.append((idx, tf.shape, tfc.shape))
            continue

        np.savez(
            dst_path,
            mean=d['mean'],
            std=d['std'],
            text_features=tf[cap_idx],     # → (77, 1024)
            text_features_c=tfc[cap_idx],  # → (512,)
            text_attention_mask=tam[cap_idx],  # → (77,)
        )
        n_done += 1

    print(f'\n  written:  {n_done:,}')
    print(f'  skipped (resume): {n_skipped:,}')
    print(f'  fallback to idx=0 (no selection): {n_fallback_idx0:,}')
    if bad_shape:
        print(f'  bad source shape: {len(bad_shape)}')
        for idx, t1, t2 in bad_shape[:5]:
            print(f'    idx {idx}: tf={t1} tfc={t2}')

    # Sanity: open one random output, confirm shapes
    if n_done:
        sample = np.load(dst_dir / '0.npz')
        print(f'\n[sanity] {dst_dir}/0.npz:')
        for k in sample.files:
            print(f'  {k}: shape={sample[k].shape} dtype={sample[k].dtype}')

    print(f'\n✅ done.')


if __name__ == '__main__':
    main()
