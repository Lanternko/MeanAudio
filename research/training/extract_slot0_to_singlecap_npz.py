"""
從 multi-cap NPZ (text_features: [5, 77, 1024], text_features_c: [5, 512]) 抽出
slot 0 作為 fixed single-cap NPZ (text_features: [77, 1024], text_features_c: [512])。

用途：Phase 9 V1 Stage 2 salvage 實驗 — S1 保留 multi-cap random 訓練，
S2 改用 fixed single-cap，驗證「multi_cap 是否只在 Stage 2 造成 MeanFlow 不收斂」。

用法：
  python extract_slot0_to_singlecap_npz.py \
    --src  ~/phase9_multicap_npz \
    --dst  ~/phase9_singlecap_slot0_npz \
    --slot 0

Slot 0 = LP-MusicCaps JSONL caption_details[0]（seed 0 generated）。
"""

import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Multi-cap NPZ source dir')
    p.add_argument('--dst', required=True, help='Single-cap NPZ output dir')
    p.add_argument('--slot', type=int, default=0, help='Which caption slot (0-4) to extract')
    p.add_argument('--skip_existing', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in os.listdir(src) if f.endswith('.npz')],
                   key=lambda x: int(x[:-4]))
    print(f'Source: {src}  ({len(files)} files)')
    print(f'Dest  : {dst}')
    print(f'Slot  : {args.slot}')

    n_skip = 0
    n_done = 0
    for f in tqdm(files, desc='extract slot'):
        out = dst / f
        if args.skip_existing and out.exists():
            n_skip += 1
            continue
        src_data = np.load(src / f)
        text_attention_mask = src_data['text_attention_mask'][args.slot] \
            if 'text_attention_mask' in src_data.files else np.ones((77,), dtype=bool)
        np.savez(
            out,
            mean=src_data['mean'],
            std=src_data['std'],
            text_features=src_data['text_features'][args.slot],       # [77, 1024]
            text_features_c=src_data['text_features_c'][args.slot],   # [512]
            text_attention_mask=text_attention_mask,                   # [77]
        )
        n_done += 1

    print(f'\nDone: wrote {n_done}, skipped existing {n_skip}')

    # Sanity: verify first file
    sample = np.load(dst / files[0])
    print(f'\nSample {files[0]}:')
    for k in sample.keys():
        print(f'  {k}: shape={sample[k].shape}')


if __name__ == '__main__':
    main()
