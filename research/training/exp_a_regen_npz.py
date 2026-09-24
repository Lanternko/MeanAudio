"""
EXP-A NPZ regen: rebuild text features for the destructured TSV
================================================================

Input:
  - destructured TSV (phase7_v1_train_destructured.tsv)
  - original NPZ dir (~/research/meanaudio_training/npz/) holding mean/std for each row
  - npz_cache_train.txt mapping TSV row N → NPZ filename

Output:
  - new NPZ dir with same filename mapping; each NPZ has:
      mean, std         — copied from original (audio latent unchanged)
      text_features     — fresh T5 (flan-t5-large) of destructured caption, padded to 77×1024
      text_features_c   — fresh CLAP (HTSAT-base) of destructured caption, 512

Cost: ~251K captions × T5+CLAP forward; on RTX 5090 with batch=64 estimate ~1.5-2 hours.

Run:
  python exp_a_regen_npz.py [--resume]

Resume support: skips NPZ files that already exist in dst dir with non-zero size.
"""

import os
import csv
import argparse
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import laion_clap
from transformers import AutoTokenizer, T5EncoderModel

CLAP_CKPT = '/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'
TSV       = Path('/home/kojiek/eval_tsvs_p100/phase7_v1_train_destructured.tsv')
SRC_NPZ   = Path('/home/kojiek/research/meanaudio_training/npz')
DST_NPZ   = Path('/home/kojiek/exps_nvme/npz_phase7v1_destructured')
CACHE_TXT = Path('/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt')
BATCH     = 32
MAX_LEN   = 77


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    DST_NPZ.mkdir(parents=True, exist_ok=True)
    print(f'src NPZ:  {SRC_NPZ}')
    print(f'dst NPZ:  {DST_NPZ}')

    # 1. Load destructured TSV
    rows = []
    with open(TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append(r)
    print(f'TSV rows: {len(rows):,}')

    # 2. Load gt_cache mapping
    with open(CACHE_TXT) as f:
        npz_files = [l.strip() for l in f if l.strip()]
    print(f'NPZ cache entries: {len(npz_files):,}')
    assert len(npz_files) == len(rows), \
        f'cache vs TSV row count mismatch: {len(npz_files)} vs {len(rows)}'

    # 3. Build skip-set if resume
    skip_idx = set()
    if args.resume:
        for i, fn in enumerate(npz_files):
            p = DST_NPZ / fn
            if p.exists() and p.stat().st_size > 0:
                skip_idx.add(i)
        print(f'Resume: skipping {len(skip_idx):,} already-done NPZ')

    todo = [(i, npz_files[i], rows[i]['caption']) for i in range(len(rows)) if i not in skip_idx]
    print(f'Todo:    {len(todo):,}')

    if not todo:
        print('Nothing to do.')
        return

    # 4. Load text encoders
    print('\n載入 T5...')
    tok = AutoTokenizer.from_pretrained('google/flan-t5-large')
    t5 = T5EncoderModel.from_pretrained('google/flan-t5-large').eval().to('cuda')
    print('載入 CLAP...')
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap.load_ckpt(CLAP_CKPT)
    clap.eval().to('cuda')

    # 5. Process in batches
    t0 = time.time()
    n_done = 0
    err = 0
    with torch.no_grad():
        for bi in tqdm(range(0, len(todo), BATCH), desc='regen'):
            batch = todo[bi:bi+BATCH]
            idxs   = [b[0] for b in batch]
            files  = [b[1] for b in batch]
            caps   = [b[2] for b in batch]

            # T5 encode (77 × 1024)
            tk = tok(caps, return_tensors='pt', padding='max_length',
                     truncation=True, max_length=MAX_LEN).to('cuda')
            t5_out = t5(**tk).last_hidden_state.cpu().numpy()    # (B, 77, 1024)
            text_attention_mask = tk.attention_mask.detach().cpu().numpy().astype(bool)

            # CLAP text embed (512)
            clap_emb = clap.get_text_embedding(caps, use_tensor=True).cpu().numpy()   # (B, 512)

            for i in range(len(batch)):
                src_path = SRC_NPZ / files[i]
                dst_path = DST_NPZ / files[i]
                if not src_path.exists():
                    err += 1
                    continue
                z = np.load(src_path)
                np.savez(dst_path,
                         mean=z['mean'].astype('float32'),
                         std=z['std'].astype('float32'),
                         text_features=t5_out[i].astype('float32'),
                         text_features_c=clap_emb[i].astype('float32'),
                         text_attention_mask=text_attention_mask[i])
                n_done += 1

    dt = time.time() - t0
    print(f'\nDone. n={n_done:,} err={err}  elapsed={dt/60:.1f} min')
    print(f'avg rate: {n_done/max(1, dt):.1f} captions/sec')


if __name__ == '__main__':
    main()
