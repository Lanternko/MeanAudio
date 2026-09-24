"""
EXP-H NPZ re-encode: build text features for EXP-H rewritten captions
======================================================================

Input:
  - EXP-H rewrite TSV (~/eval_tsvs_p100/expH_rewrite_train.tsv)
  - source NPZ dir (npz_phase8v4) for audio mean/std features
  - npz_cache_train.txt mapping TSV row N → NPZ filename

Output:
  - ~/exps_nvme/npz_expH_rewrite/  each NPZ has:
      mean, std           — copied from source (audio latent unchanged)
      text_features       — fresh T5 (flan-t5-large) of rewritten caption (77×1024)
      text_features_c     — fresh CLAP (HTSAT-base) of rewritten caption (512,)
      text_attention_mask — T5 tokenizer attention mask (77,)

Cost: ~251K captions × T5+CLAP forward; estimate ~1.5-2 hours on RTX 5090.

Run:
  python gen_expH_npz.py [--resume]

Resume support: skips NPZ files that already exist in dst dir with non-zero size.
"""

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
TSV       = Path('/home/kojiek/eval_tsvs_p100/expH_rewrite_train.tsv')
# npz_phase8v4 has valid audio mean/std; original lpmc_singlecap_npz_archive was
# deleted by disk cleanup but npz_phase8v4 carries the same audio features.
SRC_NPZ   = Path('/home/kojiek/research/meanaudio_training/npz_phase8v4')
DST_NPZ   = Path('/home/kojiek/exps_nvme/npz_expH_rewrite')
CACHE_TXT = Path('/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt')
BATCH     = 32
MAX_LEN   = 77


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    DST_NPZ.mkdir(parents=True, exist_ok=True)
    print(f'src NPZ : {SRC_NPZ}')
    print(f'dst NPZ : {DST_NPZ}')
    print(f'TSV     : {TSV}')
    print(f'cache   : {CACHE_TXT}')

    # 1. Load EXP-H rewrite TSV
    rows = []
    with open(TSV) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            rows.append(r)
    print(f'TSV rows: {len(rows):,}')
    assert len(rows) == 251599, f'Expected 251599 rows, got {len(rows)}'

    # 2. Load gt_cache mapping (row index → NPZ filename)
    with open(CACHE_TXT) as f:
        npz_files = [l.strip() for l in f if l.strip()]
    print(f'Cache entries: {len(npz_files):,}')
    assert len(npz_files) == len(rows), \
        f'cache vs TSV mismatch: {len(npz_files)} vs {len(rows)}'

    # 3. Build skip-set if resume
    skip_idx = set()
    if args.resume:
        for i, fn in enumerate(npz_files):
            p = DST_NPZ / fn
            if p.exists() and p.stat().st_size > 0:
                skip_idx.add(i)
        print(f'Resume: skipping {len(skip_idx):,} already-done')

    todo = [(i, npz_files[i], rows[i]['caption'])
            for i in range(len(rows)) if i not in skip_idx]
    print(f'Todo   : {len(todo):,}')

    if not todo:
        print('Nothing to do.')
        return

    # 4. Sanity: check a few source NPZ files exist
    missing_src = sum(1 for _, fn, _ in todo[:100] if not (SRC_NPZ / fn).exists())
    if missing_src > 0:
        print(f'WARNING: {missing_src}/100 source NPZ missing in first 100 — check SRC_NPZ path')

    # 5. Load text encoders
    print('\nLoading T5 (flan-t5-large)...')
    tok = AutoTokenizer.from_pretrained('google/flan-t5-large')
    t5  = T5EncoderModel.from_pretrained('google/flan-t5-large').eval().to('cuda')
    print('Loading CLAP (HTSAT-base)...')
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap.load_ckpt(CLAP_CKPT)
    clap.eval().to('cuda')
    print('Encoders ready.')

    # 6. Process in batches
    t0 = time.time()
    n_done = n_err = n_src_missing = 0

    with torch.no_grad():
        for bi in tqdm(range(0, len(todo), BATCH), desc='npz-regen'):
            batch = todo[bi: bi + BATCH]
            idxs  = [b[0] for b in batch]
            files = [b[1] for b in batch]
            caps  = [b[2] for b in batch]

            # T5 encode → (B, 77, 1024)
            tk = tok(caps, return_tensors='pt', padding='max_length',
                     truncation=True, max_length=MAX_LEN).to('cuda')
            t5_out = t5(**tk).last_hidden_state.cpu().numpy()
            text_attention_mask = tk.attention_mask.detach().cpu().numpy().astype(bool)

            # CLAP text embed → (B, 512)
            clap_emb = clap.get_text_embedding(caps, use_tensor=True).cpu().numpy()

            for i in range(len(batch)):
                src_path = SRC_NPZ / files[i]
                dst_path = DST_NPZ / files[i]
                if not src_path.exists():
                    n_src_missing += 1
                    n_err += 1
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
    print(f'\n=== Done ===')
    print(f'  written    : {n_done:,}')
    print(f'  errors     : {n_err:,}  (src missing: {n_src_missing:,})')
    print(f'  elapsed    : {dt/60:.1f} min')
    print(f'  rate       : {n_done/max(1,dt):.1f} captions/sec')
    print(f'  dst dir    : {DST_NPZ}')

    # 7. Quick sanity on one output NPZ
    sample_fn = npz_files[0]
    sample_path = DST_NPZ / sample_fn
    if sample_path.exists():
        z = np.load(sample_path)
        print(f'\n[sanity] {sample_path.name}:')
        for k in z.files:
            print(f'  {k}: shape={z[k].shape} dtype={z[k].dtype}')
        expected = {
            'mean': (312, 20), 'std': (312, 20),
            'text_features': (77, 1024), 'text_features_c': (512,),
            'text_attention_mask': (77,)
        }
        for k, shape in expected.items():
            ok = z[k].shape == shape
            print(f'  {k} shape {"✅" if ok else "❌"}: expected {shape}, got {z[k].shape}')


if __name__ == '__main__':
    main()
