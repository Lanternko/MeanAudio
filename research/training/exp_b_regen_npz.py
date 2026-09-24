"""
EXP-B NPZ regen: rebuild text features for qwen_slot0_train.tsv

Reuse audio mean/std from /home/kojiek/research/meanaudio_training/npz/
(via npz_cache_train.txt), recompute T5 + CLAP for slot 0 caption.

Output: /home/kojiek/exps_nvme/npz_qwen_slot0
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
TSV       = Path('/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv')
SRC_NPZ   = Path('/home/kojiek/research/meanaudio_training/npz_phase8v4')  # 5/10: original npz/ was deleted by disk cleanup; npz_phase8v4 has same audio mean/std (text features overwritten anyway)
DST_NPZ   = Path('/home/kojiek/exps_nvme/npz_qwen_slot0')
CACHE_TXT = Path('/mnt/HDD/kojiek/phase4_jamendo_data/npz_cache_train.txt')
BATCH     = 32
MAX_LEN   = 77


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dst', type=Path, default=DST_NPZ)
    ap.add_argument('--batch_size', type=int, default=BATCH)
    args = ap.parse_args()

    dst_npz = args.dst.expanduser()
    dst_npz.mkdir(parents=True, exist_ok=True)
    print(f'src: {SRC_NPZ}\ndst: {dst_npz}')

    rows = []
    with open(TSV) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows.append(r)
    print(f'TSV rows: {len(rows):,}')

    with open(CACHE_TXT) as f:
        npz_files = [l.strip() for l in f if l.strip()]
    assert len(npz_files) == len(rows), f'cache vs TSV mismatch: {len(npz_files)} vs {len(rows)}'

    skip_idx = set()
    if args.resume:
        for i, fn in enumerate(npz_files):
            p = dst_npz / fn
            if p.exists() and p.stat().st_size > 0:
                skip_idx.add(i)
        print(f'Resume: skip {len(skip_idx):,}')

    todo = [(i, npz_files[i], rows[i]['caption']) for i in range(len(rows)) if i not in skip_idx]
    print(f'Todo: {len(todo):,}')
    if not todo: return

    print('載入 T5 + CLAP...')
    tok = AutoTokenizer.from_pretrained('google/flan-t5-large')
    t5 = T5EncoderModel.from_pretrained('google/flan-t5-large').eval().to('cuda')
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap.load_ckpt(CLAP_CKPT)
    clap.eval().to('cuda')

    t0 = time.time(); n_done = 0; err = 0
    with torch.no_grad():
        for bi in tqdm(range(0, len(todo), args.batch_size), desc='regen'):
            batch = todo[bi:bi+args.batch_size]
            files = [b[1] for b in batch]
            caps  = [b[2] for b in batch]
            tk = tok(caps, return_tensors='pt', padding='max_length',
                     truncation=True, max_length=MAX_LEN).to('cuda')
            t5_out = t5(**tk).last_hidden_state.cpu().numpy()
            text_attention_mask = tk.attention_mask.detach().cpu().numpy().astype(bool)
            clap_emb = clap.get_text_embedding(caps, use_tensor=True).cpu().numpy()
            for i in range(len(batch)):
                src = SRC_NPZ / files[i]
                dst = dst_npz / files[i]
                if not src.exists(): err += 1; continue
                z = np.load(src)
                np.savez(dst,
                         mean=z['mean'].astype('float32'),
                         std=z['std'].astype('float32'),
                         text_features=t5_out[i].astype('float32'),
                         text_features_c=clap_emb[i].astype('float32'),
                         text_attention_mask=text_attention_mask[i])
                n_done += 1
    dt = time.time() - t0
    print(f'\nDone n={n_done:,} err={err} elapsed={dt/60:.1f} min ({n_done/max(1,dt):.1f}/s)')


if __name__ == '__main__':
    main()
