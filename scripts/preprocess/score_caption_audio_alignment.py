#!/usr/bin/env python
"""Audio-caption alignment of several caption columns for the same training clips.

Audio: segments_no_vocals mp3, FIRST 10 s (the window the c2p0 / MF slice10 captioners
saw), 48 kHz, int16 round-trip, one clip per forward -- the laion_clap input contract.
Text: laion_clap truncates at 77 RoBERTa tokens, i.e. the CLAP path sees what training
sees. T5 token counts (FLAN-T5 tokenizer, the other training text path) are reported too.

Per column: mean cosine, retrieval R@1/R@10 within the pool (text -> audio), T5 length
p50 and share > 77. Paired differences vs --ref with a bootstrap CI.

This is a corpus diagnostic, not a training filter (no row is dropped by CLAP).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

AUDIO_ROOT = Path('/mnt/HDD/hsiehyian/segments_no_vocals')
CLAP_CKPT = '/home/kojiek/MeanAudio/weights/music_speech_audioset_epoch_15_esc_89.98.pt'


def id_to_audio_path(clip_id):
    parts = clip_id.split('_')
    s = parts.index('segment')
    return AUDIO_ROOT / '_'.join(parts[:s - 1]) / parts[s - 1] / f'segment_{parts[s + 1]}.mp3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--cols', required=True, help='comma-separated caption columns')
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out', required=True, help='json summary; per-clip tsv next to it')
    a = ap.parse_args()
    cols = a.cols.split(',')

    with open(a.tsv, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))

    import librosa
    import torch
    import laion_clap
    from transformers import AutoTokenizer
    from tqdm import tqdm

    tok = AutoTokenizer.from_pretrained('google/flan-t5-large')
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(CLAP_CKPT)
    model.eval().to('cuda')

    A, keep = [], []
    with torch.no_grad():
        for r in tqdm(rows, desc='audio'):
            p = id_to_audio_path(r['id'])
            if not p.exists():
                continue
            w, _ = librosa.load(p, sr=48000, duration=10.0)
            w = (np.clip(w, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32767
            w = np.pad(w, (0, max(0, 480000 - len(w))))[:480000]
            A.append(model.get_audio_embedding_from_data(x=w[None], use_tensor=False)[0])
            keep.append(r)
    A = np.stack(A)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    summary, per = {'n': len(keep), 'missing_audio': len(rows) - len(keep)}, {}
    with torch.no_grad():
        for c in cols:
            T = np.stack([model.get_text_embedding([r[c]], use_tensor=False)[0] for r in tqdm(keep, desc=c)])
            T /= np.linalg.norm(T, axis=1, keepdims=True)
            S = T @ A.T
            diag = np.diag(S)
            rank = (S > diag[:, None]).sum(1)
            lens = np.array([len(tok(r[c]).input_ids) for r in keep])
            per[c] = diag
            summary[c] = {'clap_mean': float(diag.mean()), 'R@1': float((rank < 1).mean()),
                          'R@10': float((rank < 10).mean()), 't5_p50': float(np.median(lens)),
                          't5_over77': float((lens > 77).mean())}

    rng = np.random.default_rng(0)
    for c in cols:
        if c == a.ref:
            continue
        d = per[c] - per[a.ref]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        summary[c][f'd_clap_vs_{a.ref}'] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    Path(a.out).write_text(json.dumps(summary, indent=1))
    with open(Path(a.out).with_suffix('.per_clip.tsv'), 'w') as f:
        f.write('id\t' + '\t'.join(cols) + '\n')
        for k, r in enumerate(keep):
            f.write(r['id'] + '\t' + '\t'.join(f'{per[c][k]:.5f}' for c in cols) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
