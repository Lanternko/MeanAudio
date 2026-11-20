#!/usr/bin/env python
"""178 features: frozen MERT, random-weight MERT and AES over the rendered stimuli.

  python scripts/eval/wrongnote_probe_178_features.py --root <dir> --what mert
  python scripts/eval/wrongnote_probe_178_features.py --root <dir> --what aes

Frozen MERT reuses load_mert / MERT_REVISION / load_24k from metric_win_tie_133.py
(prereg §2). Each clip -> 13 layer means, shape (13, 768). The random-weight control is
AutoModel.from_config(seed 0) through the same extractor.

Chunks of CHUNK rows are written to <root>/features/<name>/chunk_XXXXX.npz (idx, x) so a
restart skips finished chunks; assemble() merges them into <name>.npy in manifest order.
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrongnote_probe_178_common as C  # noqa: E402
import metric_win_tie_133 as M133  # noqa: E402

CHUNK = 128
BATCH = 16
AES_KEYS = ('CE', 'CU', 'PC', 'PQ')


def chunk_dir(root, name):
    d = Path(root) / 'features' / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def chunks_of(n):
    return [list(range(i, min(i + CHUNK, n))) for i in range(0, n, CHUNK)]


def done_chunks(root, name):
    d = Path(root) / 'features' / name
    if not d.is_dir():
        return set()
    return {int(p.stem.split('_')[-1]) for p in d.glob('chunk_*.npz')}


def save_chunk(root, name, ci, idx, x):
    tmp = chunk_dir(root, name) / f'chunk_{ci:05d}.npz.tmp.npz'
    np.savez(tmp, idx=np.asarray(idx, dtype=np.int64), x=np.asarray(x, dtype=np.float32))
    os.replace(tmp, chunk_dir(root, name) / f'chunk_{ci:05d}.npz')


def assemble(root, name, n):
    """Merge chunks into <name>.npy in manifest order; returns (array, meta)."""
    d = Path(root) / 'features' / name
    parts = sorted(d.glob('chunk_*.npz'))
    if not parts:
        raise SystemExit(f'[FAIL] no chunks for {name}')
    idx_all, xs = [], []
    for p in parts:
        z = np.load(p)
        idx_all.append(z['idx'])
        xs.append(z['x'])
    idx = np.concatenate(idx_all)
    x = np.concatenate(xs)
    order = np.argsort(idx)
    idx, x = idx[order], x[order]
    if len(idx) != n or not np.array_equal(idx, np.arange(n)):
        raise SystemExit(f'[FAIL] {name}: {len(idx)} rows / idx mismatch vs manifest n={n}')
    if not np.all(np.isfinite(x)):
        raise SystemExit(f'[FAIL] {name}: non-finite features')
    np.save(d / f'{name}.npy', x)
    meta = {'name': name, 'n': int(n), 'shape': list(x.shape), 'chunks': len(parts),
            'finite': True, 'mean': float(x.mean()), 'std': float(x.std())}
    C.write_json(d / 'meta.json', meta)
    return x, meta


def load_mert_random(device):
    import torch
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(M133.MERT_REPO, revision=M133.MERT_REVISION,
                                     trust_remote_code=True)
    torch.manual_seed(0)
    mdl = AutoModel.from_config(cfg).to(device).eval()
    return mdl


def embed_layers(wavs, feat, mdl, device):
    """(B, 13, 768): each hidden state time-averaged (prereg §2)."""
    import torch
    inputs = feat(wavs, sampling_rate=C.EMBED_SR, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        hs = mdl(**inputs, output_hidden_states=True).hidden_states
    return np.stack([h.mean(dim=1).float().cpu().numpy() for h in hs], axis=1)


def run_mert(root, rows, device):
    import torch
    feat, mdl = M133.load_mert(device)
    rnd = load_mert_random(device)
    n = len(rows)
    paths = [r['path'] for r in rows]
    have = done_chunks(root, 'mert') & done_chunks(root, 'mert_random')
    todo = [ci for ci, idx in enumerate(chunks_of(n)) if ci not in have]
    print(f'[feat mert] {n} clips, {len(todo)} chunks to embed', flush=True)
    t0 = time.time()
    for k, ci in enumerate(todo):
        idx = chunks_of(n)[ci]
        wavs = [M133.load_24k(paths[i]) for i in idx]
        outs = []
        for s in range(0, len(wavs), BATCH):
            outs.append(embed_layers(wavs[s:s + BATCH], feat, mdl, device))
        save_chunk(root, 'mert', ci, idx, np.concatenate(outs))
        outs = []
        for s in range(0, len(wavs), BATCH):
            outs.append(embed_layers(wavs[s:s + BATCH], feat, rnd, device))
        save_chunk(root, 'mert_random', ci, idx, np.concatenate(outs))
        del wavs
        torch.cuda.empty_cache()
        print(f'[feat mert] chunk {ci} {k + 1}/{len(todo)} '
              f'({(time.time() - t0) / (k + 1):.2f} s/chunk)', flush=True)
    return {'mert': assemble(root, 'mert', n), 'mert_random': assemble(root, 'mert_random', n)}


def run_aes(root, rows):
    from eval_metrics import score_aes
    n = len(rows)
    paths = [r['path'] for r in rows]
    have = done_chunks(root, 'aes')
    todo = [ci for ci, idx in enumerate(chunks_of(n)) if ci not in have]
    print(f'[feat aes] {n} clips, {len(todo)} chunks to score', flush=True)
    t0 = time.time()
    for k, ci in enumerate(todo):
        idx = chunks_of(n)[ci]
        per, failed = score_aes([paths[i] for i in idx], batch_size=32, progress=False)
        if failed:
            raise SystemExit(f'[FAIL] AES failed on {len(failed)} clips: {list(failed)[:3]}')
        x = np.asarray([[per[paths[i]][key] for key in AES_KEYS] for i in idx], dtype=np.float32)
        save_chunk(root, 'aes', ci, idx, x)
        print(f'[feat aes] chunk {ci} {k + 1}/{len(todo)} '
              f'({(time.time() - t0) / (k + 1):.2f} s/chunk)', flush=True)
    return {'aes': assemble(root, 'aes', n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(C.ROOT))
    ap.add_argument('--what', choices=('mert', 'aes'), required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    rows = C.read_tsv(Path(a.root) / 'manifest.tsv')
    t = time.time()
    out = run_mert(a.root, rows, a.device) if a.what == 'mert' else run_aes(a.root, rows)
    for name, (x, meta) in out.items():
        print(f'[feat] {name} shape={x.shape} mean={meta["mean"]:.4f} std={meta["std"]:.4f}')
    print(f'[feat {a.what}] {len(rows)} clips in {time.time() - t:.1f} s', flush=True)


if __name__ == '__main__':
    main()
