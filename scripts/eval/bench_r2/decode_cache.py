#!/usr/bin/env python3
"""Decode every source once and write the variant cache (preregistered §3.1–3.3).

Cache  : <ROOT>/derived/<dataset>/<unit_key>[__w<k>|__full].wav   float32 mono at the SOURCE sample rate
Index  : <ROOT>/derived/<dataset>/index.tsv  (one row per unit; invalid units are listed with a reason)
Rules  : mono = channel mean; no resampling, no VAD, no zero padding.
         V5/V10/V30 = centre crop (shorter audio -> full length).
         VFULL = non-overlapping 30 s windows from the start, tail < 5 s dropped (files `__w<k>`), plus the
                 whole track as `__full` (for MEva, which scores whole files).
         norm = pyloudnorm integrated LUFS -> -23.0, pure gain, no limiter, over_fs recorded; -inf -> invalid.
                The gain is measured on the audio that is scored (the crop for V*, the whole track for VFULL).
Usage  : python decode_cache.py <dataset> [--job J] [--limit N] [--workers W]
Run with `-I` (downloaded audio is untrusted; nothing from the data directories is executed).
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'  # forked pool workers: avoid BLAS oversubscription (pyloudnorm / scipy)
import numpy as np

import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location('bench_r2_datasets', Path(__file__).with_name('bench_datasets.py'))
D = importlib.util.module_from_spec(_sp)  # not `import datasets`: would shadow the HuggingFace package
sys.modules['bench_r2_datasets'] = D
_sp.loader.exec_module(D)
DER, PLAN, in_subset, read_manifest, unit_id, unit_key = D.DER, D.PLAN, D.in_subset, D.read_manifest, D.unit_id, D.unit_key

TARGET_LUFS = -23.0
WIN = 30.0
MIN_TAIL = 5.0
CROP = {'V5': 5.0, 'V10': 10.0, 'V30': 30.0}
IDX_COLS = ['unit_id', 'key', 'variant', 'loud', 'sr', 'n_samples', 'dur', 'n_win', 'files', 'full', 'invalid',
            'gain_db', 'over_fs', 'peak', 'lufs_in', 'lufs_out']


def hms(s):
    h, m, sec = s.split(':')
    return int(h) * 3600 + int(m) * 60 + float(sec)


def read_source(item):
    """-> (float32 mono, sr)."""
    import soundfile as sf
    src = item['src']
    if src.startswith('aime:'):
        _, shard, row, b, e = src.split(':')
        return _aime_cut(shard, int(row), hms(b), hms(e))
    a, sr = sf.read(src, dtype='float32', always_2d=True)
    return a.mean(axis=1).astype(np.float32), int(sr)


_shard_cache = {}


def _aime_cut(shard, row, b, e):
    import soundfile as sf
    import pyarrow.parquet as pq
    if shard not in _shard_cache:
        _shard_cache.clear()
        _shard_cache[shard] = pq.read_table(shard, columns=['audio']).column('audio').to_pylist()
    a, sr = sf.read(io.BytesIO(_shard_cache[shard][row]['bytes']), dtype='float32', always_2d=True)
    a = a.mean(axis=1).astype(np.float32)
    i, j = int(round(b * sr)), int(round(e * sr))
    return a[i:min(j, len(a))], int(sr)


def centre(a, sr, sec):
    n = int(round(sec * sr))
    if len(a) <= n:
        return a
    s = (len(a) - n) // 2
    return a[s:s + n]


def windows(a, sr):
    n = int(WIN * sr)
    out = []
    for s in range(0, len(a), n):
        w = a[s:s + n]
        if len(w) >= MIN_TAIL * sr or (not out and len(w) > 0 and s == 0):
            out.append(w)
    return out


def loudness(a, sr):
    import pyloudnorm as pyln
    if len(a) < int(0.4 * sr) + 1:
        return -math.inf
    with np.errstate(all='ignore'):
        v = pyln.Meter(sr).integrated_loudness(a.astype(np.float64))
    return float(v) if np.isfinite(v) else -math.inf


def save(path, a, sr):
    import soundfile as sf
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.wav')
    sf.write(str(tmp), a, sr, subtype='FLOAT')
    os.replace(tmp, path)


def decode_item(args):
    item, todo, outdir = args  # todo = [(variant, loud)]
    outdir = Path(outdir)
    rows = []
    try:
        a, sr = read_source(item)
    except Exception as e:  # unreadable source -> every unit invalid
        return [dict(unit_id=unit_id(item['dataset'], item['item_id'], v, l), key=unit_key(item, v, l), variant=v,
                     loud=l, invalid=f'read_error:{type(e).__name__}:{e}'[:200]) for v, l in todo]
    for variant, loud in todo:
        key = unit_key(item, variant, loud)
        row = dict(unit_id=unit_id(item['dataset'], item['item_id'], variant, loud), key=key, variant=variant,
                   loud=loud, sr=sr, invalid='')
        try:
            if variant == 'VFULL':
                segs, full = windows(a, sr), a
            elif variant in CROP:
                segs, full = [centre(a, sr, CROP[variant])], None
            else:
                segs, full = [a], None
            ref = full if full is not None else segs[0]
            gain, lin = 0.0, loudness(ref, sr)
            row['lufs_in'] = round(lin, 3) if np.isfinite(lin) else '-inf'
            if loud == 'norm':
                if not np.isfinite(lin) or len(ref) == 0:
                    row['invalid'] = 'silent_or_too_short'
                    rows.append(row)
                    continue
                gain = TARGET_LUFS - lin
            g = 10 ** (gain / 20)
            segs = [(s * g).astype(np.float32) for s in segs]
            if full is not None:
                full = (full * g).astype(np.float32)
            if not segs or any(len(s) == 0 for s in segs):
                row['invalid'] = 'empty'
                rows.append(row)
                continue
            files = []
            for k, s in enumerate(segs):
                name = f'{key}__w{k}.wav' if variant == 'VFULL' else f'{key}.wav'
                if not (outdir / name).exists():
                    save(outdir / name, s, sr)
                files.append(name)
            if full is not None:
                name = f'{key}__full.wav'
                if not (outdir / name).exists():
                    save(outdir / name, full, sr)
                row['full'] = name
            cat = np.concatenate(segs)
            row.update(n_samples=int(sum(len(s) for s in segs)), dur=round(sum(len(s) for s in segs) / sr, 4),
                       n_win=len(segs), files=';'.join(files), gain_db=round(gain, 4),
                       over_fs=round(float(np.mean(np.abs(cat) > 1.0)), 6), peak=round(float(np.abs(cat).max()), 5),
                       lufs_out=(round(loudness(ref * g, sr), 3) if np.isfinite(lin) else '-inf'))
        except Exception as e:
            row['invalid'] = f'error:{type(e).__name__}:{e}'[:200]
        rows.append(row)
    return rows


def load_index(path):
    if not Path(path).exists():
        return {}
    with open(path, newline='') as f:
        return {r['unit_id']: r for r in csv.DictReader(f, delimiter='\t')}


def needed(dataset, jobs=None):
    """{item_id: [(variant, loud)]} from PLAN (optionally only some jobs)."""
    items = read_manifest(dataset)
    need = {}
    for job, spec in PLAN[dataset].items():
        if jobs and job not in jobs:
            continue
        for variant, loud, subset in spec:
            for it in items:
                if in_subset(it, subset):
                    need.setdefault(it['item_id'], set()).add((variant, loud))
    return items, need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--job', action='append')
    ap.add_argument('--limit', type=int, default=0, help='first N items (smoke)')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    outdir = DER / a.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    idx_path = outdir / 'index.tsv'
    done = load_index(idx_path)
    items, need = needed(a.dataset, a.job)
    work = []
    for it in items:
        if it['item_id'] not in need:
            continue
        todo = sorted(vl for vl in need[it['item_id']]
                      if unit_id(a.dataset, it['item_id'], *vl) not in done)
        if todo:
            work.append((it, todo, str(outdir)))
    if a.limit:
        work = work[:a.limit]
    print(f'{a.dataset}: {len(work)} items to decode ({len(done)} units already indexed)', flush=True)
    new = not idx_path.exists()
    t0 = time.time()
    # AIME: group by shard so each worker re-uses its parquet read
    if a.dataset == 'aime':
        work.sort(key=lambda w: (w[0]['src'].split(':')[1], int(w[0]['src'].split(':')[2])))
    with open(idx_path, 'a', newline='') as f, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, fieldnames=IDX_COLS, delimiter='\t')
        if new:
            w.writeheader()
        for n, rows in enumerate(ex.map(decode_item, work, chunksize=4), 1):
            for r in rows:
                w.writerow({k: r.get(k, '') for k in IDX_COLS})
            if n % 200 == 0:
                f.flush()
                print(f'  {n}/{len(work)} {time.time() - t0:.0f}s', flush=True)
    idx = load_index(idx_path)
    bad = [r for r in idx.values() if r['invalid']]
    print(f'{a.dataset}: index {len(idx)} units, invalid {len(bad)}', flush=True)
    for r in bad[:5]:
        print('  invalid:', r['unit_id'], r['invalid'])


if __name__ == '__main__':
    main()
