#!/usr/bin/env python
"""178 audio rule baselines (no MIDI): chroma out-of-key energy ratio and madmom key confidence.

Run each mode in its own venv, exactly like mir_metrics.py:

  ~/venvs/mir/bin/python     scripts/eval/wrongnote_probe_178_rules.py --mode chroma --root <dir>
  ~/venvs/madmom/bin/python  scripts/eval/wrongnote_probe_178_rules.py --mode madmom --root <dir>

chroma_out_ratio = 1 - (chroma energy on the 7 degrees of the best KK template) / (all chroma),
mean over non-silent frames of per-frame normalised chroma_cqt.
key_cnn_conf     = max of madmom's 24-way CNN key posterior (same convention as mir_metrics.py).

Output: <root>/features/rule_<mode>.tsv (path, value), appended per finished clip so a
restart skips paths that are already there.
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
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrongnote_probe_178_common as C  # noqa: E402

HOP = 512
SR44 = 44100


def _temper_rank(v, profiles):
    c = v - v.mean()
    nc = float(np.linalg.norm(c))
    best = -2.0
    for mode, prof in profiles:
        t = np.asarray(prof, dtype=float)
        t = t - t.mean()
        nt = float(np.linalg.norm(t))
        for tonic in range(12):
            r = float(c @ np.roll(t, tonic) / (nc * nt + 1e-12))
            if r > best:
                best, scale = r, {(tonic + d) % 12 for d in C.SCALE_DEGREES[mode]}
    return best, scale


_PROFILES = (('major', C.KK_MAJOR), ('minor', C.KK_MINOR))


def chroma_one(path):
    import librosa
    import soundfile as sf
    try:
        wav, sr = sf.read(path, dtype='float32', always_2d=True)
        wav = wav.mean(axis=1)
        ch = librosa.feature.chroma_cqt(y=wav, sr=sr, hop_length=HOP)
        rms = librosa.feature.rms(y=wav, hop_length=HOP)[0][:ch.shape[1]]
        ok = rms > 1e-3
        if ok.sum() < 10:
            return path, float('nan')
        c = ch[:, ok] / (ch[:, ok].sum(axis=0, keepdims=True) + 1e-12)
        v = c.mean(axis=1)
        _r, scale = _temper_rank(v, _PROFILES)
        tot = float(v.sum())
        return path, float(1.0 - v[sorted(scale)].sum() / tot) if tot > 0 else float('nan')
    except Exception as e:
        print(f'[WARN] chroma {path}: {e!r}', file=sys.stderr)
        return path, float('nan')


_P = {}


def _init_key():
    from madmom.features.key import CNNKeyRecognitionProcessor
    _P['key'] = CNNKeyRecognitionProcessor()


def key_one(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from madmom.audio.signal import Signal
    try:
        wav, sr = sf.read(path, dtype='float32', always_2d=True)
        wav = wav.mean(axis=1)
        if sr != SR44:
            g = np.gcd(sr, SR44)
            wav = resample_poly(wav, SR44 // g, sr // g).astype(np.float32)
        sig = Signal(wav, sample_rate=SR44, num_channels=1)
        return path, float(np.max(_P['key'](sig)))
    except Exception as e:
        print(f'[WARN] key {path}: {e!r}', file=sys.stderr)
        return path, float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('chroma', 'madmom'), required=True)
    ap.add_argument('--root', default=str(C.ROOT))
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    rows = C.read_tsv(Path(a.root) / 'manifest.tsv')
    paths = [r['path'] for r in rows]
    out = Path(a.root) / 'features' / f'rule_{a.mode}.tsv'
    out.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if out.is_file():
        with open(out, encoding='utf-8') as f:
            for ln in f:
                if ln.strip():
                    have.add(ln.split('\t')[0])
    todo = [p for p in paths if p not in have]
    print(f'[rules {a.mode}] {len(paths)} clips, {len(have)} done, {len(todo)} to do', flush=True)
    fn = chroma_one if a.mode == 'chroma' else key_one
    init = None if a.mode == 'chroma' else _init_key
    t0 = time.time()
    with open(out, 'a', encoding='utf-8') as f, Pool(a.workers, initializer=init) as pool:
        for i, (p, v) in enumerate(pool.imap_unordered(fn, todo, chunksize=8)):
            f.write(f'{p}\t{v:.6g}\n')
            if (i + 1) % 500 == 0:
                f.flush()
                print(f'[rules {a.mode}] {i + 1}/{len(todo)} '
                      f'({(time.time() - t0) / (i + 1) * 1000:.1f} ms/clip)', flush=True)
    got = {}
    with open(out, encoding='utf-8') as f:
        for ln in f:
            p, v = ln.rstrip('\n').split('\t')
            got[p] = float(v)
    missing = [p for p in paths if p not in got]
    nan = [p for p in paths if p in got and not np.isfinite(got[p])]
    info = {'mode': a.mode, 'clips': len(paths), 'scored': len(got), 'missing': len(missing),
            'nan': len(nan), 'seconds': round(time.time() - t0, 1),
            'ms_per_clip': round((time.time() - t0) / max(1, len(todo)) * 1000, 1)}
    C.write_json(Path(a.root) / 'features' / f'rule_{a.mode}.json', info)
    print(f'[rules {a.mode}] ' + str(info), flush=True)
    if missing:
        raise SystemExit(f'[FAIL] {a.mode}: {len(missing)} clips unscored')


if __name__ == '__main__':
    main()
