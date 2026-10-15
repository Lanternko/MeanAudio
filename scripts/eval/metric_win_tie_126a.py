#!/usr/bin/env python3
"""126a: per-pair win/tie/lose of the quality metrics under one-variable signal ladders.

Design: docs/experiments/metric_win_tie_126_20261016.md (signal block on real + TTM audio).
Every lvl30 clip of 5 systems (MusicCaps real, ours CFG3+neg / CFG0 s14159265, MusicGen,
MusicLDM; 5,131 clips each, from p2 115) is rewritten under 53 variants, one variable at a
time: 23 that should tie (identity, gain, polarity, lead silence, formats) and 30 that should
lose (noise, clipping, bitcrush, clicks, dropouts, lowpass, blasts, block shuffle, reverse).

Each (variant, system) unit writes its audio to a temp dir, scores it and deletes it:
AES (4 axes), MEva, CLAP (batch 1, original caption), VGGish clip-mean embedding and set
statistics for FAD. Per-clip scores stay; audio does not (it regenerates deterministically,
the noise seed is crc32(system/variant/id)).

Reading (analyze): per pair d = score(variant) - score(clean of the same clip):
tie if |d| <= m, else better/worse; m = 0.10 S, S = range of the 5 systems' clean means.
Should-tie variants pass at tie rate >= 80 %, should-lose ones at worse rate >= 80 %.
PC has no quality direction and is read only for ties.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.pop('MKL_NUM_THREADS', None)  # MKL_NUM_THREADS hangs the forked loader pool in the dac venv

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(REPO / 'scripts/eval'))
EV = Path('/home/kojiek/eval_output_nvme')
OUT = EV / 'metric_win_tie_126a_20261016'
ROOT = OUT
REPORT = REPO / 'docs/experiments/results/metric_win_tie_126a_summary.json'
TSV = '/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv'
R115 = EV / 'ttm_metrics_115_20261009'
MEVA_PY = '/home/kojiek/MeanAudio/runtime/meva_20261003/venv/bin/python'
MIN_FREE = 40 * 2 ** 30
SYSTEMS = {'real': 'musiccaps_real', 'ours_cfg3neg': 'meanaudio_cfg3_neg', 'ours_cfg0': 'meanaudio_cfg0',
           'musicgen': 'musicgen_medium', 'musicldm': 'musicldm'}
METRICS = ('PQ', 'CE', 'CU', 'PC', 'meva', 'clap')
DIRECTED = ('PQ', 'CE', 'CU', 'meva', 'clap')
M_FRAC, PASS = 0.10, 0.80
M_CURVE = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ── variants ──
def _v(name, kind, op, arg=None, rms=False, subtype='PCM_24'):
    return {'name': name, 'kind': kind, 'op': op, 'arg': arg, 'rms_match': rms, 'subtype': subtype}


def variants():
    v = [_v('clean', 'ref', 'identity'), _v('clean_shuf', 'tie', 'identity')]
    v += [_v(f'gain{g:+g}', 'tie', 'gain', g) for g in (-0.5, -1, -2, -3, -6, -10, -15, -20, -30, 1, 3, 6)]
    v += [_v('polarity', 'tie', 'polarity'), _v('pcm16', 'tie', 'identity', subtype='PCM_16'),
          _v('resample44k', 'tie', 'resample')]
    v += [_v(f'lead{ms}ms', 'tie', 'lead', ms) for ms in (10, 50, 200)]
    v += [_v(f'mp3_{k}k', 'tie', 'mp3', k) for k in (160, 64, 32)]
    v += [_v(f'noise_snr{s}', 'lose', 'noise', s, rms=True) for s in (50, 40, 30, 25, 20, 15, 10, 5, 0)]
    v += [_v(f'clip{d}db', 'lose', 'clip', d, rms=True) for d in (3, 6, 12, 20)]
    v += [_v(f'bitcrush{b}', 'lose', 'bitcrush', b, rms=True) for b in (8, 6, 4)]
    v += [_v(f'clicks{r}ps', 'lose', 'clicks', r) for r in (1, 5, 20)]
    v += [_v(f'dropout20ms_x{n}', 'lose', 'dropout', n) for n in (3, 10, 30)]
    v += [_v(f'lowpass{f // 1000}k', 'lose', 'lowpass', f, rms=True) for f in (4000, 2000, 1000)]
    v += [_v(f'blast80ms_x{n}', 'lose', 'blast', n) for n in (1, 3)]
    v += [_v(f'shuffle{b}s', 'lose', 'shuffle', b) for b in (0.25, 1.0)]
    v += [_v('reverse', 'lose', 'reverse')]
    return v


VARIANTS = {x['name']: x for x in variants()}


def rng_for(*key):
    return np.random.default_rng(zlib.crc32('/'.join(map(str, key)).encode()))


def transform(x, var, rng, sr=16000):
    """x: float64 mono at its native rate (MusicGen 32 kHz, the rest 16 kHz), lvl30. Returns (y, n_samples_over_full_scale_before_cap)."""
    op, a, n = var['op'], var['arg'], len(x)
    peak = float(np.max(np.abs(x))) or 1e-9
    if op == 'identity':
        y = x
    elif op == 'gain':
        y = x * 10 ** (a / 20)
    elif op == 'polarity':
        y = -x
    elif op == 'resample':
        import soxr
        y = soxr.resample(soxr.resample(x, sr, 44100, quality='VHQ'), 44100, sr, quality='VHQ')[:n]
        y = np.pad(y, (0, n - len(y)))
    elif op == 'lead':
        k = int(sr * a / 1000)
        y = np.concatenate([np.zeros(k), x[:n - k]])
    elif op == 'mp3':
        y = mp3_roundtrip(x, a, sr)
    elif op == 'noise':
        p = float(np.mean(x ** 2)) or 1e-12
        y = x + rng.standard_normal(n) * math.sqrt(p / 10 ** (a / 10))
    elif op == 'clip':
        t = peak * 10 ** (-a / 20)
        y = np.clip(x, -t, t)
    elif op == 'bitcrush':
        q = peak / 2 ** (a - 1)
        y = np.round(x / q) * q
    elif op == 'clicks':
        y = x.copy()
        k = int(a * n / sr)
        pos = rng.choice(n, k, replace=False)
        y[pos] = rng.choice([-1.0, 1.0], k) * min(0.999, 4 * peak)
    elif op == 'dropout':
        y, w = x.copy(), int(0.02 * sr)
        for s in rng.choice(n - w, a, replace=False):
            y[s:s + w] = 0
    elif op == 'lowpass':
        from scipy.signal import butter, sosfiltfilt
        y = sosfiltfilt(butter(8, a, fs=sr, output='sos'), x)
    elif op == 'blast':
        y, w = x.copy(), int(0.08 * sr)
        for s in rng.choice(n - w, a, replace=False):
            y[s:s + w] = np.clip(y[s:s + w] * 10 ** (24 / 20), -0.999, 0.999)
    elif op == 'shuffle':
        w = int(a * sr)
        blocks = [x[i:i + w] for i in range(0, n, w)]
        y = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
    elif op == 'reverse':
        y = x[::-1].copy()
    else:
        raise ValueError(op)
    if var['rms_match']:
        y = y * math.sqrt(float(np.mean(x ** 2)) / max(float(np.mean(y ** 2)), 1e-20))
    over = int(np.sum(np.abs(y) >= 1.0))
    return np.clip(y, -0.999, 0.999), over


def mp3_roundtrip(x, kbps, sr):
    with tempfile.TemporaryDirectory() as td:
        src, mp3, back = Path(td) / 'a.wav', Path(td) / 'a.mp3', Path(td) / 'b.wav'
        sf.write(src, x, sr, subtype='FLOAT')
        q = ['-hide_banner', '-loglevel', 'error', '-y']
        subprocess.run(['ffmpeg', *q, '-i', str(src), '-codec:a', 'libmp3lame', '-b:a', f'{kbps}k', str(mp3)], check=True)
        subprocess.run(['ffmpeg', *q, '-i', str(mp3), '-ar', str(sr), '-ac', '1', '-c:a', 'pcm_f32le', str(back)], check=True)
        y, sr2 = sf.read(back, dtype='float64')
    assert sr2 == sr
    # LAME adds encoder delay; align by cross-correlation within 4,000 samples so this tests the codec, not a shift
    lag = best_lag(x, y, 4000, sr)
    y = y[lag:lag + len(x)]
    return np.pad(y, (0, len(x) - len(y)))


def best_lag(x, y, max_lag, sr):
    seg = x[:sr * 2]
    scores = [float(np.dot(seg, y[k:k + len(seg)])) if k + len(seg) <= len(y) else -np.inf for k in range(max_lag)]
    return int(np.argmax(scores))


def make_one(job):
    src, dst, system, vname = job
    var = VARIANTS[vname]
    x, sr = sf.read(src, dtype='float64')
    assert x.ndim == 1, src
    y, over = transform(x, var, rng_for(system, vname, Path(src).stem), sr)
    sf.write(dst, y, sr, format='FLAC', subtype=var['subtype'])
    return Path(src).stem, over


# ── inputs ──
def items(limit=None):
    caps = {}
    with open(TSV, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            caps[r['id']] = r['caption']
    ids = None
    for s in SYSTEMS.values():
        got = {p.stem for p in (R115 / s / 'lvl30/audio').glob('*.flac')}
        ids = got if ids is None else ids & got
    ids = sorted(i for i in ids if i in caps)
    if limit:
        ids = ids[:limit]
    return [(i, caps[i]) for i in ids]


def units():
    return [(v['name'], s) for v in variants() for s in SYSTEMS]  # variant-major: a partial run covers every system


# ── scoring ──
def unit_dir(vname, system):
    return ROOT / 'units' / vname / system


def unit_done(d, ids):
    try:
        rows = list(csv.DictReader(open(d / 'scores.tsv'), delimiter='\t'))
        meta = json.loads((d / 'done.json').read_text())
    except (OSError, ValueError):
        return False
    return [r['id'] for r in rows] == ids and meta.get('n') == len(ids) and (d / 'fad_stats.npz').is_file()


def run_unit(vname, system, rows, embedder, clap_model, tmp_root):
    from eval_metrics import score_aes, score_clap
    d, ids = unit_dir(vname, system), [i for i, _ in rows]
    if unit_done(d, ids):
        return
    t0 = time.time()
    d.mkdir(parents=True, exist_ok=True)
    src_dir = R115 / SYSTEMS[system] / 'lvl30/audio'
    tmp = Path(tempfile.mkdtemp(prefix=f'{vname}_{system}_', dir=tmp_root))
    try:
        jobs = [(str(src_dir / f'{i}.flac'), str(tmp / f'{i}.flac'), system, vname) for i in ids]
        over = dict(embedder.pool.imap_unordered(make_one, jobs, chunksize=16))
        t_gen = time.time() - t0
        meva_out = d / 'meva'
        shutil.rmtree(meva_out, ignore_errors=True)
        meva_out.mkdir()
        meva = subprocess.Popen([MEVA_PY, str(REPO / 'scripts/eval/meva_score_dir.py'), '--audio_dir', str(tmp),
                                 '--out_dir', str(meva_out)], cwd=REPO, stdout=open(d / 'meva.log', 'w'),
                                stderr=subprocess.STDOUT)
        paths = [tmp / f'{i}.flac' for i in ids]
        if vname == 'clean_shuf':  # same audio as clean; only the AES batch composition changes
            random.Random(126).shuffle(paths)
        aes, failed = score_aes(paths, progress=False)
        if failed or len(aes) != len(ids):
            raise SystemExit(f'{vname}/{system}: AES incomplete {len(aes)}/{len(ids)}')
        aes = {Path(p).stem: v for p, v in aes.items()}
        clap = score_clap(rows, tmp, model=clap_model, progress=False)
        if len(clap) != len(ids):
            raise SystemExit(f'{vname}/{system}: CLAP incomplete {len(clap)}/{len(ids)}')
        import torch
        emb = {}
        with torch.no_grad():
            from ttm_fad_lvl30 import load_one
            for key, x in embedder.pool.imap(load_one, [(i, str(tmp / f'{i}.flac'), False) for i in ids], chunksize=8):
                emb[key] = embedder.fad.model.forward(torch.from_numpy(x)).detach().cpu().numpy().astype(np.float32)
        frames = np.concatenate([emb[i] for i in ids])
        np.savez(d / 'fad_stats.npz', mu=frames.mean(0), sigma=np.cov(frames, rowvar=False), n_frames=len(frames))
        np.save(d / 'emb_clipmean.npy', np.stack([emb[i].mean(0) for i in ids]).astype(np.float16))
        if meva.wait() != 0:
            raise SystemExit(f'{vname}/{system}: MEva failed, see {d / "meva.log"}')
        mv = {r['id']: float(r['meva_raw']) for r in csv.DictReader(open(meva_out / 'meva_per_clip.tsv'), delimiter='\t')}
        if set(mv) != set(ids):
            raise SystemExit(f'{vname}/{system}: MEva incomplete {len(mv)}/{len(ids)}')
        with open(d / 'scores.tsv', 'w', newline='') as fh:
            w = csv.writer(fh, delimiter='\t')
            w.writerow(['id', *METRICS, 'over_fs'])
            for i in ids:
                vals = [aes[i]['PQ'], aes[i]['CE'], aes[i]['CU'], aes[i]['PC'], mv[i], clap[i]]
                if not all(map(math.isfinite, vals)):
                    raise SystemExit(f'{vname}/{system}: nonfinite score for {i}')
                w.writerow([i, *[f'{x:.6f}' for x in vals], over[i]])
        (d / 'done.json').write_text(json.dumps({'n': len(ids), 'variant': VARIANTS[vname], 'system': system,
                                                 'seconds': time.time() - t0, 'gen_seconds': t_gen}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f'unit done {vname}/{system} n={len(ids)} {time.time() - t0:.0f}s')


def run(rows):
    import ttm_fad_lvl30 as F
    embedder = F.Embedder()  # forks the worker pool first, while this process is single-threaded
    from eval_metrics import load_clap
    clap_model = load_clap()
    tmp_root = ROOT / 'tmp'
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True)
    todo = units()
    for k, (vname, system) in enumerate(todo):
        if shutil.disk_usage(ROOT).free < MIN_FREE // 2:
            raise SystemExit('disk below half the preflight minimum')
        log(f'[{k + 1}/{len(todo)}] {vname}/{system}')
        run_unit(vname, system, rows, embedder, clap_model, tmp_root)
    embedder.pool.close()
    shutil.rmtree(tmp_root, ignore_errors=True)


# ── analysis ──
def load_scores(vname, system):
    d = unit_dir(vname, system)
    if not (d / 'done.json').is_file():
        return None
    rows = list(csv.DictReader(open(d / 'scores.tsv'), delimiter='\t'))
    return {m: np.array([float(r[m]) for r in rows]) for m in (*METRICS, 'over_fs')}


def frechet(a, b):
    from ttm_fad_lvl30 import frechet as fr
    return fr((a['mu'], a['sigma']), (b['mu'], b['sigma']))


def judge(d, m, metric, kind):
    tie = float(np.mean(np.abs(d) <= m))
    worse, better = float(np.mean(d < -m)), float(np.mean(d > m))
    out = {'tie': tie, 'worse': worse, 'better': better}
    if kind == 'tie':
        out['pass'] = tie >= PASS
    elif metric in DIRECTED:
        out['pass'] = worse >= PASS
    return out


def analyze(rows):
    clean = {s: load_scores('clean', s) for s in SYSTEMS}
    if any(v is None for v in clean.values()):
        raise SystemExit('clean units missing')
    S = {m: float(np.ptp([clean[s][m].mean() for s in SYSTEMS])) for m in METRICS}
    fad_clean = {s: dict(np.load(unit_dir('clean', s) / 'fad_stats.npz')) for s in SYSTEMS}
    res = {}
    for v in variants():
        if v['kind'] == 'ref':
            continue
        for s in SYSTEMS:
            sc = load_scores(v['name'], s)
            if sc is None:
                continue
            cell = {'kind': v['kind'], 'n': len(sc['PQ']), 'clips_over_fs': int(np.sum(sc['over_fs'] > 0)), 'metrics': {}}
            for m in METRICS:
                d = sc[m] - clean[s][m]
                cell['metrics'][m] = {
                    'mean_clean': float(clean[s][m].mean()), 'mean': float(sc[m].mean()), 'mean_delta': float(d.mean()),
                    'median_delta': float(np.median(d)), 'abs_delta_p90': float(np.quantile(np.abs(d), 0.9)),
                    'at_m': judge(d, M_FRAC * S[m], m, v['kind']),
                    'curve': {str(f): judge(d, f * S[m], m, v['kind']) for f in M_CURVE}}
            fs = dict(np.load(unit_dir(v['name'], s) / 'fad_stats.npz'))
            cell['fad_vs_own_clean'] = frechet(fs, fad_clean[s])
            cell['fad_vs_real_clean'] = frechet(fs, fad_clean['real'])
            res.setdefault(v['name'], {})[s] = cell
    thresholds = ladder_readouts(res)
    summary = {'protocol': {'n_clips': len(rows), 'systems': SYSTEMS, 'm_frac': M_FRAC, 'pass_rate': PASS, 'S': S,
                            'S_definition': 'range of the 5 systems clean means (this run)',
                            'variants': [v for v in variants()]},
               'clean_fad_vs_real': {s: frechet(fad_clean[s], fad_clean['real']) for s in SYSTEMS},
               'cells': res, 'ladders': thresholds}
    for p in (ROOT / 'summary.json', REPORT if ROOT == OUT else ROOT / 'report.json'):
        p.write_text(json.dumps(summary, indent=1))
    log(f'ANALYZE -> {ROOT / "summary.json"} ({sum(len(x) for x in res.values())} cells)')


LADDERS = {'gain_down': ['gain-0.5', 'gain-1', 'gain-2', 'gain-3', 'gain-6', 'gain-10', 'gain-15', 'gain-20', 'gain-30'],
           'gain_up': ['gain+1', 'gain+3', 'gain+6'], 'lead': ['lead10ms', 'lead50ms', 'lead200ms'],
           'mp3': ['mp3_160k', 'mp3_64k', 'mp3_32k'],
           'noise': [f'noise_snr{s}' for s in (50, 40, 30, 25, 20, 15, 10, 5, 0)],
           'clip': ['clip3db', 'clip6db', 'clip12db', 'clip20db'], 'bitcrush': ['bitcrush8', 'bitcrush6', 'bitcrush4'],
           'clicks': ['clicks1ps', 'clicks5ps', 'clicks20ps'],
           'dropout': ['dropout20ms_x3', 'dropout20ms_x10', 'dropout20ms_x30'],
           'lowpass': ['lowpass4k', 'lowpass2k', 'lowpass1k'], 'blast': ['blast80ms_x1', 'blast80ms_x3'],
           'shuffle': ['shuffle1.0s', 'shuffle0.25s']}


def ladder_readouts(res):
    """tie ladders: last dose still passing (tolerance); lose ladders: first dose passing (detection threshold)."""
    out = {}
    for lad, steps in LADDERS.items():
        kind = VARIANTS[steps[0]]['kind']
        for s in SYSTEMS:
            for m in METRICS:
                if kind == 'lose' and m not in DIRECTED:
                    continue
                passes = [(st, res.get(st, {}).get(s, {}).get('metrics', {}).get(m, {}).get('at_m', {}).get('pass'))
                          for st in steps]
                if any(p is None for _, p in passes):
                    continue
                if kind == 'tie':
                    ok = [st for st, p in passes if p]
                    first_fail = next((st for st, p in passes if not p), None)
                    r = {'tolerance': None if not passes[0][1] else
                         (steps[steps.index(first_fail) - 1] if first_fail else steps[-1]),
                         'n_pass': len(ok)}
                else:
                    r = {'detection': next((st for st, p in passes if p), None)}
                out.setdefault(lad, {}).setdefault(s, {})[m] = r
    return out


def preflight():
    rows = items()
    OUT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUT).free
    absent = [p for p in [MEVA_PY, '/usr/bin/ffmpeg'] if not Path(p).exists()]
    counts = {s: len(list((R115 / d / 'lvl30/audio').glob('*.flac'))) for s, d in SYSTEMS.items()}
    ok = len(rows) == 5131 and all(c >= 5131 for c in counts.values()) and not absent and free >= MIN_FREE \
        and len(VARIANTS) == 53
    print(json.dumps({'n': len(rows), 'counts': counts, 'absent': absent, 'free_gb': free / 2 ** 30,
                      'n_variants': len(VARIANTS), 'n_units': len(units()), 'ok': ok}))
    return 0 if ok else 1


def validate(rows):
    ids = [i for i, _ in rows]
    for vname, s in units():
        if not unit_done(unit_dir(vname, s), ids):
            print(json.dumps({'ok': False, 'incomplete': f'{vname}/{s}'}))
            return 1
    summ = json.loads((ROOT / 'summary.json').read_text())
    n_cells = sum(len(x) for x in summ['cells'].values())
    ok = summ['protocol']['n_clips'] == len(rows) and n_cells == (len(VARIANTS) - 1) * len(SYSTEMS)
    print(json.dumps({'n': len(rows), 'cells': n_cells, 'ok': ok}))
    return 0 if ok else 1


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--variants', help='comma list, smoke only')
    a = ap.parse_args()
    if a.preflight:
        return preflight()
    if a.limit:
        ROOT = OUT / 'smoke'
    if a.variants:
        keep = set(a.variants.split(',')) | {'clean'}
        global variants
        base = variants
        variants = lambda: [v for v in base() if v['name'] in keep]  # noqa: E731
    rows = items(a.limit)
    if a.validate_only:
        return validate(rows)
    if not a.analyze_only:
        run(rows)
    analyze(rows)
    return validate(rows)


if __name__ == '__main__':
    sys.exit(main())
