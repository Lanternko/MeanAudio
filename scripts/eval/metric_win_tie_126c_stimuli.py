#!/usr/bin/env python3
"""126c stimuli: MUSDB18-HQ stem edits (remove / solo / rebalance / misalign / detune / swap) and a separation control.

Design: docs/experiments/metric_win_tie_126c_musdb_20261019.md. Every clip is 10 s, mono 16 kHz,
FLAC PCM_24, -30 LUFS (each case is re-levelled, so a removal is not a loudness change).

Stages (run in order; each is resumable):
  prep    dac venv, CPU pool. Read each of the 150 tracks (train 100 + test 50) straight from the zip,
          mono 16 kHz, pick <= 4 non-overlapping 10 s windows from 15 s on where all four stems are
          active, cache the five true signals (mixture, vocals, drums, bass, other) with 0.5 s of
          context on each side -> OUT/cache/<win>.npz, OUT/windows.json.
  demucs  karaoke-jp-align venv (demucs 4.0.1, htdemucs = torch hub 955717e8, the model used by the
          10/01 Codex removal ablation), GPU, ~minutes. Separates each window's 44.1 kHz stereo mixture
          (2 s more context each side) -> OUT/cache/<win>_dmx.npz (16 kHz mono, same 11 s span).
  build   dac venv, CPU pool. Writes all cases and OUT/items.json (same row schema as 126b).

Cases per window (base = the mixture at -30 LUFS, kind as in 126: tie / soft should tie, lose should
be worse, descriptive has no prescribed direction):
  ref          mix
  tie          stemsum (sum of the 4 true stems), dmx_resum (sum of the 4 htdemucs stems),
               rebal_<stem>_{+1,-1}dB, signal ties (gain-1, gain-3, polarity, pcm16, lead10ms, mp3_64k)
  soft         rebal_<stem>_{+3,-3,+6,-6}dB (a mix-balance choice), vocal_pv0 (rubberband +50 then -50 cents)
  lose         drums_lag{20,50,100,250}ms, vocals_lag{50,100,250}ms, vocals_detune{25,50,100}c,
               bass_semi+1, swap_vocals / swap_drums (the stem from the same window index of the next
               track in the same split), signal loses (noise_snr40/30/20/10, clip6db, lowpass4k, clicks5ps)
  descriptive  no_<stem> (true removal), only_<stem> (solo stem), dmx_no_<stem> (htdemucs removal)

  PYTHONPATH= ~/venvs/dac/bin/python scripts/eval/metric_win_tie_126c_stimuli.py --stage prep [--limit N]
  PYTHONPATH= ~/venvs/karaoke-jp-align/bin/python scripts/eval/metric_win_tie_126c_stimuli.py --stage demucs [--limit N]
  PYTHONPATH= ~/venvs/dac/bin/python scripts/eval/metric_win_tie_126c_stimuli.py --stage build [--limit N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')  # 126a best_lag (mp3 alignment) is 200x slower with oversubscribed BLAS threads
os.environ.pop('MKL_NUM_THREADS', None)

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

ZIP = Path('/mnt/seagate/kojiek_data/musdb18hq/musdb18hq.zip')
# NVMe, not /mnt/seagate: the USB disk stalls on ext4 journal commits with many small files (126b)
OUT = Path('/home/kojiek/eval_output_nvme/metric_win_tie_126c_20261019/stimuli')
SR, DUR, PAD, DPAD = 16000, 10.0, 0.5, 2.0
N, P = int(SR * DUR), int(SR * PAD)
STEMS = ('vocals', 'drums', 'bass', 'other')
START, STEP, MAX_WIN = 15.0, 10.0, 4
ACT_ABS, ACT_REL, ACT_FRAMES = -45.0, -20.0, 0.6  # stem LUFS >= -45 and >= mix LUFS - 20; active in 60 % of 1 s frames
TARGET, PEAK_CAP = -30.0, 0.999
CAPTION = 'music'
SIG_TIE = ['gain-1', 'gain-3', 'polarity', 'pcm16', 'lead10ms', 'mp3_64k']
SIG_LOSE = ['noise_snr40', 'noise_snr30', 'noise_snr20', 'noise_snr10', 'clip6db', 'lowpass4k', 'clicks5ps']


def slug(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def tracks():
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    out = sorted({tuple(n.split('/')[:2]) for n in names if n.endswith('/mixture.wav')})
    return [{'split': sp, 'track': tr, 'slug': f'{sp}_{slug(tr)}'} for sp, tr in out]


def read_zip(z, split, track, stem):
    x, sr = sf.read(io.BytesIO(z.read(f'{split}/{track}/{stem}.wav')), dtype='float32', always_2d=True)
    return x, sr


def to16k(x, sr):
    import soxr
    return soxr.resample(x.mean(1).astype(np.float64), sr, SR, quality='VHQ')


def lufs(x):
    import pyloudnorm
    return float(pyloudnorm.Meter(SR).integrated_loudness(x))


def frame_db(x):
    f = x[:len(x) // SR * SR].reshape(-1, SR)
    return 10 * np.log10(np.mean(f ** 2, 1) + 1e-20)


# ── prep ──
def prep_track(t):
    with zipfile.ZipFile(ZIP) as z:
        sig = {}
        for s in ('mixture', *STEMS):
            x, sr = read_zip(z, t['split'], t['track'], s)
            sig[s] = to16k(x, sr)
    total = min(len(v) for v in sig.values())
    cand = []
    w0 = START
    while (w0 + DUR + PAD) * SR <= total:
        a = int(w0 * SR)
        seg = {s: v[a:a + N] for s, v in sig.items()}
        ml = lufs(seg['mixture'])
        if np.isfinite(ml) and ml > -40:
            mdb, ok, st = frame_db(seg['mixture']), True, {}
            for s in STEMS:
                sl = lufs(seg[s])
                frac = float(np.mean(frame_db(seg[s]) >= mdb - 30))
                st[s] = {'lufs': sl, 'active_frac': frac}
                ok &= bool(np.isfinite(sl) and sl >= ACT_ABS and sl >= ml + ACT_REL and frac >= ACT_FRAMES)
            if ok:
                cand.append((w0, ml, st))
        w0 += STEP
    if len(cand) > MAX_WIN:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, MAX_WIN).round().astype(int)]
    wins = []
    for k, (w0, ml, st) in enumerate(cand):
        wid = f'{t["slug"]}_w{k}'
        a = int(w0 * SR) - P
        np.savez(OUT / 'cache' / f'{wid}.npz', **{s: v[a:a + N + 2 * P].astype(np.float32) for s, v in sig.items()})
        wins.append({**t, 'win': wid, 'k': k, 'w0': w0, 'mix_lufs_raw': ml, 'stems': st})
    return t['slug'], wins


def stage_prep(limit, workers):
    (OUT / 'cache').mkdir(parents=True, exist_ok=True)
    ts = tracks()
    if limit:
        ts = [t for sp in ('train', 'test') for t in [x for x in ts if x['split'] == sp][:limit]]
    wins, empty = [], []
    with Pool(workers) as pool:
        for i, (s, w) in enumerate(pool.imap_unordered(prep_track, ts)):
            wins += w
            if not w:
                empty.append(s)
            print(f'prep {i + 1}/{len(ts)} {s} windows={len(w)}', flush=True)
    wins.sort(key=lambda w: w['win'])
    (OUT / 'windows.json').write_text(json.dumps({'windows': wins, 'tracks': len(ts), 'no_window': sorted(empty)}, indent=0))
    print(json.dumps({'tracks': len(ts), 'windows': len(wins), 'no_window': len(empty)}))


# ── demucs (karaoke-jp-align venv) ──
def stage_demucs():
    import torch
    import torchaudio.functional as AF
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    model = get_model('htdemucs').cuda().eval()
    wins = json.loads((OUT / 'windows.json').read_text())['windows']
    by = {}
    for w in wins:
        by.setdefault((w['split'], w['track']), []).append(w)
    done = 0
    with zipfile.ZipFile(ZIP) as z:
        for (split, track), ws in sorted(by.items()):
            todo = [w for w in ws if not (OUT / 'cache' / f'{w["win"]}_dmx.npz').is_file()]
            if not todo:
                continue
            x, sr = read_zip(z, split, track, 'mixture')
            assert sr == model.samplerate, sr
            for w in todo:
                a = int(round((w['w0'] - PAD - DPAD) * sr))
                b = int(round((w['w0'] + DUR + PAD + DPAD) * sr))
                seg = x[max(a, 0):b]
                lead = max(-a, 0)
                wav = torch.from_numpy(np.pad(seg, ((lead, max(b - a - lead - len(seg), 0)), (0, 0))).T.copy()).cuda()
                ref = wav.mean(0)
                with torch.no_grad():
                    est = apply_model(model, ((wav - ref.mean()) / ref.std())[None], device='cuda', split=True,
                                      overlap=0.25, progress=False)[0] * ref.std() + ref.mean()
                out = {}
                for i, s in enumerate(model.sources):
                    y = AF.resample(est[i].mean(0).double().cpu(), sr, SR).numpy()
                    c = int(round(DPAD * SR))
                    out[s] = y[c:c + N + 2 * P].astype(np.float32)
                np.savez(OUT / 'cache' / f'{w["win"]}_dmx.npz', **out)
                done += 1
            print(f'demucs {split}/{track} windows={len(todo)} total={done}', flush=True)
    print(json.dumps({'separated': done, 'windows': len(wins)}))


# ── build ──
def level(x):
    lu, peak = lufs(x), float(np.max(np.abs(x)))
    if not np.isfinite(lu) or peak <= 0:
        raise RuntimeError('unmeasurable loudness')
    g = 10 ** ((TARGET - lu) / 20)
    capped = peak * g > PEAK_CAP
    return x * (PEAK_CAP / peak if capped else g), lu, int(capped)


def crop(x, lag=0):
    """the 10 s window from a padded 11 s signal; lag > 0 delays the signal by lag samples"""
    return x[P - lag:P - lag + N]


def rubberband(x, cents):
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / 'a.wav', Path(td) / 'b.wav'
        sf.write(src, x, SR, subtype='FLOAT')
        q = ['-hide_banner', '-loglevel', 'error', '-y']
        subprocess.run(['ffmpeg', *q, '-i', str(src), '-af', f'rubberband=pitch={2 ** (cents / 1200):.8f}:pitchq=quality',
                        '-c:a', 'pcm_f32le', str(dst)], check=True)
        y, _ = sf.read(dst, dtype='float64')
    return np.pad(y, (0, max(len(x) - len(y), 0)))[:len(x)]


def build_window(job):
    import metric_win_tie_126a as A
    w, partner = job
    c = {k: v.astype(np.float64) for k, v in np.load(OUT / 'cache' / f'{w["win"]}.npz').items()}
    d = {k: v.astype(np.float64) for k, v in np.load(OUT / 'cache' / f'{w["win"]}_dmx.npz').items()}
    pc = {k: v.astype(np.float64) for k, v in np.load(OUT / 'cache' / f'{partner}.npz').items()}
    stem = {s: crop(c[s]) for s in STEMS}
    acc = sum(stem.values())
    cases = {'stemsum': ('tie', 'identity', acc), 'dmx_resum': ('tie', 'separation', crop(sum(d[s] for s in STEMS)))}
    for s in STEMS:
        for db in (1, -1, 3, -3, 6, -6):
            cases[f'rebal_{s}_{db:+d}dB'] = ('tie' if abs(db) == 1 else 'soft', f'rebal_{s}_{"up" if db > 0 else "down"}',
                                            acc + (10 ** (db / 20) - 1) * stem[s])
        cases[f'no_{s}'] = ('descriptive', 'remove', acc - stem[s])
        cases[f'only_{s}'] = ('descriptive', 'solo', stem[s])
        cases[f'dmx_no_{s}'] = ('descriptive', 'remove_dmx', crop(sum(d[t] for t in STEMS if t != s)))
    rest_d = acc - stem['drums']
    for ms in (20, 50, 100, 250):
        cases[f'drums_lag{ms}ms'] = ('lose', 'drums_lag', rest_d + crop(c['drums'], SR * ms // 1000))
    rest_v = acc - stem['vocals']
    for ms in (50, 100, 250):
        cases[f'vocals_lag{ms}ms'] = ('lose', 'vocals_lag', rest_v + crop(c['vocals'], SR * ms // 1000))
    cases['vocal_pv0'] = ('soft', 'vocal_pv0', rest_v + crop(rubberband(rubberband(c['vocals'], 50), -50)))
    for ct in (25, 50, 100):
        cases[f'vocals_detune{ct}c'] = ('lose', 'vocals_detune', rest_v + crop(rubberband(c['vocals'], ct)))
    cases['bass_semi+1'] = ('lose', 'bass_semi', acc - stem['bass'] + crop(rubberband(c['bass'], 100)))
    cases['swap_vocals'] = ('lose', 'swap', rest_v + crop(pc['vocals']))
    cases['swap_drums'] = ('lose', 'swap', rest_d + crop(pc['drums']))

    base = {'block': 'musdb', 'source': 'musdb', 'group': w['split'], 'piece': w['track'], 'excerpt': w['win'],
            'caption': CAPTION, 'w0': w['w0'], 'partner': partner}
    adir = OUT / 'audio' / w['split']
    adir.mkdir(parents=True, exist_ok=True)
    clean, lu0, cap0 = level(crop(c['mixture']))
    key0 = f'musdb__{w["win"]}__mix'
    dst = adir / f'{w["win"]}__mix.flac'
    sf.write(dst, clean, SR, format='FLAC', subtype='PCM_24')
    rows = [{**base, 'key': key0, 'case': 'mix', 'kind': 'ref', 'ladder': 'clean', 'path': str(dst), 'base': '',
             'over_fs': 0, 'capped': cap0, 'raw_lufs': lu0, 'mir': True}]
    for case, (kind, ladder, y) in cases.items():
        y, lu, cap = level(y)
        dst = adir / f'{w["win"]}__{case}.flac'
        sf.write(dst, y, SR, format='FLAC', subtype='PCM_24')
        rows.append({**base, 'key': f'musdb__{w["win"]}__{case}', 'case': case, 'kind': kind, 'ladder': ladder,
                     'path': str(dst), 'base': key0, 'over_fs': 0, 'capped': cap, 'raw_lufs': lu,
                     'mir': not case.startswith('rebal_') or abs(int(case.split('_')[-1][:-2])) == 6})
    lad = {st: l for l, steps in A.LADDERS.items() for st in steps}
    for name in SIG_TIE + SIG_LOSE:
        var = A.VARIANTS[name]
        y, over = A.transform(clean, var, A.rng_for('musdb', w['win'], name), SR)
        dst = adir / f'{w["win"]}__{name}.flac'
        sf.write(dst, y, SR, format='FLAC', subtype=var['subtype'])
        rows.append({**base, 'key': f'musdb__{w["win"]}__{name}', 'case': name, 'kind': var['kind'],
                     'ladder': lad.get(name, name), 'path': str(dst), 'base': key0, 'over_fs': over, 'capped': 0,
                     'mir': False})
    return rows


def stage_build(workers):
    wins = json.loads((OUT / 'windows.json').read_text())['windows']
    missing = [w['win'] for w in wins if not (OUT / 'cache' / f'{w["win"]}_dmx.npz').is_file()]
    if missing:
        raise SystemExit(f'demucs stage incomplete: {len(missing)} windows')
    jobs = []
    for sp in ('train', 'test'):
        ts = sorted({w['track'] for w in wins if w['split'] == sp})
        by = {t: [w for w in wins if w['split'] == sp and w['track'] == t] for t in ts}
        for i, t in enumerate(ts):
            nxt = by[ts[(i + 1) % len(ts)]]
            jobs += [(w, nxt[w['k'] % len(nxt)]['win']) for w in by[t]]
    rows = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(build_window, jobs, chunksize=2)):
            rows += r
            if (i + 1) % 50 == 0:
                print(f'build {i + 1}/{len(jobs)}', flush=True)
    rows.sort(key=lambda r: r['key'])
    keys = [r['key'] for r in rows]
    assert len(keys) == len(set(keys)), 'duplicate keys'
    (OUT / 'items.json').write_text(json.dumps(rows, indent=0))
    count = {}
    for r in rows:
        count[r['kind']] = count.get(r['kind'], 0) + 1
    print(json.dumps({'clips': len(rows), 'windows': len(jobs), 'capped': sum(r['capped'] for r in rows),
                      'mir': sum(r['mir'] for r in rows), 'by_kind': count}, indent=1))


def main():
    global OUT
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('prep', 'demucs', 'build'), required=True)
    ap.add_argument('--limit', type=int, help='first N tracks per split (smoke)')
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    if a.limit:
        OUT = OUT / 'smoke'
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'prep':
        stage_prep(a.limit, a.workers)
    elif a.stage == 'demucs':
        stage_demucs()
    else:
        stage_build(a.workers)


if __name__ == '__main__':
    raise SystemExit(main())
