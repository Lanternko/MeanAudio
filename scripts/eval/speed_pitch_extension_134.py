#!/usr/bin/env python3
"""134: dense tempo grid 0.10-2.00 and pitch +-1..12 semitones, per-pair win/tie/lose of AES/MEva/CLAP/MIR.

Design: docs/experiments/speed_pitch_extension_20261020.md (preregistered bands in section 4).
Judging is 126's rule unchanged (imported from 126a/126b): d = score(variant) - score(clean of the same clip),
tie if |d| <= m, m = 0.10 S_m0 (S = 126a S = range of the 5 TTM systems' clean means; MIR: SD of all clean clips here),
pass = 80 % of pairs.

Stimuli (10 s, mono 16 kHz FLAC PCM_24, -30 LUFS, built on CPU inside this script, idempotent per excerpt):
  maestro    the 150 MAESTRO excerpts of 123 (GeneralUser), notes up to 21 s rendered
  piano126b  100 seeded windows of the 126b Mutopia classical pieces (start moved to the first onset)
  ttm        60 seeded prompts x 5 TTM systems (115 lvl30 clips)
Implementations: rb = ffmpeg rubberband on the waveform (all sources); midi = notes warped / transposed and
re-synthesised (MIDI sources only). Tempo output = first 10 s of the changed signal; tempo > 1 on the 10 s TTM clips is
shorter than 10 s (10/tempo s) and is matched by a 'lenctl' crop of the clean clip (no processing).

Scoring per unit (<= 3000 clips, one source): AES, MEva, CLAP(batch 1); MIR (pulse_clarity, ibi_cv + the other two)
runs as a CPU subprocess per unit. No FAD, no VGGish.

  PYTHONPATH= ~/venvs/dac/bin/python scripts/eval/speed_pitch_extension_134.py [--preflight|--validate-only|--analyze-only]
      [--limit N --speeds 0.5,1.05,2.0 --semis 1,12]   # smoke: N excerpts per source, writes under OUT/smoke
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.pop('MKL_NUM_THREADS', None)  # MKL_NUM_THREADS hangs the forked loader pool in the dac venv

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(REPO / 'scripts/eval'))
import metric_win_tie_126a as A  # noqa: E402
import metric_win_tie_126b as B  # noqa: E402  (judge/cell)
import midi_realpiece_stimuli as S  # noqa: E402  (renderer, warp, transpose)

OUT = Path('/home/kojiek/eval_output_nvme/speed_pitch_extension_134_20261020')
REPORT = REPO / 'docs/experiments/results/speed_pitch_extension_134_summary.json'
MAESTRO_EX = Path('/home/kojiek/eval_output_nvme/midi_realpiece_123/excerpts.json')
PIECES = Path('/mnt/seagate/kojiek_data/126_pieces')
SR, N, LEAD, WIN, BANK = 16000, 160000, 0.25, 9.0, 'generaluser'
LONG, SHORT = 21.0, 12.0  # synthesis lengths (s): long source for the waveform path, short for MIDI-level variants
N_PIANO, N_TTM_PROMPTS, MIN_ONSETS = 100, 60, 6
UNIT, MIN_FREE, REP = 3000, 30 * 2 ** 30, 400
TEMPO = sorted({round(0.05 * k, 2) for k in range(2, 41)} | {0.97, 0.98, 0.99, 1.01, 1.02, 1.03})  # 0.10..2.00, 45 values
SEMIS = [-12, -5, -2, -1, 1, 2, 5, 12]
TIE_BAND, LOSE_LOW, LOSE_HIGH = (0.95, 1.05), 0.80, 1.25  # preregistered tempo bands
PASS = B.PASS
MIR2 = ('pulse_clarity', 'ibi_cv')
METRICS = (*A.METRICS, *MIR2)
log = A.log


def h(*k):
    return zlib.crc32('/'.join(map(str, k)).encode())


def tempo_kind(v):
    if v == 1.0:
        return 'tie'  # rubberband identity control
    if TIE_BAND[0] <= v <= TIE_BAND[1]:
        return 'tie'
    return 'lose' if v <= LOSE_LOW or v >= LOSE_HIGH else 'transition'


def pitch_kind(n):
    return 'tie' if abs(n) == 1 else 'lose'


def cname(axis, impl, v):
    return f'{axis}_{impl}_{v:.2f}' if axis == 'tempo' else f'{axis}_{impl}_{v:+d}'


# ── audio helpers ──
def level(x):
    try:
        y, _, capped = S.level(np.asarray(x, np.float64))
    except RuntimeError:  # silent / unmeasurable: the case is skipped (its pair is simply absent)
        return None, 0
    return y, capped


def write(dst, y):
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, SR, format='FLAC', subtype='PCM_24')


def rubberband(x, tempo=1.0, pitch=1.0):
    x = np.ascontiguousarray(x, np.float32)
    q = ':pitchq=quality' if pitch != 1.0 else ''
    p = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-',
                        '-af', f'rubberband=tempo={tempo:.6f}:pitch={pitch:.8f}{q}', '-f', 'f32le', '-'],
                       input=x.tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).astype(np.float64)


def synth(notes, dur):
    S.DUR, S.LEAD = dur, LEAD
    return np.asarray(S.synth([tuple(n) for n in notes], BANK, 0, 0), np.float64)


# ── excerpts ──
def maestro_excerpts(limit):
    ex = json.loads(MAESTRO_EX.read_text())
    return [{'id': e['id'], 'cluster': e['id'], 'notes': e['notes']} for e in ex[:limit or None]]


def piano_excerpts(limit):
    cands = []
    manifest = json.loads((PIECES / 'manifest.json').read_text())
    for key, p in manifest.items():
        if p['genre'] != 'classical':
            continue
        notes = sorted(S.read_notes(PIECES / 'classical' / key / 'mutopia.mid'))
        t0, last = notes[0][0], max(n[0] for n in notes)
        for k in range(16):  # the 126b windows
            w0 = t0 + k * WIN
            if w0 + WIN > last + 1.0:
                break
            on = sorted({round(n[0], 2) for n in notes if w0 <= n[0] < w0 + WIN})
            if len(on) < MIN_ONSETS:
                continue
            w1 = min(n[0] for n in notes if n[0] >= w0)  # start on the first onset so 0.1x is never empty
            sel = [(a - w1, min(b - w1, a - w1 + 8.0), q, v) for a, b, q, v in notes if w1 <= a < w1 + 21.0]
            cands.append({'id': f'{key}_w{k:02d}', 'cluster': key, 'notes': sel})
    cands.sort(key=lambda e: h('piano134', e['id']))
    out = sorted(cands[:N_PIANO], key=lambda e: e['id'])
    return out[:limit] if limit else out


def ttm_excerpts(limit):
    ids = [i for i, _ in A.items()]
    caps = dict(A.items())
    ids = sorted(ids, key=lambda i: h('ttm134', i))[:N_TTM_PROMPTS]
    ids = sorted(ids)[:limit] if limit else sorted(ids)
    return [{'id': i, 'cluster': i, 'caption': caps[i]} for i in ids]


# ── stimulus builders (run in a Pool; one job = one excerpt, one source) ──
def row(source, system, ex, case, axis, impl, v, kind, base, dst, caption, capped, dur):
    return {'key': f'{source}{system}__{ex["id"]}__{case}', 'source': source, 'system': system, 'excerpt': ex['id'],
            'cluster': ex['cluster'], 'case': case, 'axis': axis, 'impl': impl, 'value': v, 'kind': kind,
            'base': base, 'path': str(dst), 'caption': caption, 'capped': capped, 'dur': dur}


def build_midi(job):
    source, ex, speeds, semis = job
    done = OUT / 'stimuli' / 'rows' / f'{source}__{ex["id"]}.json'
    if done.is_file():
        return json.loads(done.read_text())
    notes, cap = ex['notes'], 'solo piano music'
    d = OUT / 'stimuli' / 'audio' / source
    x_long = synth(notes, LONG)
    clean, c0 = level(x_long[:N])
    assert clean is not None, ex['id']
    p0 = d / f'{ex["id"]}__clean.flac'
    write(p0, clean)
    k0 = f'{source}__{ex["id"]}__clean'
    rows = [row(source, '', ex, 'clean', 'clean', 'ref', 0, 'ref', '', p0, cap, c0, 10.0)]
    rows[0]['key'] = k0

    def emit(axis, impl, v, kind, x):
        y, c = level(x[:N])
        if y is None:
            return
        case = cname(axis, impl, v)
        dst = d / f'{ex["id"]}__{case}.flac'
        write(dst, y)
        rows.append(row(source, '', ex, case, axis, impl, v, kind, k0, dst, cap, c, len(y) / SR))
    for v in speeds:
        emit('tempo', 'rb', v, tempo_kind(v), rubberband(x_long[:int(min(LONG, 10 * v + 0.6) * SR)], tempo=v))
        if v != 1.0:
            warped = S.tempo(v)([tuple(n) for n in notes], None)
            emit('tempo', 'midi', v, tempo_kind(v), synth(warped, SHORT))
    for n in semis:
        emit('pitch', 'rb', n, pitch_kind(n), rubberband(x_long[:int(10.6 * SR)], pitch=2 ** (n / 12)))
        emit('pitch', 'midi', n, pitch_kind(n), synth(S.transpose(n)([tuple(q) for q in notes], None), SHORT))
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rows))
    return rows


def build_ttm(job):
    ex, speeds, semis = job
    done = OUT / 'stimuli' / 'rows' / f'ttm__{ex["id"]}.json'
    if done.is_file():
        return json.loads(done.read_text())
    rows = []
    for sname, sdir in A.SYSTEMS.items():
        src = A.R115 / sdir / 'lvl30/audio' / f'{ex["id"]}.flac'
        x, sr = sf.read(src, dtype='float64')
        assert x.ndim == 1, src
        if sr != SR:  # MusicGen lvl30 clips are 32 kHz; every system's clean base is the 16 kHz copy written here
            import soxr
            x = soxr.resample(x, sr, SR, quality='VHQ')
        d = OUT / 'stimuli' / 'audio' / 'ttm' / sname
        k0 = f'ttm{sname}__{ex["id"]}__clean'
        write(d / f'{ex["id"]}__clean.flac', x)
        rows.append(row('ttm', sname, ex, 'clean', 'clean', 'ref', 0, 'ref', '', d / f'{ex["id"]}__clean.flac', ex['caption'], 0, len(x) / SR))
        rows[-1]['key'] = k0

        def emit(axis, impl, v, kind, y):
            case = cname(axis, impl, v) if impl != 'ctl' else f'lenctl_{v:.2f}'
            y, c = level(y)
            if y is None:
                return
            dst = d / f'{ex["id"]}__{case}.flac'
            write(dst, y)
            rows.append(row('ttm', sname, ex, case, axis, impl, v, kind, k0, dst, ex['caption'], c, len(y) / SR))
        for v in speeds:
            y = rubberband(x, tempo=v)
            emit('tempo', 'rb', v, tempo_kind(v), y[:N])
            if v > 1.0:
                emit('tempo', 'ctl', v, 'ctl', x[:int(round(len(x) / v))])
        for n in semis:
            emit('pitch', 'rb', n, pitch_kind(n), rubberband(x, pitch=2 ** (n / 12))[:N])
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rows))
    return rows


def build(limit, speeds, semis):
    """stimuli -> OUT/stimuli/items.json (idempotent per excerpt)"""
    if (OUT / 'stimuli' / 'items.json').is_file():
        return json.loads((OUT / 'stimuli' / 'items.json').read_text())
    jobs_m = [('maestro', e, speeds, semis) for e in maestro_excerpts(limit)]
    jobs_p = [('piano126b', e, speeds, semis) for e in piano_excerpts(limit)]
    jobs_t = [(e, speeds, semis) for e in ttm_excerpts(limit)]
    rows, t0 = [], time.time()
    with Pool(16) as pool:
        for name, fn, jobs in (('midi', build_midi, jobs_m + jobs_p), ('ttm', build_ttm, jobs_t)):
            for i, r in enumerate(pool.imap(fn, jobs, chunksize=1)):
                rows += r
                if (i + 1) % 25 == 0:
                    log(f'build {name} {i + 1}/{len(jobs)} {time.time() - t0:.0f}s')
    keys = [r['key'] for r in rows]
    assert len(keys) == len(set(keys)), 'duplicate keys'
    (OUT / 'stimuli' / 'items.json').write_text(json.dumps(rows))
    log(f'built {len(rows)} clips in {time.time() - t0:.0f}s')
    return rows


# ── scoring ──
def units(rows):
    by = {}
    for r in rows:
        by.setdefault(r['source'], []).append(r)
    return [(f'{s}_{k // UNIT:03d}', by[s][k:k + UNIT]) for s in sorted(by) for k in range(0, len(by[s]), UNIT)]


def unit_done(d, keys):
    try:
        got = [r['key'] for r in csv.DictReader(open(d / 'scores.tsv'), delimiter='\t')]
        return got == keys and json.loads((d / 'done.json').read_text()).get('n') == len(keys)
    except (OSError, ValueError):
        return False


def run_unit(name, rs, clap_model):
    """126b.run_unit without VGGish; keeps the unit dir (MIR writes next to it)"""
    from eval_metrics import score_aes, score_clap
    d, keys = OUT / 'units' / name, [r['key'] for r in rs]
    if unit_done(d, keys):
        return
    t0 = time.time()
    audio = d / 'audio'
    shutil.rmtree(audio, ignore_errors=True)
    shutil.rmtree(d / 'meva', ignore_errors=True)
    audio.mkdir(parents=True)
    for r in rs:
        (audio / f'{r["key"]}.flac').symlink_to(r['path'])
    (d / 'meva').mkdir()
    meva = subprocess.Popen([A.MEVA_PY, str(REPO / 'scripts/eval/meva_score_dir.py'), '--audio_dir', str(audio),
                             '--out_dir', str(d / 'meva')], cwd=REPO, stdout=open(d / 'meva.log', 'w'), stderr=subprocess.STDOUT)
    aes, failed = score_aes([audio / f'{k}.flac' for k in keys], progress=False)
    if failed or len(aes) != len(keys):
        raise SystemExit(f'{name}: AES incomplete {len(aes)}/{len(keys)}')
    aes = {Path(p).name[:-5]: v for p, v in aes.items()}
    clap = score_clap([(r['key'], r['caption']) for r in rs], audio, model=clap_model, progress=False)
    if len(clap) != len(keys):
        raise SystemExit(f'{name}: CLAP incomplete {len(clap)}/{len(keys)}')
    if meva.wait() != 0:
        raise SystemExit(f'{name}: MEva failed, see {d / "meva.log"}')
    mv = {r['id']: float(r['meva_raw']) for r in csv.DictReader(open(d / 'meva' / 'meva_per_clip.tsv'), delimiter='\t')}
    if set(mv) != set(keys):
        raise SystemExit(f'{name}: MEva incomplete {len(mv)}/{len(keys)}')
    with open(d / 'scores.tsv', 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['key', *A.METRICS])
        for k in keys:
            vals = [aes[k]['PQ'], aes[k]['CE'], aes[k]['CU'], aes[k]['PC'], mv[k], clap[k]]
            if not all(map(math.isfinite, vals)):
                raise SystemExit(f'{name}: nonfinite score for {k}')
            w.writerow([k, *[f'{x:.6f}' for x in vals]])
    (d / 'done.json').write_text(json.dumps({'n': len(keys), 'seconds': time.time() - t0}))
    shutil.rmtree(audio)
    log(f'unit done {name} n={len(keys)} {time.time() - t0:.0f}s')


def mir_worker(rows):
    from mir_metrics import score_mir
    for name, rs in units(rows):
        d = OUT / 'units' / name
        if (d / 'mir_metrics.json').is_file():
            continue
        d.mkdir(parents=True, exist_ok=True)
        per = score_mir([r['path'] for r in rs], madmom_workers=10, chroma_workers=4)
        by = {r['path']: r['key'] for r in rs}
        tmp = d / 'mir_metrics.json.tmp'
        tmp.write_text(json.dumps({by[p]: v for p, v in per.items()}))
        tmp.rename(d / 'mir_metrics.json')
        log(f'mir done {name}')
    return 0


def run(rows, limit):
    from eval_metrics import load_clap
    clap_model = load_clap()
    cmd = [sys.executable, __file__, '--mir-worker'] + (['--limit', str(limit)] if limit else [])
    mir = subprocess.Popen(cmd + _smoke_args(), stdout=open(OUT / 'mir.log', 'w'), stderr=subprocess.STDOUT)
    todo = units(rows)
    for k, (name, rs) in enumerate(todo):
        if shutil.disk_usage(OUT).free < MIN_FREE // 2:
            raise SystemExit('disk below half the preflight minimum')
        log(f'[{k + 1}/{len(todo)}] {name}')
        run_unit(name, rs, clap_model)
    log('waiting for MIR')
    if mir.wait() != 0:
        raise SystemExit(f'MIR failed, see {OUT / "mir.log"}')


# ── analysis ──
def load_all(rows):
    sc = {}
    for name, rs in units(rows):
        d = OUT / 'units' / name
        for r in csv.DictReader(open(d / 'scores.tsv'), delimiter='\t'):
            sc[r['key']] = {m: float(r[m]) for m in A.METRICS}
        for k, v in json.loads((d / 'mir_metrics.json').read_text()).items():
            for m in MIR2:
                x = v.get(m, float('nan'))
                sc[k][m] = B.MIR[m] * x if math.isfinite(x) else float('nan')  # sign-flipped: higher = better
    return sc


def crossing(V, r, thr, up):
    """walk outward from 1.0 (V sorted, anchor at the value 0/1 given by caller); first rate on the wrong side of thr,
    linearly interpolated with the previous point. up: wrong side is r < thr (tie exit); else r >= thr (lose entry)."""
    prev = None
    for v, x in zip(V, r):
        if np.isfinite(x) and ((x < thr) if up else (x >= thr)):
            if prev is None:
                return float(v)
            (pv, px) = prev
            return float(pv + (thr - px) * (v - pv) / (x - px)) if x != px else float(v)
        prev = (v, x)
    return float('nan')


def curve_stats(sc, S, rows_by, cases, m, rng):
    """tie/worse rates per grid value with cluster-bootstrap CI + crossing speeds.
    cases: {value: case}; rows_by: case -> [row]; returns dict"""
    vals = sorted(cases)
    clusters = sorted({r['cluster'] for c in cases.values() for r in rows_by.get(c, [])})
    ci = {c: i for i, c in enumerate(clusters)}
    C, Vn = len(clusters), len(vals)
    tie, worse, n = (np.zeros((C, Vn)) for _ in range(3))
    dsum = np.zeros((C, Vn))
    for j, v in enumerate(vals):
        for r in rows_by.get(cases[v], []):
            d = sc[r['key']][m] - sc[r['base']][m]
            if not np.isfinite(d):
                continue
            i = ci[r['cluster']]
            tie[i, j] += abs(d) <= 0.10 * S[m]
            worse[i, j] += d < -0.10 * S[m]
            dsum[i, j] += d
            n[i, j] += 1
    idx = rng.integers(0, C, (REP, C))

    def rates(a, ix=None):
        a2, n2 = (a, n) if ix is None else (a[ix].sum(1), n[ix].sum(1))
        return (a2.sum(0) / n2.sum(0).clip(1)) if ix is None else a2 / n2.clip(1)
    pt = {k: rates(a) for k, a in (('tie', tie), ('worse', worse), ('mean_delta', dsum))}
    bs = {k: rates(a, idx) for k, a in (('tie', tie), ('worse', worse), ('mean_delta', dsum))}
    lo = lambda a: np.percentile(a, 2.5, axis=0)  # noqa: E731
    hi = lambda a: np.percentile(a, 97.5, axis=0)  # noqa: E731
    out = {'values': vals, 'n': n.sum(0).astype(int).tolist()}
    for k in pt:
        out[k] = pt[k].tolist()
        out[k + '_lo'], out[k + '_hi'] = lo(bs[k]).tolist(), hi(bs[k]).tolist()
    if axis_of(cases) == 'tempo':
        anchor = vals.index(1.0) if 1.0 in vals else None
        sides = {'low': ([v for v in vals if v < 1.0][::-1], 'low'), 'high': ([v for v in vals if v > 1.0], 'high')}
        cr = {}
        for side, (vv, _) in sides.items():
            ix = [vals.index(v) for v in vv]
            for kind, key, thr, up in (('tie_exit', 'tie', PASS, True), ('lose_entry', 'worse', PASS, False)):
                est = crossing(vv, pt[key][ix], thr, up)
                boot = np.array([crossing(vv, bs[key][b][ix], thr, up) for b in range(REP)])
                fin = boot[np.isfinite(boot)]
                cr[f'{kind}_{side}'] = {'est': est, 'ci': [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) else None,
                                        'n_boot_none': int(REP - len(fin))}
        out['crossings'] = cr
        out['has_anchor'] = anchor is not None
    return out


def axis_of(cases):
    return 'tempo' if any(isinstance(v, float) for v in cases) else 'pitch'


def band_stats(sc, S, rows_by, cases_by_kind, m, rng):
    """pooled rate over the cases of a preregistered band (tie rate for tie band, worse rate for lose band)"""
    out = {}
    for kind, cs in cases_by_kind.items():
        if kind not in ('tie', 'lose') or not cs:
            continue
        per = {}
        for c in cs:
            ds = [(r['cluster'], sc[r['key']][m] - sc[r['base']][m]) for r in rows_by.get(c, [])]
            ds = [(cl, d) for cl, d in ds if np.isfinite(d)]
            ok = [abs(d) <= 0.10 * S[m] if kind == 'tie' else d < -0.10 * S[m] for _, d in ds]
            per[c] = float(np.mean(ok)) if ok else float('nan')
        cl = sorted({r['cluster'] for c in cs for r in rows_by.get(c, [])})
        ci = {c: i for i, c in enumerate(cl)}
        s, n = np.zeros(len(cl)), np.zeros(len(cl))
        for c in cs:
            for r in rows_by.get(c, []):
                d = sc[r['key']][m] - sc[r['base']][m]
                if np.isfinite(d):
                    s[ci[r['cluster']]] += (abs(d) <= 0.10 * S[m]) if kind == 'tie' else (d < -0.10 * S[m])
                    n[ci[r['cluster']]] += 1
        ix = rng.integers(0, len(cl), (REP, len(cl)))
        bs = s[ix].sum(1) / n[ix].sum(1).clip(1)
        out[kind] = {'pooled_rate': float(s.sum() / max(n.sum(), 1)), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     'n_cases': len(cs), 'frac_cases_pass': float(np.nanmean([p >= PASS for p in per.values()])),
                     'all_cases_pass': bool(all(p >= PASS for p in per.values())), 'case_rates': per,
                     'judged': kind == 'tie' or m in B.DIRECTED}
    return out


def meva_monotone(sc, rows_by, cases, m, rng, lo=0.5, hi=2.0):
    """per-clip Spearman between tempo (lo..hi, 0.05 grid) and score; cluster-mean; bootstrap CI"""
    from scipy.stats import rankdata
    vals = [v for v in sorted(cases) if lo <= v <= hi and abs(v * 20 - round(v * 20)) < 1e-6]
    if len(vals) < 5:
        return None
    by = {}
    for j, v in enumerate(vals):
        for r in rows_by.get(cases[v], []):
            by.setdefault((r['cluster'], r['system'], r['excerpt']), {})[j] = sc[r['key']][m]
    clips = [(k, [d[j] for j in range(len(vals))]) for k, d in by.items() if len(d) == len(vals)]
    X = np.array([x for _, x in clips])
    ok = np.isfinite(X).all(1)
    X, ks = X[ok], [k for (k, _), o in zip(clips, ok) if o]
    if not len(X):
        return None
    rv = rankdata(vals)
    rho = np.array([np.corrcoef(rv, rankdata(x))[0, 1] for x in X])
    cl = sorted({k[0] for k in ks})
    cm = np.array([np.nanmean([r for r, k in zip(rho, ks) if k[0] == c]) for c in cl])
    bs = np.array([np.nanmean(cm[rng.integers(0, len(cm), len(cm))]) for _ in range(REP)])
    mean_curve = np.nanmean(X, 0)
    return {'n_clips': len(X), 'n_clusters': len(cl), 'rho_mean': float(np.nanmean(cm)), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            'rho_curve': float(np.corrcoef(rv, rankdata(mean_curve))[0, 1]), 'speeds': vals}


def analyze(rows):
    smoke = OUT.name == 'smoke'
    S_ = dict(json.loads(A_REPORT.read_text())['protocol']['S'])
    sc = load_all(rows)
    clean = [r for r in rows if r['kind'] == 'ref']
    for m in MIR2:
        S_[m] = float(np.nanstd([sc[r['key']][m] for r in clean]))
    S_ = {m: S_[m] for m in METRICS}
    rng = np.random.default_rng(134)
    groups = {}
    for r in rows:
        if r['base']:
            groups.setdefault((r['source'], r['impl'], r['axis']), {}).setdefault(r['case'], []).append(r)
    cells, curves, bands, mono = {}, {}, {}, {}
    for (src, impl, axis), by_case in sorted(groups.items()):
        if impl == 'ctl':
            continue
        cases = {}
        kinds = {}
        for c, rs in by_case.items():
            v = rs[0]['value']
            cases[v if axis == 'pitch' else float(v)] = c
            kinds[c] = rs[0]['kind']
        for c, rs in by_case.items():
            cells[f'{src}/{impl}/{c}'] = B.cell([(r['key'], r['base']) for r in rs], sc, S_, kinds[c] if kinds[c] in ('tie', 'lose') else 'descriptive')
            cells[f'{src}/{impl}/{c}']['kind'] = kinds[c]
        for m in METRICS:
            k = f'{src}/{impl}/{axis}/{m}'
            curves[k] = curve_stats(sc, S_, by_case, cases, m, rng)
            bands[k] = band_stats(sc, S_, by_case, {kd: [c for c in by_case if kinds[c] == kd] for kd in ('tie', 'lose')}, m, rng)
            if axis == 'tempo':
                mono[k] = meva_monotone(sc, by_case, cases, m, rng)
    ctrl = {}
    ix = {(r['source'], r['excerpt'], r['system'], r['case']): r['key'] for r in rows}
    for src in sorted({r['source'] for r in rows}):
        ex = sorted({(r['excerpt'], r['system']) for r in rows if r['source'] == src})
        for c in sorted({r['case'] for r in rows if r['source'] == src and r['impl'] == 'ctl'}):  # lenctl vs clean and vs rb tempo
            v = c.split('_')[1]
            ctrl[f'{src}/{c}_vs_clean'] = B.cell([(ix[(src, e, s, c)], ix[(src, e, s, 'clean')]) for e, s in ex if (src, e, s, c) in ix], sc, S_, 'descriptive')
            ctrl[f'{src}/tempo_rb_{v}_vs_{c}'] = B.cell([(ix[(src, e, s, f'tempo_rb_{v}')], ix[(src, e, s, c)]) for e, s in ex
                                                         if (src, e, s, f'tempo_rb_{v}') in ix and (src, e, s, c) in ix], sc, S_, 'descriptive')
        for c in sorted({r['case'] for r in rows if r['source'] == src and r['impl'] == 'rb'}):
            cm = c.replace('_rb_', '_midi_')
            pairs = [(ix[(src, e, s, c)], ix[(src, e, s, cm)]) for e, s in ex if (src, e, s, cm) in ix and (src, e, s, c) in ix]
            if pairs:
                ctrl[f'{src}/{c}_vs_{cm}'] = B.cell(pairs, sc, S_, 'descriptive')
    summary = {'protocol': {'n_clips': len(rows), 'S': S_, 'S_definition': '126a S (m0 reading); MIR: SD of all clean clips here, sign-flipped (higher=better)',
                            'm_frac': B.M_FRAC, 'pass_rate': PASS, 'rep': REP, 'tempo_grid': TEMPO, 'semitones': SEMIS,
                            'bands': {'tie': list(TIE_BAND), 'lose_low': LOSE_LOW, 'lose_high': LOSE_HIGH,
                                      'transition': 'other grid values: no prediction'},
                            'n_by_source': {s: sum(1 for r in rows if r['source'] == s) for s in sorted({r['source'] for r in rows})}},
               'cells': cells, 'curves': curves, 'bands': bands, 'monotone_tempo_0p5_2p0': mono, 'controls': ctrl}
    for p in (OUT / 'summary.json', OUT / 'report.json' if smoke else REPORT):
        p.write_text(json.dumps(summary, indent=1))
    with open(OUT / 'per_clip.tsv', 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['key', 'source', 'system', 'excerpt', 'case', 'axis', 'impl', 'value', 'kind', 'dur', *METRICS])
        for r in rows:
            w.writerow([r['key'], r['source'], r['system'], r['excerpt'], r['case'], r['axis'], r['impl'], r['value'], r['kind'],
                        f'{r["dur"]:.3f}', *[sc[r['key']][m] for m in METRICS]])
    log(f'ANALYZE -> {OUT / "summary.json"} ({len(cells)} cells)')


A_REPORT = A.REPORT


def preflight():
    absent = [p for p in [A.MEVA_PY, '/home/kojiek/venvs/madmom/bin/python', '/home/kojiek/venvs/mir/bin/python', str(MAESTRO_EX),
                          str(PIECES / 'manifest.json'), str(A_REPORT), str(S.BANKS[BANK])] if not Path(p).exists()]
    ff = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], capture_output=True, text=True).stdout
    ttm = {s: len(list((A.R115 / d / 'lvl30/audio').glob('*.flac'))) for s, d in A.SYSTEMS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUT).free
    rows = json.loads((OUT / 'stimuli/items.json').read_text()) if (OUT / 'stimuli/items.json').is_file() else []
    missing = [r['path'] for r in rows if not Path(r['path']).is_file()]
    ok = not absent and 'rubberband' in ff and min(ttm.values()) >= 5000 and free >= MIN_FREE and not missing
    print(json.dumps({'absent': absent, 'rubberband': 'rubberband' in ff, 'ttm': ttm, 'free_gb': free / 2 ** 30,
                      'n_items': len(rows), 'missing': len(missing), 'ok': ok}))
    return 0 if ok else 1


def validate(rows):
    for name, rs in units(rows):
        d = OUT / 'units' / name
        if not unit_done(d, [r['key'] for r in rs]) or not (d / 'mir_metrics.json').is_file():
            print(json.dumps({'ok': False, 'incomplete': name}))
            return 1
    summ = json.loads((OUT / 'summary.json').read_text())
    ok = summ['protocol']['n_clips'] == len(rows) and len(summ['cells']) > 100 and len(summ['curves']) > 20 and bool(summ['monotone_tempo_0p5_2p0'])
    print(json.dumps({'n': len(rows), 'cells': len(summ['cells']), 'ok': ok}))
    return 0 if ok else 1


def _smoke_args():
    a = _ARGS
    return (['--speeds', a.speeds] if a.speeds else []) + (['--semis', a.semis] if a.semis else [])


def main():
    global OUT, _ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--mir-worker', action='store_true')
    ap.add_argument('--limit', type=int, help='smoke: first N excerpts per source (writes under OUT/smoke)')
    ap.add_argument('--speeds', help='smoke: comma list replacing the tempo grid')
    ap.add_argument('--semis', help='smoke: comma list replacing the semitone list')
    a = _ARGS = ap.parse_args()
    if a.preflight:
        return preflight()
    if a.limit:
        OUT = OUT / 'smoke'
    OUT.mkdir(parents=True, exist_ok=True)
    speeds = [float(x) for x in a.speeds.split(',')] if a.speeds else TEMPO
    semis = [int(x) for x in a.semis.split(',')] if a.semis else SEMIS
    rows = build(a.limit, speeds, semis)
    if a.mir_worker:
        return mir_worker(rows)
    if a.validate_only:
        return validate(rows)
    if not a.analyze_only:
        run(rows, a.limit)
    analyze(rows)
    return 0 if a.limit else validate(rows)


if __name__ == '__main__':
    sys.exit(main())
