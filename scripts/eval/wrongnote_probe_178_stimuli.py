#!/usr/bin/env python
"""178 stimuli builder: MAESTRO excerpts -> 12 rendered variants each (prereg §1).

Reuses read_notes / synth / BANKS / MAESTRO from scripts/eval/midi_realpiece_stimuli.py
(the 123 module); only DUR / LEAD / TARGET are patched to this probe's 10 s, -23 LUFS.

  python scripts/eval/wrongnote_probe_178_stimuli.py --root <dir> [--limit 4] [--workers 20]

Resumable: a clip whose FLAC already exists (size > 0) is not rendered again; the
manifest is rewritten from excerpts.json + what is on disk.
"""
from __future__ import annotations

# BLAS pinning must happen before numpy / tinysoundfont are imported (multi-process Pool).
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midi_realpiece_stimuli as stim  # noqa: E402
import wrongnote_probe_178_common as C  # noqa: E402

stim.DUR = C.DUR
stim.LEAD = C.LEAD
stim.TARGET = C.TARGET_LUFS
MAESTRO = stim.MAESTRO
MAX_SOURCE = 300.0     # only look for windows in the first 5 min of a piece
MIN_NOTES = 20
WINDOWS_PER_PERF = 3
MAESTRO_SPLIT = {'train': 'train', 'val': 'validation', 'test': 'test'}
EXCERPT_PREFIX = {'train': 'tr', 'val': 'va', 'test': 'te'}


def maestro_rows():
    with open(MAESTRO / 'maestro-v3.0.0.csv', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _read_one(rec):
    try:
        return rec['midi_filename'], stim.read_notes(MAESTRO / rec['midi_filename'])
    except Exception as e:  # a broken MIDI must not kill selection
        print(f'[WARN] read {rec["midi_filename"]}: {e!r}', file=sys.stderr)
        return rec['midi_filename'], []


def candidate_windows(notes, midi):
    """Up to WINDOWS_PER_PERF non-overlapping 10 s windows holding >= 20 onsets, seed-ordered."""
    if not notes:
        return []
    ons = np.asarray(sorted(n[0] for n in notes), dtype=float)
    span = min(float(ons[-1]), MAX_SOURCE)
    starts = np.arange(0.0, max(0.0, span - C.DUR + 0.5), 0.5)
    if not len(starts):
        return []
    lo = np.searchsorted(ons, starts, 'left')
    hi = np.searchsorted(ons, starts + C.DUR, 'left')
    starts = starts[(hi - lo) >= MIN_NOTES]
    picked = []
    for s in sorted(starts, key=lambda s: C.h('window', C.SEED, midi, round(float(s), 1))):
        if all(abs(s - p) >= C.DUR + 2.0 for p in picked):
            picked.append(float(s))
        if len(picked) == WINDOWS_PER_PERF:
            break
    return picked


def excerpt_notes(notes, t0):
    sel = [(on - t0, min(off, t0 + C.DUR) - t0, int(p), int(v))
           for on, off, p, v in notes if t0 <= on < t0 + C.DUR]
    sel = [n for n in sel if n[1] > n[0] and n[0] < C.MUSIC_END]
    return sorted(sel)


def select_split(split, n, pool):
    recs = [r for r in maestro_rows() if r['split'] == MAESTRO_SPLIT[split]]
    recs = sorted(recs, key=lambda r: C.h('perf', C.SEED, r['midi_filename']))
    notes_by = dict(pool.map(_read_one, recs, chunksize=4))
    wins = {r['midi_filename']: candidate_windows(notes_by[r['midi_filename']], r['midi_filename'])
            for r in recs}
    picks = []
    # pass 1: one excerpt per performance; later passes: a second window of a performance
    for w_i in range(WINDOWS_PER_PERF):
        seen = set()      # one excerpt per performance per pass
        for r in recs:
            if len(picks) >= n:
                break
            midi = r['midi_filename']
            if midi in seen or len(wins[midi]) <= w_i:
                continue
            t0 = wins[midi][w_i]
            notes = excerpt_notes(notes_by[midi], t0)
            if len(notes) < MIN_NOTES:
                continue
            seen.add(midi)
            key = C.estimate_key(notes)
            picks.append({'split': split, 'perf': midi, 'midi': midi, 't0': t0,
                          'window': 'first' if w_i == 0 else f'extra{w_i}', 'notes': notes,
                          'key': key, 'variants': C.build_variants(notes, key)})
        if len(picks) >= n:
            break
    for i, p in enumerate(picks):   # prefixes must differ: 'train' and 'test' both start with t
        p['excerpt'] = f'{EXCERPT_PREFIX[split]}{i:03d}'
    if len(picks) < n:
        raise SystemExit(f'[FAIL] {split}: only {len(picks)} excerpts available, need {n}')
    return picks


def clip_path(audio_dir, ex, font, v):
    return Path(audio_dir) / font / f'{ex["excerpt"]}__{v}.flac'


def variant_rows(ex, font, audio_dir):
    """Manifest rows for one excerpt x one soundfont (no audio needed)."""
    key = ex['key']
    clean = ex['variants']['clean']
    rows = []
    for v in C.VARIANTS:
        notes = ex['variants'][v]
        n_wrong = sum(1 for a, b in zip(notes, clean) if a[2] != b[2])
        kind = v.split('_')[-1] if v.startswith('wrong_') else ''
        rows.append({
            'split': ex['split'], 'excerpt': ex['excerpt'], 'perf': ex['perf'], 'midi': ex['midi'],
            'font': font, 'variant': v, 'label': 'wrong' if v.startswith('wrong_') else 'ok',
            'wrong_kind': kind, 'wrong_k': v.split('_')[1] if kind else '',
            'n_notes': len(notes), 'n_wrong': n_wrong, 'key': key['name'],
            'oracle_out_ratio': round(C.out_ratio(notes, key), 6),
            'path': str(clip_path(audio_dir, ex, font, v)), 't0': ex['t0'], 'window': ex['window']})
    return rows


def render_job(job):
    """One excerpt x one soundfont -> render the FLACs that are missing. Returns (n_written, secs)."""
    ex, font, audio_dir = job
    t0 = time.time()
    n = 0
    x_clean = None
    for v in C.VARIANTS:
        dst = clip_path(audio_dir, ex, font, v)
        if dst.is_file() and dst.stat().st_size > 0:
            continue
        if x_clean is None:
            x_clean = stim.synth(ex['variants']['clean'], font, 0, 0)
        x = x_clean if v == 'clean' else (
            x_clean * 10 ** (-6 / 20.0) if v == 'gain_m6' else stim.synth(ex['variants'][v], font, 0, 0))
        y, _lufs, _capped = stim.level(np.asarray(x, dtype=np.float64))
        sf.write(str(dst), y, C.SR, format='FLAC', subtype='PCM_24')
        n += 1
    return n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(C.ROOT))
    ap.add_argument('--limit', type=int, help='smoke: excerpts per split')
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    root = Path(a.root)
    t_start = time.time()
    (root / 'audio').mkdir(parents=True, exist_ok=True)
    for font in {f for fs in C.FONTS.values() for f in fs}:
        (root / 'audio' / font).mkdir(parents=True, exist_ok=True)
    cache = root / 'excerpts.json'
    if cache.is_file():
        picks = json.loads(cache.read_text())
        print(f'[stimuli] excerpts.json reused: {len(picks)}', flush=True)
    else:
        picks = []
        with Pool(a.workers) as pool:
            for split, n in C.SPLITS.items():
                got = select_split(split, a.limit or n, pool)
                print(f'[stimuli] {split}: {len(got)} excerpts '
                      f'({sum(1 for p in got if p["window"] != "first")} extra windows)', flush=True)
                picks += got
        C.write_json(cache, picks)
    t_sel = time.time()

    jobs = [(ex, font, root / 'audio') for ex in picks for font in C.FONTS[ex['split']]]
    todo = [j for j in jobs if any(not (clip_path(j[2], j[0], j[1], v).is_file() and
                                        clip_path(j[2], j[0], j[1], v).stat().st_size > 0)
                                   for v in C.VARIANTS)]
    print(f'[stimuli] {len(jobs)} excerpt-font jobs, {len(todo)} need rendering', flush=True)
    rows = [r for ex, font, adir in jobs for r in variant_rows(ex, font, adir)]
    written, secs = 0, 0.0
    with Pool(a.workers) as pool:
        for i, (n, dt) in enumerate(pool.imap_unordered(render_job, todo, chunksize=1)):
            written += n
            secs += dt
            if (i + 1) % 50 == 0:
                print(f'[stimuli] rendered {i + 1}/{len(todo)} jobs '
                      f'({(time.time() - t_sel) / (i + 1):.2f} s/job wall)', flush=True)
    rows.sort(key=lambda r: (r['split'], r['excerpt'], r['font'], C.VARIANTS.index(r['variant'])))
    C.write_tsv(root / 'manifest.tsv', rows, C.MANIFEST_COLS)
    clips = sum(1 for r in rows)
    missing = [r['path'] for r in rows if not (Path(r['path']).is_file() and Path(r['path']).stat().st_size > 0)]
    info = {'excerpts': len(picks), 'clips': clips, 'missing': len(missing),
            'per_split': {s: sum(1 for p in picks if p['split'] == s) for s in C.SPLITS},
            'extra_windows': sum(1 for p in picks if p['window'] != 'first'),
            'jobs': len(jobs), 'jobs_rendered': len(todo), 'clips_written': written,
            'seconds_select': round(t_sel - t_start, 1),
            'seconds_render': round(time.time() - t_sel, 1),
            'seconds_per_written_clip': round(secs / max(1, written), 3)}
    C.write_json(root / 'stimuli.json', info)
    print('[stimuli] ' + json.dumps(info), flush=True)
    if missing:
        raise SystemExit(f'[FAIL] {len(missing)} clips missing after render')


if __name__ == '__main__':
    main()
