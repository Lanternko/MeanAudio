#!/usr/bin/env python3
"""126b stimuli: famous pieces as MIDI / real / poor performances, one-variable ladders, floors.

Design: docs/experiments/metric_win_tie_126_20261016.md (MIDI block, real-recording block, floors).
CPU only; run before queueing 126b. Every clip is 10 s, mono 16 kHz, FLAC PCM_24.

Blocks
  midi     Mutopia classical piano MIDI (9 pieces, <=16 windows each) and the 50 CSD English nursery
           melodies (keys a/b, <=3 windows each, note times aligned to the sung wav), cut into contiguous 9 s windows, rendered piano-only
           (GM program 0) on two SoundFonts with 0.25 s lead, -30 LUFS. Each window gets
           ~35 MIDI-level cases (wrong / random pitch, note positions shuffled, jitter, unsteady tempo,
           block shuffle, dropped / extra notes, one note made a clipped blast, and the "should not
           matter" edits: transpose +-1, tempo +-2/5 %, velocity x0.85) and the 52 signal variants
           of 126a (gain, polarity, lead silence, formats, noise, clipping, ...) on its GeneralUser clean render.
  rec      real recordings, <=8 10 s crops from 15 s on: pro piano and presumed-poor piano (YouTube,
           listed in 126_pieces/manifest.json), CSD professional singing (same windows as the MIDI
           block, so real singing and its piano melody are paired), presumed-poor child singing
           (YouTube). Each crop gets the 52 signal variants.
  floor    signals worse than white noise (binary +-1 and 0/1 noise, square waves, DC, digital
           silence, a sine, a sweep, a pulse train, white noise itself), each at -30 LUFS and at
           full scale.

Output: OUT/audio/<block>/<source>/<excerpt>__<case>.flac and OUT/items.json (one row per clip:
key, block, group, source, piece, excerpt, case, kind, ladder, caption, base, path, mir, ...).

  PYTHONPATH= ~/venvs/dac/bin/python scripts/eval/metric_win_tie_126b_stimuli.py [--limit N] [--workers 12]
"""
import argparse
import csv
import json
import math
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midi_realpiece_stimuli as S  # noqa: E402  (123 renderer + MIDI interventions; adds midi-packages to sys.path)
import metric_win_tie_126a as A  # noqa: E402  (126a signal variants)

PIECES = Path('/mnt/seagate/kojiek_data/126_pieces')
CSD = Path('/mnt/seagate/kojiek_data/csd/CSD/english')
# NVMe, not /mnt/seagate: the USB disk stalls on ext4 journal commits with ~100k small files (279 ms per write)
OUT = Path('/home/kojiek/eval_output_nvme/metric_win_tie_126b_20261016/stimuli')
SR, DUR, LEAD, WIN = 16000, 10.0, 0.25, 9.0
S.DUR, S.LEAD, S.MUSIC = DUR, LEAD, WIN  # the 123 renderer and interventions read these module globals
MIN_ONSETS, REC_START, REC_STEP, REC_MAX = 6, 15.0, 10.0, 8
MAX_WIN = {'classical': 16, 'nursery': 3}
CAPTION = {'classical': 'solo piano music', 'nursery': "a children's song", 'floor': 'music'}
CSD_SKIP_POOR = {'en046'}  # CSD en046 is "Together"; the fetch searched a different song under that id


def h(*k):
    return zlib.crc32('/'.join(map(str, k)).encode())


# ── MIDI-level cases (notes: (on, off, pitch, vel), window clock) ──
def one_wrong(semis):
    def f(notes, r):
        mid = [i for i, n in enumerate(notes) if 2.0 <= n[0] <= WIN - 2.0] or list(range(len(notes)))
        i = mid[len(mid) // 2]
        d = semis * (1 if r.random() < 0.5 or semis == 6 else -1)
        return [(on, off, int(np.clip(p + d, 21, 108)) if j == i else p, v) for j, (on, off, p, v) in enumerate(notes)]
    return f


def rand_pitch(frac):
    def f(notes, r):
        lo, hi = min(n[2] for n in notes), max(n[2] for n in notes)
        hit = r.random(len(notes)) < frac
        return [(on, off, int(r.integers(lo, hi + 1)) if hit[j] else p, v) for j, (on, off, p, v) in enumerate(notes)]
    return f


def pos_shuffle(frac):
    """the selected notes keep pitch, velocity and length but get a uniformly random onset"""
    def f(notes, r):
        out = []
        for on, off, p, v in notes:
            if r.random() < frac:
                d = off - on
                on = float(r.uniform(0, max(WIN - min(d, 1.0), 0.1)))
                off = on + d
            out.append((on, off, p, v))
        return sorted(out)
    return f


def unsteady(x):
    return lambda n, r: S.warp(n, 0.5, 1 + r.uniform(-x, x, size=int(S.SRC_SPAN / 0.5) + 8))


def block_shuffle(block):
    def f(notes, r):
        nb = int(round(WIN / block))
        perm = r.permutation(nb)
        new = {int(b): i for i, b in enumerate(perm)}
        return sorted((new[int(on // block)] * block + on % block, new[int(on // block)] * block + on % block + off - on, p, v)
                      for on, off, p, v in notes if on < nb * block)
    return f


def blast_pick(k):
    """indices of k notes spread over the window: they are rendered alone at velocity 127, +24 dB, clipped"""
    def f(notes, r):
        cand = [i for i, n in enumerate(notes) if 0.5 <= n[0] <= WIN - 0.5] or list(range(len(notes)))
        return sorted(int(i) for i in r.choice(cand, min(k, len(cand)), replace=False))
    return f


MIDI_CASES = {  # name: (kind, ladder, fn)  kind: lose = should lose, soft = we judge it unimportant (should tie)
    'wrong1_semi': ('lose', 'wrong', one_wrong(1)), 'wrong1_tritone': ('lose', 'wrong', one_wrong(6)),
    **{f'wrong{p}pct': ('lose', 'wrong', S.wrong_notes(p / 100)) for p in (5, 10, 30, 100)},
    **{f'randpitch{p}pct': ('lose', 'randpitch', rand_pitch(p / 100)) for p in (10, 30, 100)},
    **{f'posshuffle{p}pct': ('lose', 'posshuffle', pos_shuffle(p / 100)) for p in (10, 30, 100)},
    'pitchshuffle2s': ('lose', 'pitchshuffle', S.pitch_shuffle(2.0)), 'pitchshuffle_all': ('lose', 'pitchshuffle', S.pitch_shuffle(0)),
    **{f'jitter{ms}ms': ('lose', 'jitter', S.jitter(ms / 1000, True)) for ms in (10, 30, 80, 200)},
    **{f'unsteady{p}pct': ('lose', 'unsteady', unsteady(p / 100)) for p in (5, 15, 30)},
    **{f'blockshuffle{b}s': ('lose', 'blockshuffle', block_shuffle(b)) for b in (3.0, 1.5, 0.5)},
    'retrograde': ('lose', 'structure', S.retrograde),
    **{f'drop{p}pct': ('lose', 'drop', S.drop(p / 100)) for p in (10, 30, 60)},
    **{f'extra{p}pct': ('lose', 'extra', S.extra(p / 100)) for p in (10, 30)},
    'vel_random': ('lose', 'dynamics', S.vel('random')),
    **{f'blastnote_x{k}': ('lose', 'blastnote', blast_pick(k)) for k in (1, 3, 10)},
    'transpose-1': ('soft', 'transpose', S.transpose(-1)), 'transpose+1': ('soft', 'transpose', S.transpose(1)),
    **{f'tempo_x{s}': ('soft', 'tempo', S.tempo(s)) for s in (0.98, 1.02, 0.95, 1.05)},
    'vel_x0.85': ('soft', 'dynamics_soft', S.vel(0.85)),
}
SIGNAL = [v for v in A.variants() if v['kind'] != 'ref']  # 22 tie + 30 lose, same definitions as 126a
SIGNAL_BANK = 'generaluser'
SIG_LADDER = {st: lad for lad, steps in A.LADDERS.items() for st in steps}


# ── sources ──
def windows(notes, group):
    notes = sorted(notes)
    t0, last = notes[0][0], max(n[0] for n in notes)
    out = []
    for k in range(MAX_WIN[group]):
        w0 = t0 + k * WIN
        if w0 + WIN > last + 1.0:
            break
        sel = [(on - w0, min(off - w0, on - w0 + 8.0), p, v) for on, off, p, v in notes if w0 <= on < w0 + WIN]
        if len({round(n[0], 2) for n in sel}) >= MIN_ONSETS:
            out.append((w0, sel))
    return out


def excerpts():
    exs = []
    manifest = json.loads((PIECES / 'manifest.json').read_text())
    for key, p in manifest.items():
        if p['genre'] != 'classical':
            continue
        notes = S.read_notes(PIECES / 'classical' / key / 'mutopia.mid')
        for k, (w0, sel) in enumerate(windows(notes, 'classical')):
            exs.append({'id': f'{key}_w{k:02d}', 'group': 'classical', 'piece': key, 'w0': w0, 'notes': sel})
    meta = json.loads((CSD / 'metadata.json').read_text())
    for song in sorted(meta):
        notes = [(float(r['start']), float(r['end']), int(r['pitch']), 80) for r in csv.DictReader(open(CSD / 'csv' / f'{song}.csv'))]
        for k, (w0, sel) in enumerate(windows(notes, 'nursery')):
            exs.append({'id': f'{song}_w{k:02d}', 'group': 'nursery', 'piece': song[:5], 'csd': song, 'w0': w0, 'notes': sel})
    return exs


def recordings():
    """(source, group, piece, path) of every real recording"""
    recs = []
    manifest = json.loads((PIECES / 'manifest.json').read_text())
    for key, p in manifest.items():
        for kind in ('pro', 'poor'):
            for it in p.get(kind, []):
                if not Path(it.get('file', '')).is_file() or it.get('veto'):  # veto: title shows it is not what was searched
                    continue
                if p['genre'] == 'classical':
                    recs.append((kind, 'classical', key, it['file']))
                elif key not in CSD_SKIP_POOR:
                    recs.append(('poor_sing', 'nursery', key, it['file']))
    return recs


# ── audio helpers ──
def load_mono16k(path):
    import soxr
    x, sr = sf.read(path, dtype='float64', always_2d=True)
    x = x.mean(axis=1)
    return soxr.resample(x, sr, SR, quality='VHQ') if sr != SR else x


def lufs(x):
    import pyloudnorm
    return float(pyloudnorm.Meter(SR).integrated_loudness(x))


def write(dst, y, subtype='PCM_24'):
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, SR, format='FLAC', subtype=subtype)


def signal_rows(base_row, x, excerpt, block, source, group, piece, extra):
    """the 52 126a signal variants of one lvl30 clean clip x"""
    rows = []
    for var in SIGNAL:
        y, over = A.transform(x, var, A.rng_for(block, source, excerpt, var['name']), SR)
        dst = OUT / 'audio' / block / source / f'{excerpt}__{var["name"]}.flac'
        write(dst, y, var['subtype'])
        rows.append({**base_row, 'key': f'{block}__{source}__{excerpt}__{var["name"]}', 'case': var['name'],
                     'kind': var['kind'], 'ladder': SIG_LADDER.get(var['name'], var['name']), 'path': str(dst),
                     'base': base_row['key'], 'over_fs': over, 'mir': False, **extra})
    return rows


# ── MIDI block ──
def render_midi(job):
    ex, bank = job
    source, block, group = f'midi_{bank}', 'midi', ex['group']
    notes = [tuple(n) for n in ex['notes']]
    clean, _, capped = S.level(S.synth(notes, bank, 0, 0))
    base = {'block': block, 'group': group, 'source': source, 'piece': ex['piece'], 'excerpt': ex['id'],
            'caption': CAPTION[group], 'bank': bank, 'w0': ex['w0']}
    key0 = f'{block}__{source}__{ex["id"]}__clean'
    dst = OUT / 'audio' / block / source / f'{ex["id"]}__clean.flac'
    write(dst, clean)
    row0 = {**base, 'key': key0, 'case': 'clean', 'kind': 'ref', 'ladder': 'clean', 'path': str(dst), 'base': '',
            'over_fs': 0, 'capped': capped, 'mir': True, 'n_notes': len(notes)}
    rows = [row0]
    for case, (kind, ladder, fn) in MIDI_CASES.items():
        r = np.random.default_rng(h('midi126b', ex['id'], case))  # same edit on both SoundFonts
        if ladder == 'blastnote':
            hit = set(fn(notes, r))
            rest = [n for i, n in enumerate(notes) if i not in hit]
            loud = [(on, off, p, 127) for i, (on, off, p, v) in enumerate(notes) if i in hit]
            y_rest, y_hit = S.synth(rest, bank, 0, 0), S.synth(loud, bank, 0, 0)
            cap = 3 * float(np.max(np.abs(S.synth(notes, bank, 0, 0))))
            x = y_rest + np.clip(y_hit * 10 ** (24 / 20), -cap, cap)
            n_out = len(notes)
        else:
            new = sorted(n for n in fn(notes, r) if n[0] < WIN + 0.5)
            x, n_out = S.synth(new, bank, 0, 0), len(new)
        y, _, capped = S.level(np.asarray(x, np.float64))
        dst = OUT / 'audio' / block / source / f'{ex["id"]}__{case}.flac'
        write(dst, y)
        rows.append({**base, 'key': f'{block}__{source}__{ex["id"]}__{case}', 'case': case, 'kind': kind,
                     'ladder': ladder, 'path': str(dst), 'base': key0, 'over_fs': 0, 'capped': capped, 'mir': True,
                     'n_notes': n_out})
    if bank == SIGNAL_BANK:  # signal variants once per excerpt; the second SoundFont only carries the MIDI-level cases
        rows += signal_rows(row0, clean, ex['id'], block, source, group, ex['piece'], {})
    return rows


# ── recording block ──
def csd_crops(song, exs):
    """CSD sung wav cut at the MIDI windows of the same song/key (aligned: same note clock)"""
    x = load_mono16k(CSD / 'wav' / f'{song}.wav')
    out = []
    for ex in exs:
        s = int(round((ex['w0'] - LEAD) * SR))
        seg = x[max(s, 0):s + int(DUR * SR)]
        if s < 0:
            seg = np.concatenate([np.zeros(-s), seg])
        seg = np.pad(seg, (0, int(DUR * SR) - len(seg)))
        out.append((ex['id'], seg))
    return out


def rec_crops(path):
    x = load_mono16k(path)
    out, k = [], 0
    t = REC_START if len(x) / SR >= REC_START + 2 * DUR else 0.0
    while t + DUR <= len(x) / SR and len(out) < REC_MAX:
        seg = x[int(t * SR):int(t * SR) + int(DUR * SR)]
        out.append((f'{Path(path).stem}_c{k:02d}', seg))
        t += REC_STEP
        k += 1
    return out


def render_rec(job):
    source, group, piece, crops_src = job
    if source == 'csd_real':
        crops = csd_crops(piece, crops_src)
        piece = piece[:5]
    else:
        crops = [(f'{piece}_{c}', seg) for c, seg in rec_crops(crops_src)]
    rows = []
    for exid, seg in crops:
        l0 = lufs(seg)
        if not np.isfinite(l0) or l0 < -50:  # silence / talking pause: not a music crop
            continue
        clean, _, capped = S.level(seg)
        key0 = f'rec__{source}__{exid}__clean'
        dst = OUT / 'audio' / 'rec' / source / f'{exid}__clean.flac'
        write(dst, clean)
        base = {'block': 'rec', 'group': group, 'source': source, 'piece': piece, 'excerpt': exid,
                'caption': CAPTION[group], 'src_lufs': l0, 'src_file': str(crops_src) if source != 'csd_real' else ''}
        row0 = {**base, 'key': key0, 'case': 'clean', 'kind': 'ref', 'ladder': 'clean', 'path': str(dst), 'base': '',
                'over_fs': 0, 'capped': capped, 'mir': True}
        rows += [row0] + signal_rows(row0, clean, exid, 'rec', source, group, piece, {})
    return rows


# ── floors ──
def floor_signal(name, seed):
    n, t = int(DUR * SR), np.arange(int(DUR * SR)) / SR
    r = np.random.default_rng(h('floor126b', name, seed))
    if name == 'white':
        return np.clip(r.standard_normal(n) * 0.3, -1, 1)
    if name == 'binary_pm1':
        return r.choice([-1.0, 1.0], n)
    if name == 'binary_01':
        return r.choice([0.0, 1.0], n)
    if name == 'binary_01_hold':  # 0/1 held for 1-50 ms runs: an on/off gated DC
        out, i = np.zeros(n), 0
        while i < n:
            k = int(r.integers(16, 800))
            out[i:i + k] = r.integers(0, 2)
            i += k
        return out
    if name == 'square100':
        return np.sign(np.sin(2 * np.pi * 100 * t + 1e-9))
    if name == 'square1k':
        return np.sign(np.sin(2 * np.pi * 1000 * t + 1e-9))
    if name == 'sine1k':
        return np.sin(2 * np.pi * 1000 * t)
    if name == 'sweep':
        f0, f1 = 20.0, 8000.0
        return np.sin(2 * np.pi * f0 * DUR / math.log(f1 / f0) * (np.exp(t / DUR * math.log(f1 / f0)) - 1))
    if name == 'pulse10ps':
        out = np.zeros(n)
        out[::SR // 10] = 1.0
        return out
    if name == 'dc':
        return np.ones(n)
    if name == 'silence':
        return np.zeros(n)
    raise ValueError(name)


FLOORS = {'white': 20, 'binary_pm1': 20, 'binary_01': 20, 'binary_01_hold': 20, 'square100': 1, 'square1k': 1,
          'sine1k': 1, 'sweep': 1, 'pulse10ps': 1, 'dc': 1, 'silence': 1}


def render_floors():
    rows = []
    for name, n_seed in FLOORS.items():
        for seed in range(n_seed):
            x = floor_signal(name, seed)
            for lvl in ('lvl30', 'fs'):
                y, note = x * 0.999, ''
                if lvl == 'lvl30':
                    l0 = lufs(x) if np.any(x) else -np.inf
                    if np.isfinite(l0):  # pure scalar gain to -30 LUFS, no peak cap: the waveform shape is the point
                        y = np.clip(x * 10 ** ((-30.0 - l0) / 20), -0.999, 0.999)
                    else:
                        y, note = x * 0.03, 'unmeasurable loudness: kept at 0.03 amplitude'
                exid = f'{name}_s{seed:02d}_{lvl}'
                dst = OUT / 'audio' / 'floor' / 'floor' / f'{exid}.flac'
                write(dst, y)
                rows.append({'key': f'floor__floor__{exid}', 'block': 'floor', 'group': 'floor', 'source': 'floor',
                             'piece': name, 'excerpt': exid, 'case': f'{name}_{lvl}', 'kind': 'floor', 'ladder': 'floor',
                             'caption': CAPTION['floor'], 'path': str(dst), 'base': '', 'over_fs': 0, 'mir': True,
                             'level': lvl, 'note': note})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, help='first N windows per group and first N recordings (smoke)')
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    global OUT
    if a.limit:
        OUT = OUT / 'smoke'
    OUT.mkdir(parents=True, exist_ok=True)
    exs = excerpts()
    if a.limit:
        exs = [e for g in ('classical', 'nursery') for e in [x for x in exs if x['group'] == g][:a.limit]]
    (OUT / 'excerpts.json').write_text(json.dumps(exs))
    by_song = {}
    for e in exs:
        if e['group'] == 'nursery':
            by_song.setdefault(e['csd'], []).append(e)
    recs = recordings()[:a.limit] if a.limit else recordings()
    jobs_midi = [(e, b) for e in exs for b in S.BANKS]
    jobs_rec = [('csd_real', 'nursery', song, v) for song, v in by_song.items()] + list(recs)
    rows = render_floors()
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(render_midi, jobs_midi, chunksize=2)):
            rows += r
            if (i + 1) % 100 == 0:
                print(f'midi {i + 1}/{len(jobs_midi)}', flush=True)
        for i, r in enumerate(pool.imap_unordered(render_rec, jobs_rec)):
            rows += r
            if (i + 1) % 20 == 0:
                print(f'rec {i + 1}/{len(jobs_rec)}', flush=True)
    rows.sort(key=lambda r: r['key'])
    keys = [r['key'] for r in rows]
    assert len(keys) == len(set(keys)), 'duplicate keys'
    (OUT / 'items.json').write_text(json.dumps(rows, indent=0))
    count = {}
    for r in rows:
        count.setdefault(f'{r["block"]}/{r["source"]}', {}).setdefault(r['kind'], 0)
        count[f'{r["block"]}/{r["source"]}'][r['kind']] += 1
    print(json.dumps({'clips': len(rows), 'excerpts': len(exs), 'recordings': len(recs), 'by_source': count}, indent=1))


if __name__ == '__main__':
    raise SystemExit(main())
