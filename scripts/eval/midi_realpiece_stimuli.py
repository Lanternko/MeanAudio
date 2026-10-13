#!/usr/bin/env python3
"""Real-piece MIDI control stimuli (2026-10-14 meeting; CPU only, run before p2 123).

Source: MAESTRO v3 performance MIDI (human Disklavier performances of canonical piano
repertoire). One performance per (composer, title), the first 28 s of music from the
first onset, sustain pedal baked into note-offs, rendered with TinySoundFont on two GM
SoundFonts. Every case is the same excerpt with one intervention, so each stimulus has
its own clean baseline on the same SoundFont.

Classes (the expected direction is fixed here, before any score exists):
  degrade    rhythm / pitch / structure / dynamics errors a listener hears as wrong
  control    ordinary production or performance choices that should not move quality
  ambiguous  no agreed direction (non-idiomatic patch, large tempo or register change)
  floor      signal-level damage meant to sit at or below white noise

Output (OUT): audio/<bank>/<excerpt>__<case>.flac (30 s, mono 16 kHz, -30 LUFS scalar
gain, peak cap 0.999, PCM_24), items.json, excerpts.json, checks.json.

  PYTHONPATH=<midi-packages> ~/venvs/dac/bin/python scripts/eval/midi_realpiece_stimuli.py [--limit N]
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

CODEX = Path('/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/work')
sys.path.insert(0, str(CODEX / 'midi-packages'))
MAESTRO = Path('/mnt/seagate/kojiek_data/maestro_v3_midi/maestro-v3.0.0')
OUT = Path('/home/kojiek/eval_output_nvme/midi_realpiece_123')
BANKS = {'generaluser': CODEX / 'soundfonts/GeneralUser-GS.sf2',
         'fluidr3': CODEX / 'soundfonts/fluid/usr/share/sounds/sf2/FluidR3_GM.sf2'}
SR, SYNTH_SR, DUR, LEAD, MUSIC, SRC_SPAN = 16000, 44100, 30.0, 0.5, 28.0, 70.0
TARGET, PEAK_CAP, N_EXCERPTS, PER_COMPOSER = -30.0, 0.999, 150, 14
COMPOSERS = ('Ludwig van Beethoven', 'Frédéric Chopin', 'Wolfgang Amadeus Mozart', 'Johann Sebastian Bach',
             'Franz Schubert', 'Franz Liszt', 'Claude Debussy', 'Joseph Haydn', 'Robert Schumann',
             'Sergei Rachmaninoff', 'Felix Mendelssohn', 'Johannes Brahms', 'Domenico Scarlatti',
             'Pyotr Ilyich Tchaikovsky')


def h(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


# ── source ──
def read_notes(path):
    """MAESTRO performance -> [(on, off, pitch, vel)] in seconds with the sustain pedal baked in."""
    import mido
    t, pedal, sounding, held, notes = 0.0, False, {}, {}, []
    for msg in mido.MidiFile(path):
        t += msg.time
        if t > SRC_SPAN + 600:
            break
        if msg.type == 'control_change' and msg.control == 64:
            down = msg.value >= 64
            if pedal and not down:
                for p, (on, vel) in held.items():
                    notes.append((on, t, p, vel))
                held = {}
            pedal = down
        elif msg.type == 'note_on' and msg.velocity > 0:
            for d in (sounding, held):
                if msg.note in d:
                    on, vel = d.pop(msg.note)
                    notes.append((on, t, msg.note, vel))
            sounding[msg.note] = (t, msg.velocity)
        elif msg.type in ('note_off', 'note_on') and msg.note in sounding:
            on, vel = sounding.pop(msg.note)
            if pedal:
                held[msg.note] = (on, vel)
            else:
                notes.append((on, t, msg.note, vel))
    for d in (sounding, held):
        notes += [(on, t, p, vel) for p, (on, vel) in d.items()]
    return sorted(n for n in notes if n[1] > n[0])


def select():
    rows = list(csv.DictReader(open(MAESTRO / 'maestro-v3.0.0.csv')))
    seen, per, picks = set(), {}, []
    for r in sorted(rows, key=lambda r: h('pick', r['canonical_composer'], r['canonical_title'], r['midi_filename'])):
        key = (r['canonical_composer'], r['canonical_title'])
        if r['canonical_composer'] not in COMPOSERS or key in seen or float(r['duration']) < 120:
            continue
        if per.get(key[0], 0) >= PER_COMPOSER:
            continue
        notes = read_notes(MAESTRO / r['midi_filename'])
        t0 = notes[0][0]
        src = [(on - t0, min(off - t0, on - t0 + 12.0), p, v) for on, off, p, v in notes if on - t0 < SRC_SPAN]
        if sum(on < MUSIC for on, *_ in src) < 60 or max(on for on, *_ in src) < SRC_SPAN - 8:
            continue
        seen.add(key)
        per[key[0]] = per.get(key[0], 0) + 1
        picks.append({'id': f'x{len(picks):03d}', 'composer': key[0], 'title': key[1], 'year': r['year'],
                      'midi': r['midi_filename'], 'notes': src})
        if len(picks) == N_EXCERPTS:
            break
    return picks


# ── MIDI-level interventions: notes are (on, off, pitch, vel), time 0 = first onset ──
def warp(notes, seg, stretch):
    """Piecewise-linear time map: source segment i (length seg) lasts seg * stretch[i]."""
    edges = np.concatenate([[0.0], np.cumsum(np.asarray(stretch) * seg)])
    f = lambda t: float(np.interp(t, np.arange(len(edges)) * seg, edges))
    return [(f(on), f(off), p, v) for on, off, p, v in notes]


def tempo_jump(k):
    return lambda n, r: warp(n, 2.0, k ** r.choice([-1.0, 1.0], size=int(SRC_SPAN / 2) + 8))


def local_speed(speed):
    return lambda n, r: warp(n, 1.0, [1 / speed if 10 <= i < 16 else 1.0 for i in range(int(SRC_SPAN) + 8)])


def tempo(speed):
    return lambda n, r: warp(n, 1.0, [1 / speed] * (int(SRC_SPAN) + 8))


def jitter(sigma, chord):
    def f(notes, r):
        out, last, shift = [], -1e9, 0.0
        for on, off, p, v in notes:
            if not chord or on - last > 0.03:
                shift = float(np.clip(r.normal(0, sigma), -3 * sigma, 3 * sigma))
            last = on
            out.append((max(on + shift, 0.0), max(on + shift, 0.0) + off - on, p, v))
        return out
    return f


def wrong_notes(frac):
    def f(notes, r):
        hit = r.random(len(notes)) < frac
        d = r.choice([-2, -1, 1, 2], size=len(notes))
        return [(on, off, int(np.clip(p + d[i], 21, 108)) if hit[i] else p, v) for i, (on, off, p, v) in enumerate(notes)]
    return f


def pitch_shuffle(window):
    def f(notes, r):
        out = list(notes)
        groups = {}
        for i, n in enumerate(notes):
            groups.setdefault(int(n[0] // window) if window else 0, []).append(i)
        for idx in groups.values():
            perm = r.permutation(idx)
            for i, j in zip(idx, perm):
                out[i] = (notes[i][0], notes[i][1], notes[j][2], notes[i][3])
        return out
    return f


def random_pitch(notes, r):
    lo, hi = min(n[2] for n in notes), max(n[2] for n in notes)
    return [(on, off, int(r.integers(lo, hi + 1)), v) for on, off, p, v in notes]


def block_shuffle(block):
    def f(notes, r):
        perm = r.permutation(int(MUSIC / block))
        new = {int(b): i for i, b in enumerate(perm)}
        return [(new[int(on // block)] * block + on % block, new[int(on // block)] * block + on % block + off - on, p, v)
                for on, off, p, v in notes if on < MUSIC]
    return f


def retrograde(notes, r):
    return [(MUSIC - min(off, MUSIC), MUSIC - on, p, v) for on, off, p, v in notes if on < MUSIC]


def drop(frac):
    return lambda n, r: [x for x, keep in zip(n, r.random(len(n)) >= frac) if keep]


def extra(frac):
    def f(notes, r):
        lo, hi = min(n[2] for n in notes), max(n[2] for n in notes)
        k = int(frac * sum(n[0] < MUSIC for n in notes))
        vel = int(np.median([n[3] for n in notes]))
        add = [(float(t), float(t) + 0.3, int(r.integers(lo, hi + 1)), vel) for t in r.uniform(0, MUSIC, k)]
        return sorted(list(notes) + add)
    return f


def vel(mode):
    def f(notes, r):
        if mode == 'random':
            return [(on, off, p, int(r.integers(20, 128))) for on, off, p, v in notes]
        if mode == 'flat':
            return [(on, off, p, 64) for on, off, p, v in notes]
        return [(on, off, p, max(1, int(round(v * mode)))) for on, off, p, v in notes]
    return f


def transpose(k):
    return lambda n, r: [(on, off, int(np.clip(p + k, 0, 127)), v) for on, off, p, v in n]


# ── audio-level effects on the mono 16 kHz render ──
def rms(x):
    return float(np.sqrt(np.mean(x * x)))


def add_noise(snr):
    return lambda x, r, o: x + r.standard_normal(len(x)) * rms(x) / 10 ** (snr / 20)


def reverb(rt60, wet):
    def f(x, r, o):
        from scipy.signal import fftconvolve
        n = int(rt60 * 1.5 * SR)
        ir = np.random.default_rng(64 + int(rt60 * 10)).standard_normal(n) * np.exp(-6.9 * np.arange(n) / (rt60 * SR))
        ir[:int(0.005 * SR)] = 0
        y = fftconvolve(x, ir / np.sqrt(np.sum(ir * ir)))[:len(x)]
        return (1 - wet) * x + wet * y
    return f


def shelf(x, r, o):
    from scipy.signal import butter, sosfilt
    hi = sosfilt(butter(2, 4000, 'high', fs=SR, output='sos'), x)
    return x + (10 ** (-3 / 20) - 1) * hi


def clicks(rate):
    def f(x, r, o):
        y = x.copy()
        idx = r.integers(0, len(x), int(rate * len(x) / SR))
        y[idx] = r.choice([-1.0, 1.0], size=len(idx)) * np.max(np.abs(x))
        return y
    return f


def dropout(x, r, o):
    y, n = x.copy(), int(0.02 * SR)
    for s in r.integers(0, len(x) - n, int(5 * len(x) / SR)):
        y[s:s + n] = 0
    return y


def sample_shuffle(ms):
    def f(x, r, o):
        n = int(ms / 1000 * SR)
        blocks = x[:len(x) // n * n].reshape(-1, n)
        return np.concatenate([blocks[r.permutation(len(blocks))].ravel(), x[len(x) // n * n:]])
    return f


def splice(x, r, o):
    y = x.copy()
    for s in range(SR, len(x), 2 * SR):
        y[s:s + SR] = o[s:s + SR]
    return y


def hardclip(x, r, o):
    return np.clip(x * 10 ** (30 / 20), -np.max(np.abs(x)), np.max(np.abs(x)))


def bitcrush(x, r, o):
    q = np.max(np.abs(x)) / 4
    return np.round(x / q) * q


CASES = {  # name: (class, family, midi fn, program, detune cents, audio fn)
    'clean': ('base', 'base', None, 0, 0, None),
    **{f'tempo_jump_k{k}': ('degrade', 'rhythm', tempo_jump(k), 0, 0, None) for k in (1.15, 1.5, 2.0)},
    'local_slow_0.6': ('degrade', 'rhythm', local_speed(0.6), 0, 0, None),
    'local_fast_2.0': ('degrade', 'rhythm', local_speed(2.0), 0, 0, None),
    **{f'jitter_chord_{ms}ms': ('degrade', 'rhythm', jitter(ms / 1000, True), 0, 0, None) for ms in (30, 80, 200)},
    'jitter_note_80ms': ('degrade', 'rhythm', jitter(0.08, False), 0, 0, None),
    **{f'wrong_notes_{p}pct': ('degrade', 'pitch', wrong_notes(p / 100), 0, 0, None) for p in (10, 30, 100)},
    'pitch_shuffle_all': ('degrade', 'pitch', pitch_shuffle(0), 0, 0, None),
    'pitch_shuffle_2s': ('degrade', 'pitch', pitch_shuffle(2.0), 0, 0, None),
    'random_pitch': ('degrade', 'pitch', random_pitch, 0, 0, None),
    'detune_split_30c': ('degrade', 'pitch', None, 0, 30, None),
    'detune_split_50c': ('degrade', 'pitch', None, 0, 50, None),
    'block_shuffle_2s': ('degrade', 'structure', block_shuffle(2.0), 0, 0, None),
    'block_shuffle_0.5s': ('degrade', 'structure', block_shuffle(0.5), 0, 0, None),
    'retrograde': ('degrade', 'structure', retrograde, 0, 0, None),
    'vel_random': ('degrade', 'dynamics', vel('random'), 0, 0, None),
    'vel_flat': ('degrade', 'dynamics', vel('flat'), 0, 0, None),
    'drop_30pct': ('degrade', 'notes', drop(0.3), 0, 0, None),
    'drop_60pct': ('degrade', 'notes', drop(0.6), 0, 0, None),
    'extra_30pct': ('degrade', 'notes', extra(0.3), 0, 0, None),
    'bright_piano': ('control', 'timbre', None, 1, 0, None),
    'reverb_room': ('control', 'production', None, 0, 0, reverb(0.4, 0.2)),
    'reverb_hall': ('control', 'production', None, 0, 0, reverb(1.8, 0.3)),
    'shelf_m3db': ('control', 'production', None, 0, 0, shelf),
    'tempo_m5pct': ('control', 'performance', tempo(0.95), 0, 0, None),
    'tempo_p5pct': ('control', 'performance', tempo(1.05), 0, 0, None),
    'vel_x0.85': ('control', 'performance', vel(0.85), 0, 0, None),
    **{f'transpose_{k:+d}': ('control', 'key', transpose(k), 0, 0, None) for k in (-2, -1, 1, 2)},
    'tempo_x0.75': ('ambiguous', 'tempo', tempo(0.75), 0, 0, None),
    'tempo_x1.33': ('ambiguous', 'tempo', tempo(1.33), 0, 0, None),
    'transpose_-12': ('ambiguous', 'register', transpose(-12), 0, 0, None),
    'transpose_+12': ('ambiguous', 'register', transpose(12), 0, 0, None),
    'patch_violin': ('ambiguous', 'patch', None, 40, 0, None),
    'patch_flute': ('ambiguous', 'patch', None, 73, 0, None),
    'patch_nylon_guitar': ('ambiguous', 'patch', None, 24, 0, None),
    'patch_warm_pad': ('ambiguous', 'patch', None, 89, 0, None),
    'white_snr20': ('floor', 'noise', None, 0, 0, add_noise(20)),
    'white_snr0': ('floor', 'noise', None, 0, 0, add_noise(0)),
    'pure_white': ('floor', 'noise', None, 0, 0, lambda x, r, o: r.standard_normal(len(x)) * rms(x)),
    'clicks_5ps': ('floor', 'glitch', None, 0, 0, clicks(5)),
    'clicks_50ps': ('floor', 'glitch', None, 0, 0, clicks(50)),
    'dropout_20ms_5ps': ('floor', 'glitch', None, 0, 0, dropout),
    'sample_shuffle_20ms': ('floor', 'glitch', None, 0, 0, sample_shuffle(20)),
    'sample_shuffle_200ms': ('floor', 'glitch', None, 0, 0, sample_shuffle(200)),
    'splice_other_1s': ('floor', 'glitch', None, 0, 0, splice),
    'hardclip_30db': ('floor', 'distortion', None, 0, 0, hardclip),
    'bitcrush_3bit': ('floor', 'distortion', None, 0, 0, bitcrush),
    'reverse_audio': ('floor', 'glitch', None, 0, 0, lambda x, r, o: x[::-1].copy()),
}

# ── render ──
_SYNTH = {}


def synth(notes, bank, program, cents):
    """notes on the excerpt clock -> mono 16 kHz float64, 30 s, music from LEAD."""
    import soxr
    import tinysoundfont
    if bank not in _SYNTH:
        s = tinysoundfont.Synth(gain=-6, samplerate=SYNTH_SR)
        _SYNTH[bank] = (s, s.sfload(str(BANKS[bank])))
    s, sid = _SYNTH[bank]
    s.sounds_off()
    s.soundfonts[sid].reset()
    s.generate_simple(4410)
    for ch in (0, 1):
        s.program_select(ch, sid, 0, program)
        s.pitchbend(ch, 8192)
    if cents:
        s.pitchbend(1, 8192 + int(round(cents / 200 * 8192)))
    ev = []
    for i, (on, off, p, v) in enumerate(notes):
        on, off = on + LEAD, min(off + LEAD, DUR - 0.2)
        if on >= DUR - 0.5 or off <= on:
            continue
        ch = 1 if cents and h('split', i) % 2 else 0
        ev += [(off, 0, ch, p, 0), (on, 1, ch, p, v)]
    x, cur = np.zeros((int(SYNTH_SR * DUR), 2), np.float32), 0
    for t, kind, ch, p, v in sorted(ev):
        end = min(int(round(t * SYNTH_SR)), len(x))
        if end > cur:
            x[cur:end] = np.frombuffer(s.generate_simple(end - cur), np.float32).reshape(-1, 2)
            cur = end
        s.noteon(ch, p, v) if kind else s.noteoff(ch, p)
    x[cur:] = np.frombuffer(s.generate_simple(len(x) - cur), np.float32).reshape(-1, 2)
    return soxr.resample(x.mean(axis=1).astype(np.float64), SYNTH_SR, SR, quality='VHQ')[:int(SR * DUR)]


def level(x):
    import pyloudnorm
    lufs, peak = float(pyloudnorm.Meter(SR).integrated_loudness(x)), float(np.max(np.abs(x)))
    if not np.isfinite(lufs) or peak <= 0:
        raise RuntimeError('unmeasurable loudness')
    g = 10 ** ((TARGET - lufs) / 20)
    capped = peak * g > PEAK_CAP
    return x * (PEAK_CAP / peak if capped else g), lufs, int(capped)


def midi_of(ex, case):
    fn = CASES[case][2]
    notes = [tuple(n) for n in ex['notes']]
    if fn is not None:
        notes = fn(notes, np.random.default_rng(h('midi', ex['id'], case)))
    return sorted(n for n in notes if n[0] < MUSIC)


def render_excerpt(job):
    ex, other, bank = job
    clean = synth(midi_of(ex, 'clean'), bank, 0, 0)
    other_clean = synth(midi_of(other, 'clean'), bank, 0, 0)
    rows = []
    for case, (cls, fam, fn, prog, cents, fx) in CASES.items():
        dst = OUT / 'audio' / bank / f'{ex["id"]}__{case}.flac'
        notes = midi_of(ex, case)
        x = clean if (fn is None and prog == 0 and not cents) else synth(notes, bank, prog, cents)
        if fx is not None:
            x = fx(x, np.random.default_rng(h('fx', ex['id'], case)), other_clean)
        y, lufs, capped = level(np.asarray(x, np.float64))
        sf.write(dst, y, SR, format='FLAC', subtype='PCM_24')
        on = np.array([n[0] for n in notes])
        rows.append({'key': f'{bank}__{ex["id"]}__{case}', 'bank': bank, 'excerpt': ex['id'], 'case': case,
                     'class': cls, 'family': fam, 'path': str(dst), 'n_notes': len(notes),
                     'ioi_cv': float(np.std(np.diff(np.unique(on))) / np.mean(np.diff(np.unique(on)))),
                     'pitch_mean': float(np.mean([n[2] for n in notes])), 'src_lufs': lufs, 'capped': capped,
                     'base': '' if case == 'clean' else f'{bank}__{ex["id"]}__clean'})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    for bank in BANKS:
        (OUT / 'audio' / bank).mkdir(parents=True, exist_ok=True)
    cache = OUT / 'excerpts.json'
    picks = json.loads(cache.read_text()) if cache.is_file() else select()
    if not cache.is_file():
        cache.write_text(json.dumps(picks))
    picks = picks[:a.limit] if a.limit else picks
    jobs = [(ex, picks[(i + 1) % len(picks)], bank) for i, ex in enumerate(picks) for bank in BANKS]
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(render_excerpt, jobs)):
            rows += r
            if (i + 1) % 20 == 0:
                print(f'rendered {i + 1}/{len(jobs)} excerpt-banks', flush=True)
    (OUT / ('items.json' if not a.limit else 'items_smoke.json')).write_text(json.dumps(rows, indent=0))
    by = {}
    for r in rows:
        by.setdefault(r['case'], []).append(r)
    checks = {c: {'n': len(v), 'class': v[0]['class'], 'notes': float(np.mean([r['n_notes'] for r in v])),
                  'ioi_cv': float(np.mean([r['ioi_cv'] for r in v])), 'pitch_mean': float(np.mean([r['pitch_mean'] for r in v])),
                  'capped': sum(r['capped'] for r in v)} for c, v in by.items()}
    (OUT / ('checks.json' if not a.limit else 'checks_smoke.json')).write_text(json.dumps(checks, indent=1))
    comp = {}
    for p in picks:
        comp[p['composer']] = comp.get(p['composer'], 0) + 1
    print(json.dumps({'excerpts': len(picks), 'clips': len(rows), 'cases': len(CASES), 'composers': comp}, ensure_ascii=False))


if __name__ == '__main__':
    raise SystemExit(main())
