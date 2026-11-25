"""Shared constants/helpers for the 178 wrong-note probe.

Prereg: docs/experiments/wrongnote_probe_mert_178_20261103.md
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/wrongnote_probe_178')
REPO = Path('/home/kojiek/MeanAudio')
RESULTS = REPO / 'docs/experiments/results'

SEED = 178
SR = 16000
EMBED_SR = 24000
DUR = 10.0          # render window
LEAD = 0.5          # silence before the music, same as midi_realpiece_stimuli
MUSIC_END = DUR - 0.6   # 9.4: last onset that still reaches the output
TARGET_LUFS = -23.0

SPLITS = {'train': 800, 'val': 150, 'test': 250}
FONTS = {'train': ('generaluser',), 'val': ('generaluser',), 'test': ('generaluser', 'fluidr3')}
MAIN_FONT = 'generaluser'

OK_VARIANTS = ('clean', 'transpose', 'tempo', 'gain_m6')
WRONG_KS = (1, 5, 10, 30)
WRONG_KINDS = ('in', 'out')
WRONG_VARIANTS = tuple(f'wrong_{k}_{kind}' for k in WRONG_KS for kind in WRONG_KINDS)
VARIANTS = OK_VARIANTS + WRONG_VARIANTS

# Krumhansl-Kessler key profiles (probe-tone ratings, scale degree 1..12)
KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
SCALE_DEGREES = {'major': (0, 2, 4, 5, 7, 9, 11), 'minor': (0, 2, 3, 5, 7, 8, 10)}
PITCH_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')


def h(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:16], 16)


# ── key estimation (prereg §1) ────────────────────────────────────────────────
def pitch_class_hist(notes):
    """Duration-weighted pitch-class histogram of the notes of one excerpt."""
    import numpy as np
    hist = np.zeros(12, dtype=float)
    for on, off, p, _v in notes:
        hist[int(p) % 12] += max(1e-3, float(off) - float(on))
    return hist


def estimate_key(notes):
    """Best of 24 KK templates by Pearson correlation with the pitch-class histogram."""
    import numpy as np
    hist = pitch_class_hist(notes)
    c = hist - hist.mean()
    nc = float(np.linalg.norm(c))
    best = None
    for mode, prof in (('major', KK_MAJOR), ('minor', KK_MINOR)):
        t = np.asarray(prof, dtype=float)
        t = t - t.mean()
        nt = float(np.linalg.norm(t))
        for tonic in range(12):
            v = np.roll(t, tonic)
            r = float(c @ v / (nc * nt + 1e-12)) if nc > 0 else 0.0
            if best is None or r > best[0]:
                best = (r, tonic, mode)
    r, tonic, mode = best
    scale = sorted({(tonic + d) % 12 for d in SCALE_DEGREES[mode]})
    return {'tonic': int(tonic), 'mode': mode, 'r': r, 'scale': scale,
            'name': f'{PITCH_NAMES[tonic]} {mode}'}


def in_scale(p, key):
    return int(p) % 12 in key['scale']


def out_ratio(notes, key):
    """Symbolic oracle: share of notes whose pitch class is outside the estimated scale."""
    if not notes:
        return 0.0
    return float(sum(0 if in_scale(n[2], key) else 1 for n in notes)) / len(notes)


# ── wrong-note interventions (prereg §1) ──────────────────────────────────────
def _target_pitch(p, key, kind, rng):
    """Nearest pitch within <= 2 semitones that is non-scale (out) or another scale tone (in)."""
    opts = []
    for d in (1, 2):
        for s in (-1, 1):
            q = p + s * d
            if not 21 <= q <= 108 or q == p:
                continue
            if (kind == 'out' and not in_scale(q, key)) or (kind == 'in' and in_scale(q, key)):
                opts.append((d, q))
    if not opts:
        if kind == 'in' and not in_scale(p, key):
            # prereg: a note that was already outside the scale moves to the NEXT scale tone
            for q in range(p + 1, min(p + 13, 109)):
                if in_scale(q, key):
                    return int(q)
        return None
    if kind == 'in' and not in_scale(p, key):
        return int(min(q for _d, q in opts))
    dmin = min(d for d, _q in opts)
    cands = [q for d, q in opts if d == dmin]
    return int(cands[int(rng.integers(len(cands)))])


def select_idxs(notes, k, rng):
    """Which notes get moved. k == 1: the note closest to the centre outside the middle 2 s."""
    n = len(notes)
    if n == 0:
        return []
    if k == 1:
        on = [float(x[0]) for x in notes]
        c = 0.5 * (min(on) + max(on))
        cand = [i for i, t in enumerate(on) if t < MUSIC_END and abs(t - c) > 1.0]
        pool = cand or [i for i, t in enumerate(on) if t < MUSIC_END] or list(range(n))
        return [min(pool, key=lambda i: (abs(on[i] - c), i))]
    take = max(1, int(round(n * k / 100.0)))
    return sorted(int(i) for i in rng.choice(n, size=min(take, n), replace=False))


def apply_wrong(notes, key, kind, k, rng):
    """Pitch-only edit; timing / velocity untouched. Returns (notes, n_changed)."""
    out = [tuple(n) for n in notes]
    changed = 0
    for i in select_idxs(notes, k, rng):
        on, off, p, v = out[i]
        q = _target_pitch(int(p), key, kind, rng)
        if q is not None and q != p:
            out[i] = (on, off, q, v)
            changed += 1
    return out, changed


def build_variants(notes, key):
    """Variant name -> notes, one rng per variant so the seeds are independent."""
    import numpy as np
    out = {'clean': [tuple(n) for n in notes]}
    r = np.random.default_rng(h('variant', 'transpose', [n[2] for n in notes][:4], len(notes)))
    step = 1 if r.integers(2) else -1
    out['transpose'] = [(on, off, int(min(108, max(21, p + step))), v) for on, off, p, v in notes]
    r = np.random.default_rng(h('variant', 'tempo', len(notes)))
    speed = 1.0 + (0.03 if r.integers(2) else -0.03)
    out['tempo'] = [(on / speed, off / speed, p, v) for on, off, p, v in notes]
    out['gain_m6'] = [tuple(n) for n in notes]
    for k in WRONG_KS:
        for kind in WRONG_KINDS:
            r = np.random.default_rng(h('wrong', k, kind, len(notes), [n[2] for n in notes][:6]))
            out[f'wrong_{k}_{kind}'] = apply_wrong(notes, key, kind, k, r)[0]
    return out


# ── io helpers ────────────────────────────────────────────────────────────────
def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=1, ensure_ascii=False, default=float) + '\n')


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def write_tsv(path, rows, cols):
    lines = ['\t'.join(cols)]
    for r in rows:
        lines.append('\t'.join(str(r.get(c, '')) for c in cols))
    atomic_text(path, '\n'.join(lines) + '\n')


def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        cols = f.readline().rstrip('\n').split('\t')
        return [dict(zip(cols, ln.rstrip('\n').split('\t'))) for ln in f if ln.strip()]


MANIFEST_COLS = ['split', 'excerpt', 'perf', 'midi', 'font', 'variant', 'label', 'wrong_kind',
                 'wrong_k', 'n_notes', 'n_wrong', 'key', 'oracle_out_ratio', 'path', 't0', 'window']
