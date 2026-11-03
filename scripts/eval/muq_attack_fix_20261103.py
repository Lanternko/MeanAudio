#!/usr/bin/env python3
"""MuQ-Eval lowpass attack -> input-fix (F0/F1/F2) -> TTM 16 kHz re-eval -> candidates 3/4/5.

Design and pre-registered criteria: docs/experiments/muq_attack_fix_20261103.md
Re-uses stimulus loaders and helpers from muq_lowpass_probe_20261028.py (import only; not modified).

  Part A  fixes on musdb / odaqref / odaqlp / mc522 / MusicEval
  Part B  TTM table (6 systems, 1000 shared filenames, F0; musicgen also F1)
  Part C  dither on silent frames / gain ladder / htdemucs stemsum (126c cache)

Output: ~/nvme_experiment_artifacts/meanaudio/muq_attack_fix_20261103/{per_clip.tsv,summary.json,run.log}
per_clip.tsv is written incrementally and resumed by key on rerun.

  python scripts/eval/muq_attack_fix_20261103.py --part all
  python scripts/eval/muq_attack_fix_20261103.py --part A --limit 5     # smoke
"""
import argparse
import csv
import itertools
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import muq_lowpass_probe_20261028 as base  # noqa: E402
from newmetrics_muqeval import score_muqeval  # noqa: E402

OUT = Path.home() / 'nvme_experiment_artifacts/meanaudio/muq_attack_fix_20261103'
TTM = Path('/home/kojiek/eval_output_nvme/ttm_metrics_115_20261009')
STEMS_CACHE = Path('/home/kojiek/eval_output_nvme/metric_win_tie_126c_20261019/stimuli/cache')
ME_TSV = base.B2 / 'manifests/me.tsv'
SYS = ('meanaudio_cfg0', 'meanaudio_cfg3_neg', 'musiccaps_real',
       'musicgen_medium', 'musicldm', 'musicldm_fid8')
FIXES = ('F0', 'F1', 'F2')
VARIANTS = ('A1', 'A3a')
HEADS = ('MI', 'TA')
SCORERS = tuple(f'{v}_{h}' for v in VARIANTS for h in HEADS)
ATK = ('clean', 'lp4k', 'lp5k', 'lp6k', 'lp8k', 'sr16k')
GAINS = [g for g in range(-30, 7, 3) if g != 0]
DITHERS = (-90, -80, -70)
STEMS = ('vocals', 'drums', 'bass', 'other')
FIELDS = ['part', 'set', 'sys', 'item', 'fix', 'transform', 'sr_in', 'sr_proc',
          'A1_MI', 'A1_TA', 'A3a_MI', 'A3a_TA', 'human', 'extra']
KEYF = FIELDS[:7]


# ── input fixes ──
def prep(y, sr, fix):
    """mono -> F0 identity / F1 soxr 16 kHz / F2 Butterworth-4k lowpass."""
    m = np.asarray(y, dtype=np.float32)
    if m.ndim > 1:
        m = m.mean(axis=1)
    if fix == 'F0':
        return m, int(sr)
    if fix == 'F1':
        return soxr.resample(m, sr, 16000, quality='HQ').astype(np.float32), 16000
    if fix == 'F2':
        return base.lp(m, sr, 4000.0), int(sr)
    raise ValueError(fix)


def score(y, sr):
    out = {}
    for v in VARIANTS:
        s = score_muqeval(y, sr, variant=v)
        for h in HEADS:
            out[f'{v}_{h}'] = s[h]
    return out


class Writer:
    def __init__(self, path):
        self.path = path
        self.rows, self.seen = [], set()
        if path.exists():
            with open(path, newline='') as f:
                for r in csv.DictReader(f, delimiter='\t'):
                    self.rows.append(r)
                    self.seen.add(self.key(r))
        self.f = open(path, 'a', newline='')
        self.w = csv.DictWriter(self.f, FIELDS, delimiter='\t')
        if not self.rows:
            self.w.writeheader()
        self.n_new = 0

    @staticmethod
    def key(r):
        return tuple(str(r[k]) for k in KEYF)

    def emit(self, key_vals, sc, sr_in, sr_proc, human='', extra=''):
        row = dict(zip(KEYF, key_vals))
        row.update({k: round(float(v), 6) for k, v in sc.items()})
        row.update({'sr_in': sr_in, 'sr_proc': sr_proc, 'human': human, 'extra': extra})
        if self.key(row) in self.seen:
            return False
        self.seen.add(self.key(row))
        self.rows.append(row)
        self.w.writerow(row)
        self.f.flush()
        self.n_new += 1
        return True

    def close(self):
        self.f.close()


def islice(gen, n):
    return gen if not n else itertools.islice(gen, n)


# ── Part A ──
def me_rows():
    for r in csv.DictReader(open(ME_TSV), delimiter='\t'):
        e = json.loads(r['extra'])
        if e.get('split') not in ('dev', 'test'):
            continue
        yield r['item_id'], r['src'], float(e['human_OVL']), e.get('split'), float(e.get('dur', 0))


def part_a(W, lim):
    for gen in (base.musdb, base.odaqref, base.odaqlp, base.mc522):
        name = gen.__name__
        for st, item, x, sr, tfs, meta in islice(gen(), lim):
            if name in ('musdb', 'odaqref'):
                tfs_use = ATK
            elif name == 'odaqlp':
                tfs_use = tfs
            else:
                tfs_use = ('clean',)
            for tf in tfs_use:
                y = x if name == 'odaqlp' else base.transform(x, sr, tf)
                if y is None:
                    continue
                for f in FIXES:
                    yy, ss = prep(y, sr, f)
                    W.emit(('A', st, '', item, f, tf), score(yy, ss), sr, ss,
                           human=meta.get('human_PQ', ''))
            print(f'A {st} {item} rows={len(W.rows)}', flush=True)
    for item, src, ovl, split, dur in islice(me_rows(), lim):
        x, sr = sf.read(src, dtype='float32', always_2d=True)
        for f in FIXES:
            yy, ss = prep(x, sr, f)
            W.emit(('A', 'me', '', item, f, 'clean'), score(yy, ss), sr, ss,
                   human=ovl, extra=split)
        print(f'A me {item} rows={len(W.rows)}', flush=True)


# ── Part B ──
def common_ttm():
    sets = []
    for s in SYS:
        sets.append({p.name for p in (TTM / s / 'lvl30/audio').glob('*.flac')})
    return sorted(set.intersection(*sets))


def part_b(W, lim):
    common = common_ttm()
    n = lim or 1000
    idx = np.random.default_rng(0).choice(len(common), size=n, replace=False)
    picks = sorted(common[i] for i in idx)
    (OUT / 'part_b_selection.json').write_text(json.dumps({'n_common': len(common), 'n_pick': n,
                                                           'seed': 0, 'files': picks}, indent=1))
    for s in SYS:
        fixes = ('F0', 'F1') if s == 'musicgen_medium' else ('F0',)
        for fn in picks:
            x, sr = sf.read(TTM / s / 'lvl30/audio' / fn, dtype='float32', always_2d=True)
            for f in fixes:
                yy, ss = prep(x, sr, f)
                W.emit(('B', 'ttm', s, fn, f, ''), score(yy, ss), sr, ss)
            print(f'B {s} {fn} rows={len(W.rows)}', flush=True)


# ── Part C ──
def clean_index(rows):
    """(set, item) -> {scorer: clean F0 score} from Part A rows."""
    out = {}
    for r in rows:
        if r['part'] == 'A' and r['fix'] == 'F0' and r['transform'] == 'clean':
            out[(r['set'], r['item'])] = {c: float(r[c]) for c in SCORERS}
    return out


def dither(m, sr, level_db, seed):
    n = max(1, int(0.02 * sr))
    rng = np.random.default_rng(seed)
    scale = 10.0 ** (level_db / 20.0)
    y = m.copy()
    for i in range(len(m) // n):
        seg = y[i * n:(i + 1) * n]
        if np.sqrt(np.mean(seg.astype(np.float64) ** 2)) < 10.0 ** (-50 / 20.0):
            seg += rng.normal(0.0, scale, size=len(seg)).astype(np.float32)
    return y


def part_c(W, lim):
    for gen in (base.musdb, base.odaqref):
        for st, item, x, sr, tfs, meta in islice(gen(), lim):
            m, _ = prep(x, sr, 'F0')
            seed = zlib.crc32(item.encode())
            for lv in DITHERS:
                y = dither(m, sr, lv, seed)
                W.emit(('C', st, '', item, 'F0', f'dither{lv}'), score(y, sr), sr, sr)
            for g in GAINS:
                y = (m * 10.0 ** (g / 20.0)).astype(np.float32)
                W.emit(('C', st, '', item, 'F0', f'gain{g:+d}'), score(y, sr), sr, sr)
            print(f'C {st} {item} rows={len(W.rows)}', flush=True)
    if lim:
        wins = sorted(STEMS_CACHE.glob('*_dmx.npz'))[:lim]
    else:
        wins = sorted(STEMS_CACHE.glob('*_dmx.npz'))
    for p in wins:
        win = p.name[:-len('_dmx.npz')]
        dmx = np.load(p)
        mixnpz = np.load(STEMS_CACHE / f'{win}.npz')
        mix = np.asarray(mixnpz['mixture'])
        dmx_sum = sum(np.asarray(dmx[k]) for k in STEMS)
        true_sum = sum(np.asarray(mixnpz[s]) for s in STEMS)
        for tf, y in (('mixture', mix), ('true_stemsum', true_sum), ('dmx_resum', dmx_sum)):
            yy, ss = prep(y, 16000, 'F0')
            W.emit(('C', 'musdb_stems', '', win, 'F0', tf), score(yy, ss), 16000, ss)
        print(f'C stems {win} rows={len(W.rows)}', flush=True)


# ── summaries ──
def mean_ci(vals):
    d = np.asarray(vals, float)
    return {'n': int(len(d)), 'mean': float(d.mean()), 'ci95': base.boot(d),
            'n_up': int((d > 0.05).sum()), 'n_down': int((d < -0.05).sum()),
            'max_abs': float(np.max(np.abs(d)))}


def a_summary(rows):
    S = {}
    idx = {}
    for r in rows:
        if r['part'] != 'A':
            continue
        idx[(r['set'], r['item'], r['fix'], r['transform'])] = r
    for st in ('musdb', 'odaqref'):
        items = sorted({k[1] for k in idx if k[0] == st})
        for f in FIXES:
            for c in SCORERS:
                clean = [(i, float(idx[(st, i, f, 'clean')][c])) for i in items
                         if (st, i, f, 'clean') in idx]
                S[f'A|attack|{st}|{f}|clean_mean|{c}'] = mean_ci([v for _, v in clean])
                for tf in ATK[1:]:
                    d = [float(idx[(st, i, f, tf)][c]) - v for i, v in clean
                         if (st, i, f, tf) in idx]
                    if d:
                        S[f'A|attack|{st}|{f}|{tf}|{c}'] = mean_ci(d)
    # ODAQ LP human anchor
    mus = {}
    for r in csv.DictReader(open(base.ODAQ / 'ODAQ_results.csv')):
        if r['method'] == 'LP':
            mus.setdefault((r['item'], r['process']), []).append(float(r['score']))
    for f in FIXES:
        for c in SCORERS:
            pts = []
            for (st, item, fx, tf), r in idx.items():
                if st == 'odaqlp' and fx == f and (item, tf) in mus:
                    pts.append((item, tf, float(np.mean(mus[(item, tf)])), float(r[c])))
            if not pts:
                continue
            S[f'A|odaqlp|{f}|muq_by_process|{c}'] = {
                p: float(np.mean([s for _, q, _, s in pts if q == p]))
                for p in sorted({q for _, q, _, _ in pts})}
            S[f'A|odaqlp|{f}|mushra_by_process'] = {
                p: float(np.mean([m for _, q, m, _ in pts if q == p]))
                for p in sorted({q for _, q, _, _ in pts})}
            S[f'A|odaqlp|{f}|rho_pooled|{c}'] = base.spearman(
                [m for *_, m, _ in pts], [s for *_, s in pts])
            S[f'A|odaqlp|{f}|rho_within_trial|{c}'] = {
                tr: base.spearman([m for t2, _, m, _ in pts if t2 == tr],
                                  [s for t2, _, _, s in pts if t2 == tr])
                for tr in sorted({t2 for t2, *_ in pts})}
            byp = S[f'A|odaqlp|{f}|muq_by_process|{c}']
            S[f'A|odaqlp|{f}|lp50_minus_ref|{c}'] = byp.get('LP50', np.nan) - byp.get('reference', np.nan)
            S[f'A|odaqlp|{f}|lp70_minus_ref|{c}'] = byp.get('LP70', np.nan) - byp.get('reference', np.nan)
    # mc522 validity
    by = {}
    for r in rows:
        if r['part'] == 'A' and r['set'] == 'mc522':
            by.setdefault(r['item'], {})[(r['fix'], r['transform'])] = r
    for f in FIXES:
        hp = [float(t[(f, 'clean')]['human']) for t in by.values() if (f, 'clean') in t]
        ids = [i for i, t in by.items() if (f, 'clean') in t]
        for c in SCORERS:
            S[f'A|mc522|{f}|rho_humanPQ|{c}'] = base.spearman(
                hp, [float(by[i][(f, 'clean')][c]) for i in ids])
    # MusicEval validity (dev+test) and all (incl. train) note
    me = {}
    for r in rows:
        if r['part'] == 'A' and r['set'] == 'me':
            me.setdefault(r['item'], {})[(r['fix'], r['transform'])] = r
    for f in FIXES:
        ids = [i for i, t in me.items() if (f, 'clean') in t]
        for c in SCORERS:
            S[f'A|me|{f}|rho_humanOVL|{c}'] = base.spearman(
                [float(me[i][(f, 'clean')]['human']) for i in ids],
                [float(me[i][(f, 'clean')][c]) for i in ids])
        splits = sorted({me[i][(f, 'clean')]['extra'] for i in ids})
        for sp in splits:
            sub = [i for i in ids if me[i][(f, 'clean')]['extra'] == sp]
            for c in SCORERS:
                S[f'A|me|{f}|rho_humanOVL_{sp}|{c}'] = base.spearman(
                    [float(me[i][(f, 'clean')]['human']) for i in sub],
                    [float(me[i][(f, 'clean')][c]) for i in sub])
    return S


def paired_boot(a, b, n=2000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {'n': int(len(d)), 'mean': float(d.mean()),
            'ci95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def b_summary(rows):
    S = {}
    idx = {}
    for r in rows:
        if r['part'] == 'B':
            idx[(r['sys'], r['item'], r['fix'])] = r
    items = sorted({k[1] for k in idx})
    for s in SYS:
        fixes = ('F0', 'F1') if s == 'musicgen_medium' else ('F0',)
        for f in fixes:
            for c in SCORERS:
                v = [float(idx[(s, i, f)][c]) for i in items if (s, i, f) in idx]
                S[f'B|mean|{s}|{f}|{c}'] = mean_ci(v)
    def rank(key):
        vals = {}
        for s in SYS:
            f = key(s)
            v = [float(idx[(s, i, f)]['A1_MI']) for i in items if (s, i, f) in idx]
            vals[s] = float(np.mean(v))
        return dict(sorted(vals.items(), key=lambda kv: -kv[1]))
    S['B|rank|A1_MI|F0'] = rank(lambda s: 'F0')
    S['B|rank|A1_MI|musicgen_F1'] = rank(lambda s: 'F1' if s == 'musicgen_medium' else 'F0')
    for c in ('A3a_MI',):
        vals = {}
        for s in SYS:
            f = 'F0'
            v = [float(idx[(s, i, f)][c]) for i in items if (s, i, f) in idx]
            vals[s] = float(np.mean(v))
        S[f'B|rank|{c}|F0'] = dict(sorted(vals.items(), key=lambda kv: -kv[1]))
        vals = {}
        for s in SYS:
            f = 'F1' if s == 'musicgen_medium' else 'F0'
            v = [float(idx[(s, i, f)][c]) for i in items if (s, i, f) in idx]
            vals[s] = float(np.mean(v))
        S[f'B|rank|{c}|musicgen_F1'] = dict(sorted(vals.items(), key=lambda kv: -kv[1]))
    # musicgen F1 - F0 per item
    for c in SCORERS:
        d = [float(idx[('musicgen_medium', i, 'F1')][c]) - float(idx[('musicgen_medium', i, 'F0')][c])
             for i in items if ('musicgen_medium', i, 'F1') in idx and ('musicgen_medium', i, 'F0') in idx]
        S[f'B|musicgen_delta_F1_F0|{c}'] = mean_ci(d)
    # paired vs meanaudio_cfg3_neg
    for c in SCORERS:
        for f in ('F0', 'F1'):
            mg = [(i, float(idx[('musicgen_medium', i, f)][c]), float(idx[('meanaudio_cfg3_neg', i, 'F0')][c]))
                  for i in items if ('musicgen_medium', i, f) in idx and ('meanaudio_cfg3_neg', i, 'F0') in idx]
            if mg:
                S[f'B|paired|{f}|musicgen_minus_cfg3neg|{c}'] = paired_boot(
                    [a for _, a, _ in mg], [b for _, _, b in mg])
    return S


def c_summary(rows):
    S = {}
    clean = clean_index(rows)
    by = {}
    for r in rows:
        if r['part'] == 'C':
            by.setdefault((r['set'], r['item'], r['transform']), r)
    for st in ('musdb', 'odaqref'):
        items = sorted({k[1] for k in by if k[0] == st})
        for tf in [f'dither{lv}' for lv in DITHERS] + [f'gain{g:+d}' for g in GAINS]:
            for c in SCORERS:
                d = [float(by[(st, i, tf)][c]) - clean[(st, i)][c]
                     for i in items if (st, i, tf) in by and (st, i) in clean]
                if d:
                    S[f'C|{st}|{tf}|{c}'] = mean_ci(d)
    items = sorted({k[1] for k in by if k[0] == 'musdb_stems'})
    for tf in ('true_stemsum', 'dmx_resum'):
        for c in SCORERS:
            d = [float(by[('musdb_stems', i, tf)][c]) - float(by[('musdb_stems', i, 'mixture')][c])
                 for i in items if ('musdb_stems', i, tf) in by and ('musdb_stems', i, 'mixture') in by]
            if d:
                S[f'C|musdb_stems|{tf}|{c}'] = mean_ci(d)
    for c in SCORERS:
        v = [float(by[('musdb_stems', i, 'mixture')][c]) for i in items
             if ('musdb_stems', i, 'mixture') in by]
        S[f'C|musdb_stems|mixture_mean|{c}'] = mean_ci(v)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=('A', 'B', 'C', 'all'), default='all')
    ap.add_argument('--limit', type=int, default=0, help='items per set (smoke); 0 = full')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    W = Writer(OUT / 'per_clip.tsv')
    if a.part in ('A', 'all'):
        part_a(W, a.limit)
    if a.part in ('B', 'all'):
        part_b(W, a.limit)
    if a.part in ('C', 'all'):
        part_c(W, a.limit)
    W.close()
    S = {'meta': {'limit': a.limit, 'part': a.part, 'n_rows': len(W.rows),
                  'new_rows': W.n_new, 'fixes': FIXES, 'sys': SYS}}
    S['A'] = a_summary(W.rows)
    S['B'] = b_summary(W.rows)
    S['C'] = c_summary(W.rows)
    json.dump(S, open(OUT / 'summary.json', 'w'), indent=1)
    for k, v in S['A'].items():
        if isinstance(v, dict) and 'mean' in v:
            print(f"{k:52s} n={v['n']:4d} m={v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")
        else:
            print(k, v)
    print('rows', len(W.rows), 'new', W.n_new)


if __name__ == '__main__':
    main()
