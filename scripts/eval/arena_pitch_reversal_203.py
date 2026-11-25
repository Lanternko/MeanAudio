#!/usr/bin/env python
"""203: 移調／變速在 Music Arena 反向的原因（MuQ-only probe）。設計：docs/experiments/arena_pitch_reversal_203_20261125.md。

  python arena_pitch_reversal_203.py [--limit N] [--out DIR]     # 生成刺激＋MuQ A1/A3a 評分（續跑）
  python arena_pitch_reversal_203.py --analyze [--out DIR] [--summary PATH]   # CPU 分析

只 import arena_consequence_200 的函式（不改該檔）；不留音檔。
"""
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena_consequence_200 as A200  # noqa: E402

REPO = HERE.parent.parent
OUT = Path.home() / 'eval_output_nvme/arena_pitch_reversal_203'
SUMMARY = REPO / 'docs/experiments/results/arena_pitch_reversal_203_summary.json'
SEED = 20261125
NBOOT = 5000
PER_SYSTEM = 50
SR16 = 16000
LP_HZ = 7127.0
CONDS = ['d16_F0', 'd16_rb_id', 'd16_ps_m2', 'd16_ts_090', 'd16_lp7127', 'ps_m2_to16', 'ts_090_to16']
COLS = ['id', 'system', 'sr', 'cond', 'lufs_out', 'hf_db', 'A1_MI', 'A3a_MI']
PS = 2 ** (-2 / 12)


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


# ═════════ subset ═════════
def f0_ids_200():
    d = {}
    for p in sorted(glob.glob(str(A200.OUT / 'per_clip' / 'F0_*.tsv'))):
        d.update(A200.tsv_read(Path(p)))
    return d


def build_subset():
    f0 = f0_ids_200()
    csv.field_size_limit(10**9)
    with open(A200.MANIFEST, newline='') as f:
        man = {r['item_id']: r for r in csv.DictReader(f, delimiter='\t')}
    bysys = {}
    for i in f0:
        if i in man:
            bysys.setdefault(man[i]['group'], []).append(i)
    rows = []
    for sy in sorted(bysys):
        ids = sorted(bysys[sy])
        if len(ids) > PER_SYSTEM:
            seed = int(hashlib.sha256(f'20261125:{sy}'.encode()).hexdigest()[:8], 16)
            ids = sorted(random.Random(seed).sample(ids, PER_SYSTEM))
        for i in ids:
            rows.append(dict(id=i, system=sy, src=man[i]['src'], sr=int(f0[i]['sr'])))
    return rows


# ═════════ stimuli ═════════
def resample16(x, sr):
    if sr == SR16:
        return np.asarray(x, np.float32)
    import soxr
    return soxr.resample(np.asarray(x, np.float32), sr, SR16, quality='HQ').astype(np.float32)


def hf_db(w, sr):
    w = np.asarray(w, np.float64)
    P = np.abs(np.fft.rfft(w)) ** 2
    f = np.fft.rfftfreq(len(w), 1.0 / sr)
    tot = float(P.sum())
    hf = float(P[(f >= LP_HZ) & (f <= 8000.0)].sum())
    if tot <= 0 or hf <= 0:
        return -120.0
    return max(-120.0, 10 * math.log10(hf / tot))


def build_clip(item):
    """-> dict cond -> float32 stimulus at 16 kHz."""
    from muq_lowpass_probe_20261028 import lp
    x, sr = A200.load_center30(item['src'])
    n = len(x)
    l0 = A200.lufs_of(x, sr)
    x16 = resample16(x, sr)
    n16 = len(x16)
    l16 = A200.lufs_of(x16, SR16)
    rl = lambda y: A200.relock(y, l16, SR16)
    fl = A200.fit_len
    rb = A200.rubberband
    out = {'d16_F0': x16}
    out['d16_rb_id'] = rl(fl(rb(x16, SR16, 1.0, 1.0), n16))
    out['d16_ps_m2'] = rl(fl(rb(x16, SR16, 1.0, PS), n16))
    out['d16_ts_090'] = rl(rb(x16, SR16, 0.9, 1.0))
    out['d16_lp7127'] = rl(lp(np.asarray(x16, np.float64), SR16, LP_HZ))
    out['ps_m2_to16'] = rl(resample16(fl(rb(x, sr, 1.0, PS), n), sr))
    out['ts_090_to16'] = rl(resample16(rb(x, sr, 0.9, 1.0), sr))
    return out, l16


# ═════════ scoring ═════════
class MuQ:
    def __init__(self):
        from muq_gen_steer_179_score import muq_hf_cache
        import newmetrics_muqeval as NMQ
        self.NMQ = NMQ
        with muq_hf_cache():
            NMQ._load('A1', 'cuda')
            NMQ._load('A3a', 'cuda')

    def score(self, w):
        return (float(self.NMQ.score_muqeval(w, SR16, 'A1')['MI']),
                float(self.NMQ.score_muqeval(w, SR16, 'A3a')['MI']))


def read_rows(path):
    if not path.is_file():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def run(out, limit):
    out.mkdir(parents=True, exist_ok=True)
    subset = build_subset()
    with open(out / 'subset.tsv', 'w') as f:
        f.write('id\tsystem\tsrc\tsr\n')
        for r in subset:
            f.write(f"{r['id']}\t{r['system']}\t{r['src']}\t{r['sr']}\n")
    if limit:
        subset = subset[:limit]
    pc = out / 'per_clip.tsv'
    done = {(r['id'], r['cond']) for r in read_rows(pc)}
    todo = [it for it in subset if any((it['id'], c) not in done for c in CONDS)]
    log(f'subset {len(subset)} clips; todo {len(todo)}')
    new = not pc.is_file()
    fh = open(pc, 'a')
    if new:
        fh.write('\t'.join(COLS) + '\n')
        fh.flush()
    if not todo:
        return
    scorer = MuQ()
    t0 = time.time()
    pool = ThreadPoolExecutor(4)
    futs = {}
    W = 6
    for k in range(min(W, len(todo))):
        futs[k] = pool.submit(build_clip, todo[k])
    for k, it in enumerate(todo):
        stim, _ = futs.pop(k).result()
        if k + W < len(todo):
            futs[k + W] = pool.submit(build_clip, todo[k + W])
        for c in CONDS:
            if (it['id'], c) in done:
                continue
            w = stim[c]
            if not np.all(np.isfinite(w)):
                raise RuntimeError(f"{it['id']} {c}: non-finite stimulus")
            a1, a3 = scorer.score(w)
            if not (math.isfinite(a1) and math.isfinite(a3)):
                raise RuntimeError(f"{it['id']} {c}: non-finite score")
            fh.write('\t'.join([it['id'], it['system'], str(it['sr']), c, repr(A200.lufs_of(w, SR16)),
                                repr(hf_db(w, SR16)), repr(a1), repr(a3)]) + '\n')
        fh.flush()
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            log(f'{k + 1}/{len(todo)} clips, {time.time() - t0:.0f}s ({(time.time() - t0) / (k + 1):.2f}s/clip)')
    fh.close()
    log(f'done {len(todo)} clips in {time.time() - t0:.1f}s ({(time.time() - t0) / len(todo):.2f}s/clip)')


# ═════════ analysis ═════════
def load_native():
    nat = {}
    for c in ('F0', 'ps_m2', 'ts_090', 'rb_id', 'lp6k'):
        d = {}
        for p in sorted(glob.glob(str(A200.OUT / 'per_clip' / f'{c}_*.tsv'))):
            d.update(A200.tsv_read(Path(p)))
        nat[c] = d
    return nat


def analyze(out, summary):
    rows = read_rows(out / 'per_clip.tsv')
    v = {}
    meta = {}
    hf = {}
    for r in rows:
        v.setdefault(r['cond'], {})[r['id']] = {'A1_MI': float(r['A1_MI']), 'A3a_MI': float(r['A3a_MI'])}
        meta[r['id']] = (r['system'], int(float(r['sr'])))
        hf.setdefault(r['cond'], []).append(float(r['hf_db']))
    ids = sorted(i for i in meta if all(i in v.get(c, {}) for c in CONDS))
    nat = load_native()
    ids = [i for i in ids if all(i in nat[c] for c in nat)]
    n = len(ids)
    rng = np.random.default_rng(SEED)
    bi = rng.integers(0, n, (NBOOT, n))
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    def g(c, s, i):
        if c.startswith('nat_'):
            return nat[c[4:]][i][s]
        return v[c][i][s]

    def diff(a, b, s, sel=None):
        sel = ids if sel is None else sel
        return np.array([g(a, s, i) - g(b, s, i) for i in sel])

    def stat(x):
        if len(x) == 0:
            return None
        bb = np.random.default_rng(SEED).integers(0, len(x), (NBOOT, len(x)))
        return {'n': int(len(x)), 'mean': float(x.mean()), 'ci': pct(x[bb].mean(1))}

    D = {'T': ('d16_ps_m2', 'd16_F0'), 'P': ('ps_m2_to16', 'd16_F0'), 'E': ('d16_ps_m2', 'ps_m2_to16'),
         'L': ('d16_lp7127', 'd16_F0'), 'RB16': ('d16_rb_id', 'd16_F0'),
         'TS16': ('d16_ts_090', 'd16_F0'), 'TSN16': ('ts_090_to16', 'd16_F0'), 'TS_E': ('d16_ts_090', 'ts_090_to16'),
         'DS': ('d16_F0', 'nat_F0'), 'NAT_PS': ('nat_ps_m2', 'nat_F0'), 'NAT_TS': ('nat_ts_090', 'nat_F0')}
    res = {'kind': 'arena_pitch_reversal_203_summary_v1', 'created': time.strftime('%FT%T%z'), 'n_clips': n, 'seed': SEED,
           'nboot': NBOOT, 'out': str(out)}
    res['diffs'] = {s: {k: stat(diff(a, b, s)) for k, (a, b) in D.items()} for s in ('A1_MI', 'A3a_MI')}
    srs = sorted({meta[i][1] for i in ids})
    systems = sorted({meta[i][0] for i in ids})
    res['by_sr'], res['by_system'] = {}, {}
    for s in ('A1_MI', 'A3a_MI'):
        res['by_sr'][s] = {k: {str(sr): stat(diff(*D[k], s, [i for i in ids if meta[i][1] == sr])) for sr in srs}
                           for k in ('T', 'P', 'E', 'L', 'TS16', 'TSN16')}
        bs = {}
        for k in ('T', 'P', 'E', 'L', 'TS16', 'TSN16'):
            per = {sy: (lambda x: {'n': int(len(x)), 'mean': float(x.mean())})(diff(*D[k], s, [i for i in ids if meta[i][0] == sy]))
                   for sy in systems}
            bs[k] = {'per_system': per, 'n_systems_pos': int(sum(p['mean'] > 0 for p in per.values())), 'n_systems': len(systems)}
        res['by_system'][s] = bs
    res['hf_db_mean'] = {c: float(np.mean(hf[c])) for c in CONDS if c in hf}
    res['frac_T_pos'] = {s: float((diff(*D['T'], s) > 0).mean()) for s in ('A1_MI', 'A3a_MI')}
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None
    res['pearson'] = {s: {'T_L': corr(diff(*D['T'], s), diff(*D['L'], s)), 'E_L': corr(diff(*D['E'], s), diff(*D['L'], s))}
                      for s in ('A1_MI', 'A3a_MI')}
    T, E = res['diffs']['A1_MI']['T'], res['diffs']['A1_MI']['E']
    if T['ci'][0] > 0 and E['ci'][0] > 0:
        verdict = '成立'
    elif E['ci'][0] > 0:
        verdict = '部分成立'
    elif E['ci'][1] <= 0 or (T['ci'][1] < 0 and E['ci'][0] <= 0 <= E['ci'][1]):
        verdict = '不成立'
    else:
        verdict = '未定'
    res['H1_verdict_A1'] = verdict
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    Path(summary).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    # markdown
    print(f'n_clips={n}  H1(A1)={verdict}\n')
    print('| diff | A1 mean [95% CI] | A3a mean [95% CI] |\n|---|---|---|')
    for k in D:
        a, b = res['diffs']['A1_MI'][k], res['diffs']['A3a_MI'][k]
        print(f"| {k} = {D[k][0]} - {D[k][1]} | {a['mean']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | "
              f"{b['mean']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] |")
    print('\nhf_db mean:', {k: round(x, 1) for k, x in res['hf_db_mean'].items()})
    print('frac T>0:', res['frac_T_pos'], ' pearson:', res['pearson'])
    for k in ('T', 'P', 'E', 'L', 'TS16', 'TSN16'):
        print(f'\n{k} by sr (A1):', {sr: (round(x['mean'], 3), x['n']) for sr, x in res['by_sr']['A1_MI'][k].items() if x})
        print(f"{k} systems with mean>0 (A1): {res['by_system']['A1_MI'][k]['n_systems_pos']}/{res['by_system']['A1_MI'][k]['n_systems']}")
    print('\nsummary ->', summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--summary', type=Path, default=SUMMARY)
    a = ap.parse_args()
    if a.analyze:
        analyze(a.out, a.summary)
    else:
        run(a.out, a.limit)


if __name__ == '__main__':
    main()
