"""C: evidence table over 126a/126b/126c per-clip scores (CPU). Rules fixed in doc §4C before results.
Cluster = clip (126a), piece (126b/126c). Bootstrap 400 reps over clusters (weights shared per #clusters)."""
import gzip, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, '/home/kojiek/MeanAudio/scripts/eval')
import common as C
import metric_win_tie_126a as A, metric_win_tie_126b as B, metric_win_tie_126c as CC
def cells_126a():
    data = C.load_126a(); S = C.S_of('126a'); clean = data['clean']; out = []
    for v, per in data.items():
        for sy, rec in per.items():
            if rec['kind'] not in ('tie', 'lose'): continue
            out.append({'cell': f'{v}|{sy}', 'ladder_step': v, 'system': sy, 'kind': rec['kind'],
                        'd': {m: rec['m'][m] - clean[sy]['m'][m] for m in C.NEURAL}})
    return out, S
REP = 400; rng = np.random.default_rng(20261020); _W = {}
def W(k):
    if k not in _W: _W[k] = rng.multinomial(k, np.ones(k) / k, size=REP).astype(np.float32)
    return _W[k]
def stat(d, cl, m0, kind, metric):
    ok = np.isfinite(d); d, cl = d[ok], cl[ok]
    if len(d) < 2: return None
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, d); n = np.bincount(inv).astype(float)
    mean = d.mean(); w = W(len(u)); bm = (w @ s) / (w @ n)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    cm = s / n
    sd = d.std(ddof=1)
    r = C.judge_rates(d, m0)
    exp_neg = kind == 'lose' and metric in C.DIRECTED
    sign = float(np.mean(cm < 0)) if exp_neg else float(np.mean(np.abs(cm) <= m0))
    return dict(n=int(len(d)), ncl=int(len(u)), mean=float(mean), lo=float(lo), hi=float(hi), dz=float(mean / sd) if sd > 0 else 0.0,
                tie=r['tie'], worse=r['worse'], better=r['better'], sign=sign, m0=m0)
def classify(st, kind, metric, top):
    if kind == 'lose' and metric in C.DIRECTED:
        if st['lo'] > 0: return '明確問題', '該判差但分數 CI 顯示上升'
        if top and st['worse'] < 0.2: return '明確問題', '最大幅度判差率 < 20%'
        if st['worse'] >= 0.8 and st['hi'] < 0: return '做得好', ''
        return '爭議', ('CI 不跨零但 |Δ| 小於 m₀' if (st['hi'] < 0 and abs(st['mean']) < st['m0']) else 'CI 跨零或判差率不足')
    if kind in ('tie', 'soft'):
        if st['tie'] >= 0.8: return '做得好', ''
        return '爭議', ('CI 不跨零但 |Δ| 小於 m₀' if (st['lo'] > 0 or st['hi'] < 0) and abs(st['mean']) < st['m0'] else '平手率不足')
    return None, ''
def main():
    rows = []
    c, S = cells_126a(); data = C.load_126a(); ids = data['clean']['real']['ids']
    top = {s: lad[-1] for lad in A.LADDERS.values() for s in [lad[-1]]}
    for ce in c:
        cl = np.array(data['clean'][ce['system']]['ids'])
        for m, d in ce['d'].items():
            st = stat(d, cl, 0.1 * S[m], ce['kind'], m)
            rows.append(('126a', ce['cell'], ce['ladder_step'], ce['system'], ce['kind'], m, st, ce['ladder_step'] in top))
    for tag, root, keyf, lad in (('126b', 'metric_win_tie_126b_20261016', lambda r: (r['block'], r['source'], r['case'], r['kind']), {**B.MIDI_LADDERS, **A.LADDERS}),
                                 ('126c', 'metric_win_tie_126c_20261019', lambda r: (r['case'], r['kind']), CC.LADDERS)):
        R = C.EV / root; S = C.S_of(tag)
        sc, items = C.load_keyed(R, R / 'stimuli/items.json', R / 'mir.json')
        groups = {}
        for r in items:
            if r['base'] and r['kind'] in ('tie', 'soft', 'lose'): groups.setdefault(keyf(r), []).append(r)
        tops = {l[-1] for l in lad.values()}
        for key, rs in groups.items():
            cl = np.array([r['piece'] for r in rs])
            for m in list(C.NEURAL) + list(C.MIR):
                if m not in S: continue
                d = np.array([sc[r['key']].get(m, np.nan) - sc[r['base']].get(m, np.nan) for r in rs])
                st = stat(d, cl, 0.1 * S[m], key[-1], m)
                rows.append((tag, '|'.join(key), key[-2] if tag == '126b' else key[0], key[1] if tag == '126b' else 'musdb', key[-1], m, st, (key[-2] if tag == '126b' else key[0]) in tops))
    out = []
    for ds, cell, step, sysm, kind, m, st, tp in rows:
        if st is None: continue
        cat, why = classify(st, kind, m, tp)
        out.append(dict(ds=ds, cell=cell, step=step, system=sysm, kind=kind, metric=m, top=tp, cat=cat, why=why, **st))
    json.dump(out, gzip.open(HERE / 'evidence_rows.json.gz', 'wt'))
    print(len(out))
main()
