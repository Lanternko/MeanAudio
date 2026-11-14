"""Analyze the 6 kHz low-pass listening test per the pre-registration (§4 of
docs/experiments/human_listen_lp6k_20261113.md). Reads ratings dump + key + predictions;
prints/writes analysis.json next to the key (not committed). Excluded raters are still
reported descriptively, flagged."""
import json, glob, math, random, sys
from pathlib import Path
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/human_listen_lp6k_20261113')
key = json.load(open(OUT / 'key.json'))
trials = json.load(open(OUT / 'page_trials.json'))['trials']
pred = json.load(open(OUT / 'predictions.json'))
pid = {(r['base'], r['id']): r for r in pred['per_id']}
METRICS = ['A1_MI', 'A3a_MI', 'PQ', 'CE', 'PAM', 'CMI_q']

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def code(t, ans):
    """map an answer to orig/lp/same for a trial"""
    if ans == 'same': return 'same'
    f = t['a'] if ans == 'A' else t['b']
    return 'orig' if key[f]['version'] == 'base' else 'lp'

raters = []
for p in sorted(glob.glob(str(OUT / 'ratings_dump/ratings/*.json'))):
    d = json.load(open(p)); ans = d['answers']
    rows = []
    for t in trials:
        a = ans.get(f"t{t['trial']}") or ans.get(str(t['trial']))
        if not a: continue
        ka, kb = key[t['a']], key[t['b']]
        rows.append(dict(trial=t['trial'], kind=ka['kind'], base=ka['base'], id=ka['id'],
                         orig_side='A' if ka['version'] == 'base' else 'B',
                         pq=code(t, a['pq']), ovl=code(t, a['ovl']), pq_raw=a['pq'], ovl_raw=a['ovl']))
    # repeat consistency: repeat trial shares (base,id) with a main trial
    main = {(r['base'], r['id']): r for r in rows if r['kind'] == 'main'}
    rep = [(main[(r['base'], r['id'])], r) for r in rows if r['kind'] == 'repeat' and (r['base'], r['id']) in main]
    incons = sum(m['pq'] != r['pq'] for m, r in rep)
    excl = []
    if d.get('familiar') == 'yes': excl.append('familiar=yes')
    if incons >= 2: excl.append(f'repeat_inconsistent {incons}/3')
    raters.append(dict(uid=Path(p).stem, n=len(rows), familiar=d.get('familiar'), excluded=excl, rows=rows,
                       repeat_pairs=[dict(base=m['base'], main=m['pq'], rep=r['pq']) for m, r in rep], repeat_incons=incons))

def summ(rows, q):
    m = [r for r in rows if r['kind'] == 'main']; c = [r for r in rows if r['kind'] == 'catch']
    side = [r for r in m if r[q] != 'same']; ko = sum(r[q] == 'orig' for r in side)
    return dict(n_main=len(m), n_side=len(side), k_orig=ko, p_orig=ko / len(side) if side else None,
                wilson=wilson(ko, len(side)), same_main=sum(r[q] == 'same' for r in m) / len(m),
                same_catch=sum(r[q] == 'same' for r in c) / len(c) if c else None,
                catch_raw=[r[q + '_raw'] for r in c],
                per_base={b: dict(orig=sum(r[q] == 'orig' for r in m if r['base'] == b), lp=sum(r[q] == 'lp' for r in m if r['base'] == b),
                                  same=sum(r[q] == 'same' for r in m if r['base'] == b)) for b in sorted({r['base'] for r in m})})

def agree(rows, q):
    out = {}
    side = [r for r in rows if r['kind'] == 'main' and r[q] != 'same']
    for c in METRICS:
        k = sum((pid[(r['base'], r['id'])][c] > 0) == (r[q] == 'lp') for r in side)
        out[c] = dict(k=k, n=len(side), rate=k / len(side) if side else None, wilson=wilson(k, len(side)))
    return out

valid = [r for r in raters if not r['excluded']]
res = dict(n_raters=len(raters), n_valid=len(valid), formal=len(valid) >= 6,
           raters=[dict(uid=r['uid'][:6], familiar=r['familiar'], excluded=r['excluded'], repeat_pairs=r['repeat_pairs'],
                        pq=summ(r['rows'], 'pq'), ovl=summ(r['rows'], 'ovl'),
                        agree_pq=agree(r['rows'], 'pq'), agree_ovl=agree(r['rows'], 'ovl')) for r in raters])
json.dump(res, open(OUT / 'analysis.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
