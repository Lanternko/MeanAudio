"""Round 10b (post-hoc, AES only): round-10 best configs with every GM >= 80 program and synth basses 38/39 swapped
for acoustic programs (all pitched programs GM 0-79, no synth bass). 12 held-out seeds 2000-2011, paired vs original."""
import json, copy
import numpy as np
import multiprocessing as mp
import run10 as R
F = json.load(open(R.M.OUT / 'scores10_final.json'))
RV = copy.deepcopy([r for r in F if r['cell'] == 'reverb' and r['rank_in_cell'] == 1][0]['cfg'])
NR = copy.deepcopy([r for r in F if r['cell'] == 'noreverb' and r['rank_in_cell'] == 1][0]['cfg'])


def sw(c, **kw):
    c = copy.deepcopy(c)
    for k, v in kw.items():
        if k in ('pad', 'counter'):
            c[k] = None if v is None else {**c[k], 'prog': v}
        else:
            c[k] = v
    return c


V = {'RV orig': RV}
for main in (46, 72, 11, 0):
    for bass in (35, 32):
        for pad in (52, 19, 48):
            for ctr in (11, 43):
                V[f'RV ac main{main} bass{bass} pad{pad} ctr{ctr}'] = sw(RV, prog=main, bass=bass, pad=pad, counter=ctr)
V['NR orig'] = NR
for bass in (35, 32, 33):
    V[f'NR ac bass{bass}'] = sw(NR, bass=bass)
SEEDS = range(2000, 2012)
if __name__ == '__main__':
    keys = [(k, s) for k in V for s in SEEDS]
    with mp.get_context('fork').Pool(12) as pool:
        res = R.Scorer10(pool).score([(V[k], s) for k, s in keys])
    out = [{'variant': k, 'seed': s, **r} for (k, s), r in zip(keys, res)]
    (R.M.OUT / 'scores10b.json').write_text(json.dumps({'out': out, 'cfgs': V}, indent=1))
    rng = np.random.default_rng(0)
    rows = []
    for k in V:
        ref = 'RV orig' if k.startswith('RV') else 'NR orig'
        ce = np.array([o['CE'] for o in out if o['variant'] == k]); b = np.array([o['CE'] for o in out if o['variant'] == ref])
        d = ce - b; bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        rows.append((ce.mean(), k, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), ce.max(), np.mean([o['PQ'] for o in out if o['variant'] == k])))
    for r in sorted(rows, reverse=True)[:15] + [r for r in rows if 'orig' in r[1] or r[1].startswith('NR')]:
        print(f'{r[1]:40s} CE {r[0]:.3f} d {r[2]:+.3f} [{r[3]:+.3f},{r[4]:+.3f}] max {r[5]:.2f} PQ {r[6]:.2f}', flush=True)
