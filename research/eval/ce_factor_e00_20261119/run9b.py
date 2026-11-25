"""Round 9b (post-hoc, AES only): X (round-9 best) with its GM 39 Synth Bass 2 swapped for real acoustic/electric basses,
to check the 8.13 doesn't hinge on a synth (sawtooth-like) bass. 12 held-out seeds 1000-1011, paired."""
import sys, json, copy
import numpy as np
import multiprocessing as mp
import run9 as R
X = json.load(open(R.M.OUT / 'scores9_final.json'))[0]['cfg']
V = {'X (bass 39 synth)': {}}
for b in (32, 33, 34, 35, 38):
    for db in (0.0, -6.0):
        V[f'bass {b} @{db:+.0f}'] = {'bass': b, 'bass_db': db}
V['no bass'] = {'bass': None}
V['no layer'] = {'layer2': None}
V['no bass, no layer'] = {'bass': None, 'layer2': None}
SEEDS = range(1000, 1012)
if __name__ == '__main__':
    cfgs = {k: {**copy.deepcopy(X), **v} for k, v in V.items()}
    items = [(cfgs[k], s) for k in V for s in SEEDS]
    with mp.get_context('fork').Pool(12) as pool:
        res = R.Scorer(pool).score(items)
    out = [{'variant': k, 'seed': s, **r} for (k, s), r in zip([(k, s) for k in V for s in SEEDS], res)]
    (R.M.OUT / 'scores9b.json').write_text(json.dumps(out, indent=1))
    base = np.array([o['CE'] for o in out if o['variant'] == 'X (bass 39 synth)'])
    rng = np.random.default_rng(0)
    for k in V:
        ce = np.array([o['CE'] for o in out if o['variant'] == k]); pq = np.mean([o['PQ'] for o in out if o['variant'] == k])
        d = ce - base; bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        print(f'{k:22s} CE {ce.mean():.3f}  d {d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]  PQ {pq:.2f}', flush=True)
