"""Round 10f (post-hoc, AES only): cumulative ladder, demo order v3 — bass first, short notes, then the rest of the
accompaniment, change the melody only in the last three steps. Timbre swap + violin layer merged into one card
(10c: each alone hurts, together +0.49). Same base, same end point B, same 12 seeds 2000-2011 as 10c."""
import json, copy
import numpy as np
import multiprocessing as mp
import run10 as R
import run10c as C
B, base = C.B, C.base
order = [('加低音線', dict(bass=B['bass'], bass_db=B['bass_db'])),
         ('短音、每秒 3.7 顆', dict(rhythm=B['rhythm'], rate=B['rate'], legato=B['legato'])),
         ('換主音色＋疊小提琴', dict(bank=B['bank'], prog=B['prog'], layers=B['layers'])),
         ('小聲的鼓', dict(drums='rock')), ('鼓型換 trap', dict(drums=B['drums'])),
         ('力度隨機', dict(vel_mean=B['vel_mean'], vel_sd=B['vel_sd'])), ('小聲和弦墊底', dict(pad=B['pad'])),
         ('短殘響', dict(reverb=B['reverb'])), ('音高種類多', dict(pdist=B['pdist'])),
         ('大跳、音域寬', dict(center=B['center'], spread=B['spread'])), ('小聲副旋律', dict(counter=B['counter']))]
steps = [('瑪麗原曲（鋼琴）', base)]
c = copy.deepcopy(base)
for nm, kw in order:
    c = dict(copy.deepcopy(c), **copy.deepcopy(kw)); steps.append((nm, c))
assert json.dumps(steps[-1][1], sort_keys=True) == json.dumps(B, sort_keys=True), 'ladder end != B'
SEEDS = C.SEEDS; n = len(SEEDS)
if __name__ == '__main__':
    with mp.get_context('fork').Pool(12) as pool:
        res = R.Scorer10(pool).score([(c, s) for _, c in steps for s in SEEDS])
    ce = [np.array([res[i * n + j]['CE'] for j in range(n)]) for i in range(len(steps))]
    rng = np.random.default_rng(0); out = []
    for i, (nm, c) in enumerate(steps):
        r = dict(name=nm, mean_CE=float(ce[i].mean()), max_CE=float(ce[i].max()), per_seed=ce[i].tolist(), cfg=c)
        if i:
            d = ce[i] - ce[i - 1]; bs = [rng.choice(d, n).mean() for _ in range(2000)]
            r.update(dCE=float(d.mean()), dCE_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        out.append(r); print(f"{nm:14s} {r['mean_CE']:.3f} {r.get('dCE', 0):+.3f} {r.get('dCE_ci', '')}", flush=True)
    (R.M.OUT / 'scores10f_ladder.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
