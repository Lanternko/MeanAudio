import json, numpy as np
R = json.load(open('scores3.json'))
W = sorted({r['width'] for r in R}); T = sorted({r['timbre'] for r in R})
rng = np.random.default_rng(0)
def cell(t, rule, w): return np.array([r['CE'] for r in R if r['timbre'] == t and r['rule'] == rule and r['width'] == w])
print('mean CE (12 tonic x 3 seed), jump / walk')
print('timbre'.ljust(9) + ''.join(f'{w:>12}' for w in W))
for t in T:
    print(t.ljust(9) + ''.join(f'{cell(t,"j",w).mean():6.2f}/{cell(t,"w",w).mean():<5.2f}' for w in W))
for rule in 'jw':
    m = [np.mean([cell(t, rule, w).mean() for t in T]) for w in W]
    print('ALL', rule, ' '.join(f'{x:.2f}' for x in m))
# paired diffs (per timbre x tonic x seed): jump vs walk at same range; width 12 -> 48
def key(r): return (r['timbre'], r['midi'], r['seed'])
def dif(a, b, filt=lambda r: True):
    A = {key(r): r['CE'] for r in R if a(r) and filt(r)}; B = {key(r): r['CE'] for r in R if b(r) and filt(r)}
    d = np.array([A[k] - B[k] for k in A if k in B]); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (d > 0).mean()
for w in (7, 12, 24, 48):
    print(f'jump-walk w{w}: %.3f [%.3f, %.3f] pos %.2f' % dif(lambda r: r['rule'] == 'j' and r['width'] == w, lambda r: r['rule'] == 'w' and r['width'] == w))
for a, b in ((2, 0), (12, 2), (24, 12), (48, 24)):
    for rule in 'jw':
        print(f'{rule} w{a}-w{b}: %.3f [%.3f, %.3f] pos %.2f' % dif(lambda r: r['rule'] == rule and r['width'] == a, lambda r: r['rule'] == rule and r['width'] == b))
# within-width: does mean |interval| predict CE?  (rank corr pooled over width>=7 controlling timbre/width by demeaning)
X = [r for r in R if r['width'] >= 7]
g = {}
for r in X: g.setdefault((r['timbre'], r['width']), []).append(r)
xa, ya, xs, ys = [], [], [], []
for rs in g.values():
    ce = np.array([r['CE'] for r in rs]); ce -= ce.mean()
    ia = np.array([r['mean_abs_int'] for r in rs]); ia -= ia.mean()
    sd = np.array([r['pitch_sd'] for r in rs]); sd -= sd.mean()
    xa += list(ia); ya += list(ce); xs += list(sd)
print('within cell corr CE~mean|int| %.3f, CE~pitch_sd %.3f' % (np.corrcoef(xa, ya)[0, 1], np.corrcoef(xs, ya)[0, 1]))
json.dump({rule: {t: {w: round(float(cell(t, rule, w).mean()), 3) for w in W} for t in T} for rule in 'jw'}, open('/tmp/claude-1005/-home-kojiek-MeanAudio/5863b2db-6efb-478f-b52d-c0d162e128a3/scratchpad/w3.json', 'w'))
