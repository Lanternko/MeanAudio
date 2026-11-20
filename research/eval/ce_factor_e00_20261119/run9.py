"""Round 9 (post-hoc, AES only): best CE WITHOUT e00 -- every pitched sound is a SoundFont program (M.A.MS.synth), no numpy oscillators.
Stages: 1 timbre screen, 2 random factor sampling, 3 local search (1+lambda ES), 4 held-out check, 5 stacking ladder."""
import sys, json, copy
import numpy as np
import multiprocessing as mp
import run as M          # sets OMP pins, SR/NS/T, M.A.MS.synth, M.C.lock
import run2              # gm_kit, events
import run4              # MEL, Q
SR, NS, T = M.SR, M.NS, M.T
OUT = M.OUT
MEL, Q = run4.MEL, run4.Q
MAJ = {0, 2, 4, 5, 7, 9, 11}
BANKS = ('generaluser', 'fluidr3')
KEYS = ('CE', 'CU', 'PC', 'PQ')

DEFAULT = dict(bank='generaluser', prog=0, rhythm='grid', rate=8.0, legato=0.9, center=66, spread=24, pdist='chrom',
               vel_mean=100, vel_sd=10, poly=1, bass=None, bass_db=-9.0, layer2=None, layer_db=-9.0,
               drums=None, drum_db=-12.0, lp_fc=None)


def tier_of(c):
    progs = [c['prog']] + ([c['bass']] if c['bass'] is not None else []) + ([c['layer2']] if c['layer2'] is not None else [])
    return 'acoustic' if all(p <= 79 for p in progs) else 'any'


def lp(buf, fc):
    f = np.fft.rfftfreq(NS, 1 / SR)
    return np.fft.irfft(np.fft.rfft(buf) / np.sqrt(1 + (f / fc) ** 8), NS)


def draw_pitches(c, n, rng, line=0):
    pd = c['pdist']
    if pd == 'mary':
        return [60 + MEL[i % len(MEL)][0] for i in range(n)]
    lo = int(round(c['center'] - c['spread'] / 2)); hi = int(round(c['center'] + c['spread'] / 2))
    pool = list(range(lo, hi + 1))
    if pd == 'major':
        pool = [p for p in pool if p % 12 in MAJ] or pool
    out, prev = [], None
    for _ in range(n):
        p = int(rng.choice(pool))
        if pd == 'norep' and len(pool) > 1:
            while p == prev:
                p = int(rng.choice(pool))
        out.append(p); prev = p
    return out


def onsets(c, shift=0.0):
    if c['rhythm'] == 'mary':
        t, ons, durs = 0.0, [], []
        for _, d in MEL:
            ons.append(t + shift); durs.append(d * Q * 0.9); t += d * Q
        return [(o, o + dd) for o, dd in zip(ons, durs)]
    per = 1.0 / c['rate']; nl = c['legato'] * per
    ons = np.arange(shift, T - 0.05, per)
    return [(float(o), float(o) + nl) for o in ons]


def synth_slice(notes, bank, prog):
    x = M.A.MS.synth(notes, bank, int(prog), 0)
    i0 = int(M.A.MS.LEAD * SR)
    return np.asarray(x[i0:i0 + NS], np.float64)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def render9(c, seed):
    rng = np.random.default_rng([int(seed), 9])
    lines = []
    nlines = c['poly']
    for ln in range(nlines):
        sh = 0.0 if ln == 0 else (0.5 / c['rate'] if c['rhythm'] == 'grid' else Q / 2)
        tm = onsets(c, sh)
        ps = draw_pitches(c, len(tm), rng, ln)
        vs = np.clip(np.round(rng.normal(c['vel_mean'], c['vel_sd'], len(tm))), 20, 127).astype(int)
        lines += [(a, b, p, int(v)) for (a, b), p, v in zip(tm, ps, vs)]
    main = synth_slice(lines, c['bank'], c['prog'])
    r0 = rms(main) or 0.05
    mix = main.copy()
    if c['bass'] is not None:
        per = 2.0 / c['rate'] if c['rhythm'] == 'grid' else Q * 2
        ons = np.arange(0.0, T - 0.05, per)
        bn = [(float(o), float(o) + c['legato'] * per, int(rng.integers(int(c['center']) - 24, int(c['center']) - 12 + 1)),
               int(np.clip(round(rng.normal(c['vel_mean'], c['vel_sd'])), 20, 127))) for o in ons]
        b = synth_slice(bn, c['bank'], c['bass'])
        mix += b * (r0 * 10 ** (c['bass_db'] / 20) / (rms(b) or 1.0))
    if c['layer2'] is not None:
        b = synth_slice(lines, c['bank'], c['layer2'])
        mix += b * (r0 * 10 ** (c['layer_db'] / 20) / (rms(b) or 1.0))
    if c['drums']:
        ev = run2.events(c['drums'], 'groove', int(seed))
        d = run2.gm_kit(ev, c['bank'])
        mix += d * (rms(mix) * 10 ** (c['drum_db'] / 20) / (rms(d) or 1.0))
    if c['lp_fc']:
        mix = lp(mix, float(c['lp_fc']))
    return np.asarray(M.C.lock(mix), np.float32)


def _r(a):
    return render9(*a)


class Scorer:
    def __init__(self, pool):
        import torch
        from audiobox_aesthetics.infer import AesPredictor
        self.torch, self.pool = torch, pool
        self.aes = AesPredictor(checkpoint_pth=None, batch_size=32)

    def score(self, items, keep=False):
        W = self.pool.map(_r, items, chunksize=4)
        rows = []
        for s0 in range(0, len(W), 32):
            rows += self.aes.forward([{'path': self.torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': SR} for w in W[s0:s0 + 32]])
        res = [{k: float(r[k]) for k in KEYS} for r in rows]
        assert all(np.isfinite(list(d.values())).all() for d in res), 'NaN score'
        return (res, W) if keep else res


# ---------------- sampling / mutation
def pool_progs(tier, top):
    return top[tier]


def sample_cfg(rng, tier, top):
    c = dict(DEFAULT)
    b, p = top[tier][int(rng.integers(len(top[tier])))]
    c.update(bank=b, prog=p, rate=float(rng.uniform(2, 16)), legato=float(rng.uniform(0.3, 2.0)),
             center=int(rng.integers(55, 80)), spread=int(rng.integers(4, 37)), pdist=str(rng.choice(['chrom', 'major', 'norep'])),
             vel_mean=float(rng.uniform(60, 120)), vel_sd=float(rng.uniform(0, 25)), poly=int(rng.integers(1, 3)))
    hi = 79 if tier == 'acoustic' else 127
    if rng.random() < 0.5:
        c['bass'] = int(rng.choice(list(range(32, 40)) + [p])); c['bass_db'] = float(rng.uniform(-18, 0))
    if rng.random() < 0.5:
        c['layer2'] = int(rng.integers(0, hi + 1)); c['layer_db'] = float(rng.uniform(-18, 0))
    c['drums'] = [None, 'rock', 'four_floor'][int(rng.integers(3))]; c['drum_db'] = float(rng.uniform(-24, 0))
    if rng.random() < 0.5:
        c['lp_fc'] = float(np.exp(rng.uniform(np.log(800), np.log(8000))))
    return c


def mutate(c, rng, tier):
    c = copy.deepcopy(c)
    hi = 79 if tier == 'acoustic' else 127
    ctrl = ['bank', 'prog', 'rate', 'legato', 'center', 'spread', 'pdist', 'vel_mean', 'vel_sd', 'poly', 'bass', 'bass_db', 'layer2', 'layer_db',
            'drums', 'drum_db', 'lp_fc']
    while True:
        n = 0
        for k in ctrl:
            if rng.random() > 0.22:
                continue
            n += 1
            if k == 'bank': c['bank'] = BANKS[1 - BANKS.index(c['bank'])]
            elif k == 'prog':
                c['prog'] = int(rng.integers(0, 80)) if tier == 'acoustic' else int(rng.integers(80, 128))
            elif k == 'rate': c['rate'] = float(np.clip(c['rate'] * np.exp(rng.normal(0, 0.2)), 2, 16))
            elif k == 'legato': c['legato'] = float(np.clip(c['legato'] + rng.normal(0, 0.25), 0.3, 2.0))
            elif k == 'center': c['center'] = int(np.clip(c['center'] + round(rng.normal(0, 4)), 55, 79))
            elif k == 'spread': c['spread'] = int(np.clip(c['spread'] + round(rng.normal(0, 5)), 4, 36))
            elif k == 'pdist': c['pdist'] = str(rng.choice(['chrom', 'major', 'norep']))
            elif k == 'vel_mean': c['vel_mean'] = float(np.clip(c['vel_mean'] + rng.normal(0, 10), 60, 120))
            elif k == 'vel_sd': c['vel_sd'] = float(np.clip(c['vel_sd'] + rng.normal(0, 5), 0, 25))
            elif k == 'poly': c['poly'] = 3 - c['poly']
            elif k == 'bass':
                if c['bass'] is not None and rng.random() < 0.4: c['bass'] = None
                else: c['bass'] = int(rng.choice(list(range(32, 40)) + [c['prog']]))
            elif k == 'layer2':
                if c['layer2'] is not None and rng.random() < 0.4: c['layer2'] = None
                else: c['layer2'] = int(rng.integers(0, hi + 1))
            elif k == 'bass_db': c['bass_db'] = float(np.clip(c['bass_db'] + rng.normal(0, 4), -18, 0))
            elif k == 'layer_db': c['layer_db'] = float(np.clip(c['layer_db'] + rng.normal(0, 4), -18, 0))
            elif k == 'drums': c['drums'] = [None, 'rock', 'four_floor'][int(rng.integers(3))]
            elif k == 'drum_db': c['drum_db'] = float(np.clip(c['drum_db'] + rng.normal(0, 5), -24, 0))
            elif k == 'lp_fc':
                if c['lp_fc'] is None: c['lp_fc'] = float(np.exp(rng.uniform(np.log(800), np.log(8000))))
                elif rng.random() < 0.3: c['lp_fc'] = None
                else: c['lp_fc'] = float(np.clip(c['lp_fc'] * np.exp(rng.normal(0, 0.3)), 800, 8000))
        if n and (tier == 'acoustic' or tier_of(c) == 'any'):
            if tier == 'acoustic' and tier_of(c) != 'acoustic':
                continue
            return c


def cell_of(c):
    return (tier_of(c), c['lp_fc'] is not None)


def boot(d, rng, n=2000):
    d = np.asarray(d); idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mean_over(res, idxs):
    return {k: float(np.mean([res[i][k] for i in idxs])) for k in KEYS}


# ---------------- stages
def stage1(S, smoke):
    base = dict(DEFAULT, rate=8, legato=0.9, center=66, spread=24, pdist='chrom', vel_mean=100, vel_sd=10)
    progs = [0, 19, 80, 81] if smoke else range(128)
    seeds = (0,) if smoke else (0, 1, 2)
    items, meta = [], []
    for b in BANKS:
        for p in progs:
            for fc in (None, 2260.0):
                for s in seeds:
                    items.append((dict(base, bank=b, prog=p, lp_fc=fc), s)); meta.append((b, p, fc))
    res = S.score(items)
    out = []
    for (b, p, fc), r in zip(meta, res):
        out.append(dict(bank=b, prog=p, filt=fc is not None, **r))
    (OUT / ('scores9_s1_smoke.json' if smoke else 'scores9_s1.json')).write_text(json.dumps(out, indent=1))
    return out


def top_from_s1(out, k=12):
    from collections import defaultdict
    ag = defaultdict(lambda: {'n': [], 'f': []})
    for r in out:
        ag[(r['bank'], r['prog'])]['f' if r['filt'] else 'n'].append(r['CE'])
    rows = [(max(np.mean(v['n']), np.mean(v['f'])), b, p, np.mean(v['n']), np.mean(v['f'])) for (b, p), v in ag.items()]
    top = {}
    top['acoustic'] = [(b, p) for _, b, p, _, _ in sorted([r for r in rows if r[2] <= 79], reverse=True)[:k]]
    top['any'] = [(b, p) for _, b, p, _, _ in sorted([r for r in rows if r[2] >= 80], reverse=True)[:k]]
    return top, rows


def stage2(S, smoke):
    top, _ = top_from_s1(json.loads((OUT / ('scores9_s1_smoke.json' if smoke else 'scores9_s1.json')).read_text()), 3 if smoke else 12)
    rng = np.random.default_rng(20261119)
    n = 10 if smoke else 2000
    cfgs = [sample_cfg(rng, t, top) for t in ('acoustic', 'any') for _ in range(n)]
    items = [(c, s) for c in cfgs for s in (0, 1)]
    res = S.score(items)
    out = []
    for i, c in enumerate(cfgs):
        d = dict(cfg=c, tier=tier_of(c), filtered=c['lp_fc'] is not None)
        d.update(mean_over(res, (2 * i, 2 * i + 1)))
        out.append(d)
    (OUT / ('scores9_s2_smoke.json' if smoke else 'scores9_s2.json')).write_text(json.dumps(out, indent=1))
    # main effects
    eff = {}
    ce = np.array([o['CE'] for o in out])
    def binned(name, vals, edges=None, cats=None):
        v = np.array(vals, dtype=object)
        if cats is not None:
            eff[name] = {str(c_): [float(ce[[x == c_ for x in vals]].mean()), int(sum(x == c_ for x in vals))] for c_ in cats if any(x == c_ for x in vals)}
        else:
            vv = np.array(vals, float); b = np.digitize(vv, edges)
            eff[name] = {f'bin{i}[{edges[i-1] if i else "-inf"},{edges[i] if i < len(edges) else "inf"})': [float(ce[b == i].mean()), int((b == i).sum())] for i in range(len(edges) + 1) if (b == i).any()}
    C = [o['cfg'] for o in out]
    binned('rate', [c['rate'] for c in C], [4, 6, 8, 10, 12, 14]); binned('legato', [c['legato'] for c in C], [0.6, 0.9, 1.2, 1.5, 1.8])
    binned('center', [c['center'] for c in C], [60, 64, 68, 72, 76]); binned('spread', [c['spread'] for c in C], [8, 12, 16, 20, 24, 28, 32])
    binned('pdist', [c['pdist'] for c in C], cats=['chrom', 'major', 'norep']); binned('vel_mean', [c['vel_mean'] for c in C], [70, 80, 90, 100, 110])
    binned('vel_sd', [c['vel_sd'] for c in C], [5, 10, 15, 20]); binned('poly', [c['poly'] for c in C], cats=[1, 2])
    binned('bass', [c['bass'] is not None for c in C], cats=[False, True]); binned('layer2', [c['layer2'] is not None for c in C], cats=[False, True])
    binned('drums', [c['drums'] for c in C], cats=[None, 'rock', 'four_floor']); binned('lp', [c['lp_fc'] is not None for c in C], cats=[False, True])
    binned('bank', [c['bank'] for c in C], cats=list(BANKS)); binned('tier', [o['tier'] for o in out], cats=['acoustic', 'any'])
    binned('drum_db(drums on)', [c['drum_db'] if c['drums'] else 99 for c in C], [-21, -18, -15, -12, -9, -6, -3, 50])
    (OUT / ('scores9_s2_effects_smoke.json' if smoke else 'scores9_s2_effects.json')).write_text(json.dumps(eff, indent=1))
    print(json.dumps(eff, indent=1), flush=True)
    return out


def stage3(S, smoke):
    s2 = json.loads((OUT / ('scores9_s2_smoke.json' if smoke else 'scores9_s2.json')).read_text())
    rng = np.random.default_rng(31)
    SS = (100, 101, 102, 103)
    kk = 2 if smoke else 10
    parents = []
    for t in ('acoustic', 'any'):
        rows = sorted([o for o in s2 if o['tier'] == t], key=lambda o: -o['CE'])[:kk]
        parents += [dict(cfg=o['cfg'], tier=t) for o in rows]
    lam, gens = (2, 1) if smoke else (8, 12)
    def ev(cfgs):
        res = S.score([(c, s) for c in cfgs for s in SS])
        return [dict(mean_over(res, range(4 * i, 4 * i + 4)), per_seed=[res[4 * i + j]['CE'] for j in range(4)]) for i in range(len(cfgs))]
    r0 = ev([p['cfg'] for p in parents])
    log = []
    for p, r in zip(parents, r0):
        p['score'] = r; log.append(dict(gen=0, cfg=p['cfg'], tier=p['tier'], accepted=True, **r))
    for g in range(1, gens + 1):
        kids = [(i, mutate(p['cfg'], rng, p['tier'])) for i, p in enumerate(parents) for _ in range(lam)]
        rk = ev([k for _, k in kids])
        for (i, kcfg), r in zip(kids, rk):
            acc = r['CE'] >= parents[i]['score']['CE'] and False
            log.append(dict(gen=g, cfg=kcfg, tier=parents[i]['tier'], accepted=False, **r))
            if r['CE'] >= parents[i]['score']['CE']:
                parents[i]['cfg'] = kcfg; parents[i]['score'] = r; log[-1]['accepted'] = True
        print('gen', g, 'best per tier', {t: max(p['score']['CE'] for p in parents if p['tier'] == t) for t in ('acoustic', 'any')}, flush=True)
    (OUT / ('scores9_s3_smoke.json' if smoke else 'scores9_s3.json')).write_text(json.dumps(log, indent=1))
    return log


def stage4(S, smoke):
    import soundfile as sf
    log = json.loads((OUT / ('scores9_s3_smoke.json' if smoke else 'scores9_s3.json')).read_text())
    cells = {}
    for l in log:
        cells.setdefault(cell_of(l['cfg']), []).append(l)
    picks = {}
    for cell, rows in cells.items():
        seen, sel = set(), []
        for l in sorted(rows, key=lambda l: -l['CE']):
            key = json.dumps(l['cfg'], sort_keys=True)
            if key in seen: continue
            seen.add(key); sel.append(l)
            if len(sel) == (1 if smoke else 3): break
        picks[cell] = sel
    SEEDS = tuple(range(1000, 1002 if smoke else 1012))
    items, meta = [], []
    for cell, sel in picks.items():
        for ci, l in enumerate(sel):
            for s in SEEDS:
                items.append((l['cfg'], s)); meta.append((cell, ci, s))
    res, W = S.score(items, keep=True)
    rng = np.random.default_rng(5)
    out, best_clip = [], {}
    for cell, sel in picks.items():
        for ci, l in enumerate(sel):
            idx = [i for i, m in enumerate(meta) if m[0] == cell and m[1] == ci]
            ces = [res[i]['CE'] for i in idx]
            d = dict(tier=cell[0], filtered=cell[1], rank_in_cell=ci, cfg=l['cfg'], search_CE=l['CE'], heldout_CE=float(np.mean(ces)),
                     CE_ci=boot(ces, rng), best_clip_CE=float(max(ces)), best_seed=int(meta[idx[int(np.argmax(ces))]][2]),
                     **{f'mean_{k}': float(np.mean([res[i][k] for i in idx])) for k in ('CU', 'PC', 'PQ')})
            out.append(d)
            i_best = idx[int(np.argmax(ces))]
            if cell not in best_clip or res[i_best]['CE'] > best_clip[cell][0]:
                best_clip[cell] = (res[i_best]['CE'], W[i_best], l['cfg'], meta[i_best][2])
    (OUT / ('scores9_final_smoke.json' if smoke else 'scores9_final.json')).write_text(json.dumps(out, indent=1))
    d = OUT / ('r9_top_smoke' if smoke else 'r9_top'); d.mkdir(exist_ok=True)
    for cell, (ce, w, c, s) in best_clip.items():
        sf.write(str(d / f'{cell[0]}_{"filt" if cell[1] else "nofilt"}_CE{ce:.2f}.wav'), w, SR, subtype='FLOAT')
    cell, (ce, w, c, s) = max(best_clip.items(), key=lambda kv: kv[1][0])
    sf.write(str(d / f'best_overall_CE{ce:.2f}.wav'), w, SR, subtype='FLOAT')
    return out


BASE_L = dict(DEFAULT, bank='generaluser', prog=0, rhythm='mary', pdist='mary', center=64, spread=12, vel_mean=100, vel_sd=0, poly=1)


def stage5(S, smoke):
    fin = json.loads((OUT / ('scores9_final_smoke.json' if smoke else 'scores9_final.json')).read_text())
    X = dict(DEFAULT, **max(fin, key=lambda d: d['heldout_CE'])['cfg'])
    SEEDS = tuple(range(1000, 1002 if smoke else 1012))
    ladder = []
    c = dict(BASE_L); ladder.append(('0 Mary/piano/C4 base', dict(c)))
    c['pdist'] = 'chrom'; ladder.append(('1 +random chromatic within 12 st', dict(c)))
    c.update(center=X['center'], spread=X['spread'], pdist=X['pdist']); ladder.append(('2 +X center/spread/pdist', dict(c)))
    c.update(rhythm='grid', rate=X['rate'], legato=X['legato']); ladder.append(('3 +X rate/legato (grid)', dict(c)))
    c.update({k: X[k] for k in ('bank', 'prog', 'poly', 'bass', 'bass_db', 'layer2', 'layer_db', 'vel_mean', 'vel_sd')}); ladder.append(('4 +X timbre/poly/bass/layer/vel', dict(c)))
    c.update(drums=X['drums'], drum_db=X['drum_db']); ladder.append(('5 +X drums', dict(c)))
    c.update(lp_fc=X['lp_fc']); ladder.append(('6 +X lowpass (=X)', dict(c)))
    loo = [('X (full)', dict(X))]
    def drop(name, **kw): loo.append((name, dict(X, **kw)))
    drop('-pitch variety (Mary pitch cycle)', pdist='mary')
    drop('-range (center64/spread12)', center=64, spread=12)
    drop('-rhythm (Mary rhythm)', rhythm='mary')
    drop('-timbre (generaluser piano)', bank='generaluser', prog=0)
    drop('-poly/bass/layer', poly=1, bass=None, layer2=None)
    drop('-velocity (100/sd0)', vel_mean=100, vel_sd=0)
    drop('-drums', drums=None)
    drop('-lowpass', lp_fc=None)
    allc = ladder + loo
    items = [(cc, s) for _, cc in allc for s in SEEDS]
    res = S.score(items)
    n = len(SEEDS); rng = np.random.default_rng(7)
    ce = {name: np.array([res[i * n + j]['CE'] for j in range(n)]) for i, (name, _) in enumerate(allc)}
    out = dict(X=X, ladder=[], loo=[])
    def row(name, cc, ref):
        d = ce[name] - ce[ref] if ref else np.zeros(n)
        r = dict(name=name, cfg=cc, mean_CE=float(ce[name].mean()), CE_ci=boot(ce[name], rng), per_seed=ce[name].tolist())
        if ref: r.update(ref=ref, dCE=float(d.mean()), dCE_ci=boot(d, rng))
        return r
    for i, (name, cc) in enumerate(ladder):
        r = row(name, cc, ladder[i - 1][0] if i else None); r['dCE_vs_base'] = float((ce[name] - ce[ladder[0][0]]).mean()); out['ladder'].append(r)
    for i, (name, cc) in enumerate(loo):
        out['loo'].append(row(name, cc, loo[0][0] if i else None))
    (OUT / ('scores9_ladder_smoke.json' if smoke else 'scores9_ladder.json')).write_text(json.dumps(out, indent=1))
    for r in out['ladder'] + out['loo']:
        print(r['name'], round(r['mean_CE'], 3), r.get('dCE'), r.get('dCE_ci'), flush=True)
    return out


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    st = [a for a in sys.argv[1:] if not a.startswith('--')]
    stages = [int(x) for x in sys.argv[sys.argv.index('--stage') + 1].split(',')] if '--stage' in sys.argv else [1, 2, 3, 4, 5]
    with mp.get_context('fork').Pool(12) as pool:
        S = Scorer(pool)
        for s in stages:
            print('== stage', s, flush=True)
            {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5}[s](S, smoke)
    print('done', flush=True)
