"""Round 10 (post-hoc, AES only): push CE of the no-e00 (SoundFont-only) recipe from 8.13 toward 8.5.
New optional ingredients on top of round 9: layers (<=4 doublings, oct shift), pad (block chords), counter (quiet 2nd line),
15 drum grooves + independent drum bank, numpy-convolution reverb (post=True). All pitched sounds are SoundFont programs.
Stages: 1 ablation on X, 2 ES search, 3 held-out (cells no-reverb / reverb), 4 loudness side-cell, 5 leave-one-out."""
import sys, json, copy
import numpy as np
import multiprocessing as mp
import run9
from run9 import M, run2, SR, NS, T, OUT, KEYS, BANKS, MAJ, draw_pitches, onsets, synth_slice, rms, lp, boot, mean_over, Q

GROOVES = list(run2.GROOVES)
PAD_PROGS = list(range(88, 96)) + list(range(48, 52)) + list(range(16, 21)) + [52, 53, 54, 4, 5]

DEFAULT10 = dict(bank='generaluser', prog=0, rhythm='grid', rate=8.0, legato=0.9, center=66, spread=24, pdist='chrom',
                 vel_mean=100, vel_sd=10, poly=1, bass=None, bass_db=-9.0, layers=[], pad=None, counter=None,
                 drums=None, drum_db=-12.0, drum_bank='generaluser', reverb=None, lp_fc=None)


def upgrade(c9):
    c = copy.deepcopy(DEFAULT10)
    for k in ('bank', 'prog', 'rhythm', 'rate', 'legato', 'center', 'spread', 'pdist', 'vel_mean', 'vel_sd', 'poly', 'bass', 'bass_db',
              'drums', 'drum_db', 'lp_fc'):
        c[k] = c9[k]
    c['layers'] = [dict(prog=int(c9['layer2']), db=float(c9['layer_db']), oct=0)] if c9.get('layer2') is not None else []
    c['drum_bank'] = c9['bank']
    return c


def lines_of(c, rng):
    lines = []
    for ln in range(c['poly']):
        sh = 0.0 if ln == 0 else (0.5 / c['rate'] if c['rhythm'] == 'grid' else Q / 2)
        tm = onsets(c, sh)
        ps = draw_pitches(c, len(tm), rng, ln)
        vs = np.clip(np.round(rng.normal(c['vel_mean'], c['vel_sd'], len(tm))), 20, 127).astype(int)
        lines += [(a, b, p, int(v)) for (a, b), p, v in zip(tm, ps, vs)]
    return lines


def pad_notes(c, lines):
    pd = c['pad']; dur = float(pd['dur'])
    notes = []
    for k in range(int(np.ceil((T - 0.05) / dur))):
        t0 = k * dur
        if pd['mode'] == 'prog':
            root, third = [(0, 4), (7, 4), (9, 3), (5, 4)][k % 4]
        else:
            w0 = np.floor(t0 / 2.0) * 2.0
            cnt = np.zeros(12)
            for a, b, p, v in lines:
                if w0 <= a < w0 + 2.0:
                    cnt[p % 12] += 1
            root = int(np.argmax(cnt)) if cnt.sum() else 0
            third = 4 if cnt[(root + 4) % 12] >= cnt[(root + 3) % 12] else 3
        r = 48 + ((root - 48) % 12)
        for iv in (0, third, 7):
            notes.append((t0, min(t0 + dur * 1.02, T), int(r + iv), 80))
    return notes


def reverb(mix, rt60, wet, rng):
    L = int(rt60 * SR); t = np.arange(L) / SR
    ir = rng.standard_normal(L) * np.exp(-6.908 * t / rt60)
    ir /= np.sqrt(np.sum(ir ** 2))
    n = NS + L
    w = np.fft.irfft(np.fft.rfft(mix, n) * np.fft.rfft(ir, n), n)[:NS]
    w *= rms(mix) / (rms(w) or 1.0)
    return (1 - wet) * mix + wet * w


def render10(c, seed, target=-30.0):
    rng = np.random.default_rng([int(seed), 9])           # same stream as render9 -> identical for round-9 configs
    lines = lines_of(c, rng)
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
    for L in c['layers']:
        ln = [(a, b, int(np.clip(p + L['oct'], 21, 108)), v) for a, b, p, v in lines]
        b = synth_slice(ln, c['bank'], L['prog'])
        mix += b * (r0 * 10 ** (L['db'] / 20) / (rms(b) or 1.0))
    if c['pad'] is not None:
        b = synth_slice(pad_notes(c, lines), c['bank'], c['pad']['prog'])
        mix += b * (r0 * 10 ** (c['pad']['db'] / 20) / (rms(b) or 1.0))
    if c['counter'] is not None:
        ct = c['counter']; rc = np.random.default_rng([int(seed), 10, 1])
        cc = dict(c, rate=c['rate'] * ct['mult'], rhythm='grid', poly=1)
        tm = onsets(cc, 0.5 / cc['rate'])
        ps = draw_pitches(cc, len(tm), rc)
        vs = np.clip(np.round(rc.normal(c['vel_mean'], c['vel_sd'], len(tm))), 20, 127).astype(int)
        b = synth_slice([(a, bb, p, int(v)) for (a, bb), p, v in zip(tm, ps, vs)], c['bank'], ct['prog'])
        mix += b * (r0 * 10 ** (ct['db'] / 20) / (rms(b) or 1.0))
    if c['drums']:
        ev = run2.events(c['drums'], 'groove', int(seed))
        d = run2.gm_kit(ev, c['drum_bank'])
        mix += d * (rms(mix) * 10 ** (c['drum_db'] / 20) / (rms(d) or 1.0))
    if c['reverb'] is not None:
        mix = reverb(mix, float(c['reverb']['rt60']), float(c['reverb']['wet']), np.random.default_rng([int(seed), 10, 2]))
    if c['lp_fc']:
        mix = lp(mix, float(c['lp_fc']))
    return np.asarray(M.C.P.fit(M.C.P.to_lufs(mix, float(target))[0]), np.float32)


def _r10(a):
    return render10(*a)


class Scorer10(run9.Scorer):
    def score(self, items, keep=False):          # items: (cfg, seed[, target])
        W = self.pool.map(_r10, items, chunksize=4)
        rows = []
        for s0 in range(0, len(W), 32):
            rows += self.aes.forward([{'path': self.torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': SR} for w in W[s0:s0 + 32]])
        res = [{k: float(r[k]) for k in KEYS} for r in rows]
        assert all(np.isfinite(list(d.values())).all() for d in res), 'NaN score'
        return (res, W) if keep else res


def X10():
    fin = json.loads((OUT / 'scores9_final.json').read_text())
    return upgrade(max(fin, key=lambda d: d['heldout_CE'])['cfg'])


def jd(x):
    return json.dumps(x, sort_keys=True)


# ---------------- mutation
def rl(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def new_layer(rng):
    return dict(prog=int(rng.integers(0, 128)), db=rl(rng, -18, 0), oct=int(rng.choice([-12, 0, 12])))


def new_pad(rng):
    return dict(prog=int(rng.choice(PAD_PROGS)), db=rl(rng, -30, -6), dur=int(rng.choice([1, 2, 4])), mode=str(rng.choice(['common', 'prog'])))


def new_counter(rng, c):
    return dict(prog=int(rng.integers(0, 128)), db=rl(rng, -24, -6), mult=float(rng.choice([0.5, 1, 2])))


def new_reverb(rng):
    return dict(rt60=rl(rng, 0.2, 2.5), wet=rl(rng, 0.02, 0.5))


def mutate(c, rng):
    c = copy.deepcopy(c)
    while True:
        n = 0
        for k in ['bank', 'prog', 'rate', 'legato', 'center', 'spread', 'pdist', 'vel_mean', 'vel_sd', 'poly', 'bass', 'bass_db', 'drums',
                  'drum_db', 'drum_bank', 'lp_fc', 'layers', 'pad', 'counter', 'reverb']:
            if rng.random() > 0.2:
                continue
            n += 1
            if k == 'bank': c['bank'] = BANKS[1 - BANKS.index(c['bank'])]
            elif k == 'prog': c['prog'] = int(rng.integers(0, 80)) if rng.random() < 0.8 else int(rng.integers(80, 128))
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
            elif k == 'bass_db': c['bass_db'] = float(np.clip(c['bass_db'] + rng.normal(0, 4), -18, 0))
            elif k == 'drums':
                c['drums'] = None if (c['drums'] and rng.random() < 0.25) else GROOVES[int(rng.integers(len(GROOVES)))]
            elif k == 'drum_db': c['drum_db'] = float(np.clip(c['drum_db'] + rng.normal(0, 5), -30, 0))
            elif k == 'drum_bank': c['drum_bank'] = BANKS[1 - BANKS.index(c['drum_bank'])]
            elif k == 'lp_fc':
                if c['lp_fc'] is None: c['lp_fc'] = float(np.exp(rng.uniform(np.log(800), np.log(8000))))
                elif rng.random() < 0.3: c['lp_fc'] = None
                else: c['lp_fc'] = float(np.clip(c['lp_fc'] * np.exp(rng.normal(0, 0.3)), 800, 8000))
            elif k == 'layers':
                u = rng.random(); L = c['layers']
                if (not L or u < 0.3) and len(L) < 4: L.append(new_layer(rng))
                elif u < 0.5 and L: L.pop(int(rng.integers(len(L))))
                elif L:
                    l = L[int(rng.integers(len(L)))]; f = rng.choice(['prog', 'db', 'oct'])
                    if f == 'prog': l['prog'] = int(rng.integers(0, 128))
                    elif f == 'db': l['db'] = float(np.clip(l['db'] + rng.normal(0, 4), -18, 0))
                    else: l['oct'] = int(rng.choice([-12, 0, 12]))
            elif k == 'pad':
                p = c['pad']
                if p is None: c['pad'] = new_pad(rng)
                elif rng.random() < 0.3: c['pad'] = None
                else:
                    f = rng.choice(['prog', 'db', 'dur', 'mode'])
                    if f == 'prog': p['prog'] = int(rng.choice(PAD_PROGS))
                    elif f == 'db': p['db'] = float(np.clip(p['db'] + rng.normal(0, 4), -30, -6))
                    elif f == 'dur': p['dur'] = int(rng.choice([1, 2, 4]))
                    else: p['mode'] = 'prog' if p['mode'] == 'common' else 'common'
            elif k == 'counter':
                p = c['counter']
                if p is None: c['counter'] = new_counter(rng, c)
                elif rng.random() < 0.3: c['counter'] = None
                else:
                    f = rng.choice(['prog', 'db', 'mult'])
                    if f == 'prog': p['prog'] = int(rng.integers(0, 128))
                    elif f == 'db': p['db'] = float(np.clip(p['db'] + rng.normal(0, 4), -24, -6))
                    else: p['mult'] = float(rng.choice([0.5, 1, 2]))
            elif k == 'reverb':
                p = c['reverb']
                if p is None: c['reverb'] = new_reverb(rng)
                elif rng.random() < 0.3: c['reverb'] = None
                else:
                    p['rt60'] = float(np.clip(p['rt60'] * np.exp(rng.normal(0, 0.3)), 0.2, 2.5)); p['wet'] = float(np.clip(p['wet'] + rng.normal(0, 0.08), 0.0, 0.5))
        if n:
            return c


def cell_of(c):
    return 'reverb' if c['reverb'] is not None else 'noreverb'


# ---------------- stage 1: ablations on X
def ablations(X):
    A = []                                  # (ingredient, name, new cfg)
    def add(ing, name, **kw):
        c = copy.deepcopy(X); c.update(copy.deepcopy(kw)); A.append((ing, name, c))
    v = X['layers'][0]
    Ls = lambda *ls: dict(layers=[dict(prog=p, db=d, oct=o) for p, d, o in ls])
    add('layers', 'X layer: violin db -3', **Ls((v['prog'], -3, 0)))
    add('layers', 'X layer: violin +12 oct', **Ls((v['prog'], v['db'], 12)))
    add('layers', 'X layer: violin -12 oct', **Ls((v['prog'], v['db'], -12)))
    add('layers', '+piano(0) -6', **Ls((v['prog'], v['db'], 0), (0, -6, 0)))
    add('layers', '+strings ens(48) -6', **Ls((v['prog'], v['db'], 0), (48, -6, 0)))
    add('layers', '+vibes(11) -9', **Ls((v['prog'], v['db'], 0), (11, -9, 0)))
    add('layers', '+harp(46) +12 oct -6', **Ls((v['prog'], v['db'], 0), (46, -6, 12)))
    add('layers', '+harp(46) -12 oct -6', **Ls((v['prog'], v['db'], 0), (46, -6, -12)))
    add('layers', '+nylon gtr(24) -6', **Ls((v['prog'], v['db'], 0), (24, -6, 0)))
    add('layers', '3 extra: 48,0,11', **Ls((v['prog'], v['db'], 0), (48, -6, 0), (0, -9, 0), (11, -9, 12)))
    for nm, p, m, d, du in [('warm pad(89) prog -18 2s', 89, 'prog', -18, 2), ('strings(48) prog -18 2s', 48, 'prog', -18, 2),
                            ('organ(19) prog -18 2s', 19, 'prog', -18, 2), ('choir(52) prog -18 2s', 52, 'prog', -18, 2),
                            ('epiano(4) prog -18 2s', 4, 'prog', -18, 2), ('strings(48) common -18 2s', 48, 'common', -18, 2),
                            ('strings(48) prog -24 2s', 48, 'prog', -24, 2), ('strings(48) prog -12 2s', 48, 'prog', -12, 2),
                            ('strings(48) prog -18 1s', 48, 'prog', -18, 1), ('strings(48) prog -18 4s', 48, 'prog', -18, 4)]:
        add('pad', nm, pad=dict(prog=p, db=d, dur=du, mode=m))
    for nm, p, d, m in [('piano(0) -18 x1', 0, -18, 1), ('piano(0) -12 x1', 0, -12, 1), ('piano(0) -6 x1', 0, -6, 1), ('vibes(11) -12 x1', 11, -12, 1),
                        ('nylon(24) -12 x1', 24, -12, 1), ('flute(73) -12 x1', 73, -12, 1), ('piano(0) -12 x0.5', 0, -12, 0.5),
                        ('piano(0) -12 x2', 0, -12, 2), ('strings(48) -12 x1', 48, -12, 1), ('vibes(11) -18 x0.5', 11, -18, 0.5)]:
        add('counter', nm, counter=dict(prog=p, db=d, mult=m))
    for g in GROOVES:
        if g != X['drums']:
            add('drums', f'groove {g}', drums=g)
    for d in (-30, -24, -12, -6, 0):
        add('drums', f'rock db {d}', drum_db=float(d))
    add('drums', 'rock kit fluidr3', drum_bank='fluidr3')
    add('drums', 'no drums', drums=None)
    for rt, w in [(0.3, 0.15), (0.6, 0.15), (1.2, 0.15), (0.6, 0.3), (1.2, 0.3), (2.0, 0.3), (1.2, 0.5), (2.5, 0.5), (0.2, 0.5), (2.0, 0.15)]:
        add('reverb', f'rt60 {rt} wet {w}', reverb=dict(rt60=rt, wet=w))
    return A


def paired(ces, ref, rng):
    d = np.asarray(ces) - np.asarray(ref)
    return float(d.mean()), boot(d, rng)


def stage1(S, smoke):
    X = X10(); A = ablations(X)
    SEEDS = tuple(range(1000, 1002 if smoke else 1012))
    if smoke: A = A[:2] + A[10:12] + A[30:31]
    allc = [('X', 'X', X)] + A
    res = S.score([(c, s) for _, _, c in allc for s in SEEDS]); n = len(SEEDS)
    ce = [np.array([res[i * n + j]['CE'] for j in range(n)]) for i in range(len(allc))]
    rng = np.random.default_rng(1); out = []
    for i, (ing, nm, c) in enumerate(allc):
        r = dict(ingredient=ing, name=nm, cfg=c, mean_CE=float(ce[i].mean()), per_seed=ce[i].tolist(),
                 **{f'mean_{k}': float(np.mean([res[i * n + j][k] for j in range(n)])) for k in ('CU', 'PC', 'PQ')})
        if i: r['dCE'], r['dCE_ci'] = paired(ce[i], ce[0], rng)
        out.append(r)
    (OUT / ('scores10_s1_smoke.json' if smoke else 'scores10_s1.json')).write_text(json.dumps(out, indent=1))
    for r in out:
        print(r['ingredient'], '|', r['name'], round(r['mean_CE'], 3), r.get('dCE') and round(r['dCE'], 3), r.get('dCE_ci') and [round(x, 3) for x in r['dCE_ci']], flush=True)
    return out


# ---------------- stage 2: ES
def stage2(S, smoke):
    s1 = json.loads((OUT / ('scores10_s1_smoke.json' if smoke else 'scores10_s1.json')).read_text())
    X = X10()
    parents = [dict(cfg=X, tag='X')]
    s3 = json.loads((OUT / 'scores9_s3.json').read_text())
    seen = {jd(upgrade(X)) if False else jd(X)}
    ac = [l for l in sorted(s3, key=lambda l: -l['CE']) if all(p <= 79 for p in [l['cfg']['prog']] + [l['cfg'][k] for k in ('bass', 'layer2') if l['cfg'][k] is not None])]
    for l in ac:
        u = upgrade(l['cfg'])
        if jd(u) in seen: continue
        seen.add(jd(u)); parents.append(dict(cfg=u, tag='r9top'))
        if sum(p['tag'] == 'r9top' for p in parents) == (2 if smoke else 10): break
    best, comb = {}, copy.deepcopy(X)
    for r in s1[1:]:
        if r['dCE_ci'][0] > 0 and (r['ingredient'] not in best or r['dCE'] > best[r['ingredient']]['dCE']):
            best[r['ingredient']] = r
    for ing, r in best.items():
        parents.append(dict(cfg=r['cfg'], tag=f'X+{ing}:{r["name"]}'))
    if len(best) > 1:
        for ing, r in best.items():
            for k in ('layers', 'pad', 'counter', 'reverb'):
                if ing == k: comb[k] = copy.deepcopy(r['cfg'][k])
            if ing == 'drums':
                for k in ('drums', 'drum_db', 'drum_bank'): comb[k] = r['cfg'][k]
        parents.append(dict(cfg=comb, tag='X+all helpful'))
    rng = np.random.default_rng(101)
    SS = (100, 101, 102, 103)
    lam = 2 if smoke else 8
    n = len(parents)
    gens = 2 if smoke else max(20, min(40, (15000 - 4 * n) // (4 * lam * n)))
    print('parents', n, [p['tag'] for p in parents], 'gens', gens, 'clips~', 4 * n + gens * 4 * lam * n, flush=True)
    def ev(cfgs):
        res = S.score([(c, s) for c in cfgs for s in SS])
        return [dict(mean_over(res, range(4 * i, 4 * i + 4)), per_seed=[res[4 * i + j]['CE'] for j in range(4)]) for i in range(len(cfgs))]
    r0 = ev([p['cfg'] for p in parents]); log = []
    for p, r in zip(parents, r0):
        p['score'] = r; log.append(dict(gen=0, tag=p['tag'], cfg=p['cfg'], accepted=True, **r))
    for g in range(1, gens + 1):
        kids = [(i, mutate(p['cfg'], rng)) for i, p in enumerate(parents) for _ in range(lam)]
        rk = ev([k for _, k in kids])
        for (i, kc), r in zip(kids, rk):
            log.append(dict(gen=g, tag=parents[i]['tag'], cfg=kc, accepted=False, **r))
            if r['CE'] >= parents[i]['score']['CE']:
                parents[i]['cfg'] = kc; parents[i]['score'] = r; log[-1]['accepted'] = True
        print('gen', g, 'best parent CE', round(max(p['score']['CE'] for p in parents), 3), 'mean', round(float(np.mean([p['score']['CE'] for p in parents])), 3), flush=True)
        (OUT / ('scores10_s2_smoke.json' if smoke else 'scores10_s2.json')).write_text(json.dumps(log))
    (OUT / ('scores10_s2_smoke.json' if smoke else 'scores10_s2.json')).write_text(json.dumps(log, indent=1))
    return log


# ---------------- stage 3: held-out
def stage3(S, smoke):
    import soundfile as sf
    log = json.loads((OUT / ('scores10_s2_smoke.json' if smoke else 'scores10_s2.json')).read_text())
    picks = {}
    for cell in ('noreverb', 'reverb'):
        seen, sel = set(), []
        for l in sorted([l for l in log if cell_of(l['cfg']) == cell], key=lambda l: -l['CE']):
            if jd(l['cfg']) in seen: continue
            seen.add(jd(l['cfg'])); sel.append(l)
            if len(sel) == (1 if smoke else 5): break
        picks[cell] = sel
    SEEDS = tuple(range(2000, 2002 if smoke else 2012))
    items, meta = [], []
    for cell, sel in picks.items():
        for ci, l in enumerate(sel):
            for s in SEEDS:
                items.append((l['cfg'], s)); meta.append((cell, ci, s))
    res, W = S.score(items, keep=True)
    rng = np.random.default_rng(5); out, best = [], {}
    for cell, sel in picks.items():
        for ci, l in enumerate(sel):
            idx = [i for i, m in enumerate(meta) if m[0] == cell and m[1] == ci]
            ces = [res[i]['CE'] for i in idx]
            out.append(dict(cell=cell, rank_in_cell=ci, tag=l['tag'], cfg=l['cfg'], search_CE=l['CE'], heldout_CE=float(np.mean(ces)), CE_ci=boot(ces, rng),
                            best_clip_CE=float(max(ces)), best_seed=int(meta[idx[int(np.argmax(ces))]][2]), per_seed=ces,
                            **{f'mean_{k}': float(np.mean([res[i][k] for i in idx])) for k in ('CU', 'PC', 'PQ')}))
            ib = idx[int(np.argmax(ces))]
            if cell not in best or res[ib]['CE'] > best[cell][0]:
                best[cell] = (res[ib]['CE'], W[ib], l['cfg'], meta[ib][2])
    (OUT / ('scores10_final_smoke.json' if smoke else 'scores10_final.json')).write_text(json.dumps(out, indent=1))
    d = OUT / ('r10_top_smoke' if smoke else 'r10_top'); d.mkdir(exist_ok=True)
    ver = []
    for cell, (ce, w, c, s) in best.items():
        p = d / f'{cell}_CE{ce:.2f}.wav'
        sf.write(str(p), w, SR, subtype='FLOAT')
        w2, sr2 = sf.read(str(p), dtype='float32')
        rr = S.aes.forward([{'path': S.torch.from_numpy(np.ascontiguousarray(w2))[None], 'sample_rate': sr2}])[0]
        ver.append(dict(cell=cell, wav=str(p), sr=sr2, stored_CE=ce, rescored_CE=float(rr['CE']), diff=float(rr['CE']) - ce))
        print('verify', ver[-1], flush=True)
    (OUT / ('scores10_verify_smoke.json' if smoke else 'scores10_verify.json')).write_text(json.dumps(ver, indent=1))
    for r in sorted(out, key=lambda r: -r['heldout_CE']):
        print(r['cell'], r['rank_in_cell'], round(r['heldout_CE'], 3), [round(x, 3) for x in r['CE_ci']], round(r['best_clip_CE'], 3),
              round(r['mean_PQ'], 2), round(r['mean_CU'], 2), round(r['mean_PC'], 2), flush=True)
    return out


# ---------------- stage 4: loudness side-cell
def stage4(S, smoke):
    import soundfile as sf
    fin = json.loads((OUT / ('scores10_final_smoke.json' if smoke else 'scores10_final.json')).read_text())
    cands = {'overall': max(fin, key=lambda r: r['heldout_CE'])}
    for cell in ('noreverb', 'reverb'):
        rows = [r for r in fin if r['cell'] == cell]
        if rows: cands[cell] = max(rows, key=lambda r: r['heldout_CE'])
    SEEDS = tuple(range(2000, 2002 if smoke else 2012)); TG = (-30.0, -23.0, -16.0)
    items, meta = [], []
    for nm, r in cands.items():
        for tg in TG:
            for s in SEEDS:
                items.append((r['cfg'], s, tg)); meta.append((nm, tg, s))
    res, W = S.score(items, keep=True)
    rng = np.random.default_rng(9); out = []
    d = OUT / ('r10_top_smoke' if smoke else 'r10_top')
    for nm, r in cands.items():
        base = None
        for tg in TG:
            idx = [i for i, m in enumerate(meta) if m[0] == nm and m[1] == tg]
            ces = [res[i]['CE'] for i in idx]
            lufs = [float(M.C.P.lufs_of(W[i].astype(np.float64))) for i in idx]
            pk = [float(np.max(np.abs(W[i]))) for i in idx]
            row = dict(cfg_name=nm, cell=r['cell'], target=tg, mean_CE=float(np.mean(ces)), CE_ci=boot(ces, rng), best_clip_CE=float(max(ces)), per_seed=ces,
                       mean_achieved_lufs=float(np.mean(lufs)), n_peak_capped=int(sum(abs(l - tg) > 0.05 for l in lufs)), mean_peak=float(np.mean(pk)),
                       **{f'mean_{k}': float(np.mean([res[i][k] for i in idx])) for k in ('CU', 'PC', 'PQ')})
            if base is None: base = ces
            else: row['dCE_vs_-30'], row['dCE_ci'] = paired(ces, base, rng)
            out.append(row); print(row['cfg_name'], tg, round(row['mean_CE'], 3), row.get('dCE_vs_-30'), row['mean_achieved_lufs'], row['n_peak_capped'], flush=True)
            if nm == 'overall' and tg != -30.0:
                ib = idx[int(np.argmax(ces))]; sf.write(str(d / f'loud{int(tg)}_CE{max(ces):.2f}.wav'), W[ib], SR, subtype='FLOAT')
    (OUT / ('scores10_loud_smoke.json' if smoke else 'scores10_loud.json')).write_text(json.dumps(out, indent=1))
    return out


# ---------------- stage 5: leave-one-out
def stage5(S, smoke):
    fin = json.loads((OUT / ('scores10_final_smoke.json' if smoke else 'scores10_final.json')).read_text())
    B = copy.deepcopy(max(fin, key=lambda r: r['heldout_CE'])['cfg'])
    print('best cfg', jd(B), flush=True)
    loo = [('B (full)', B)]
    def drop(name, cond, **kw):
        if cond: loo.append((name, dict(copy.deepcopy(B), **kw)))
        else: print('absent in B:', name, flush=True)
    drop('-pitch variety (Mary pitch cycle)', True, pdist='mary')
    drop('-range (center64/spread12)', True, center=64, spread=12)
    drop('-rhythm (Mary rhythm)', True, rhythm='mary')
    drop('-timbre (generaluser piano)', True, bank='generaluser', prog=0)
    drop('-poly (1 line)', B['poly'] == 2, poly=1)
    drop('-bass', B['bass'] is not None, bass=None)
    drop('-velocity (100/sd0)', True, vel_mean=100, vel_sd=0)
    drop('-lowpass', B['lp_fc'] is not None, lp_fc=None)
    for i, L in enumerate(B['layers']):
        drop(f'-layer{i + 1} (prog {L["prog"]}, {L["db"]:.0f} dB, oct {L["oct"]})', True, layers=[l for j, l in enumerate(B['layers']) if j != i])
    drop('-all layers', len(B['layers']) > 1, layers=[])
    drop('-pad', B['pad'] is not None, pad=None)
    drop('-counter', B['counter'] is not None, counter=None)
    drop('-drums', B['drums'] is not None, drums=None)
    drop('-drum groove (->rock)', B['drums'] not in (None, 'rock'), drums='rock')
    drop('-drum kit (->melody bank)', B['drum_bank'] != B['bank'] and B['drums'] is not None, drum_bank=B['bank'])
    drop('-reverb', B['reverb'] is not None, reverb=None)
    SEEDS = tuple(range(2000, 2002 if smoke else 2012)); n = len(SEEDS)
    res = S.score([(c, s) for _, c in loo for s in SEEDS])
    ce = [np.array([res[i * n + j]['CE'] for j in range(n)]) for i in range(len(loo))]
    rng = np.random.default_rng(7); out = []
    for i, (nm, c) in enumerate(loo):
        r = dict(name=nm, cfg=c, mean_CE=float(ce[i].mean()), CE_ci=boot(ce[i], rng), per_seed=ce[i].tolist())
        if i: r['dCE'], r['dCE_ci'] = paired(ce[i], ce[0], rng)
        out.append(r); print(nm, round(r['mean_CE'], 3), r.get('dCE') and round(r['dCE'], 3), r.get('dCE_ci') and [round(x, 3) for x in r['dCE_ci']], flush=True)
    (OUT / ('scores10_loo_smoke.json' if smoke else 'scores10_loo.json')).write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    stages = [int(x) for x in sys.argv[sys.argv.index('--stage') + 1].split(',')] if '--stage' in sys.argv else [1, 2, 3, 4, 5]
    with mp.get_context('fork').Pool(12) as pool:
        S = Scorer10(pool)
        for s in stages:
            print('== stage', s, flush=True)
            {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5}[s](S, smoke)
    print('done', flush=True)
