#!/usr/bin/env python
"""196: AES CE push (eval-only driver).

Design: docs/experiments/ce_push_196_20261114.md (pre-registered; this file only implements it).
Question: with simple post-processing that never looks at test, how high can AES CE be pushed on the 9 MusicCaps bases?
No specificity requirement (PQ/CU/PC/A1 may rise too); only dCE CI lower bound > m_CE = 0.119 per base.

Stages (all resumable from per_clip/*.tsv; frozen JSONs are written once):
  S1   dev (200 ids) x 9 bases x 15 cells, tier 1 (MuQ+AES+CLAP): base, pno{12,6,0}, syn{12,6,0,m6,m12}, lp{2k,3k,4500},
       tilt{3,6}, synonly.
  S2   dev stacks, chosen from the S1 pooled dCE on the 6 listenable bases (stage2_spec.json frozen first):
       stackA / stackB / stackC (+ synm18 if best_syn == synm12).
  sel  selection.json: star_any / star_keep (dCLAP >= -0.02) / best_pno / best_syn / best_filt with dev numbers.
  test 800 ids x 9 bases, tier 2 (adds PAM, CMI-RM caption): base, star_any, star_keep, best_pno, best_filt, synonly (dedup).
Reuses 195 (ce_weakness_195.py: key_of, arp, synth_E, run_cell, split ...; not edited) by importing it and pointing its OUT/X at 196's.
The 195 D `base` cells (tier 2, same split/ids) are copied instead of rescored when id set and columns match.
"""
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ce_weakness_195 as C  # noqa: E402

M, P, H, Q = C.M, C.P, C.H, C.Q
SR, NS = C.SR, C.NS
REPO = C.REPO
OUT195 = H.EVAL_ROOT / 'ce_weakness_195'
OUT = H.EVAL_ROOT / 'ce_push_196'
KIND = 'ce_push_196_summary_v1'
SEL_KIND = 'ce_push_196_selection_v1'
SEED = 20261114
NB = 5000
M_CE_FRAC = 0.10
KEEP_DCLAP = -0.02
MIN_FREE = 20 * 1024**3
D_BASES_FULL, MC_BASES = C.BASES, C.MC_BASES
DELTA = C.DELTA + ['CLAP']

# ── cell registry: name -> dict(pno=r|None, syn=r|None, filt=[kind, param]|None, only=bool) ──
S1_CELLS = {
    'base': {},
    'pno12': {'pno': 12}, 'pno6': {'pno': 6}, 'pno0': {'pno': 0},
    'syn12': {'syn': 12}, 'syn6': {'syn': 6}, 'syn0': {'syn': 0}, 'synm6': {'syn': -6}, 'synm12': {'syn': -12},
    'lp2k': {'filt': ['lp', 2000]}, 'lp3k': {'filt': ['lp', 3000]}, 'lp4500': {'filt': ['lp', 4500]},
    'tilt3': {'filt': ['tilt', -3]}, 'tilt6': {'filt': ['tilt', -6]},
    'synonly': {'only': True},
}
S1_ORDER = list(S1_CELLS)
PNO_C = ['pno12', 'pno6', 'pno0']
SYN_C = ['syn12', 'syn6', 'syn0', 'synm6', 'synm12']
FILT_C = ['lp2k', 'lp3k', 'lp4500', 'tilt3', 'tilt6']
PNO_R = (12, 6, 0)
SYN_R = (12, 6, 0, -6, -12, -18)

SYN = {}       # (tonic, mode, r) -> float32 synth locked to -30 - r LU (r=None: -30 -> synonly)
KEYC = C.KEYC  # (base, id) -> (tonic, mode)
X = None
IS_SMOKE = False


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


# ═════════ stimuli ═════════
def e00_params():
    return list(json.loads((OUT195 / 'e_top10.json').read_text())['top10'][0]['p'])


def build_syn(p0):
    for mode in (0, 1):
        for t in range(12):
            p = list(p0)
            p[4] = (t + 0.5) / 12
            p[3] = 0.25 if mode else 0.08
            x = np.asarray(C.synth_E(p), np.float64)      # locked to -30 LUFS
            SYN[(t, mode, None)] = x.astype(np.float32)
            for r in SYN_R:
                SYN[(t, mode, r)] = P.fit(P.to_lufs(x, -30.0 - r)[0])


def build_pno(pool):
    keys = [(t, m) for m in (0, 1) for t in range(12)]
    raw = pool.map(C._job_arp, keys)
    for k, x in zip(keys, raw):
        for r in PNO_R:
            C.ARPS[(k[0], k[1], r)] = P.fit(P.to_lufs(x, -30.0 - r)[0])


def apply_filt(w, filt):
    w = np.asarray(w, np.float64)
    n = len(w)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    kind, a = filt
    if kind == 'lp':
        g = 1.0 / np.sqrt(1.0 + (f / float(a)) ** 8)
    else:
        g = 10 ** (float(a) * np.log2(np.maximum(f, 50.0) / 1000.0) / 20)
    return np.fft.irfft(np.fft.rfft(w) * g, n)


def stim(w, key, d):
    """add arps (pno, then syn), filter the whole mix, lock to -30 LUFS."""
    if d.get('only'):
        return np.asarray(SYN[(key[0], key[1], None)], np.float32)
    y = np.asarray(w, np.float64)
    if d.get('pno') is not None:
        y = y + C.ARPS[(key[0], key[1], d['pno'])]
    if d.get('syn') is not None:
        y = y + SYN[(key[0], key[1], d['syn'])]
    if d.get('filt'):
        y = apply_filt(y, d['filt'])
    if not d:
        return np.asarray(w)      # base: untouched, as 195
    return C.lock(y)


def builders(base, wavs, defs):
    def key_for(i):
        if (base, i) not in KEYC:
            KEYC[(base, i)] = C.key_of(wavs[i])
        return KEYC[(base, i)]
    out = {}
    for name, d in defs.items():
        if not d:
            out[name] = lambda i: wavs[i]
        else:
            out[name] = (lambda d_: (lambda i: stim(wavs[i], key_for(i), d_)))(d)
    return out


# ═════════ run machinery ═════════
def setup(smoke):
    C.OUT = OUT
    C.IS_SMOKE = smoke
    C.setup(smoke)


def get_ids(smoke):
    return C.get_ids(smoke)


def splits(smoke):
    mc, _ = get_ids(smoke)
    sp = C.get_split(mc)
    return sp['dev_mc'], sp['test_mc']


def bases(smoke):
    return C.d_bases(smoke)


def lbases(smoke):
    return C.listen_bases(smoke)


def run_cell(key, ids, build, caps, tier):
    """C.run_cell with CUDA-OOM retry/backoff (a partial cell is rescored from scratch)."""
    for att in range(8):
        try:
            return C.run_cell(key, ids, build, caps, tier, True, 'caption')
        except Exception as e:  # noqa: BLE001
            oom = 'out of memory' in str(e).lower() or type(e).__name__ == 'OutOfMemoryError'
            if not oom or att == 7:
                raise
            P.SC.torch.cuda.empty_cache()
            log(f'{key}: CUDA OOM attempt {att}, sleep {60 * (att + 1)}s')
            time.sleep(60 * (att + 1))


def done(key, ids, tier):
    return C.cell_done(key, ids, tier, True, True)


def reuse_195(new_key, old_key, ids, tier):
    """copy a 195 base cell when ids and required columns match."""
    dst = OUT / 'per_clip' / f'{new_key}.tsv'
    if done(new_key, ids, tier):
        return True
    src = OUT195 / 'per_clip' / f'{old_key}.tsv'
    if not src.is_file():
        return False
    if M.rows_ok(M.tsv_read(src), ids, Q.req_cols(tier, True, True)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        log(f'{new_key}: reused 195 {old_key}')
        return True
    return False


def dev_base(smoke, b, dev):
    return reuse_195(f'dev_{b}_base', f'Ddev_{b}_base', dev, 1)


def test_base(smoke, b, test):
    return reuse_195(f'test_{b}_base', f'D_{b}_base', test, 2)


def run_dev(smoke, caps_all, defs):
    dev, _ = splits(smoke)
    caps = {i: caps_all[i] for i in dev}
    for b in bases(smoke):
        have_base = dev_base(smoke, b, dev)
        todo = [c for c in defs if c != 'base' and not done(f'dev_{b}_{c}', dev, 1)] + ([] if have_base else ['base'])
        if not todo:
            continue
        wavs = M.load_base(b, dev)
        bl = builders(b, wavs, defs if 'base' in todo or True else defs)
        for c in todo:
            run_cell(f'dev_{b}_{c}', dev, bl[c], caps, 1)
        del wavs


def dev_dce(cells, smoke):
    """pooled (mean over the listenable bases of per-base mean) dCE and dCLAP vs base, plus per-base dCE."""
    dev, _ = splits(smoke)
    lb = lbases(smoke)
    out = {}
    for c in cells:
        ce, cl, per = [], [], {}
        for b in lb:
            a = M.tsv_read(OUT / 'per_clip' / f'dev_{b}_{c}.tsv')
            r = M.tsv_read(OUT / 'per_clip' / f'dev_{b}_base.tsv')
            d_ce = float(np.nanmean(M.arr(a, dev, 'CE') - M.arr(r, dev, 'CE')))
            ce.append(d_ce)
            cl.append(float(np.nanmean(M.arr(a, dev, 'CLAP') - M.arr(r, dev, 'CLAP'))))
            per[b] = d_ce
        out[c] = {'dCE': float(np.mean(ce)), 'dCLAP': float(np.mean(cl)), 'per_base_dCE': per}
    return out


def stage2_defs(smoke):
    """freeze stage-2 spec from S1 dev results (written once)."""
    p = OUT / 'stage2_spec.json'
    if p.is_file():
        return json.loads(p.read_text())
    s1 = dev_dce([c for c in S1_ORDER if c not in ('base', 'synonly')], smoke)
    bp = max(PNO_C, key=lambda c: s1[c]['dCE'])
    bs = max(SYN_C, key=lambda c: s1[c]['dCE'])
    bf = max(FILT_C, key=lambda c: s1[c]['dCE'])
    top = bp if s1[bp]['dCE'] >= s1[bs]['dCE'] else bs
    filt = S1_CELLS[bf]['filt']
    defs = {
        'stackA': {**S1_CELLS[top], 'filt': filt},
        'stackB': {**S1_CELLS[bp], **S1_CELLS[bs]},
        'stackC': {**S1_CELLS[bp], **S1_CELLS[bs], 'filt': filt},
    }
    if bs == 'synm12':
        defs['synm18'] = {'syn': -18}
    spec = {'best_pno': bp, 'best_syn': bs, 'best_filt': bf, 'stackA_arp': top, 'defs': defs,
            'dev_dCE_s1': {c: v['dCE'] for c, v in s1.items()}, 'written_before_stage2_scoring': True}
    P.jdump(p, spec)
    log(f'stage-2 spec frozen: {json.dumps({k: v for k, v in spec.items() if k not in ("defs", "dev_dCE_s1")})}')
    return spec


def select(smoke):
    p = OUT / 'selection.json'
    if p.is_file():
        return json.loads(p.read_text())
    spec = json.loads((OUT / 'stage2_spec.json').read_text())
    cells = [c for c in S1_ORDER if c not in ('base', 'synonly')] + list(spec['defs'])
    dv = dev_dce(cells, smoke)
    star_any = max(cells, key=lambda c: dv[c]['dCE'])
    keep = [c for c in cells if dv[c]['dCLAP'] >= KEEP_DCLAP]
    star_keep = max(keep, key=lambda c: dv[c]['dCE']) if keep else max(cells, key=lambda c: dv[c]['dCLAP'])
    sel = {'kind': SEL_KIND, 'seed': SEED, 'bases_used': lbases(smoke), 'star_any': star_any, 'star_keep': star_keep,
           'star_keep_rule': f'dCLAP >= {KEEP_DCLAP}; fallback max dCLAP if none',
           'star_keep_fallback': not keep,
           'best_pno': spec['best_pno'], 'best_syn': spec['best_syn'], 'best_filt': spec['best_filt'],
           'dev_by_cell': dv, 'cell_defs': {**{c: S1_CELLS[c] for c in S1_ORDER}, **spec['defs']},
           'written_before_test_aggregates': True}
    P.jdump(p, sel)
    log(f'selection frozen: star_any={star_any} star_keep={star_keep} best_pno={spec["best_pno"]} '
        f'best_syn={spec["best_syn"]} best_filt={spec["best_filt"]}')
    return sel


def test_cells(sel):
    out = []
    for c in (sel['star_any'], sel['star_keep'], sel['best_pno'], sel['best_filt'], 'synonly'):
        if c not in out:
            out.append(c)
    return out


def run_test(smoke, caps_all, sel):
    _, test = splits(smoke)
    caps = {i: caps_all[i] for i in test}
    cells = test_cells(sel)
    defs = {c: sel['cell_defs'][c] for c in cells}
    for b in bases(smoke):
        have_base = test_base(smoke, b, test)
        todo = [c for c in cells if not done(f'test_{b}_{c}', test, 2)] + ([] if have_base else ['base'])
        if not todo:
            continue
        wavs = M.load_base(b, test)
        bl = builders(b, wavs, {'base': {}, **defs})
        for c in todo:
            run_cell(f'test_{b}_{c}', test, bl[c], caps, 2)
        del wavs


def sanity(smoke, base_wavs):
    """stim sanity: synth is finite, locked, in key; writes two example wavs."""
    import soundfile as sf
    d = OUT / 'sanity'
    d.mkdir(parents=True, exist_ok=True)
    rows, bad = {}, []
    hit = 0
    for (t, m, r), x in SYN.items():
        if r is not None:
            continue
        L = float(P.to_lufs(np.asarray(x, np.float64), -30.0)[1]) if False else None  # noqa
        k = C.key_of(x)
        hit += int(k == (t, m))
        if not np.isfinite(x).all() or float(np.max(np.abs(x))) > 1.0:
            bad.append((t, m))
    rows['synth_key_recovered'] = hit
    rows['n_keys'] = 24
    rows['bad_finite_or_clip'] = bad
    for name, (t, m) in (('synonly_Cmaj', (0, 0)), ('synonly_Amin', (9, 1))):
        sf.write(d / f'{name}.wav', SYN[(t, m, None)], SR)
    P.jdump(d / 'sanity.json', rows)
    log(f'sanity: synth key recovered {hit}/24, non-finite/clipping {bad}')
    return rows


def run_all(smoke):
    global X
    setup(smoke)
    C.EXC = json.loads(C.EXCERPTS.read_text()) if C.EXCERPTS.is_file() else None
    get_ids(smoke)
    splits(smoke)
    caps_all = P.caps_all()
    P.progress('start', '', 0, 0)
    pool = mp.get_context('fork').Pool(P.NTHREAD)         # forked before CUDA / scorers exist
    build_pno(pool)
    pool.close()
    build_syn(e00_params())
    sanity(smoke, None)
    X = C.Z()
    X.load_all()
    C.X = X
    log(f'scorers loaded; torch max alloc {P.SC.torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    run_dev(smoke, caps_all, S1_CELLS)
    spec = stage2_defs(smoke)
    run_dev(smoke, caps_all, spec['defs'])
    sel = select(smoke)
    run_test(smoke, caps_all, sel)
    P.jdump(OUT / 'gpu_peak.json', {'torch_max_alloc_gib': P.SC.torch.cuda.max_memory_allocated() / 2**30,
                                    'torch_max_reserved_gib': P.SC.torch.cuda.max_memory_reserved() / 2**30,
                                    'nvidia_smi_used_max_mib': X.peak_smi})
    P.progress('scored', '', 0, 0)
    project(smoke)


def project(smoke):
    tp = OUT / 'timing.json'
    if not tp.is_file():
        return None
    t = json.loads(tp.read_text())

    def rate(prefix):
        v = [x for k, x in t['keys'].items() if k.startswith(prefix) and x['n'] and not k.endswith('_base')]
        n = sum(x['n'] for x in v)
        return (sum(x['wall_s'] for x in v) / n) if n else None
    rd, rt = rate('dev_'), rate('test_')
    n_dev, n_test, nb = 200, 800, 9
    est = {'dev_S1': (rd * (14 * nb * n_dev + 3 * n_dev) if rd else None),
           'dev_S2': (rd * 4 * nb * n_dev if rd else None),
           'test': (rt * 5 * nb * n_test if rt else None)}
    tot = sum(v for v in est.values() if v) / 3600
    rec = {'per_clip_s': {'dev_tier1': rd, 'test_tier2': rt}, 'est_stage_hours': {k: (v / 3600 if v else None) for k, v in est.items()},
           'projected_total_hours': tot,
           'note': 'smoke timings include warm-up / small-chunk overhead, so this over-estimates; 195 D bases reused (dev 6/9, test 9/9)'}
    P.jdump(OUT / 'projection.json', rec)
    log('PROJECTION stage hours: ' + json.dumps({k: (round(v / 3600, 2) if v else None) for k, v in est.items()}))
    log(f'PROJECTION full run: {tot:.2f} h')
    return rec


# ═════════ summary ═════════
class Boot:
    def __init__(self, n):
        self.n = n
        self.idx = np.random.default_rng(SEED).integers(0, n, (NB, n))

    def ci(self, d):
        d = np.asarray(d, float)
        if not np.isfinite(d).all():
            d = np.where(np.isfinite(d), d, np.nan)
            m = np.isfinite(d)
            if not m.any():
                return None
            d = d[m]
            idx = np.random.default_rng(SEED).integers(0, len(d), (NB, len(d)))
        else:
            idx = self.idx
        b = d[idx].mean(1)
        return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(d))]


def reading(n):
    return 'pushable' if n >= 5 else ('base_dependent' if n >= 2 else 'not_pushable')


def summarize(smoke):
    setup(smoke)
    dev, test = splits(smoke)
    sel = select(smoke)
    cells = test_cells(sel)
    S_CE = H.S['CE_lvl30']
    m_ce = M_CE_FRAC * S_CE
    lb, allb = lbases(smoke), bases(smoke)
    bt = Boot(len(test))
    rr = M.tsv_read(OUT / 'per_clip' / 'test_real_base.tsv')
    real_ce = M.arr(rr, test, 'CE')
    real_med, real_p90 = float(np.median(real_ce)), float(np.percentile(real_ce, 90))
    out = {'kind': KIND, 'smoke': smoke, 'boot': {'NB': NB, 'seed': SEED}, 'S_CE': S_CE, 'm_CE': m_ce, 'n_test': len(test),
           'selection': {k: v for k, v in sel.items() if k not in ('dev_by_cell', 'cell_defs')},
           'dev_by_cell': sel['dev_by_cell'], 'cell_defs': sel['cell_defs'],
           'real_base_test_CE': {'mean': float(real_ce.mean()), 'median': real_med, 'p90': real_p90},
           'listenable_bases': lb, 'bases': {}}
    for b in allb:
        base = M.tsv_read(OUT / 'per_clip' / f'test_{b}_base.tsv')
        rec = {'base_CE_mean': float(M.arr(base, test, 'CE').mean())}
        for c in cells:
            a = M.tsv_read(OUT / 'per_clip' / f'test_{b}_{c}.tsv')
            ce = M.arr(a, test, 'CE')
            delta = {}
            for col in DELTA:
                ci = bt.ci(M.arr(a, test, col) - M.arr(base, test, col))
                if ci is not None:
                    delta[col] = ci
            rec[c] = {'delta': delta, 'CE_after_mean': float(ce.mean()),
                      'frac_above_real_median': float((ce > real_med).mean()), 'frac_above_real_p90': float((ce > real_p90).mean()),
                      'success': bool(delta['CE'][1] > m_ce)}
        out['bases'][b] = rec
    head = {}
    for role in ('star_any', 'star_keep', 'best_pno', 'best_filt'):
        c = sel[role]
        who = [b for b in lb if out['bases'][b][c]['success']]
        head[role] = {'cell': c, 'n_success_of_listenable': len(who), 'n_listenable': len(lb), 'bases': who, 'reading': reading(len(who)),
                      'mean_CE_after_listenable': float(np.mean([out['bases'][b][c]['CE_after_mean'] for b in lb])),
                      'mean_dCE_listenable': float(np.mean([out['bases'][b][c]['delta']['CE'][0] for b in lb]))}
    out['headline'] = head
    out['how_high'] = {'star_any_CE_after_mean_listenable': head['star_any']['mean_CE_after_listenable'],
                       'real_base_CE_mean': float(M.arr(rr, test, 'CE').mean()), 'e00_195_CE': 8.0898,
                       'synonly_CE_after_mean_listenable': float(np.mean([out['bases'][b]['synonly']['CE_after_mean'] for b in lb]))}
    out['verdicts'] = {'star_any': f'{head["star_any"]["cell"]}: {head["star_any"]["n_success_of_listenable"]}/{len(lb)} -> {head["star_any"]["reading"]}',
                       'star_keep': f'{head["star_keep"]["cell"]}: {head["star_keep"]["n_success_of_listenable"]}/{len(lb)} -> {head["star_keep"]["reading"]}'}
    P.jdump(OUT / 'summary.json', out)
    P.atomic(OUT / 'summary_table.md', table_md(out))
    log('summary written; verdicts: ' + json.dumps(out['verdicts'], ensure_ascii=False))
    return out


def fmt(ci):
    return 'n/a' if not ci else f'{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]'


def table_md(o):
    sel = o['selection']
    L = [f'# 196 summary ({"smoke" if o["smoke"] else "full"}; S_CE = {o["S_CE"]:.4f}, m_CE = {o["m_CE"]:.4f}; bootstrap {NB}, seed {SEED}; test n = {o["n_test"]})', '',
         '## Verdicts', '']
    L += [f'- {k}: **{v}**' for k, v in o['verdicts'].items()]
    hh = o['how_high']
    L += ['', f'how high: star_any mean CE on listenable bases {hh["star_any_CE_after_mean_listenable"]:.3f}; real base mean {hh["real_base_CE_mean"]:.3f}; '
              f'e00 (195) {hh["e00_195_CE"]}; synonly {hh["synonly_CE_after_mean_listenable"]:.3f}', '',
          f'frozen: star_any = {sel["star_any"]}, star_keep = {sel["star_keep"]}, best_pno = {sel["best_pno"]}, best_syn = {sel["best_syn"]}, best_filt = {sel["best_filt"]}', '',
          '## Dev (pooled over listenable bases)', '', '| cell | dCE | dCLAP |', '|---|---|---|']
    for c, v in sorted(o['dev_by_cell'].items(), key=lambda kv: -kv[1]['dCE']):
        L.append(f'| {c} | {v["dCE"]:+.3f} | {v["dCLAP"]:+.3f} |')
    L += ['', '## Test, per base', '',
          '| base | cell | success | CE after | dCE | > real median | > real p90 | dPQ | dCU | dPC | dA1_MI | dCLAP | dPAM | dCMI_q |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for b, rec in o['bases'].items():
        for c, e in rec.items():
            if c == 'base_CE_mean':
                continue
            d = e['delta']
            L.append(f'| {b} | {c} | {e["success"]} | {e["CE_after_mean"]:.3f} | {fmt(d.get("CE"))} | {e["frac_above_real_median"]:.2f} | '
                     f'{e["frac_above_real_p90"]:.2f} | {fmt(d.get("PQ"))} | {fmt(d.get("CU"))} | {fmt(d.get("PC"))} | {fmt(d.get("A1_MI"))} | '
                     f'{fmt(d.get("CLAP"))} | {fmt(d.get("PAM"))} | {fmt(d.get("CMI_q"))} |')
    L += ['', '## Headline', '']
    for k, h in o['headline'].items():
        L.append(f'- {k} ({h["cell"]}): {h["n_success_of_listenable"]}/{h["n_listenable"]} -> {h["reading"]}; '
                 f'mean CE after {h["mean_CE_after_listenable"]:.3f}, mean dCE {h["mean_dCE_listenable"]:+.3f}')
    return '\n'.join(L) + '\n'


# ═════════ validate / preflight ═════════
def validate(smoke):
    errs = []
    try:
        setup(smoke)
        dev, test = splits(smoke)
        for f in ('split.json', 'stage2_spec.json', 'selection.json', 'summary.json', 'summary_table.md'):
            if not (OUT / f).is_file():
                errs.append(f'missing {f}')

        def chk(key, ids, tier):
            p = OUT / 'per_clip' / f'{key}.tsv'
            if not p.is_file():
                errs.append(f'missing cell {p.name}')
            elif not M.rows_ok(M.tsv_read(p), ids, Q.req_cols(tier, True, True)):
                errs.append(f'cell {p.name}: id set or required columns not finite')
        if (OUT / 'selection.json').is_file() and (OUT / 'stage2_spec.json').is_file():
            sel = json.loads((OUT / 'selection.json').read_text())
            spec = json.loads((OUT / 'stage2_spec.json').read_text())
            for b in bases(smoke):
                for c in list(S1_CELLS) + list(spec['defs']):
                    chk(f'dev_{b}_{c}', dev, 1)
                for c in ['base'] + test_cells(sel):
                    chk(f'test_{b}_{c}', test, 2)
        if (OUT / 'summary.json').is_file():
            s = json.loads((OUT / 'summary.json').read_text())
            if s.get('kind') != KIND:
                errs.append('summary.json wrong kind')
            bad = C._finite_json(s)
            if bad:
                errs.append(f'non-finite numbers in summary.json, e.g. {bad[:3]}')
            for k in ('star_any', 'star_keep'):
                if not s.get('verdicts', {}).get(k):
                    errs.append(f'verdict {k} missing')
            if (OUT / 'selection.json').stat().st_mtime > (OUT / 'summary.json').stat().st_mtime:
                errs.append('selection.json newer than summary.json')
    except SystemExit as e:
        errs.append(str(e))
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    if errs:
        for e in errs[:30]:
            log(f'VALIDATE FAIL {e}')
        log(f'VALIDATE FAIL {len(errs)} problem(s)')
        return 1
    log('VALIDATE OK')
    return 0


def preflight(smoke):
    errs = []
    H.EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(H.EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    for p in (H.PARTB / 'part_b_selection.json', H.TTM115, H.MC_TSV, P.REF_DIR, H.PY, OUT195 / 'e_top10.json',
              OUT195 / 'split.json', C.MS.BANKS['generaluser']):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    try:
        mc, win = get_ids(smoke)
        want = 16 if smoke else 1000
        if len(mc) != want:
            errs.append(f'{len(mc)} ids with reference audio, expected {want}')
        for b in bases(smoke):
            if b == 'real':
                continue
            miss = [i for i in mc if not M.base_path(b, i).is_file()]
            if miss:
                errs.append(f'base {b}: {len(miss)} missing audio files, e.g. {miss[0]}')
        if not smoke:
            old = json.loads((OUT195 / 'split.json').read_text())
            if old != C.make_split(mc):
                errs.append('195 split.json differs from the split 196 would build')
        import importlib
        for mod in ('tinysoundfont', 'soxr', 'librosa', 'pyloudnorm', 'soundfile'):
            importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if errs:
        return 1
    log(f'PREFLIGHT OK (smoke={smoke})')
    if not (OUT / 'summary.json').is_file():
        need_mem = 8 if smoke else 20
        if C.mem_available_gb() < need_mem:
            log(f'PREFLIGHT WAIT MemAvailable {C.mem_available_gb():.1f} GiB < {need_mem} GiB')
            return 75
        f = P.free_mib()
        if f < 12 * 1024:
            log(f'PREFLIGHT WAIT GPU free {f} MiB < {12 * 1024} MiB')
            return 75
    return 0


def main():
    global OUT, IS_SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--root', help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.chdir(REPO)
    if a.smoke:
        IS_SMOKE = True
        OUT = H.EVAL_ROOT / 'ce_push_196_smoke'
    if a.root:
        OUT = Path(a.root)
    if a.preflight:
        return preflight(a.smoke)
    if a.validate_only:
        return validate(a.smoke)
    if a.summary_only:
        summarize(a.smoke)
        return validate(a.smoke)
    rc = preflight(a.smoke)
    if rc:
        return rc
    run_all(a.smoke)
    summarize(a.smoke)
    return validate(a.smoke)


if __name__ == '__main__':
    sys.exit(main())
