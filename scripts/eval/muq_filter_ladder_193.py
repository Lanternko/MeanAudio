#!/usr/bin/env python
"""193: MuQ A1 filter ladder (eval-only driver).

Design: docs/experiments/muq_filter_ladder_193_20261113.md (pre-registered; this file only implements it).

Cells (all at 16 kHz, 10 s, renormalised to -30 LUFS after filtering; stimuli built on the fly, never stored):
  base                                      the base itself
  lp{7,6,5,4,3,2,1}k                        Butterworth order 8 low-pass, sosfiltfilt (muq_lowpass_probe_20261028.lp)
  hp{100,200,500}                           Butterworth order 4 high-pass, sosfiltfilt
  eq{125,250,500,1k,2k,4k}_{p10,m10}        RBJ peaking biquad, Q = sqrt(2), gain +-10 dB, sosfilt (one direction)
Tier 3 only on base / lp6k / lp4k / lp2k / hp200, tier 2 elsewhere.

Reuses 190 (Y scorer class, lvl_sr, analyse, req_cols) and 189 (loaders, tsv, M.XScorer) unchanged; neither is edited.
Dev/test split: seed 20261113, 200 of 1000 Part B ids dev (musdb ~20% of tracks dev).  lp_star / oneliner_star are chosen on
dev only and frozen in selection.json before any test aggregate is computed.  Everything is resumable per cell TSV.
"""
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muq_overlay_matrix_189 as M  # noqa: E402
import muq_postproc_attack_190 as Q  # noqa: E402

P = M.P
H = M.H
REPO = P.REPO
SR, NS = P.SR, P.NS
OUT = H.EVAL_ROOT / 'muq_filter_ladder_193'
KIND = 'muq_filter_ladder_193_summary_v1'
SEL_KIND = 'muq_filter_ladder_193_selection_v1'
IS_SMOKE = False
SEED = 20261113
NB = 5000
M_A1 = M.M_A1
BASES, MC_BASES, SEEDS = M.BASES, M.MC_BASES, M.SEEDS
LISTEN = ['real'] + [f'ma_cfg3neg_s{s}' for s in SEEDS] + ['musicgen', 'musicldm']
SMOKE_BASES = ['real', 'ma_cfg3neg_s14159265', 'musicgen', 'musdb']
MIN_FREE = 20 * 1024**3
DELTA_COLS, MUQ_COLS, AES_COLS, SCORE_COLS = M.DELTA_COLS, M.MUQ_COLS, M.AES_COLS, M.SCORE_COLS

LP_FC = [7000, 6000, 5000, 4000, 3000, 2000, 1000]
HP_FC = [100, 200, 500]
EQ_F = [('125', 125), ('250', 250), ('500', 500), ('1k', 1000), ('2k', 2000), ('4k', 4000)]
LP_CELLS = [f'lp{fc // 1000}k' for fc in LP_FC]
HP_CELLS = [f'hp{fc}' for fc in HP_FC]
EQ_CELLS = [f'eq{n}_{s}' for n, _ in EQ_F for s in ('p10', 'm10')]
ONELINER = HP_CELLS + EQ_CELLS
TIER3 = {'base', 'lp6k', 'lp4k', 'lp2k', 'hp200'}
CELLS = {'base': (3, SR, True)}
for _c in LP_CELLS + HP_CELLS + EQ_CELLS:
    CELLS[_c] = (3 if _c in TIER3 else 2, SR, True)
LP_HZ = dict(zip(LP_CELLS, LP_FC))
EQ_HZ = {f'eq{n}_{s}': (f0, 10.0 if s == 'p10' else -10.0) for n, f0 in EQ_F for s in ('p10', 'm10')}

X = Q.Y()


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def cell_key(base, ov):
    return f'T_{base}_{ov}'


def bases_of(smoke):
    return SMOKE_BASES if smoke else BASES


def get_ids(smoke):
    return Q.get_ids(smoke)


# ── filters ──────────────────────────────────────────────
def hp_sos(fc, sr=SR):
    from scipy.signal import butter
    return butter(4, fc, 'high', fs=sr, output='sos')


def eq_sos(f0, gain_db, sr=SR, q=math.sqrt(2)):
    """RBJ cookbook peaking EQ."""
    from scipy.signal import tf2sos
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / sr
    al = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b = np.array([1 + al * A, -2 * cw, 1 - al * A])
    a = np.array([1 + al / A, -2 * cw, 1 - al / A])
    return tf2sos(b, a)


def apply_cell(w, ov):
    from scipy.signal import sosfilt, sosfiltfilt
    w = np.asarray(w, np.float64)
    if ov == 'base':
        return np.asarray(w, np.float32)
    if ov in LP_HZ:
        import muq_lowpass_probe_20261028 as LPm
        y = LPm.lp(w, SR, LP_HZ[ov])
    elif ov in HP_CELLS:
        y = sosfiltfilt(hp_sos(int(ov[2:])), w)
    elif ov in EQ_HZ:
        f0, g = EQ_HZ[ov]
        y = sosfilt(eq_sos(f0, g), w)
    else:
        raise KeyError(ov)
    return Q.lvl_sr(y, SR)


def builder(ov, wavs):
    return lambda i: apply_cell(wavs[i], ov)


def filter_response():
    """Measured magnitude (dB) of every filter at test frequencies; written once to filter_response.json."""
    from scipy.signal import butter, sosfreqz
    fs = [62.5, 125, 250, 500, 1000, 2000, 4000, 6000, 7000]
    rec = {}
    for ov, fc in LP_HZ.items():
        s = butter(8, fc, 'low', fs=SR, output='sos')
        w, h = sosfreqz(s, worN=[fc, 2 * fc if 2 * fc < 8000 else 7999], fs=SR)
        rec[ov] = {'single_pass_db_at_fc': float(20 * np.log10(abs(h[0]))), 'filtfilt_db_at_fc': float(40 * np.log10(abs(h[0])))}
    for ov in HP_CELLS:
        fc = int(ov[2:])
        w, h = sosfreqz(hp_sos(fc), worN=[fc], fs=SR)
        rec[ov] = {'single_pass_db_at_fc': float(20 * np.log10(abs(h[0]))), 'filtfilt_db_at_fc': float(40 * np.log10(abs(h[0])))}
    for ov, (f0, g) in EQ_HZ.items():
        fr = [f0, f0 * 4 if f0 * 4 < 8000 else None, f0 / 4]
        fr = [f for f in fr if f]
        w, h = sosfreqz(eq_sos(f0, g), worN=fr, fs=SR)
        rec[ov] = {'gain_db_at_f0': float(20 * np.log10(abs(h[0]))),
                   'gain_db_two_octaves_away': [float(20 * np.log10(abs(x))) for x in h[1:]]}
    return rec


# ── split ────────────────────────────────────────────────
def make_split(mc, win):
    rng = np.random.default_rng(SEED)
    ids = sorted(mc)
    n_dev = 200 if len(ids) == 1000 else max(2, int(round(0.2 * len(ids))))
    dev = set(np.array(ids)[rng.permutation(len(ids))[:n_dev]].tolist())
    tmap = M.track_map()
    tracks = sorted({tmap[i] for i in win})
    nt = max(1, int(round(0.2 * len(tracks))))
    dev_tr = set(np.array(tracks)[np.random.default_rng(SEED + 1).permutation(len(tracks))[:nt]].tolist())
    sp = {'seed': SEED, 'dev_mc': [i for i in mc if i in dev], 'test_mc': [i for i in mc if i not in dev],
          'dev_win': [i for i in win if tmap[i] in dev_tr], 'test_win': [i for i in win if tmap[i] not in dev_tr],
          'dev_tracks': sorted(dev_tr)}
    return sp


def get_split(mc, win):
    sp = make_split(mc, win)
    p = OUT / 'split.json'
    if not p.is_file():
        P.jdump(p, sp)
    return sp


# ── run ──────────────────────────────────────────────────
def setup(smoke):
    M.OUT = OUT
    P.OUT = OUT
    Q.OUT = OUT
    P.IS_SMOKE = smoke
    P.CFG['chunk'] = 16 if smoke else 64
    P.POOL = ThreadPoolExecutor(P.NTHREAD)
    H.load_scales()
    OUT.mkdir(parents=True, exist_ok=True)


def run_cell(key, ids, build, caps, ov, cmi_mode):
    tier, sr, clap = CELLS[ov]
    path = OUT / 'per_clip' / f'{key}.tsv'
    req = Q.req_cols(tier, bool(caps), clap)
    if path.is_file():
        rows = M.tsv_read(path)
        if M.rows_ok(rows, ids, req):
            return rows
    t0, t_before, n_before = time.time(), Counter(X.t), Counter(X.n)
    ch = P.CFG['chunk']
    chunks = [ids[k:k + ch] for k in range(0, len(ids), ch)]
    res, over, mx = {}, 0, 0.0
    nxt = [P.POOL.submit(build, x) for x in chunks[0]]
    for k, c in enumerate(chunks):
        cur = nxt
        if k + 1 < len(chunks):
            nxt = [P.POOL.submit(build, x) for x in chunks[k + 1]]
        wavs = [f.result() for f in cur]
        for w in wavs:
            if not np.all(np.isfinite(w)):
                raise RuntimeError(f'{key}: non-finite stimulus')
            a = float(np.max(np.abs(w)))
            mx = max(mx, a)
            over += int(a > 1.0)
        rows = X.score(wavs, [caps[x] for x in c] if caps else None, tier, cmi_mode, sr=sr, clap=clap)
        for x, r in zip(c, rows):
            bad = [q for q in req if not math.isfinite(r.get(q, math.nan))]
            if bad:
                raise RuntimeError(f'{key}: non-finite {bad} for {x}: {r}')
            res[x] = r
        P.progress('test', key, len(res), len(ids))
    M.tsv_write(path, res, ids)
    wall = time.time() - t0
    dt = {k: X.t[k] - t_before.get(k, 0.0) for k in X.t}
    P.add_timing(key, len(ids), wall, dt)
    M.cost_add(dt, {k: X.n[k] - n_before.get(k, 0) for k in X.n})
    P.jdump(OUT / 'stim_stats' / f'{key}.json', {'n': len(ids), 'n_abs_over_1': over, 'max_abs': mx, 'sr': sr})
    log(f'{key}: tier {tier}, {len(ids)} stimuli in {wall:.1f}s ({len(ids) / wall:.2f}/s); {over} with |x|>1 (max {mx:.3f})')
    return res


def run_all(smoke):
    setup(smoke)
    mc, win = get_ids(smoke)
    get_split(mc, win)
    caps_all = P.caps_all()
    P.jdump(OUT / 'filter_response.json', filter_response())
    P.progress('start', '', 0, 0)
    X.load_all()
    log(f'all scorers loaded; torch max alloc {P.SC.torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    for base in bases_of(smoke):
        ids = win if base == 'musdb' else mc
        wavs = M.load_base(base, ids)
        caps = None if base == 'musdb' else {i: caps_all[i] for i in ids}
        cmi_mode = 'empty' if base == 'musdb' else 'caption'
        for ov in CELLS:
            run_cell(cell_key(base, ov), ids, builder(ov, wavs), caps, ov, cmi_mode)
        del wavs
    P.jdump(OUT / 'gpu_peak.json', {'torch_max_alloc_gib': P.SC.torch.cuda.max_memory_allocated() / 2**30,
                                    'torch_max_reserved_gib': P.SC.torch.cuda.max_memory_reserved() / 2**30,
                                    'nvidia_smi_used_max_mib': X.peak_smi})
    P.progress('scored', '', 0, 0)
    project(smoke)


def project(smoke):
    tp, cp = OUT / 'timing.json', OUT / 'cost.json'
    if not tp.is_file():
        return None
    t = json.loads(tp.read_text())
    per_ov = {}
    for k, v in t['keys'].items():
        if not k.startswith('T_'):
            continue
        for ov in CELLS:
            if k.endswith('_' + ov):
                per_ov.setdefault(ov, []).append(v['wall_s'] / v['n'])
    wall = sum(float(np.mean(per_ov[o])) * (9 * 1000 + 551) for o in CELLS if o in per_ov)
    cost = json.loads(cp.read_text()) if cp.is_file() else {'t': {}, 'n': {}}
    per_scorer = {k: cost['t'][k] / cost['n'][k] for k in cost['t'] if cost['n'].get(k)}
    rec = {'per_cell_type_s_per_clip': {o: float(np.mean(v)) for o, v in per_ov.items()},
           'per_scorer_s_per_clip': per_scorer, 'projected_total_hours': wall / 3600,
           'note': 'full = 10 bases (9 MusicCaps x 1000 + 551 musdb) per cell; smoke includes warm-up, so over-estimates'}
    P.jdump(OUT / 'projection.json', rec)
    log('PROJECTION per-scorer: ' + ', '.join(f'{k} {v * 1000:.1f}ms' for k, v in per_scorer.items()))
    log(f'PROJECTION full run: {rec["projected_total_hours"]:.1f} h')
    return rec


# ── summary ──────────────────────────────────────────────
class Boot:
    def __init__(self, clusters):
        _, self.k = np.unique(np.asarray(clusters), return_inverse=True)
        self.K = int(self.k.max()) + 1
        self.idx = np.random.default_rng(SEED).integers(0, self.K, (NB, self.K))

    def ci(self, d):
        d = np.asarray(d, float)
        m = np.isfinite(d)
        if m.sum() == 0:
            return None
        s = np.bincount(self.k[m], weights=d[m], minlength=self.K)
        cn = np.bincount(self.k[m], minlength=self.K).astype(float)
        den = cn[self.idx].sum(1)
        ok = den > 0
        b = s[self.idx].sum(1)[ok] / den[ok]
        return [float(d[m].mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(m.sum())]


def select(cells, dev_mc, mc_bases):
    """Dev-only selection (section 2.2).  Written to selection.json BEFORE any test aggregate exists; frozen if present."""
    p = OUT / 'selection.json'
    if p.is_file():
        return json.loads(p.read_text())

    def dm(ov, col):
        v = []
        for b in mc_bases:
            d = M.arr(cells[(b, ov)], dev_mc, col) - M.arr(cells[(b, 'base')], dev_mc, col)
            v.append(float(np.nanmean(d)))
        return float(np.mean(v))

    a1 = {ov: dm(ov, 'A1_MI') for ov in CELLS if ov != 'base'}
    ce = {ov: dm(ov, 'CE') for ov in CELLS if ov != 'base'}
    lp_star = max(LP_CELLS, key=lambda o: a1[o])
    cand = [o for o in ONELINER if ce[o] < 0]
    one = max(cand if cand else ONELINER, key=lambda o: a1[o])
    top3 = sorted(a1, key=lambda o: -a1[o])[:3]
    sel = {'kind': SEL_KIND, 'seed': SEED, 'n_dev_mc': len(dev_mc), 'bases_used': mc_bases, 'lp_star': lp_star,
           'oneliner_star': one, 'oneliner_has_CE_down': bool(cand), 'dev_top3_dA1': top3,
           'dev_mean_dA1': a1, 'dev_mean_dCE': ce, 'written_before_test_aggregates': True}
    P.jdump(p, sel)
    log(f'selection frozen: lp_star={lp_star} oneliner_star={one} top3={top3}')
    return sel


def e1_reading(n):
    return 'established' if n >= 5 else ('base_dependent' if n >= 2 else 'not_established')


def summarize(smoke):
    setup(smoke)
    mc, win = get_ids(smoke)
    sp = get_split(mc, win)
    dev_mc, test_mc, test_win = sp['dev_mc'], sp['test_mc'], sp['test_win']
    bases = bases_of(smoke)
    tmap = M.track_map()
    S_CE = H.S['CE_lvl30']
    S_PQ = H.S['PQ_lvl30']
    m_pq = 0.10 * S_PQ
    cells = {}
    for b in bases:
        for ov in CELLS:
            p = OUT / 'per_clip' / f'{cell_key(b, ov)}.tsv'
            if p.is_file():
                cells[(b, ov)] = M.tsv_read(p)
    mc_bases = [b for b in bases if b != 'musdb']
    sel = select(cells, dev_mc, mc_bases)        # dev only; frozen before the test aggregates below
    listen = [b for b in LISTEN if b in bases]

    def bmean(b, c):
        v = M.arr(cells[(b, 'base')], mc, c)
        return float(np.nanmean(v)) if np.isfinite(v).any() else math.nan

    S = {}
    if smoke:
        have = [b for b in mc_bases if (b, 'base') in cells]
        for c in DELTA_COLS:
            g = [bmean(b, c) for b in have]
            S[c] = float(max(g) - min(g)) if all(math.isfinite(x) for x in g) else None
    else:
        for c in DELTA_COLS:
            g = [bmean('real', c), bmean('musicgen', c), bmean('musicldm', c),
                 float(np.mean([bmean(f'ma_cfg0_s{s}', c) for s in SEEDS])),
                 float(np.mean([bmean(f'ma_cfg3neg_s{s}', c) for s in SEEDS]))]
            S[c] = float(max(g) - min(g)) if all(math.isfinite(x) for x in g) else None
    out = {'kind': KIND, 'smoke': smoke, 'M_A1': M_A1, 'S_CE': S_CE, 'S_PQ_H': S_PQ, 'm_PQ': m_pq,
           'S_tie_from_bases': S, 'boot': {'NB': NB, 'seed': SEED}, 'n_test_mc': len(test_mc),
           'n_test_win': len(test_win), 'selection': sel, 'listenable_bases': listen, 'bases': {}}
    boots = {}
    for b in bases:
        ids = test_win if b == 'musdb' else test_mc
        boots[b] = Boot([tmap[i] for i in ids] if b == 'musdb' else list(range(len(ids))))
        if (b, 'base') not in cells:
            continue
        bt = boots[b]
        rec = {'n_test': len(ids), 'cells': {}}
        for ov in CELLS:
            if ov == 'base' or (b, ov) not in cells:
                continue
            e = Q.analyse(cells, b, ids, bt, S, S_CE, ov, 'base', None)
            d = e.get('delta', {})
            a1, a3 = d.get('A1_MI'), d.get('A3a_MI')
            e['A3a_transfer'] = (float(a3[0] / a1[0]) if (a1 and a3 and a1[0] > 0.01) else None)
            pq, ce = d.get('PQ'), d.get('CE')
            e['pq_success'] = bool(pq and ce and pq[1] >= m_pq and ce[2] < 0)
            rec['cells'][ov] = e
        out['bases'][b] = rec

    def cellrec(b, ov):
        return out['bases'].get(b, {}).get('cells', {}).get(ov)

    # E1
    e1 = {}
    for tag in ('lp_star', 'oneliner_star'):
        ov = sel[tag]
        who = [b for b in listen if (cellrec(b, ov) or {}).get('verdict_v3') == 'success']
        mr = cellrec('musdb', ov)
        md = None
        if mr:
            a1, ce = mr['delta'].get('A1_MI'), mr['delta'].get('CE')
            md = bool(a1 and ce and a1[1] > 0 and ce[2] < 0)
        e1[tag] = {'cell': ov, 'n_success_of_listenable': len(who), 'n_listenable': len(listen), 'bases': who,
                   'reading': e1_reading(len(who)), 'musdb_dA1_lo_gt0_and_dCE_hi_lt0': md}
    # E2
    ls = sel['lp_star']
    if ls == 'lp6k':
        e2 = {'lp_star': ls, 'reading': '6k_is_top', 'note': 'lp_star == lp6k'}
    else:
        per, n_hi = {}, 0
        for b in listen:
            if (b, ls) not in cells or (b, 'lp6k') not in cells:
                continue
            ids = test_mc
            d = M.arr(cells[(b, ls)], ids, 'A1_MI') - M.arr(cells[(b, 'lp6k')], ids, 'A1_MI')
            ci = boots[b].ci(d)
            per[b] = ci
            n_hi += int(ci is not None and ci[1] > 0)
        e2 = {'lp_star': ls, 'per_base_diff_A1': per, 'n_lo_gt0': n_hi,
              'reading': 'higher_than_6k' if n_hi >= 4 else '6k_near_top'}
    # dose curves
    dose = {}
    for b in bases:
        cur = {}
        for ov in LP_CELLS:
            r = cellrec(b, ov)
            if r:
                cur[ov] = {c: r['delta'].get(c) for c in ('A1_MI', 'CE', 'PQ')}
        ok = [LP_HZ[ov] for ov in LP_CELLS if cur.get(ov, {}).get('A1_MI') and cur[ov]['A1_MI'][1] >= M_A1]
        dose[b] = {'curve': cur, 'highest_cutoff_hz_dA1_ge_m': (max(ok) if ok else None)}
    # per-cell counts over listenable bases + guards on success cells
    cellcount, guards = {}, []
    for ov in CELLS:
        if ov == 'base':
            continue
        rs = {b: cellrec(b, ov) for b in listen if cellrec(b, ov)}
        cellcount[ov] = {'n_success': sum(r.get('verdict_v3') == 'success' for r in rs.values()),
                         'n_pq_success': sum(bool(r.get('pq_success')) for r in rs.values()),
                         'n_a1_ge_m': sum(bool(r['delta'].get('A1_MI') and r['delta']['A1_MI'][1] >= M_A1) for r in rs.values()),
                         'mean_dA1': float(np.mean([r['delta']['A1_MI'][0] for r in rs.values() if r['delta'].get('A1_MI')]))
                         if rs else None,
                         'musdb_label': (cellrec('musdb', ov) or {}).get('verdict_v3')}
        for b in bases:
            r = cellrec(b, ov)
            if r and r.get('verdict_v3') == 'success':
                g = r.get('guards', {})
                guards.append({'base': b, 'cell': ov,
                               'PAM_ci_below_zero': g.get('PAM', {}).get('ci_below_zero'),
                               'PAM_beyond_tie_down': g.get('PAM', {}).get('beyond_tie_down'),
                               'CMI_q_ci_below_zero': g.get('CMI_q', {}).get('ci_below_zero'),
                               'CMI_q_beyond_tie_down': g.get('CMI_q', {}).get('beyond_tie_down'),
                               'independent_scorers_blind': (None if r.get('independent_unseen') is None
                                                             else bool(r['independent_unseen']))})
    out.update({'E1': e1, 'E2': e2, 'dose_curves': dose, 'cell_counts_listenable': cellcount,
                'guards_success_cells': guards,
                'note': 'scorer movements only; all judgments on test ids; selection on dev only'})
    P.jdump(OUT / 'summary.json', out)
    P.atomic(OUT / 'summary_table.md', table_md(out))
    log('summary written; E1: ' + json.dumps({k: (v['cell'], v['n_success_of_listenable'], v['reading']) for k, v in e1.items()})
        + f'; E2: {e2["reading"]}')
    return out


def fmt(ci):
    return 'n/a' if not ci else f'{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]'


def table_md(out):
    sel = out['selection']
    L = [f'# 193 summary ({"smoke" if out["smoke"] else "full"}; m = {M_A1}, S_CE = {out["S_CE"]:.4f}, '
         f'm_PQ = {out["m_PQ"]:.4f}; bootstrap {NB}, seed {SEED}; judged on test ids, musdb clustered by track)', '',
         f'Selection (dev only): lp_star = **{sel["lp_star"]}**, oneliner_star = **{sel["oneliner_star"]}** '
         f'(CE-down candidate exists: {sel["oneliner_has_CE_down"]}); dev top-3 dA1: {", ".join(sel["dev_top3_dA1"])}', '',
         '## E1 / E2', '', '| star | cell | success bases / listenable | reading | musdb |', '|---|---|---|---|---|']
    for t, v in out['E1'].items():
        L.append(f'| {t} | {v["cell"]} | {v["n_success_of_listenable"]} / {v["n_listenable"]} | {v["reading"]} | '
                 f'{v["musdb_dA1_lo_gt0_and_dCE_hi_lt0"]} |')
    L += ['', f'E2 (lp_star vs lp6k): **{out["E2"]["reading"]}**' + (f', bases with CI lower > 0: {out["E2"]["n_lo_gt0"]}'
                                                                   if 'n_lo_gt0' in out['E2'] else ''), '',
          '## Per-cell counts over listenable bases', '',
          '| cell | n success | n pq_success | n dA1>=m | mean dA1 | musdb |', '|---|---|---|---|---|---|']
    for ov, v in out['cell_counts_listenable'].items():
        L.append(f'| {ov} | {v["n_success"]} | {v["n_pq_success"]} | {v["n_a1_ge_m"]} | '
                 f'{"n/a" if v["mean_dA1"] is None else f"{v["mean_dA1"]:+.3f}"} | {v["musdb_label"]} |')
    L += ['', '## LP dose curves (dA1 / dCE / dPQ per base)', '',
          '| base | cell | dA1_MI | dCE | dPQ |', '|---|---|---|---|---|']
    for b, d in out['dose_curves'].items():
        for ov, c in d['curve'].items():
            L.append(f'| {b} | {ov} | {fmt(c["A1_MI"])} | {fmt(c["CE"])} | {fmt(c["PQ"])} |')
        L.append(f'| {b} | highest cutoff with dA1 CI lower >= m | {d["highest_cutoff_hz_dA1_ge_m"]} | | |')
    L += ['', '## Independent guards on success cells', '',
          '| base | cell | PAM<0 | PAM beyond tie | CMI<0 | CMI beyond tie | independent blind |', '|---|---|---|---|---|---|---|']
    for g in out['guards_success_cells']:
        L.append(f'| {g["base"]} | {g["cell"]} | {g["PAM_ci_below_zero"]} | {g["PAM_beyond_tie_down"]} | '
                 f'{g["CMI_q_ci_below_zero"]} | {g["CMI_q_beyond_tie_down"]} | {g["independent_scorers_blind"]} |')
    return '\n'.join(L) + '\n'


# ── validate / preflight ─────────────────────────────────
def validate(smoke):
    errs = []
    try:
        setup(smoke)
        mc, win = get_ids(smoke)
        for b in bases_of(smoke):
            ids = win if b == 'musdb' else mc
            for ov, (tier, sr, clap) in CELLS.items():
                p = OUT / 'per_clip' / f'{cell_key(b, ov)}.tsv'
                if not p.is_file():
                    errs.append(f'missing cell {p.name}')
                elif not M.rows_ok(M.tsv_read(p), ids, Q.req_cols(tier, b != 'musdb', clap)):
                    errs.append(f'cell {p.name}: id set or required columns not finite (tier {tier})')
        for f in ('split.json', 'selection.json', 'summary.json', 'summary_table.md', 'filter_response.json'):
            if not (OUT / f).is_file():
                errs.append(f'missing {f}')
        if (OUT / 'summary.json').is_file():
            s = json.loads((OUT / 'summary.json').read_text())
            if s.get('kind') != KIND:
                errs.append('summary.json wrong kind')
            if (OUT / 'selection.json').is_file() and \
                    (OUT / 'selection.json').stat().st_mtime > (OUT / 'summary.json').stat().st_mtime:
                errs.append('selection.json newer than summary.json')
            for k in ('E1', 'E2', 'dose_curves'):
                if k not in s:
                    errs.append(f'summary.json lacks {k}')
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
    if shutil.disk_usage(H.EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    for p in (H.PARTB / 'part_b_selection.json', H.TTM115, H.MC_TSV, P.REF_DIR, M.MUSDB_DIR / 'windows.json',
              M.MUSDB_DIR / 'cache', M.SE_CKPT, H.PY):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    try:
        mc, win = get_ids(smoke)
        want = 16 if smoke else 1000
        if len(mc) != want:
            errs.append(f'{len(mc)} Part B ids with reference audio, expected {want}')
        for b in (bases_of(smoke) if smoke else MC_BASES):
            if b in ('real', 'musdb'):
                continue
            miss = [i for i in mc if not M.base_path(b, i).is_file()]
            if miss:
                errs.append(f'base {b}: {len(miss)} missing audio files, e.g. {miss[0]}')
        miss = [i for i in win if not (M.MUSDB_DIR / 'cache' / f'{i}.npz').is_file()]
        if miss:
            errs.append(f'musdb: {len(miss)} missing cache files, e.g. {miss[0]}')
        if not smoke and len(win) != 551:
            errs.append(f'{len(win)} musdb windows, expected 551')
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if errs:
        return 1
    log(f'PREFLIGHT OK (smoke={smoke})')
    if not (OUT / 'summary.json').is_file():
        need_mem = 8 if smoke else 25
        if Q.mem_available_gb() < need_mem:
            log(f'PREFLIGHT WAIT MemAvailable {Q.mem_available_gb():.1f} GiB < {need_mem} GiB')
            return 75
        f = P.free_mib()
        if f < 20 * 1024:
            log(f'PREFLIGHT WAIT GPU free {f} MiB < {20 * 1024} MiB')
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
        OUT = H.EVAL_ROOT / 'muq_filter_ladder_193_smoke'
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
