#!/usr/bin/env python
"""198: AES CE overlay interpolation ladder (eval-only driver, AES scoring only).

Design: docs/experiments/ce_overlay_interp_198_20261119.md (pre-registered; this file only implements it).
Question: when something is overlaid on a clip, does post-overlay CE sit between the base and the overlay alone, is it
the exact middle, and does CE rise monotonically with the overlay's loudness?
Grid: 9 bases (196's MusicCaps bases) x the same 200 dev ids (196 split.json) x 8 overlays x r_rel in
{-24..+24 step 6} LU (positive = overlay louder than the base) + base + overlay-alone (-30 LUFS). Scored with AES only
(PQ/CE/CU/PC, 16 kHz, -30 LUFS after the mix, same call as 196).
Gate: e00 cells at r_rel in {-12,-6,0,6,12,18} must reproduce 196 dev_<base>_syn{12,6,0,m6,m12,m18}.tsv
(|dCE| per clip: median <= 2e-3, max <= 0.02), else the job fails.
Resumable from per_clip/*.tsv. Reuses 195/196/188/189 helpers by import (none of them is edited).
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
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ce_weakness_195 as C  # noqa: E402

M, P, H, Q = C.M, C.P, C.H, C.Q
SR, NS = C.SR, C.NS
REPO = C.REPO
OUT195 = H.EVAL_ROOT / 'ce_weakness_195'
OUT196 = H.EVAL_ROOT / 'ce_push_196'
OUT = H.EVAL_ROOT / 'ce_overlay_interp_198'
KIND = 'ce_overlay_interp_198_summary_v1'
SEED = 198
NB = 2000
MONO_TOL = 0.119           # = m_CE (0.10 * S_CE_lvl30)
MONO_TOL_PQ = 0.0914
GATE_MED, GATE_MAX = 2e-3, 0.02
MIN_FREE = 20 * 1024**3
AES_COLS = ['PQ', 'CE', 'CU', 'PC']
BASES_ALL = C.MC_BASES
SMOKE_BASES = ['real', 'musicgen']
SMOKE_N = 5
CH = 64

OVERLAYS = ['e00', 'pnoarp', 'pnoblock', 'sine1k', 'white', 'pink', 'music', 'drums']
KEY_OV = ('e00', 'pnoarp', 'pnoblock')
PITCHED = {'e00': 1, 'pnoarp': 1, 'pnoblock': 1, 'sine1k': 1, 'music': 1, 'white': 0, 'pink': 0, 'drums': 0}
RR = [-24, -18, -12, -6, 0, 6, 12, 18, 24]
GATE_MAP = {-12: 'syn12', -6: 'syn6', 0: 'syn0', 6: 'synm6', 12: 'synm12'}
INFO_MAP = {18: 'synm18'}   # 196 peak-capped (0.999) the overlay here; 198 does not -> reported, not gated
OVW = {}                   # name -> {subkey: (w64 at -30 LUFS, lufs, w32)}; subkey: (t, mode) | None | music id
X = None
IS_SMOKE = False


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def tag(r):
    return f'm{-r}' if r < 0 else f'p{r}'


def cell_key(b, o, r):
    return f'{b}_{o}_only' if r is None else f'{b}_{o}_{tag(r)}'


# ═════════ ids / bases ═════════
def get_ids(smoke):
    ids = json.loads((OUT196 / 'split.json').read_text())['dev_mc']
    return ids[:SMOKE_N] if smoke else ids


def bases(smoke):
    return SMOKE_BASES if smoke else BASES_ALL


def setup(smoke):
    C.OUT = OUT
    C.IS_SMOKE = smoke
    C.setup(smoke)
    P.CFG['chunk'] = CH


# ═════════ overlays ═════════
def _put(name, sub, w):
    w = np.asarray(w, np.float64)
    w32 = C.lock(w)
    w64 = np.asarray(w32, np.float64)
    OVW.setdefault(name, {})[sub] = (w64, float(P.lufs_of(w64)), w32)


def _job_block(key):
    t, mode = key
    iv = [0, 3 if mode else 4, 7, 12]
    root = 48 + t
    notes = [(1.0 * k, 1.0 * k + 0.9, root + v, 80) for k in range(10) for v in iv]
    x = C.MS.synth(notes, 'generaluser', 0, 0)
    return np.asarray(x[int(C.MS.LEAD * SR):int(C.MS.LEAD * SR) + 160000], np.float64)


def _rng(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def drums_wave():
    n, T = NS, NS / SR
    rn = _rng('drums_noise').standard_normal(n)
    out = np.zeros(n)
    tt = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 * tt + 75 * 0.03 * (1 - np.exp(-tt / 0.03)))) * np.exp(-tt / 0.05)
    snare = rn[:len(tt)] * np.exp(-tt / 0.04)
    tt2 = np.arange(int(0.05 * SR)) / SR
    hat = np.diff(rn[:len(tt2) + 1]) * np.exp(-tt2 / 0.01)

    def hit(pat, per, off, amp):
        for on in np.arange(off, T, per):
            i0 = int(on * SR)
            m = min(len(pat), n - i0)
            if m > 0:
                out[i0:i0 + m] += amp * pat[:m]
    hit(kick, 0.5, 0.0, 1.0)
    hit(snare, 1.0, 0.5, 0.7)
    hit(hat, 0.25, 0.0, 0.35)
    return out


def pink_wave():
    n = NS
    w = _rng('pink').standard_normal(n)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    g = 1.0 / np.sqrt(np.maximum(f, f[1]))
    return np.fft.irfft(np.fft.rfft(w) * g, n)


def build_fixed():
    t = np.arange(NS) / SR
    _put('sine1k', None, np.sin(2 * np.pi * 1000.0 * t))
    _put('white', None, _rng('white').standard_normal(NS))
    _put('pink', None, pink_wave())
    _put('drums', None, drums_wave())


def e00_params():
    return list(json.loads((OUT195 / 'e_top10.json').read_text())['top10'][0]['p'])


def build_e00(p0):
    for mode in (0, 1):
        for t in range(12):
            p = list(p0)
            p[4] = (t + 0.5) / 12
            p[3] = 0.25 if mode else 0.08
            _put('e00', (t, mode), C.synth_E(p))


def build_pool_overlays(pool):
    keys = [(t, m) for m in (0, 1) for t in range(12)]
    for name, fn in (('pnoarp', C._job_arp), ('pnoblock', _job_block)):
        for k, x in zip(keys, pool.map(fn, keys)):
            _put(name, k, x)


def build_music(ids):
    real = M.load_base('real', ids)
    n = len(ids)
    for k, i in enumerate(ids):
        w = real[ids[(k + 1) % n]]            # cyclic derangement: another real recording
        _put('music', i, w)


def get_ov(name, key, i):
    if name in KEY_OV:
        return OVW[name][key]
    if name == 'music':
        return OVW[name][i]
    return OVW[name][None]


def make_mix(base_w, ov, r):
    w64, lufs, w32 = ov
    if r is None:
        return w32
    g = 10 ** ((-30.0 + r - lufs) / 20)
    return C.lock(np.asarray(base_w, np.float64) + w64 * g)


# ═════════ scoring ═════════
def aes_rows(wavs):
    return X.aes4(wavs, SR)


def score_cell(key, ids, build):
    path = OUT / 'per_clip' / f'{key}.tsv'
    if cell_ok(key, ids):
        return
    for att in range(8):
        try:
            return _score_cell(key, ids, build, path)
        except Exception as e:  # noqa: BLE001
            oom = 'out of memory' in str(e).lower() or type(e).__name__ == 'OutOfMemoryError'
            if not oom or att == 7:
                raise
            P.SC.torch.cuda.empty_cache()
            log(f'{key}: CUDA OOM attempt {att}, sleep {60 * (att + 1)}s')
            time.sleep(60 * (att + 1))


def _score_cell(key, ids, build, path):
    t0 = time.time()
    chunks = [ids[k:k + CH] for k in range(0, len(ids), CH)]
    res, mx = {}, 0.0
    nxt = [P.POOL.submit(build, x) for x in chunks[0]]
    for k, c in enumerate(chunks):
        cur = nxt
        if k + 1 < len(chunks):
            nxt = [P.POOL.submit(build, x) for x in chunks[k + 1]]
        wavs = [f.result() for f in cur]
        for w in wavs:
            if not np.all(np.isfinite(w)):
                raise RuntimeError(f'{key}: non-finite stimulus')
            mx = max(mx, float(np.max(np.abs(w))))
        out = aes_rows(wavs)
        for j, x in enumerate(c):
            row = {q: out[q][j] for q in AES_COLS}
            bad = [q for q in AES_COLS if not math.isfinite(row[q])]
            if bad:
                raise RuntimeError(f'{key}: non-finite {bad} for {x}')
            res[x] = row
        P.progress('score', key, len(res), len(ids))
    lines = ['id\t' + '\t'.join(AES_COLS)] + [x + '\t' + '\t'.join(repr(res[x][q]) for q in AES_COLS) for x in ids]
    P.atomic(path, '\n'.join(lines) + '\n')
    wall = time.time() - t0
    P.add_timing(key, len(ids), wall, {})
    log(f'{key}: {len(ids)} stimuli in {wall:.1f}s ({len(ids) / wall:.1f}/s); max|x| {mx:.3f}')


def read_tsv(path):
    return M.tsv_read(path)


def cell_ok(key, ids):
    p = OUT / 'per_clip' / f'{key}.tsv'
    if not p.is_file():
        return False
    try:
        return M.rows_ok(read_tsv(p), ids, AES_COLS)
    except Exception:  # noqa: BLE001
        return False


# ═════════ run ═════════
def key_table(b, wavs, ids):
    kp = OUT / 'keys.json'
    allk = json.loads(kp.read_text()) if kp.is_file() else {}
    if b in allk and set(allk[b]) >= set(ids):
        return {i: tuple(allk[b][i]) for i in ids}
    ks = {i: tuple(C.key_of(wavs[i])) for i in ids}
    allk[b] = {i: list(k) for i, k in ks.items()}
    P.jdump(kp, allk)
    return ks


def gate(b, ids, smoke):
    """e00 cells vs 196 syn cells; fails the job when outside tolerance."""
    gp = OUT / 'gate.json'
    g = json.loads(gp.read_text()) if gp.is_file() else {'tol': {'median': GATE_MED, 'max': GATE_MAX}, 'bases': {}}
    rec = {}
    for r, old in GATE_MAP.items():
        a = read_tsv(OUT / 'per_clip' / f'{cell_key(b, "e00", r)}.tsv')
        o = read_tsv(OUT196 / 'per_clip' / f'dev_{b}_{old}.tsv')
        d = np.abs(M.arr(a, ids, 'CE') - M.arr(o, ids, 'CE'))
        rec[tag(r)] = {'196_cell': old, 'median_abs_dCE': float(np.median(d)), 'max_abs_dCE': float(d.max()), 'n': len(ids)}
    for r, old in INFO_MAP.items():
        a = read_tsv(OUT / 'per_clip' / f'{cell_key(b, "e00", r)}.tsv')
        o = read_tsv(OUT196 / 'per_clip' / f'dev_{b}_{old}.tsv')
        d = np.abs(M.arr(a, ids, 'CE') - M.arr(o, ids, 'CE'))
        rec['info_' + tag(r)] = {'196_cell': old, 'median_abs_dCE': float(np.median(d)), 'max_abs_dCE': float(d.max()),
                                 'note': '196 capped overlay peak at 0.999; 198 uncapped; not gated'}
    a = read_tsv(OUT / 'per_clip' / f'{b}_base.tsv')
    o = read_tsv(OUT196 / 'per_clip' / f'dev_{b}_base.tsv')
    d = np.abs(M.arr(a, ids, 'CE') - M.arr(o, ids, 'CE'))
    rec['base_info'] = {'median_abs_dCE': float(np.median(d)), 'max_abs_dCE': float(d.max())}
    a = read_tsv(OUT / 'per_clip' / f'{cell_key(b, "e00", None)}.tsv')
    o = read_tsv(OUT196 / 'per_clip' / f'dev_{b}_synonly.tsv')
    d = np.abs(M.arr(a, ids, 'CE') - M.arr(o, ids, 'CE'))
    rec['only_info'] = {'median_abs_dCE': float(np.median(d)), 'max_abs_dCE': float(d.max())}
    gate_cells = [v for k, v in rec.items() if k in {tag(r) for r in GATE_MAP}]
    ok = all(v['median_abs_dCE'] <= GATE_MED and v['max_abs_dCE'] <= GATE_MAX for v in gate_cells)
    rec['ok'] = ok
    g['bases'][b] = rec
    g['ok'] = all(v['ok'] for v in g['bases'].values())
    g['median_of_medians'] = float(np.median([v['median_abs_dCE'] for x in g['bases'].values() for k, v in x.items()
                                              if k in {tag(r) for r in GATE_MAP}]))
    g['max_abs_dCE'] = float(max(v['max_abs_dCE'] for x in g['bases'].values() for k, v in x.items()
                                 if k in {tag(r) for r in GATE_MAP}))
    P.jdump(gp, g)
    log(f'GATE {b}: ' + ('OK' if ok else 'FAIL') + ' ' + json.dumps({k: (round(v['median_abs_dCE'], 5), round(v['max_abs_dCE'], 5))
                                                                    for k, v in rec.items() if isinstance(v, dict)}))
    if not ok:
        raise RuntimeError(f'replication gate failed on base {b} (see gate.json)')


def run_base(b, ids, smoke):
    todo = [(o, r) for o in OVERLAYS for r in RR + [None] if not cell_ok(cell_key(b, o, r), ids)]
    need_base = not cell_ok(f'{b}_base', ids)
    if not todo and not need_base:
        if not smoke or True:
            gate(b, ids, smoke)
        return
    wavs = M.load_base(b, ids)
    ks = key_table(b, wavs, ids)
    if need_base:
        score_cell(f'{b}_base', ids, lambda i: wavs[i])
    for o in OVERLAYS:            # e00 first so the gate fires early
        for r in [None] + RR:
            key = cell_key(b, o, r)
            if cell_ok(key, ids):
                continue
            score_cell(key, ids, (lambda o_, r_: lambda i: make_mix(wavs[i], get_ov(o_, ks[i], i), r_))(o, r))
        if o == 'e00':
            gate(b, ids, smoke)
    del wavs


def features(ids, bs):
    """spectral-centroid features (scale-invariant) for the summary's Spearman block; written once."""
    fp = OUT / 'features.json'
    if fp.is_file():
        f = json.loads(fp.read_text())
        if set(f.get('bases', [])) >= set(bs):
            return f
    import librosa

    def cent(w):
        return float(np.mean(librosa.feature.spectral_centroid(y=np.asarray(w, np.float32), sr=SR, n_fft=2048, hop_length=512)))
    f = {'bases': list(bs), 'cent_base': {}, 'cent_ov_key': {}, 'cent_ov_fixed': {}, 'cent_ov_music': {}}
    for b in bs:
        wavs = M.load_base(b, ids)
        f['cent_base'][b] = {i: cent(wavs[i]) for i in ids}
    for o in KEY_OV:
        f['cent_ov_key'][o] = {f'{t}_{m}': cent(OVW[o][(t, m)][2]) for m in (0, 1) for t in range(12)}
    for o in ('sine1k', 'white', 'pink', 'drums'):
        f['cent_ov_fixed'][o] = cent(OVW[o][None][2])
    f['cent_ov_music'] = {i: cent(OVW['music'][i][2]) for i in ids}
    P.jdump(fp, f)
    return f


def sanity(ids):
    import soundfile as sf
    d = OUT / 'sanity'
    d.mkdir(parents=True, exist_ok=True)
    rows = {}
    for o in OVERLAYS:
        sub = {'e00': (0, 0), 'pnoarp': (0, 0), 'pnoblock': (0, 0), 'music': ids[0]}.get(o)
        w64, lufs, w32 = OVW[o][sub]
        rows[o] = {'lufs': lufs, 'peak': float(np.max(np.abs(w32))), 'finite': bool(np.isfinite(w32).all()), 'n': int(len(w32))}
        sf.write(d / f'{o}.wav', w32, SR)
    P.jdump(d / 'sanity.json', rows)
    log('sanity: ' + json.dumps({k: (round(v['lufs'], 2), round(v['peak'], 3)) for k, v in rows.items()}))


def run_all(smoke):
    global X
    setup(smoke)
    ids = get_ids(smoke)
    P.progress('start', '', 0, 0)
    json_split = OUT / 'split.json'
    if not json_split.is_file():
        P.jdump(json_split, {'source': str(OUT196 / 'split.json'), 'dev_mc': ids})
    pool = mp.get_context('fork').Pool(P.NTHREAD)         # forked before CUDA exists
    build_pool_overlays(pool)
    pool.close()
    build_e00(e00_params())
    build_fixed()
    build_music(ids)
    sanity(ids)
    features(ids, bases(smoke))
    X = Q.Y()
    import torch
    P.SC.torch = torch
    P.SC._aes()
    w = (np.random.default_rng(0).standard_normal(NS) * 0.05).astype(np.float32)
    aes_rows([w, w])
    P.SC.torch.cuda.reset_peak_memory_stats()
    log(f'AES loaded; ids {len(ids)}, bases {bases(smoke)}')
    peak_free = [P.free_mib()]
    for b in bases(smoke):
        run_base(b, ids, smoke)
        peak_free.append(P.free_mib())
    P.jdump(OUT / 'gpu_peak.json', {'torch_max_alloc_gib': P.SC.torch.cuda.max_memory_allocated() / 2**30,
                                    'torch_max_reserved_gib': P.SC.torch.cuda.max_memory_reserved() / 2**30,
                                    'nvidia_smi_min_free_mib_between_bases': int(min(peak_free))})
    P.progress('scored', '', 0, 0)
    project(smoke)


def project(smoke):
    tp = OUT / 'timing.json'
    if not tp.is_file():
        return None
    t = json.loads(tp.read_text())
    n = sum(x['n'] for x in t['keys'].values())
    wall = sum(x['wall_s'] for x in t['keys'].values())
    per = wall / n if n else None
    full_n = 9 * (1 + len(OVERLAYS) * (len(RR) + 1)) * 200
    rec = {'per_clip_s': per, 'full_n_stimuli': full_n, 'projected_scoring_hours': (per * full_n / 3600 if per else None),
           'note': 'smoke timings include warm-up / short-cell overhead, so this over-estimates; adds ~10 min setup'}
    P.jdump(OUT / 'projection.json', rec)
    log(f'PROJECTION full run: {rec["projected_scoring_hours"]} h scoring')
    return rec


# ═════════ summary ═════════
class Boot:
    def __init__(self, n):
        self.idx = np.random.default_rng(SEED).integers(0, n, (NB, n))


def ratio_boot(dm, do, bt):
    """p = mean(dm)/mean(do); returns (p, lo, hi) with clip-bootstrap."""
    if abs(do.mean()) < 1e-3:       # gap ~0 (e.g. real base x music overlay: cyclic shift of the same set) -> undefined
        return float('nan'), float('nan'), float('nan')
    with np.errstate(all='ignore'):
        p = float(dm.mean() / do.mean())
        pb = dm[bt.idx].mean(1) / do[bt.idx].mean(1)
    pb = pb[np.isfinite(pb)]
    if len(pb) < NB // 2 or not np.isfinite(p):
        return p, float('nan'), float('nan')
    return p, float(np.percentile(pb, 2.5)), float(np.percentile(pb, 97.5))


def sp_corr(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return None
    r, p = spearmanr(a[m], b[m])
    return {'rho': float(r), 'p': float(p), 'n': int(m.sum())}


def energy_w(r):
    x = 10 ** (r / 10)
    return float(x / (1 + x))


def summarize(smoke):
    setup(smoke)
    ids = get_ids(smoke)
    bs = bases(smoke)
    bt = Boot(len(ids))
    S_CE, S_PQ = H.S['CE_lvl30'], H.S['PQ_lvl30'] if 'PQ_lvl30' in H.S else None
    tol = {'CE': 0.10 * S_CE, 'PQ': MONO_TOL_PQ}
    fe = json.loads((OUT / 'features.json').read_text())
    keys = json.loads((OUT / 'keys.json').read_text())
    D = {}                                           # (b, o, r|'only'|'base') -> {col: array}
    for b in bs:
        D[(b, 'base')] = {c: M.arr(read_tsv(OUT / 'per_clip' / f'{b}_base.tsv'), ids, c) for c in AES_COLS}
        for o in OVERLAYS:
            for r in RR + [None]:
                D[(b, o, r)] = {c: M.arr(read_tsv(OUT / 'per_clip' / f'{cell_key(b, o, r)}.tsv'), ids, c) for c in AES_COLS}
    out = {'kind': KIND, 'smoke': smoke, 'boot': {'NB': NB, 'seed': SEED}, 'n_ids': len(ids), 'bases': bs, 'overlays': OVERLAYS,
           'r_rel': RR, 'tol': tol, 'cells': {}, 'pooled': {}, 'curves': {}, 'spearman_r0': {}, 'headline': {}}
    for col in ('CE', 'PQ'):
        m = tol[col]
        for o in OVERLAYS:
            for r in RR:
                per_base, boots = {}, []
                for b in bs:
                    base, only, mix = D[(b, 'base')][col], D[(b, o, None)][col], D[(b, o, r)][col]
                    dm, do = mix - base, only - base
                    p, lo, hi = ratio_boot(dm, do, bt)
                    lo_c, hi_c = np.minimum(base, only), np.maximum(base, only)
                    per_base[b] = {'p': p, 'ci': [lo, hi], 'gap': float(do.mean()), 'gap_small': bool(abs(do.mean()) < m),
                                   'dm': float(dm.mean()), 'frac_above_max': float((mix > hi_c).mean()),
                                   'frac_below_min': float((mix < lo_c).mean()),
                                   'frac_inside': float(((mix >= lo_c) & (mix <= hi_c)).mean())}
                    with np.errstate(all='ignore'):
                        boots.append(dm[bt.idx].mean(1) / do[bt.idx].mean(1) if abs(do.mean()) >= 1e-3
                                     else np.full(NB, np.nan))
                pv = np.array([per_base[b]['p'] for b in bs])
                pb = np.nanmean(np.vstack(boots), 0)
                lo, hi = float(np.nanpercentile(pb, 2.5)), float(np.nanpercentile(pb, 97.5))
                w = energy_w(r)
                pooled = float(np.nanmean(pv))
                out['cells'].setdefault(col, {}).setdefault(o, {})[tag(r)] = {
                    'pooled_p': pooled, 'pooled_ci': [lo, hi], 'w_energy': w,
                    'middle': bool(lo <= 0.5 <= hi), 'energy_weighted': bool(abs(pooled - w) < 0.1),
                    'frac_above_max_mean': float(np.mean([per_base[b]['frac_above_max'] for b in bs])),
                    'frac_below_min_mean': float(np.mean([per_base[b]['frac_below_min'] for b in bs])),
                    'n_gap_small_bases': int(sum(per_base[b]['gap_small'] for b in bs)), 'per_base': per_base}
        # curves / monotonicity
        for o in OVERLAYS:
            per_b = {}
            for b in bs:
                base, only = D[(b, 'base')][col], D[(b, o, None)][col]
                mat = np.vstack([D[(b, o, r)][col] for r in RR])            # (9, n)
                curve = mat.mean(1)
                up = bool(np.all(np.diff(curve) >= -m))
                dn = bool(np.all(np.diff(curve) <= m))
                direction = 'up' if only.mean() >= base.mean() else 'down'
                exp_ok = up if direction == 'up' else dn
                d = np.diff(mat, axis=0)
                clip_up = np.all(d >= -m, 0)
                clip_dn = np.all(d <= m, 0)
                clip_exp = clip_up if direction == 'up' else clip_dn
                per_b[b] = {'curve': curve.tolist(), 'direction': direction, 'mono_up': up, 'mono_down': dn,
                            'mono_expected': exp_ok, 'per_clip_mono_expected': float(clip_exp.mean()),
                            'per_clip_mono_up': float(clip_up.mean()), 'per_clip_mono_down': float(clip_dn.mean())}
            out['curves'].setdefault(col, {})[o] = {
                'per_base': per_b,
                'n_bases_mono_expected': int(sum(v['mono_expected'] for v in per_b.values())),
                'mean_per_clip_mono_expected': float(np.mean([v['per_clip_mono_expected'] for v in per_b.values()]))}
        # Spearman at r_rel = 0 over (base, overlay) cells, n = len(bs) * 8
        rows = []
        for b in bs:
            cb = np.array([fe['cent_base'][b][i] for i in ids])
            for o in OVERLAYS:
                base, only, mix = D[(b, 'base')][col], D[(b, o, None)][col], D[(b, o, 0)][col]
                dm, do = mix - base, only - base
                if o in KEY_OV:
                    co = np.array([fe['cent_ov_key'][o]['%d_%d' % tuple(keys[b][i])] for i in ids])
                elif o == 'music':
                    co = np.array([fe['cent_ov_music'][i] for i in ids])
                else:
                    co = np.full(len(ids), fe['cent_ov_fixed'][o])
                rows.append({'p': float(dm.mean() / do.mean()) if abs(do.mean()) >= 1e-3 else float('nan'), 'gap': float(do.mean()),
                             'base_CE': float(base.mean()), 'log_cent_ratio': float(np.log(co.mean() / cb.mean())),
                             'pitched': PITCHED[o], 'abs_gap': float(abs(do.mean()))})
        sp = {}
        for k in ('gap', 'base_CE', 'log_cent_ratio', 'pitched'):
            sp[k] = sp_corr([x['p'] for x in rows], [x[k] for x in rows])
        big = [x for x in rows if x['abs_gap'] >= m]
        sp['n_cells'] = len(rows)
        sp['n_cells_gap_ge_m'] = len(big)
        sp['restricted_gap_ge_m'] = {k: sp_corr([x['p'] for x in big], [x[k] for x in big])
                                     for k in ('gap', 'base_CE', 'log_cent_ratio', 'pitched')}
        out['spearman_r0'][col] = sp
    # headline (CE): per overlay count of r where the pooled CI contains 0.5 / |p-w|<0.1
    for o in OVERLAYS:
        cs = out['cells']['CE'][o]
        out['headline'][o] = {'n_middle_of_9': int(sum(v['middle'] for v in cs.values())),
                              'n_energy_weighted_of_9': int(sum(v['energy_weighted'] for v in cs.values())),
                              'p_r0': cs['p0']['pooled_p'],
                              'n_bases_mono_expected': out['curves']['CE'][o]['n_bases_mono_expected'],
                              'per_clip_mono_expected': out['curves']['CE'][o]['mean_per_clip_mono_expected']}
    g = json.loads((OUT / 'gate.json').read_text())
    out['gate'] = {'ok': g['ok'], 'median_of_medians': g['median_of_medians'], 'max_abs_dCE': g['max_abs_dCE']}
    out = nan2none(out)
    P.jdump(OUT / 'summary.json', out)
    P.atomic(OUT / 'summary_table.md', table_md(out))
    log('summary written; headline: ' + json.dumps(out['headline']))
    return out


def nan2none(o):
    if isinstance(o, dict):
        return {k: nan2none(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nan2none(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def table_md(o):
    L = [f'# 198 summary ({"smoke" if o["smoke"] else "full"}; bootstrap {NB}, seed {SEED}; n ids = {o["n_ids"]}; '
         f'mono tol CE {o["tol"]["CE"]:.4f})', '', f'gate ok = {o["gate"]["ok"]}, median {o["gate"]["median_of_medians"]:.5f}, '
         f'max {o["gate"]["max_abs_dCE"]:.5f}', '', '## CE pooled p by overlay x r_rel (9-base mean)', '',
         '| overlay | ' + ' | '.join(tag(r) for r in RR) + ' | mid/9 | energy/9 | mono bases | per-clip mono |',
         '|---|' + '---|' * (len(RR) + 4)]
    for ov in o['overlays']:
        cs = o['cells']['CE'][ov]
        h = o['headline'][ov]
        L.append(f'| {ov} | ' + ' | '.join((f'{cs[tag(r)]["pooled_p"]:+.2f}' if cs[tag(r)]["pooled_p"] is not None else 'n/a') for r in RR) +
                 f' | {h["n_middle_of_9"]} | {h["n_energy_weighted_of_9"]} | {h["n_bases_mono_expected"]}/{len(o["bases"])} | '
                 f'{h["per_clip_mono_expected"]:.2f} |')
    L += ['', '## Spearman at r_rel = 0 (p vs ...)', '']
    for col, sp in o['spearman_r0'].items():
        L.append(f'- {col}: ' + '; '.join(f'{k} rho {v["rho"]:+.2f} (p {v["p"]:.3g})' for k, v in sp.items()
                                          if isinstance(v, dict) and 'rho' in v) + f' (n = {sp["n_cells"]})')
    return '\n'.join(L) + '\n'


# ═════════ validate / preflight ═════════
def validate(smoke):
    errs = []
    try:
        setup(smoke)
        ids = get_ids(smoke)
        for f in ('split.json', 'keys.json', 'features.json', 'gate.json', 'summary.json', 'summary_table.md'):
            if not (OUT / f).is_file():
                errs.append(f'missing {f}')
        for b in bases(smoke):
            for key in [f'{b}_base'] + [cell_key(b, o, r) for o in OVERLAYS for r in RR + [None]]:
                if not cell_ok(key, ids):
                    errs.append(f'bad/missing cell {key}')
        if (OUT / 'gate.json').is_file() and not json.loads((OUT / 'gate.json').read_text()).get('ok'):
            errs.append('gate not ok')
        if (OUT / 'summary.json').is_file():
            s = json.loads((OUT / 'summary.json').read_text())
            if s.get('kind') != KIND:
                errs.append('summary.json wrong kind')
            bad = C._finite_json(s)
            if bad:
                errs.append(f'non-finite numbers in summary.json, e.g. {bad[:3]}')
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
    for p in (OUT196 / 'split.json', OUT195 / 'e_top10.json', P.REF_DIR, C.MS.BANKS['generaluser']):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    try:
        ids = get_ids(smoke)
        want = SMOKE_N if smoke else 200
        if len(ids) != want:
            errs.append(f'{len(ids)} dev ids, expected {want}')
        miss = [i for i in ids if not P.ref_path(i).is_file()]
        if miss:
            errs.append(f'{len(miss)} ids without reference audio, e.g. {miss[0]}')
        for b in bases(smoke):
            if b != 'real':
                miss = [i for i in ids if not M.base_path(b, i).is_file()]
                if miss:
                    errs.append(f'base {b}: {len(miss)} missing audio, e.g. {miss[0]}')
            if not (OUT196 / 'per_clip' / f'dev_{b}_syn0.tsv').is_file():
                errs.append(f'missing 196 gate cells for {b}')
        import importlib
        for mod in ('tinysoundfont', 'soxr', 'librosa', 'pyloudnorm', 'soundfile', 'scipy'):
            importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if errs:
        return 1
    log(f'PREFLIGHT OK (smoke={smoke})')
    if not (OUT / 'summary.json').is_file():
        need_mem = 8 if smoke else 14
        if C.mem_available_gb() < need_mem:
            log(f'PREFLIGHT WAIT MemAvailable {C.mem_available_gb():.1f} GiB < {need_mem} GiB')
            return 75
        f = P.free_mib()
        if f < 8 * 1024:
            log(f'PREFLIGHT WAIT GPU free {f} MiB < {8 * 1024} MiB')
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
        OUT = H.EVAL_ROOT / 'ce_overlay_interp_198_smoke'
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
