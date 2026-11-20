#!/usr/bin/env python
"""194: CMI-RM weakness probes (eval-only driver).

Design: docs/experiments/cmirm_weakness_194_20261113.md (pre-registered; this file only implements it).

Part A  text manipulation: 9 bases x 1000 MusicCaps ids x 8 texts (cap / empty / irr / cap_hq / hq_only / cap_lq / hqpre /
        lq_only); CMI-RM quality + alignment for every cell (the 189/190 XScorer exposes no alignment column, so this driver
        calls newmetrics_cmirm.score_cmirm directly).  irr = one id-level derangement (seed 20261113) shared by all bases.
Part C  30 s window: MUSDB18-HQ mixtures, 60 s from 15 s, 16 kHz mono, -30 LUFS; cells base60 / base30 / noise_{pre,post}
        (white noise SNR 10 dB re that half's RMS) / lp_{pre,post} (2 kHz Butterworth 8, sosfiltfilt); no re-normalisation;
        scored in 'empty' mode and with the fixed text "A piece of music.".
Parts B and D of the design are not re-run (reuse existing scores).
Deviations: D1 no dev/test split in Part A (no selection step); D2 Part C on MUSDB only; D3 base30 vs base60 consistency
tolerance 0.01 on mean |dq|; D4 the 1e-4 post tolerance is below the resampler edge effect (~1e-3, measured by
max|base30-base60|), so the primary verdict uses max(1e-4, that floor) and the strict 1e-4 verdict is reported alongside.
Resumable per cell TSV under <root>/per_clip/.  Neither 189/190 nor 188 is edited.
"""
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
import argparse  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
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
SR = P.SR
OUT = H.EVAL_ROOT / 'cmirm_weakness_194'
KIND = 'cmirm_weakness_194_summary_v1'
SEED = 20261113
NB = 5000
BASES = M.MC_BASES
ZIP = Path('/mnt/seagate/kojiek_data/musdb18hq/musdb18hq.zip')
S_Q_CAP, S_Q_EMPTY = 0.657, 1.450          # design: S for CMI quality (caption mode / empty mode), fixed constants
M_Q_CAP, M_Q_EMPTY = 0.10 * S_Q_CAP, 0.10 * S_Q_EMPTY
FIXED_TEXT = 'A piece of music.'
HQ = 'High quality, professional studio recording.'
LQ = 'Low quality, noisy.'
TEXTS = ['cap', 'empty', 'irr', 'cap_hq', 'hq_only', 'cap_lq', 'hqpre', 'lq_only']
C_CELLS = ['base60', 'base30', 'noise_pre', 'noise_post', 'lp_pre', 'lp_post']
C_MODES = ['empty', 'fixed']
MIN_FREE = 20 * 1024**3
POST_TOL = 1e-4
BASE_TOL = 0.01
IS_SMOKE = False
N_TRACKS = 150


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


# ── CMI scorer ───────────────────────────────────────────
class Cmi:
    def __init__(self):
        self.t = 0.0
        self.n = 0

    def load(self):
        from muq_gen_steer_179_score import muq_hf_cache
        with muq_hf_cache():
            import newmetrics_cmirm
            pip_muq = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'muq' or k.startswith('muq.')}
            try:
                newmetrics_cmirm._load('cuda', False)
            finally:
                for k in [k for k in sys.modules if k == 'muq' or k.startswith('muq.')]:
                    del sys.modules[k]
                sys.modules.update(pip_muq)
        self.nm = newmetrics_cmirm
        w = (np.random.default_rng(0).standard_normal(SR * 10) * 0.05).astype(np.float32)
        self.one(w, 'caption', 'warm up')
        self.t, self.n = 0.0, 0

    def one(self, w, mode, text):
        t0 = time.time()
        r = self.nm.score_cmirm(w, sr=SR, prompt_mode=mode, caption=text)
        self.t += time.time() - t0
        self.n += 1
        return float(r['quality']), float(r['alignment'])


CMI = Cmi()


# ── tsv ──────────────────────────────────────────────────
def tsv_write(path, rows, ids):
    lines = ['id\tquality\talignment'] + [f'{i}\t{rows[i][0]!r}\t{rows[i][1]!r}' for i in ids]
    P.atomic(path, '\n'.join(lines) + '\n')


def tsv_read(path):
    with open(path, newline='') as f:
        return {r['id']: (float(r['quality']), float(r['alignment'])) for r in csv.DictReader(f, delimiter='\t')}


def rows_ok(rows, ids):
    return set(rows) == set(ids) and all(math.isfinite(v) for i in ids for v in rows[i])


def col(rows, ids, k):
    return np.array([rows[i][k] for i in ids], float)


# ── ids / texts ──────────────────────────────────────────
def get_ids(smoke):
    mc, _ = M.get_ids(False)
    return mc[:8] if smoke else mc


def derangement(ids):
    ids = sorted(ids)
    rng = np.random.default_rng(SEED)
    while True:
        p = rng.permutation(len(ids))
        if len(ids) < 2 or not np.any(p == np.arange(len(ids))):
            return {ids[k]: ids[int(p[k])] for k in range(len(ids))}


def text_of(kind, i, caps, irr):
    c = caps[i]
    return {'cap': ('caption', c), 'empty': ('empty', ''), 'irr': ('caption', caps[irr[i]]),
            'cap_hq': ('caption', f'{c} {HQ}'), 'hq_only': ('caption', HQ), 'cap_lq': ('caption', f'{c} {LQ}'),
            'hqpre': ('caption', f'High quality recording. {c}'), 'lq_only': ('caption', LQ)}[kind]


def setup(smoke):
    P.OUT = M.OUT = Q.OUT = OUT
    P.IS_SMOKE = smoke
    P.POOL = ThreadPoolExecutor(P.NTHREAD)
    H.load_scales()
    OUT.mkdir(parents=True, exist_ok=True)


def score_cell(key, ids, items):
    """items: id -> (mode, text, wav).  Resumable per-cell TSV."""
    path = OUT / 'per_clip' / f'{key}.tsv'
    if path.is_file():
        rows = tsv_read(path)
        if rows_ok(rows, ids):
            return rows
    t0 = time.time()
    rows = {}
    for k, i in enumerate(ids):
        mode, text, w = items[i]
        rows[i] = CMI.one(w, mode, text)
        if k % 100 == 0:
            P.progress('score', key, k, len(ids))
    tsv_write(path, rows, ids)
    P.add_timing(key, len(ids), time.time() - t0, {'cmi': time.time() - t0})
    log(f'{key}: {len(ids)} in {time.time() - t0:.1f}s ({len(ids) / (time.time() - t0):.1f}/s)')
    return rows


# ── Part A ───────────────────────────────────────────────
def run_a(smoke):
    ids = get_ids(smoke)
    caps = P.caps_all()
    irr = derangement(ids)
    P.jdump(OUT / 'irr_map.json', irr)
    for b in BASES:
        keys = [f'A_{b}_{t}' for t in TEXTS]
        if all((OUT / 'per_clip' / f'{k}.tsv').is_file() and rows_ok(tsv_read(OUT / 'per_clip' / f'{k}.tsv'), ids)
               for k in keys):
            continue
        wavs = M.load_base(b, ids)
        for t in TEXTS:
            score_cell(f'A_{b}_{t}', ids, {i: (*text_of(t, i, caps, irr), wavs[i]) for i in ids})
        del wavs


# ── Part C ───────────────────────────────────────────────
def track_list(smoke):
    with zipfile.ZipFile(ZIP) as z:
        names = sorted({n.split('/')[1] + '|' + n.split('/')[0] for n in z.namelist() if n.endswith('/mixture.wav')})
    tr = [tuple(n.split('|')) for n in names]            # (track, split)
    return tr[:4] if smoke else tr


def prep_track(track, split):
    import soundfile as sf
    import soxr
    p = OUT / 'c_base' / f'{track.replace("/", "_")}.npy'
    if p.is_file():
        return np.load(p)
    with zipfile.ZipFile(ZIP) as z:
        x, sr = sf.read(io.BytesIO(z.read(f'{split}/{track}/mixture.wav')), dtype='float32', always_2d=True)
    if x.shape[0] < 75 * sr:
        return None
    y = x[15 * sr:75 * sr].mean(1).astype(np.float64)
    y = soxr.resample(y, sr, SR, quality='VHQ')[:60 * SR]
    y = np.asarray(P.to_lufs(y, -30.0)[0], np.float32)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, y)
    return y


def degrade(y, cell, track):
    from scipy.signal import butter, sosfiltfilt
    y = np.asarray(y, np.float64).copy()
    h = 30 * SR
    if cell in ('base60',):
        return y.astype(np.float32)
    if cell == 'base30':
        return y[:h].astype(np.float32)
    sl = slice(0, h) if cell.endswith('_pre') else slice(h, 2 * h)
    seg = y[sl]
    if cell.startswith('noise'):
        rng = P.rng_for('194', track, cell)
        rms = float(np.sqrt(np.mean(seg ** 2)))
        y[sl] = seg + rng.standard_normal(len(seg)) * rms * 10 ** (-10 / 20)
    else:
        y[sl] = sosfiltfilt(butter(8, 2000, 'low', fs=SR, output='sos'), seg)
    return y.astype(np.float32)


def run_c(smoke):
    tr = track_list(smoke)
    keep = []
    for t, sp in tr:
        if len(keep) >= N_TRACKS:
            break
        y = prep_track(t, sp)
        if y is not None:
            keep.append((t, sp))
    names = [t for t, _ in keep]
    P.jdump(OUT / 'c_tracks.json', {'tracks': names})
    for cell in C_CELLS:
        for mode in C_MODES:
            key = f'C_{cell}_{mode}'
            items = {}
            for t, sp in keep:
                y = np.load(OUT / 'c_base' / f'{t.replace("/", "_")}.npy')
                items[t] = ('empty' if mode == 'empty' else 'caption', '' if mode == 'empty' else FIXED_TEXT,
                            degrade(y, cell, t))
            score_cell(key, names, items)
    return names


def run_all(smoke):
    setup(smoke)
    P.progress('start', '', 0, 0)
    CMI.load()
    log('CMI-RM loaded')
    run_a(smoke)
    run_c(smoke)
    P.jdump(OUT / 'gpu_peak.json', {'nvidia_smi_note': 'CMI-RM only'})
    P.progress('scored', '', 0, 0)
    project(smoke)


def project(smoke):
    tp = OUT / 'timing.json'
    if not tp.is_file():
        return None
    t = json.loads(tp.read_text())['keys']
    a = [v['wall_s'] / v['n'] for k, v in t.items() if k.startswith('A_')]
    c = [v['wall_s'] / v['n'] for k, v in t.items() if k.startswith('C_')]
    tot = (np.mean(a) * 9 * 8 * 1000 if a else 0) + (np.mean(c) * 12 * 150 if c else 0)
    rec = {'s_per_clip_A': float(np.mean(a)) if a else None, 's_per_clip_C': float(np.mean(c)) if c else None,
           'projected_total_hours': tot / 3600, 'note': 'A: 9x8x1000 scorings; C: 12 cells x 150 tracks (60 s clips)'}
    P.jdump(OUT / 'projection.json', rec)
    log(f'PROJECTION full run: {rec["projected_total_hours"]:.2f} h ({rec})')
    return rec


# ── summary ──────────────────────────────────────────────
class Boot:
    def __init__(self, n):
        self.idx = np.random.default_rng(SEED).integers(0, n, (NB, n))

    def ci(self, d):
        d = np.asarray(d, float)
        b = d[self.idx].mean(1)
        return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(d))]


def read_cells(prefix, names, ids):
    return {n: tsv_read(OUT / 'per_clip' / f'{prefix}{n}.tsv') for n in names
            if (OUT / 'per_clip' / f'{prefix}{n}.tsv').is_file()}


def reading(n):
    return 'manipulable_or_mixed' if n >= 7 else ('base_dependent' if n >= 3 else 'not_established')


def summarize(smoke):
    from scipy.stats import kendalltau
    setup(smoke)
    ids = get_ids(smoke)
    bt = Boot(len(ids))
    A = {b: read_cells(f'A_{b}_', TEXTS, ids) for b in BASES}
    s_align = None
    capal = [float(np.mean(col(A[b]['cap'], ids, 1))) for b in BASES if 'cap' in A[b]]
    s_align = float(max(capal) - min(capal)) if capal else None
    m_align = 0.10 * s_align if s_align else None
    out = {'kind': KIND, 'smoke': smoke, 'seed': SEED, 'boot': {'NB': NB}, 'n_ids': len(ids), 'm_q': M_Q_CAP,
           'S_align': s_align, 'm_align': m_align, 'A': {'bases': {}}}
    e1, e2 = [], []
    for b in BASES:
        a = A[b]
        rec = {}

        def dq(t, ref, k=0):
            return bt.ci(col(a[t], ids, k) - col(a[ref], ids, k))
        rec['E1_cap_hq_vs_cap_q'] = dq('cap_hq', 'cap')
        rec['E2_irr_vs_cap_q'] = dq('irr', 'cap')
        rec['E1_success'] = bool(rec['E1_cap_hq_vs_cap_q'][1] > M_Q_CAP)
        rec['E2_success'] = bool(rec['E2_irr_vs_cap_q'][2] < -M_Q_CAP)
        rec['sec_cap_lq_vs_cap_q'] = dq('cap_lq', 'cap')
        rec['sec_hqpre_vs_cap_q'] = dq('hqpre', 'cap')
        rec['sec_hq_only_vs_empty_q'] = dq('hq_only', 'empty')
        rec['sec_lq_only_vs_empty_q'] = dq('lq_only', 'empty')
        rec['align_vs_cap'] = {t: dq(t, 'cap', 1) for t in TEXTS if t != 'cap'}
        rec['irr_align_drop_check'] = bool(rec['align_vs_cap']['irr'][2] < 0)
        rec['mean_q'] = {t: float(np.mean(col(a[t], ids, 0))) for t in TEXTS}
        e1.append(rec['E1_success'])
        e2.append(rec['E2_success'])
        out['A']['bases'][b] = rec
    out['A']['E1'] = {'n_success': int(sum(e1)), 'n_bases': len(BASES), 'reading': reading(sum(e1))}
    out['A']['E2'] = {'n_success': int(sum(e2)), 'n_bases': len(BASES), 'reading': reading(sum(e2)),
                      'meaning': 'irrelevant caption lowers quality head => text mixed into quality'}
    qc = [out['A']['bases'][b]['mean_q']['cap'] for b in BASES]
    qh = [out['A']['bases'][b]['mean_q']['cap_hq'] for b in BASES]
    tau = kendalltau(qc, qh)
    out['A']['kendall_tau_cap_vs_cap_hq'] = {'tau': float(tau.statistic) if np.isfinite(tau.statistic) else None,
                                              'n_systems': len(BASES)}
    out['A']['n_irr_align_drop'] = int(sum(r['irr_align_drop_check'] for r in out['A']['bases'].values()))

    # Part C
    names = json.loads((OUT / 'c_tracks.json').read_text())['tracks'] if (OUT / 'c_tracks.json').is_file() else []
    C = {m: {c: tsv_read(OUT / 'per_clip' / f'C_{c}_{m}.tsv') for c in C_CELLS
             if (OUT / 'per_clip' / f'C_{c}_{m}.tsv').is_file()} for m in C_MODES}
    cb = Boot(len(names)) if names else None
    out['C'] = {'n_tracks': len(names), 'modes': {}}
    for m in C_MODES:
        cc = C[m]
        mq = M_Q_EMPTY if m == 'empty' else M_Q_CAP
        if 'base60' not in cc or 'base30' not in cc:
            continue
        b60, b30 = col(cc['base60'], names, 0), col(cc['base30'], names, 0)
        consistency = float(np.mean(np.abs(b60 - b30)))
        if consistency > BASE_TOL:
            raise RuntimeError(f'194 Part C {m}: base30 vs base60 mean |dq| {consistency:.4f} > {BASE_TOL}: '
                               'implementation problem (windowing)')
        floor = float(np.max(np.abs(b60 - b30)))   # resampler edge effect: same audio in first 30 s, differs only at the boundary
        rec = {'m_q': mq, 'base30_vs_base60_mean_abs_dq': consistency, 'edge_floor_max_abs_dq': floor, 'families': {}}
        for fam in ('noise', 'lp'):
            pre, post = cc[f'{fam}_pre'], cc[f'{fam}_post']
            d_pre = col(pre, names, 0) - b60
            d_post = col(post, names, 0) - b60
            a_pre = col(pre, names, 1) - col(cc['base60'], names, 1)
            a_post = col(post, names, 1) - col(cc['base60'], names, 1)
            ci_pre = cb.ci(d_pre)
            max_post = float(np.max(np.abs(d_post)))
            def verdict(tol):
                if max_post >= tol:
                    return 'sees_beyond_30s'
                return 'blind_spot_beyond_30s' if ci_pre[2] < -mq else 'not_visible_within_window'
            v = verdict(max(POST_TOL, floor))
            v_strict = verdict(POST_TOL)
            rec['families'][fam] = {'verdict': v, 'verdict_strict_1e-4': v_strict, 'pre_dq': ci_pre, 'pre_dalign': cb.ci(a_pre),
                                    'post_dq_max_abs': max_post, 'post_dq_mean': float(d_post.mean()),
                                    'post_dalign_max_abs': float(np.max(np.abs(a_post))),
                                    'post_tol': POST_TOL}
        out['C']['modes'][m] = rec
    P.jdump(OUT / 'summary.json', out)
    P.atomic(OUT / 'summary_table.md', table_md(out))
    log(f'summary written; A E1 {out["A"]["E1"]}, E2 {out["A"]["E2"]}; C ' +
        json.dumps({m: {f: v['verdict'] for f, v in r['families'].items()} for m, r in out['C']['modes'].items()}))
    return out


def fmt(ci):
    return 'n/a' if not ci else f'{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]'


def table_md(out):
    A = out['A']
    L = [f'# 194 summary ({"smoke" if out["smoke"] else "full"}; m_q = {out["m_q"]:.4f}, m_align = {out["m_align"]}; '
         f'bootstrap {NB}, seed {SEED})', '', '## Part A text manipulation', '',
         f'E1 (cap_hq vs cap, CI lower > m_q): {A["E1"]["n_success"]}/{A["E1"]["n_bases"]} -> **{A["E1"]["reading"]}**; '
         f'E2 (irr vs cap, CI upper < -m_q): {A["E2"]["n_success"]}/{A["E2"]["n_bases"]} -> **{A["E2"]["reading"]}**; '
         f'Kendall tau cap vs cap_hq {A["kendall_tau_cap_vs_cap_hq"]["tau"]}; irr alignment drop on {A["n_irr_align_drop"]} bases', '',
         '| base | dq cap_hq | dq irr | dq cap_lq | dq hqpre | dq hq_only-empty | dq lq_only-empty | dalign irr |',
         '|---|---|---|---|---|---|---|---|']
    for b, r in A['bases'].items():
        L.append(f'| {b} | {fmt(r["E1_cap_hq_vs_cap_q"])} | {fmt(r["E2_irr_vs_cap_q"])} | {fmt(r["sec_cap_lq_vs_cap_q"])} | '
                 f'{fmt(r["sec_hqpre_vs_cap_q"])} | {fmt(r["sec_hq_only_vs_empty_q"])} | {fmt(r["sec_lq_only_vs_empty_q"])} | '
                 f'{fmt(r["align_vs_cap"]["irr"])} |')
    L += ['', '## Part C 30 s window', '', '| mode | family | verdict | pre dq | post max abs dq |', '|---|---|---|---|---|']
    for m, r in out['C']['modes'].items():
        for f, v in r['families'].items():
            L.append(f'| {m} | {f} | {v["verdict"]} | {fmt(v["pre_dq"])} | {v["post_dq_max_abs"]:.2e} |')
    return '\n'.join(L) + '\n'


# ── validate / preflight ─────────────────────────────────
def validate(smoke):
    errs = []
    try:
        setup(smoke)
        ids = get_ids(smoke)
        for b in BASES:
            for t in TEXTS:
                p = OUT / 'per_clip' / f'A_{b}_{t}.tsv'
                if not p.is_file():
                    errs.append(f'missing {p.name}')
                elif not rows_ok(tsv_read(p), ids):
                    errs.append(f'{p.name}: id set mismatch or NaN')
        names = json.loads((OUT / 'c_tracks.json').read_text())['tracks'] if (OUT / 'c_tracks.json').is_file() else []
        want = 4 if smoke else N_TRACKS
        if len(names) != want:
            errs.append(f'Part C tracks {len(names)} != {want}')
        for c in C_CELLS:
            for m in C_MODES:
                p = OUT / 'per_clip' / f'C_{c}_{m}.tsv'
                if not p.is_file():
                    errs.append(f'missing {p.name}')
                elif not rows_ok(tsv_read(p), names):
                    errs.append(f'{p.name}: id set mismatch or NaN')
        for f in ('summary.json', 'summary_table.md', 'irr_map.json'):
            if not (OUT / f).is_file():
                errs.append(f'missing {f}')
        if (OUT / 'summary.json').is_file():
            s = json.loads((OUT / 'summary.json').read_text())
            if s.get('kind') != KIND:
                errs.append('wrong kind')
            for b in BASES:
                r = s['A']['bases'].get(b, {})
                for k in ('E1_success', 'E2_success', 'E1_cap_hq_vs_cap_q', 'E2_irr_vs_cap_q'):
                    if k not in r:
                        errs.append(f'summary A {b} lacks {k}')
            for k in ('E1', 'E2'):
                if s['A'].get(k, {}).get('reading') is None:
                    errs.append(f'summary A lacks {k}.reading')
            for m in C_MODES:
                for f in ('noise', 'lp'):
                    v = s['C']['modes'].get(m, {}).get('families', {}).get(f, {}).get('verdict')
                    if v is None:
                        errs.append(f'summary C {m}/{f} lacks verdict')
            txt = json.dumps(s)
            if 'NaN' in txt:
                errs.append('summary.json contains NaN')
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
    from newmetrics_common import REPOS
    for p in (H.PARTB / 'part_b_selection.json', H.MC_TSV, P.REF_DIR, ZIP, REPOS / 'CMI-RewardBench', H.PY):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    try:
        ids = get_ids(smoke)
        if len(ids) != (8 if smoke else 1000):
            errs.append(f'{len(ids)} ids')
        for b in BASES:
            if b == 'real':
                continue
            miss = [i for i in ids if not M.base_path(b, i).is_file()]
            if miss:
                errs.append(f'base {b}: {len(miss)} missing, e.g. {miss[0]}')
        caps = P.caps_all()
        if any(i not in caps for i in ids):
            errs.append('captions missing')
        if len(track_list(smoke)) < (4 if smoke else N_TRACKS):
            errs.append('fewer MUSDB mixture tracks than needed')
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
        OUT = H.EVAL_ROOT / 'cmirm_weakness_194_smoke'
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
