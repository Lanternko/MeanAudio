#!/usr/bin/env python
"""199: objective change amount of the 197 attacks (eval-only driver).

Design: docs/experiments/attack_change_amount_199_20261125.md (pre-registered; this file only implements it).
Regenerates the 197 stimuli (197 / 193 / 190 / 189 builders imported unchanged; nothing is kept on disk but per-clip numbers
and CLAP embeddings for fd_clap) and writes, per cell, one TSV with columns
    id  rho  lag  mel  d_clap  d_mert
into ~/eval_output_nvme/attack_change_amount_199[_smoke]/per_clip/{T1_<base>_<cell>,T2A_<bank>_<cell>}.tsv.

  rho   min over lag (+-0.6 s) and scalar gain of ||y - g x(lag)||^2 / ||y||^2 on the overlap (0 same, 1 unrelated)
  lag   best lag in samples (y[n] ~ x[n - lag])
  mel   64-band log-mel (25 ms / 10 ms hop, floor = 80 dB below the reference max), unit-RMS normalised, same lag; mean |dB diff|
  d_clap, d_mert   1 - cos of LAION-CLAP / MERT-v1-95M embeddings (20 s stimuli: mean over front and back 10 s windows)

No MuQ, no rescoring of A1 / PQ / CE / CMI-RM.  Resume: a cell whose TSV is complete is skipped.
  python scripts/eval/attack_change_amount_199.py [--smoke] [--part T1|T2A|all]
"""
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attack_metric_matrix_197 as A  # noqa: E402

C, M, P, H, Q = A.C, A.M, A.P, A.H, A.Q
SR = A.SR
SEED = 20261117
OUT = H.EVAL_ROOT / 'attack_change_amount_199'
KIND = 'attack_change_amount_199_v1'
LAG_MAX = 9600                      # +-0.6 s
CLAP_BATCH, MERT_BATCH = 8, 16      # CLAP batch > 8 changes padding (cos diff 0.08 at 32); <= 8 equals per-file (3e-7)
WIN = 160000                        # 10 s window for 20 s stimuli / CLAP crop
log = A.log
SELF = Path(__file__).resolve()

# 197's T1 tokens (in 197 table order, see attack_metric_matrix_tables.T1_ROWS)
T1_197 = ['gain_p6', 'gain_m6', 'sil05', 'rb_id', 'ps_p1', 'ps_m1', 'ps_p2', 'ps_m2', 'ts_090', 'ts_110',
          'kick05_r12', 'kickrand_r12', 'click05_r12', 'sq220_r18', 'nburst05_r12', 'pno0']
T1_193 = ['lp6k', 'lp4k', 'lp2k', 'hp200', 'eq250_p10']
T1_190 = ['ctr_snr10', 'side_snr10', 'demucs']
T1_189 = ['sine1k_r18', 'osynth_r18']
T1_TREAT = T1_197 + T1_193 + T1_190 + T1_189
R_LIST = [24, 18, 12, 6, 0, -6, -12, -18]
WN_LIST = [40, 30, 20, 10, 0]


def rname(r):
    return f'mix_r{r}' if r >= 0 else f'mix_rm{-r}'


MIX_CELLS = [rname(r) for r in R_LIST]
WN_CELLS = [f'wn_snr{s}' for s in WN_LIST]
T1_LADDER = ['identity', 'other_clip'] + MIX_CELLS + WN_CELLS
T2A_TREAT = [c for c in A.T2A_CELLS if c != 'base']
T2A_LADDER = ['identity', 'other_excerpt', 'other_bank'] + MIX_CELLS
SEED_NEXT = {f'ma_cfg3neg_s{a}': f'ma_cfg3neg_s{b}' for a, b in
             zip(A.SEEDS, list(A.SEEDS[1:]) + list(A.SEEDS[:1]))}
COLS = ['id', 'rho', 'lag', 'mel', 'd_clap', 'd_mert']
TIMING = {}


def tadd(k, dt):
    TIMING[k] = TIMING.get(k, 0.0) + dt


def t1_cells(base):
    return T1_TREAT + T1_LADDER + (['other_seed'] if base in SEED_NEXT else [])


def derange(n):
    """fixed-seed derangement: partner[i] != i."""
    order = np.random.default_rng(SEED).permutation(n)
    partner = np.empty(n, int)
    for k in range(n):
        partner[order[k]] = order[(k + 1) % n]
    return partner


# ═════════ signal-layer quantities ═════════
_MELFB = None


def melfb():
    global _MELFB
    if _MELFB is None:
        import librosa
        _MELFB = librosa.filters.mel(sr=SR, n_fft=512, n_mels=64).astype(np.float64)
    return _MELFB


def align(x, y, lag_max=LAG_MAX):
    """-> (rho, lag, m0, m1): y[m + lag] ~ g x[m] on m in [m0, m1)."""
    from scipy import fft as sfft
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    nx, ny = len(x), len(y)
    n = sfft.next_fast_len(nx + ny - 1, real=True)
    c = sfft.irfft(sfft.rfft(y, n) * np.conj(sfft.rfft(x, n)), n)       # c[k] = sum_m y[m+k] x[m] (circular, no alias)
    taus = np.arange(-lag_max, lag_max + 1)
    cc = c[taus % n]
    cx = np.concatenate([[0.0], np.cumsum(x * x)])
    cy = np.concatenate([[0.0], np.cumsum(y * y)])
    m0 = np.maximum(0, -taus)
    m1 = np.minimum(nx, ny - taus)
    ok = m1 - m0 > 1000
    m0c, m1c = np.where(ok, m0, 0), np.where(ok, m1, 1)
    ex = cx[m1c] - cx[m0c]
    ey = cy[m1c + taus * ok] - cy[m0c + taus * ok]
    good = ok & (ex > 1e-12) & (ey > 1e-12)
    if not good.any():
        return 1.0, 0, 0, min(nx, ny)
    rho_t = np.where(good, 1.0 - cc * cc / np.where(good, ex * ey, 1.0), np.inf)
    j = int(np.argmin(rho_t))
    return float(max(rho_t[j], 0.0)), int(taus[j]), int(m0[j]), int(m1[j])


def logmel(s, floor_ref=None):
    from numpy.lib.stride_tricks import sliding_window_view
    s = np.asarray(s, np.float64)
    s = s / (math.sqrt(float(np.mean(s * s))) + 1e-12)
    fr = sliding_window_view(s, 400)[::160] * np.hanning(400)
    pw = np.abs(np.fft.rfft(fr, 512, axis=1)) ** 2
    db = 10 * np.log10(np.maximum(pw @ melfb().T, 1e-10))
    return db


def signal_layer(x, y):
    rho, lag, m0, m1 = align(x, y)
    xs = np.asarray(x, np.float64)[m0:m1]
    ys = np.asarray(y, np.float64)[m0 + lag:m1 + lag]
    if len(xs) < 800 or float(np.mean(xs * xs)) < 1e-14 or float(np.mean(ys * ys)) < 1e-14:
        return rho, lag, float('nan')
    dx, dy = logmel(xs), logmel(ys)
    fl = dx.max() - 80.0
    dx, dy = np.maximum(dx, fl), np.maximum(dy, fl)
    return rho, lag, float(np.mean(np.abs(dx - dy)))


# ═════════ embedding layer ═════════
class Emb:
    def __init__(self):
        self.clap = self.mert = self.feat = None
        self.torch = None

    def load(self):
        import torch
        self.torch = torch
        import eval_metrics as EM
        from muq_gen_steer_179_score import standard_hf_cache
        with standard_hf_cache():
            self.clap = EM.load_clap()
            import metric_win_tie_133 as W
            self.W = W
            self.feat, self.mert = W.load_mert()
        log('CLAP + MERT loaded')

    @staticmethod
    def windows(w):
        w = np.asarray(w, np.float32)
        if len(w) > 200000:                       # 20 s stimulus: front and back 10 s
            return [w[:WIN], w[-WIN:]]
        return [w]

    def _prep_clap(self, w):
        import librosa
        w = np.asarray(w, np.float32)[:WIN]       # CLAP crops > 10 s at random: truncate deterministically
        x48 = librosa.resample(w, orig_sr=SR, target_sr=48000)
        return (np.round(np.clip(x48, -1, 1) * 32767.0) / 32767.0).astype(np.float32)

    def _prep_mert(self, w):
        import librosa
        return librosa.resample(np.asarray(w, np.float32), orig_sr=SR, target_sr=24000).astype(np.float32)

    def _run(self, items, prep, fn, bs):
        """items: list of 1-D arrays -> (N, D) L2-normalised, batching only equal lengths."""
        pre = list(P.POOL.map(prep, items))
        out = [None] * len(pre)
        by = {}
        for k, a in enumerate(pre):
            by.setdefault(len(a), []).append(k)
        for ks in by.values():
            for s in range(0, len(ks), bs):
                b = ks[s:s + bs]
                e = np.asarray(fn([pre[k] for k in b]), np.float64)
                e = e / np.linalg.norm(e, axis=1, keepdims=True)
                for k, v in zip(b, e):
                    out[k] = v
        return np.stack(out)

    def clap_fn(self, arrs, per_file=False):
        with self.torch.no_grad():
            if per_file:
                return np.concatenate([self.clap.get_audio_embedding_from_data(a[None], use_tensor=False) for a in arrs])
            return self.clap.get_audio_embedding_from_data(np.stack(arrs), use_tensor=False)

    def mert_fn(self, arrs):
        return self.W.embed_mert_batch([a for a in arrs], self.feat, self.mert, 'cuda')

    def embed(self, ws):
        """ws: list of waveforms -> {'clap': (N, W, 512), 'mert': (N, W, 768)} (W = 1 or 2 windows)."""
        wins = [self.windows(w) for w in ws]
        nw = len(wins[0])
        res = {}
        for name, prep, fn, bs in (('clap', self._prep_clap, self.clap_fn, CLAP_BATCH),
                                   ('mert', self._prep_mert, self.mert_fn, MERT_BATCH)):
            t0 = time.time()
            flat = [a for w in wins for a in w]
            e = self._run(flat, prep, fn, bs)
            res[name] = e.reshape(len(ws), nw, -1)
            tadd(f'embed_{name}', time.time() - t0)
        return res


def dist(ex, ey):
    """(N, W, D) normalised embeddings -> d (N,) = mean over windows of 1 - cos."""
    return 1.0 - (ex * ey).sum(-1).mean(-1)


# ═════════ io / resume ═════════
def tsv_path(key):
    return OUT / 'per_clip' / f'{key}.tsv'


def cell_done(key, ids):
    p = tsv_path(key)
    if not p.is_file():
        return False
    try:
        rows = [ln.rstrip('\n').split('\t') for ln in p.read_text().splitlines()]
    except OSError:
        return False
    if len(rows) != len(ids) + 1 or rows[0] != COLS or {r[0] for r in rows[1:]} != set(ids):
        return False
    return all(len(r) == len(COLS) and all(math.isfinite(float(v)) for v in r[1:]) for r in rows[1:])


def write_tsv(key, ids, rec):
    p = tsv_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        f.write('\t'.join(COLS) + '\n')
        for k, i in enumerate(ids):
            f.write('\t'.join([str(i)] + [repr(float(v[k])) for v in rec]) + '\n')
    tmp.replace(p)


def save_emb(key, e):
    d = OUT / 'emb'
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f'{key}.tmp.npy', e['clap'].mean(1).astype(np.float32))
    (d / f'{key}.tmp.npy').replace(d / f'{key}.npy')


def score_cell(key, ids, xs, ys, ref_emb, X, save=True):
    """xs, ys: lists (aligned with ids). ref_emb: embeddings of xs. Writes the TSV."""
    t0 = time.time()
    sig = list(P.POOL.map(lambda a: signal_layer(*a), zip(xs, ys)))
    tadd('signal', time.time() - t0)
    e = X.embed(ys)
    if save:
        save_emb(key, e)
    rho = np.array([s[0] for s in sig])
    lag = np.array([s[1] for s in sig], float)
    mel = np.array([s[2] for s in sig])
    write_tsv(key, ids, [rho, lag, mel, dist(ref_emb['clap'], e['clap']), dist(ref_emb['mert'], e['mert'])])
    return rho, lag, mel


# ═════════ T1 ═════════
OV189 = {}


def build_overlays_189():
    sel = json.loads((H.EVAL_ROOT / 'muq_overlay_matrix_189' / 'selection.json').read_text())
    P.ATT.update(json.loads((H.EVAL_ROOT / 'muq_attractor_probe_188' / 'attractor.json').read_text()))
    OV189['osynth'] = P.build_C(sel['osynth'])
    OV189['sine1k'] = P.build_C(P.C_NAME['sine1k'])
    for k, v in OV189.items():
        assert len(v) == A.NS and np.all(np.isfinite(v)), k


def demucs_of(base, ids, wavs):
    """htdemucs separate-and-re-sum via 190's worker (karaoke-jp-align venv). -> dir with <id>.npy."""
    dout = OUT / 'demucs' / base
    need = [i for i in ids if not (dout / f'{i}.npy').is_file()]
    if need:
        din = OUT / 'demucs_in' / base
        shutil.rmtree(din, ignore_errors=True)
        din.mkdir(parents=True)
        for i in need:
            np.save(din / f'{i}.npy', wavs[i].astype(np.float32))
        env = {k: v for k, v in os.environ.items() if k != 'MKL_NUM_THREADS'}
        env.update(OMP_NUM_THREADS='2')
        t0 = time.time()
        r = subprocess.run([str(Q.KJ_PY), '-I', str(Path(Q.__file__).resolve()), '--demucs-worker', str(din), str(dout)],
                           cwd=A.REPO, env=env)
        tadd('demucs', time.time() - t0)
        shutil.rmtree(din, ignore_errors=True)
        if r.returncode or any(not (dout / f'{i}.npy').is_file() for i in need):
            raise RuntimeError(f'demucs worker failed for {base} rc={r.returncode}')
        log(f'demucs {base}: {len(need)} clips in {time.time() - t0:.1f}s')
    return dout


def t1_y(cell, base, i, k, ids, wavs, partner, other_w, dem_dir):
    w = wavs[i]
    if cell == 'identity':
        return w
    if cell == 'other_clip':
        return wavs[ids[partner[k]]]
    if cell == 'other_seed':
        return other_w[i]
    if cell.startswith('mix_r'):
        r = int(cell[5:]) if not cell.startswith('mix_rm') else -int(cell[6:])
        return P.mix(w, wavs[ids[partner[k]]], r)
    if cell.startswith('wn_snr'):
        snr = int(cell[6:])
        rng = P.rng_for(base, i, 'wn', snr)
        pw = float(np.mean(np.asarray(w, np.float64) ** 2))
        return P.lvl30(np.asarray(w, np.float64) + rng.standard_normal(len(w)) * math.sqrt(pw / 10 ** (snr / 10)))[0]
    if cell in T1_197:
        return A.t1_stim(cell, w, i, base)
    if cell in T1_193:
        import muq_filter_ladder_193 as F
        return F.apply_cell(w, cell)
    if cell in ('ctr_snr10', 'side_snr10'):
        return Q.build_f2(w, 10, 'ctr' if cell.startswith('ctr') else 'side', base, i, cell)
    if cell == 'demucs':
        return P.lvl30(P.fit(np.load(dem_dir / f'{i}.npy')))[0]
    if cell in ('sine1k_r18', 'osynth_r18'):
        return P.mix(w, OV189[cell.split('_')[0]], 18)
    raise KeyError(cell)


def stage_T1(smoke, X, bases, ids):
    n = len(ids)
    partner = derange(n)
    for base in bases:
        cells = t1_cells(base)
        todo = [c for c in cells if not cell_done(f'T1_{base}_{c}', ids)]
        if not todo:
            log(f'T1 {base}: all done')
            continue
        tb = time.time()
        wavs = M.load_base(base, ids)
        other_w = M.load_base(SEED_NEXT[base], ids) if base in SEED_NEXT else None
        dem_dir = demucs_of(base, ids, wavs) if 'demucs' in todo else None
        xs0 = [wavs[i] for i in ids]
        exts = [Q.ext_of(w) for w in xs0]
        ref = {'base': X.embed(xs0)}
        if any(c in ('ctr_snr10', 'side_snr10') for c in todo):
            ref['ext'] = X.embed(exts)
        save_emb(f'T1_{base}_ref_base', ref['base'])
        if 'ext' in ref:
            save_emb(f'T1_{base}_ref_ext', ref['ext'])
        for c in todo:
            t0 = time.time()
            key = f'T1_{base}_{c}'
            is_ext = c in ('ctr_snr10', 'side_snr10')
            ys = list(P.POOL.map(lambda a: t1_y(c, base, a[1], a[0], ids, wavs, partner, other_w, dem_dir), enumerate(ids)))
            tadd('build', time.time() - t0)
            xs = exts if is_ext else xs0
            rho, lag, mel = score_cell(key, ids, xs, ys, ref['ext' if is_ext else 'base'], X)
            log(f'{key}: rho {np.median(rho):.4f} mel {np.nanmedian(mel):.3f} lag {np.median(lag):.0f}  ({time.time() - t0:.1f}s)')
        shutil.rmtree(OUT / 'demucs' / base, ignore_errors=True)
        TIMING[f'base_wall_{base}'] = time.time() - tb
        del wavs, other_w


# ═════════ T2A ═════════
def stage_T2A(smoke, X, pool):
    ids = C.b_ids(smoke)
    n = len(ids)
    nex = n // 2
    banks = A.BANKS
    # render every cell once (fork pool), keep in memory (300 x 10 s x 13 cells float32 ~ 0.1 GB)
    wav = {}
    for cell in A.T2A_CELLS:
        t0 = time.time()
        arr = pool.map(A._job_A, [(k // 2, banks[k % 2], cell) for k in range(n)], chunksize=2)
        wav[cell] = [np.asarray(a, np.float32) for a in arr]
        tadd('render_T2A', time.time() - t0)
        log(f'T2A render {cell}: {time.time() - t0:.1f}s')
    base_w = wav['base']
    for b, bank in enumerate(banks):
        ks = [k for k in range(n) if k % 2 == b]
        bids = [ids[k] for k in ks]
        cells = T2A_TREAT + T2A_LADDER
        todo = [c for c in cells if not cell_done(f'T2A_{bank}_{c}', bids)]
        if not todo:
            continue
        xs = [base_w[k] for k in ks]
        ref = X.embed(xs)
        save_emb(f'T2A_{bank}_ref_base', ref)
        for c in todo:
            t0 = time.time()
            ys = []
            for k in ks:
                part_ex = (k + 2) % n                 # same bank, next excerpt
                if c in wav:
                    y = wav[c][k]
                elif c == 'identity':
                    y = base_w[k]
                elif c == 'other_excerpt':
                    y = base_w[part_ex]
                elif c == 'other_bank':
                    y = base_w[k ^ 1]
                else:
                    r = int(c[5:]) if not c.startswith('mix_rm') else -int(c[6:])
                    y = P.mix(base_w[k], base_w[part_ex], r)
                ys.append(y)
            rho, lag, mel = score_cell(f'T2A_{bank}_{c}', bids, xs, ys, ref, X)
            log(f'T2A_{bank}_{c}: rho {np.median(rho):.4f} mel {np.nanmedian(mel):.3f}  ({time.time() - t0:.1f}s)')


# ═════════ main ═════════
def run_all(smoke, part, bases):
    A.setup(smoke)             # sets C.OUT = A.OUT; repoint first
    ids_t1 = C.get_ids(smoke)[0]
    with open(C.EXCERPTS) as f:
        C.EXC = json.load(f)
    build_overlays_189()
    pool = mp.get_context('fork').Pool(P.NTHREAD)       # fork BEFORE CUDA / scorers exist
    if part in ('all', 'T1'):
        A.build_pno(pool)
    X = Emb()
    X.load()
    if part in ('all', 'T2A'):
        stage_T2A(smoke, X, pool)
    pool.close()
    if part in ('all', 'T1'):
        stage_T1(smoke, X, bases, ids_t1)
    P.jdump(OUT / 'timing.json', TIMING)
    log(f'timing {json.dumps({k: round(v, 1) for k, v in TIMING.items()})}')


# ═════════ validate / preflight / finalize ═════════
RESULT_PREFIX = REPO_RES = Path(__file__).resolve().parents[2] / 'docs/experiments/results/attack_change_amount_199'
MIN_FREE_PRE = 20 * 2**30


def all_cells(smoke, bases):
    A.setup(smoke)
    with open(C.EXCERPTS) as f:
        C.EXC = json.load(f)
    ids1 = C.get_ids(smoke)[0]
    ids2 = C.b_ids(smoke)
    out = [(f'T1_{b}_{c}', ids1) for b in bases for c in t1_cells(b)]
    for k, bank in enumerate(A.BANKS):
        bids = [ids2[j] for j in range(len(ids2)) if j % 2 == k]
        out += [(f'T2A_{bank}_{c}', bids) for c in T2A_TREAT + T2A_LADDER]
    return out


def validate(smoke, bases):
    errs = []
    try:
        for key, ids in all_cells(smoke, bases):
            if not cell_done(key, ids):
                errs.append(f'bad/missing cell {key}')
        sj = Path(str(RESULT_PREFIX) + '_summary.json')
        if not smoke:
            if not sj.is_file():
                errs.append('missing results summary.json')
            else:
                txt = sj.read_text()
                if 'NaN' in txt or 'Infinity' in txt:
                    errs.append('non-finite numbers in summary.json')
                d = json.loads(txt)
                if sorted(d.get('acceptance', {})) != ['d_clap', 'd_mert', 'mel', 'rho']:
                    errs.append('summary.json acceptance block incomplete')
            if not (OUT / 'timing.json').is_file():
                errs.append('missing timing.json')
    except SystemExit as e:
        errs.append(str(e))
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs[:30]:
        log(f'VALIDATE FAIL {e}')
    log('VALIDATE FAIL %d problem(s)' % len(errs) if errs else 'VALIDATE OK')
    return 1 if errs else 0


def preflight(smoke):
    errs = []
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUT).free < MIN_FREE_PRE:
        errs.append('less than 20 GiB free on NVMe')
    for p in (C.EXCERPTS, P.REF_DIR):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if errs:
        return 1
    if not (OUT / 'timing.json').is_file():
        if C.mem_available_gb() < (8 if smoke else 14):
            log('PREFLIGHT WAIT MemAvailable low')
            return 75
        if P.free_mib() < 8 * 1024:
            log('PREFLIGHT WAIT GPU free < 8 GiB')
            return 75
    log('PREFLIGHT OK')
    return 0


def finalize(smoke):
    if smoke:
        return
    r = subprocess.run([sys.executable, '-I', str(HERE.parent / 'analysis' / 'attack_change_amount_tables.py'),
                        '--root', str(OUT)])
    if r.returncode:
        raise SystemExit('tables script failed')


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--part', default='all', choices=['all', 'T1', 'T2A'])
    ap.add_argument('--bases', default=None, help='comma list (default: 6 listenable bases; smoke: real + one MeanAudio)')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    OUT = H.EVAL_ROOT / ('attack_change_amount_199_smoke' if a.smoke else 'attack_change_amount_199')
    A.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    bases = a.bases.split(',') if a.bases else (['real', 'ma_cfg3neg_s14159265'] if a.smoke else A.LISTEN)
    if a.preflight:
        return preflight(a.smoke)
    if a.validate_only:
        return validate(a.smoke, bases)
    free = shutil.disk_usage(OUT).free
    if free < A.MIN_FREE:
        raise SystemExit(f'free disk {free / 2**30:.1f} GiB < {A.MIN_FREE / 2**30:.0f} GiB')
    t0 = time.time()
    run_all(a.smoke, a.part, bases)
    log(f'done in {time.time() - t0:.0f}s')
    finalize(a.smoke)
    return validate(a.smoke, bases)


if __name__ == '__main__':
    sys.exit(main())
