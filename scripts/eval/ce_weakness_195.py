#!/usr/bin/env python
"""195: AES CE weakness (eval-only driver).

Design: docs/experiments/ce_weakness_195_20261113.md (pre-registered; this file only implements it).

Parts (all 16 kHz, 10 s, locked to -30 LUFS; AES = CE is the endpoint, MuQ/PAM/CMI-RM are descriptive guards):
  B  MAESTRO MIDI note-density ladder: 150 excerpts x 2 SoundFonts x 16 cells (extra_harm / extra_rand / wrong x 10,30,60,100 %,
     vel_rand, jitter_20ms, jitter_50ms); cluster = performance.
  C  MUSDB 126c windows: per-stem gain surface (4 stems x {-12,-6,-3,+3,+6,+12 dB, remove}); analysis only on windows where the
     stem is active; top-3 (stem, dB) by dCE are frozen in c_top3.json.
  D  recipe attack on 9 MusicCaps bases (same dev 200 / test 800 split as 193): mix* (htdemucs stems re-mixed with the C recipes),
     arp* (key-matched arpeggio under the music); selection on dev over the 6 listenable bases, frozen in selection.json before
     any test cell is scored.
  E  black-box search over a ~20 parameter synthesiser: separable numpy ES, 3 restarts, 20,000 AES evaluations, objective = CE;
     top-10 are frozen and run through all scorers, CE percentile vs the real MusicCaps clips.
Stage order: B, C, D-dev, select, D-test, E.  Every stage resumes from per-cell TSVs / es/ checkpoints.
Reuses 190 (Y scorer, analyse, req_cols), 189 (loaders, tsv), 123 stimuli; none of them is edited.
"""
import os
import sys

# ── demucs per-stem worker: runs before any heavy import (karaoke-jp-align venv has no MuQ stack) ──
if len(sys.argv) > 1 and sys.argv[1] == '--demucs-stems-worker':
    def _stems_worker(din, dout):
        import time as _t
        from pathlib import Path as _P

        import numpy as _np
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from scipy.signal import resample_poly
        din, dout = _P(din), _P(dout)
        dout.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in din.glob('*.npy') if not (dout / p.name).is_file())
        model = get_model('htdemucs').cuda().eval()
        assert list(model.sources) == ['drums', 'bass', 'other', 'vocals'], model.sources
        t0 = _t.time()
        for k in range(0, len(files), 4):
            fs = files[k:k + 4]
            xs, stats = [], []
            for p in fs:
                w = _np.load(p).astype(_np.float64)
                up = resample_poly(w, 441, 160).astype(_np.float32)
                st = torch.from_numpy(_np.stack([up, up]))
                ref = st.mean(0)
                m, s = ref.mean(), ref.std()
                xs.append((st - m) / s)
                stats.append((float(m), float(s), len(w)))
            T = max(x.shape[-1] for x in xs)
            batch = torch.stack([torch.nn.functional.pad(x, (0, T - x.shape[-1])) for x in xs])
            src = None
            for att in range(8):   # OOM retry (2026-11-14: 195 died when other procs filled VRAM)
                try:
                    with torch.no_grad():
                        if att == 0:
                            src = apply_model(model, batch, shifts=0, split=True, overlap=0.25, device='cuda', progress=False)
                        else:   # fall back to one clip at a time; same per-clip math
                            src = torch.cat([apply_model(model, batch[q:q + 1], shifts=0, split=True, overlap=0.25,
                                                         device='cuda', progress=False) for q in range(len(batch))])
                    break
                except torch.OutOfMemoryError:
                    src = None
                    torch.cuda.empty_cache()
                    print(f'demucs stems worker: CUDA OOM at batch {k} attempt {att}, sleep {60 * (att + 1)}s', flush=True)
                    _t.sleep(60 * (att + 1))
            if src is None:
                raise RuntimeError(f'demucs stems worker: CUDA OOM persisted at batch {k}')
            for j, p in enumerate(fs):
                n = xs[j].shape[-1]
                st = src[j, :, :, :n].float().cpu().numpy().mean(1) * stats[j][1]      # [4, n] mono, input domain (DC dropped)
                down = resample_poly(st.astype(_np.float64), 160, 441, axis=1)
                nn = stats[j][2]
                down = down[:, :nn] if down.shape[1] >= nn else _np.pad(down, ((0, 0), (0, nn - down.shape[1])))
                tmp = dout / (p.name + '.tmp.npy')
                _np.save(tmp, down.astype(_np.float16))
                tmp.replace(dout / p.name)
        print(f'demucs stems worker: {len(files)} clips in {_t.time() - t0:.1f}s', flush=True)
    _stems_worker(sys.argv[2], sys.argv[3])
    sys.exit(0)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muq_overlay_matrix_189 as M  # noqa: E402
import muq_postproc_attack_190 as Q  # noqa: E402
import midi_realpiece_stimuli as MS  # noqa: E402

P = M.P
H = M.H
REPO = P.REPO
SELF = Path(__file__).resolve()
SR, NS = P.SR, P.NS
OUT = H.EVAL_ROOT / 'ce_weakness_195'
KIND = 'ce_weakness_195_summary_v1'
SEL_KIND = 'ce_weakness_195_selection_v1'
IS_SMOKE = False
SEED = 20261113
NB = 5000
KJ_PY = Q.KJ_PY
DEMUCS_CKPT = Q.DEMUCS_CKPT
EXCERPTS = H.EVAL_ROOT / 'midi_realpiece_123/excerpts.json'
MC_BASES, BASES, SEEDS = M.MC_BASES, M.BASES, M.SEEDS
LISTEN = ['real'] + [f'ma_cfg3neg_s{s}' for s in SEEDS] + ['musicgen', 'musicldm']
SMOKE_D_BASES = ['real', 'musicgen']
MIN_FREE = 20 * 1024**3
STEMS = ['drums', 'bass', 'other', 'vocals']          # htdemucs source order
CSTEMS = ['vocals', 'drums', 'bass', 'other']          # 126c window cache order
DELTA = ['A1_MI', 'A3a_MI', 'PQ', 'CE', 'CU', 'PC', 'PAM', 'CMI_q']
M_CE_FRAC = 0.10

EXC = None            # excerpts, loaded before the process pool forks
ARPS = {}             # (tonic, mode, r) -> float32 arpeggio locked to -30 - r LU
KEYC = {}             # (base, id) -> (tonic, mode)

# ── Part B cells ──
DOSES = [10, 30, 60, 100]
B_CELLS = ['base'] + [f'{f}_{d}' for f in ('extra_harm', 'extra_rand', 'wrong') for d in DOSES] + \
          ['vel_rand', 'jitter_20ms', 'jitter_50ms']
BANKS_B = ['generaluser', 'fluidr3']
# ── Part C cells ──
C_TAGS = {'m12': -12.0, 'm6': -6.0, 'm3': -3.0, 'p3': 3.0, 'p6': 6.0, 'p12': 12.0, 'rm': 'rm'}
C_CELLS = [f'{s}_{t}' for s in CSTEMS for t in C_TAGS]
# ── Part D cells ──
D_DEV = ['base', 'mixA', 'mixB', 'mixC', 'mixABC', 'arp12', 'arp18']
D_TEST = ['base', 'mix_star', 'mix_ctrl', 'arp_star', 'arp_ctrl']

X = None


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def mem_available_gb():
    return Q.mem_available_gb()


class Z(Q.Y):
    """190's Y scorer without SongEval (tier 2 never uses it); keeps the CMI-RM `muq` sys.modules swap."""

    def load_all(self):
        from muq_gen_steer_179_score import muq_hf_cache
        P.SC._muq()
        P.SC._aes()
        P.SC._clap()
        with muq_hf_cache():
            import newmetrics_pam
            import newmetrics_cmirm
            newmetrics_pam._load('cuda')
            pip_muq = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'muq' or k.startswith('muq.')}
            try:
                newmetrics_cmirm._load('cuda', False)
            finally:
                for k in [k for k in sys.modules if k == 'muq' or k.startswith('muq.')]:
                    del sys.modules[k]
                sys.modules.update(pip_muq)
        w = (np.random.default_rng(0).standard_normal(NS) * 0.05).astype(np.float32)
        self.score([w, w], ['warm up', 'warm up'], 2, 'caption')
        self.t.clear()
        self.n.clear()
        P.SC.torch.cuda.reset_peak_memory_stats()


def lock(w):
    return P.fit(P.to_lufs(w, -30.0)[0])


# ═════════ Part B: MIDI notes ═════════
def _pitch_pool(notes):
    lo, hi = min(n[2] for n in notes), max(n[2] for n in notes)
    pcs = np.zeros(12)
    for n in notes:
        pcs[n[2] % 12] += 1
    cand = np.arange(lo, hi + 1)
    return lo, hi, cand, pcs[cand % 12]


def op_extra(frac, harmonic):
    def f(notes, r):
        lo, hi, cand, wt = _pitch_pool(notes)
        k = max(1, int(round(frac * len(notes))))
        vel = int(np.median([n[3] for n in notes]))
        if harmonic and wt.sum() > 0:
            p = r.choice(cand, size=k, p=wt / wt.sum())
        else:
            p = r.integers(lo, hi + 1, size=k)
        add = [(float(t), float(t) + 0.3, int(pp), vel) for t, pp in zip(r.uniform(0, 10.0, k), p)]
        return sorted(list(notes) + add)
    return f


def b_op(cell):
    if cell == 'base':
        return None
    if cell.startswith('extra_harm_'):
        return op_extra(int(cell.split('_')[-1]) / 100, True)
    if cell.startswith('extra_rand_'):
        return op_extra(int(cell.split('_')[-1]) / 100, False)
    if cell.startswith('wrong_'):
        return MS.wrong_notes(int(cell.split('_')[-1]) / 100)
    if cell == 'vel_rand':
        return MS.vel('random')
    if cell == 'jitter_20ms':
        return MS.jitter(0.020, False)
    if cell == 'jitter_50ms':
        return MS.jitter(0.050, False)
    raise KeyError(cell)


def b_ids(smoke):
    n = 4 if smoke else len(EXC)
    return [f'{EXC[k]["id"]}_{b}' for k in range(n) for b in BANKS_B]


def _job_B(job):
    exi, bank, cell = job
    ex = EXC[exi]
    notes = sorted(tuple(n) for n in ex['notes'] if n[0] < 10.0)
    fn = b_op(cell)
    if fn is not None:
        notes = sorted(fn(notes, np.random.default_rng(MS.h('195B', ex['id'], cell, SEED))))
    notes = [n for n in notes if n[0] < 10.0]
    x = MS.synth(notes, bank, 0, 0)
    y = x[int(MS.LEAD * SR):int(MS.LEAD * SR) + 160000]
    return lock(y)


# ═════════ Part C: stems ═════════
def win_info():
    return {w['win']: w for w in M.windows()}


def stem_active(w, s):
    st = w['stems'][s]
    return bool(st['lufs'] >= -45 and st['lufs'] >= w['mix_lufs_raw'] - 20 and st['active_frac'] >= 0.6)


def load_stems_c(wid):
    z = np.load(M.MUSDB_DIR / 'cache' / f'{wid}.npz')
    return {s: np.asarray(z[s][8000:168000], np.float64) for s in CSTEMS}


def c_wave(stems, s, tag):
    mix = np.zeros(160000)
    for k in CSTEMS:
        g = 1.0
        if k == s:
            g = 0.0 if C_TAGS[tag] == 'rm' else 10 ** (C_TAGS[tag] / 20)
        mix += g * stems[k]
    return lock(mix)


# ═════════ Part D: recipes ═════════
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_of(w):
    import librosa
    ch = librosa.feature.chroma_stft(y=np.asarray(w, np.float32), sr=SR, n_fft=2048, hop_length=512).mean(1)
    if not np.isfinite(ch).all() or ch.std() < 1e-9:
        return (0, 0)
    best, arg = -9.0, (0, 0)
    for mode, prof in ((0, KS_MAJ), (1, KS_MIN)):
        for t in range(12):
            c = float(np.corrcoef(ch, np.roll(prof, t))[0, 1])
            if c > best:
                best, arg = c, (t, mode)
    return arg


def _job_arp(key):
    t, mode = key
    iv = [0, 3 if mode else 4, 7, 12]
    root = 48 + t
    notes = [(0.25 * k, 0.25 * k + 0.3, root + iv[k % 4], 80) for k in range(40)]
    x = MS.synth(notes, 'generaluser', 0, 0)
    return np.asarray(x[int(MS.LEAD * SR):int(MS.LEAD * SR) + 160000], np.float64)


def build_arps(pool):
    keys = [(t, m) for m in (0, 1) for t in range(12)]
    raw = pool.map(_job_arp, keys)
    for k, x in zip(keys, raw):
        for r in (12, 18):
            ARPS[(k[0], k[1], r)] = P.fit(P.to_lufs(x, -30.0 - r)[0])


def arp_wave(w, key, r, ctrl):
    t, m = key
    if ctrl:
        t = (t + 6) % 12
    return lock(np.asarray(w, np.float64) + ARPS[(t, m, r)])


def mix_wave(st, recipe):
    g = np.ones(4)
    for si, db in recipe:
        g[si] = 0.0 if db == 'rm' else 10 ** (db / 20)
    y = (np.asarray(st, np.float32) * g[:, None].astype(np.float32)).sum(0)
    return lock(y)


def ctrl_recipe(recipe):
    return [(si, (12.0 if db == 'rm' else -db)) for si, db in recipe]


def recipes_from_top3(top3):
    rec = [[(STEMS.index(e['stem']), e['db'])] for e in top3]
    while len(rec) < 3:
        rec.append(rec[-1])
    return {'mixA': rec[0], 'mixB': rec[1], 'mixC': rec[2], 'mixABC': rec[0] + rec[1] + rec[2]}


# ═════════ Part E: synthesiser ═════════
NP = 20
_CH = [[0, 4, 7], [0, 3, 7], [0, 5, 7], [0, 7, 12], [0, 3, 6], [0, 4, 7, 11]]
_rng0 = np.random.default_rng(195)
_NOISE = _rng0.standard_normal(16000).astype(np.float64)
_VELR = _rng0.random(256)


def synth_E(p):
    p = np.clip(np.asarray(p, float), 0, 1)
    n = NS
    T = n / SR
    wts = p[0:3] + 1e-3
    wts = wts / wts.sum()
    iv = _CH[min(int(p[3] * 6), 5)]
    midi = 12 * (2 + min(int(p[5] * 4), 3) + 1) + int(p[4] * 12) % 12
    nlen = 0.05 + p[6] * 0.95
    A, D, S, R = 0.002 + p[7] * 0.2, 0.01 + p[8] * 0.5, p[9], 0.01 + p[10] * 0.5
    step = 1.0 / (1.0 + p[11] * 7.0)
    nv = min(1 + min(int(p[12] * 4), 3), len(iv))
    tilt, lp = -12 + p[13] * 18, 500.0 * 16.0 ** p[14]
    lk, ls, lh = p[15], p[16], p[17]
    dyn, cents = p[18], p[19] * 30.0
    buf = np.zeros(n)
    for k, on in enumerate(np.arange(0.0, T - 0.05, step)):
        f = 440.0 * 2 ** ((midi + iv[k % nv] - 69) / 12)
        i0 = int(on * SR)
        L = min(int((nlen + R) * SR), n - i0)
        if L < 8:
            continue
        t = np.arange(L) / SR
        xp = np.maximum.accumulate(np.array([0.0, A, A + D, nlen, nlen + R]) + np.arange(5) * 1e-6)
        env = np.interp(t, xp, [0.0, 1.0, S, S, 0.0])
        fs = [f] + ([f * 2 ** (cents / 1200)] if cents > 0.5 else [])
        w = 0.0
        for ff in fs:
            ph = ff * t
            w = w + wts[0] * np.sin(2 * np.pi * ph) + wts[1] * (2 * (ph % 1) - 1) + wts[2] * np.sign(np.sin(2 * np.pi * ph))
        buf[i0:i0 + L] += w / len(fs) * env * (1.0 - dyn * _VELR[k % 256])
    r0 = float(np.sqrt(np.mean(buf ** 2))) or 0.05
    dr = np.zeros(n)

    def hit(pat, at, per, off):
        for on in np.arange(off, T, per):
            i0 = int(on * SR)
            m = min(len(pat), n - i0)
            if m > 0:
                dr[i0:i0 + m] += pat[:m]
    tt = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 * tt + 75 * 0.03 * (1 - np.exp(-tt / 0.03)))) * np.exp(-tt / 0.05)
    snare = _NOISE[:len(tt)] * np.exp(-tt / 0.04)
    tt2 = np.arange(int(0.05 * SR)) / SR
    hat = np.diff(_NOISE[:len(tt2) + 1]) * np.exp(-tt2 / 0.01)
    buf = buf + 0.0
    if lk > 0:
        d = np.zeros(n)
        dr = d
        hit(kick * lk * r0 * 2.5, 0, 0.5, 0.0)
        buf += dr
    if ls > 0:
        dr = np.zeros(n)
        hit(snare * ls * r0 * 1.5, 0, 1.0, 0.5)
        buf += dr
    if lh > 0:
        dr = np.zeros(n)
        hit(hat * lh * r0 * 2.0, 0, 0.25, 0.0)
        buf += dr
    X_ = np.fft.rfft(buf)
    f_ = np.fft.rfftfreq(n, 1 / SR)
    g = 10 ** (tilt * np.log2(np.maximum(f_, 50.0) / 1000.0) / 20) / np.sqrt(1 + (f_ / lp) ** 8)
    buf = np.fft.irfft(X_ * g, n)
    if not np.isfinite(buf).all() or np.max(np.abs(buf)) < 1e-9:
        buf = np.random.default_rng(1).standard_normal(n) * 1e-4
    return lock(buf)


def _job_E(p):
    return synth_E(p)


# ═════════ run machinery ═════════
def setup(smoke):
    M.OUT = OUT
    P.OUT = OUT
    Q.OUT = OUT
    P.IS_SMOKE = smoke
    P.CFG['chunk'] = 16 if smoke else 64
    P.POOL = ThreadPoolExecutor(P.NTHREAD)
    H.load_scales()
    OUT.mkdir(parents=True, exist_ok=True)


def get_ids(smoke):
    mc, win = M.get_ids(smoke)
    mc = mc[:16] if smoke else mc
    return mc, win


def make_split(mc):
    rng = np.random.default_rng(SEED)
    ids = sorted(mc)
    n_dev = 200 if len(ids) == 1000 else 8
    dev = set(np.array(ids)[rng.permutation(len(ids))[:n_dev]].tolist())
    return {'seed': SEED, 'dev_mc': [i for i in mc if i in dev], 'test_mc': [i for i in mc if i not in dev]}


def get_split(mc):
    sp = make_split(mc)
    p = OUT / 'split.json'
    if not p.is_file():
        P.jdump(p, sp)
    return sp


def d_bases(smoke):
    return SMOKE_D_BASES if smoke else MC_BASES


def listen_bases(smoke):
    return [b for b in d_bases(smoke) if b in LISTEN]


def run_cell(key, ids, build, caps, tier, clap, cmi_mode, prefetch=None):
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
        rows = X.score(wavs, [caps[x] for x in c] if caps else None, tier, cmi_mode, sr=SR, clap=clap)
        for x, r in zip(c, rows):
            bad = [q for q in req if not math.isfinite(r.get(q, math.nan))]
            if bad:
                raise RuntimeError(f'{key}: non-finite {bad} for {x}: {r}')
            res[x] = r
        P.progress('score', key, len(res), len(ids))
    M.tsv_write(path, res, ids)
    wall = time.time() - t0
    dt = {k: X.t[k] - t_before.get(k, 0.0) for k in X.t}
    P.add_timing(key, len(ids), wall, dt)
    M.cost_add(dt, {k: X.n[k] - n_before.get(k, 0) for k in X.n})
    P.jdump(OUT / 'stim_stats' / f'{key}.json', {'n': len(ids), 'n_abs_over_1': over, 'max_abs': mx})
    log(f'{key}: {len(ids)} stimuli in {wall:.1f}s ({len(ids) / wall:.2f}/s); {over} with |x|>1 (max {mx:.3f})')
    return res


def cell_done(key, ids, tier, clap, has_caps):
    p = OUT / 'per_clip' / f'{key}.tsv'
    return p.is_file() and M.rows_ok(M.tsv_read(p), ids, Q.req_cols(tier, has_caps, clap))


# ── Part B ──
def stage_B(smoke, pool):
    ids = b_ids(smoke)
    nex = len(ids) // 2
    idx = {i: k for k, i in enumerate(ids)}
    todo = [c for c in B_CELLS if not cell_done(f'B_{c}', ids, 2, False, False)]

    def submit(cell):
        jobs = [(k // 2, BANKS_B[k % 2], cell) for k in range(len(ids))]
        return pool.map_async(_job_B, jobs, chunksize=2)
    P.progress('B', '', 0, len(todo))
    pend = submit(todo[0]) if todo else None
    for n, cell in enumerate(todo):
        t0 = time.time()
        arr = pend.get()
        render_s = time.time() - t0
        pend = submit(todo[n + 1]) if n + 1 < len(todo) else None
        arr = {i: arr[idx[i]] for i in ids}
        run_cell(f'B_{cell}', ids, lambda i, a=arr: a[i], None, 2, False, 'empty')
        log(f'B {cell}: wait for render {render_s:.1f}s')
    log(f'Part B done ({nex} excerpts x 2 banks = {len(ids)} ids)')


# ── Part C ──
def c_top3_path():
    return OUT / 'c_top3.json'


def c_active(win):
    wi = win_info()
    return {s: [w for w in win if stem_active(wi[w], s)] for s in CSTEMS}


def stage_C(smoke):
    mc, win = get_ids(smoke)
    wi = win_info()
    act = c_active(win)
    P.jdump(OUT / 'c_active.json', {s: len(v) for s, v in act.items()} | {'n_windows': len(win)})
    t0 = time.time()
    stem_cache = {w: load_stems_c(w) for w in win}
    log(f'loaded {len(win)} window stems in {time.time() - t0:.1f}s')
    run_cell('C_base', win, lambda i: lock(sum(stem_cache[i].values())), None, 2, False, 'empty')
    for s in CSTEMS:
        for t in C_TAGS:
            ids = act[s]
            if not ids:
                continue
            run_cell(f'C_{s}_{t}', ids, lambda i, s=s, t=t: c_wave(stem_cache[i], s, t), None, 2, False, 'empty')
    del stem_cache
    freeze_c_top3(win)


def freeze_c_top3(win):
    p = c_top3_path()
    if p.is_file():
        return json.loads(p.read_text())
    act = c_active(win)
    tmap = M.track_map()
    base = M.tsv_read(OUT / 'per_clip/C_base.tsv')
    res = {}
    for s in CSTEMS:
        for t in C_TAGS:
            f = OUT / 'per_clip' / f'C_{s}_{t}.tsv'
            if not f.is_file() or not act[s]:
                continue
            rows = M.tsv_read(f)
            d = M.arr(rows, act[s], 'CE') - M.arr(base, act[s], 'CE')
            res[(s, t)] = float(np.nanmean(d))
    best = {}
    for (s, t), v in res.items():
        if s not in best or v > best[s][1]:
            best[s] = (t, v)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][1])[:3]
    top3 = [{'stem': s, 'tag': t, 'db': C_TAGS[t], 'mean_dCE_active': v} for s, (t, v) in ranked]
    out = {'kind': 'ce_weakness_195_c_top3_v1', 'top3': top3,
           'all_mean_dCE': {f'{s}_{t}': v for (s, t), v in res.items()},
           'rule': 'best dB per stem by mean dCE over active windows, top-3 stems'}
    P.jdump(p, out)
    log('C top-3 frozen: ' + ', '.join(f'{e["stem"]}_{e["tag"]} {e["mean_dCE_active"]:+.3f}' for e in top3))
    return out


# ── Part D ──
def demucs_stems(base, ids):
    dout = OUT / 'stems' / base
    need = [i for i in ids if not (dout / f'{i}.npy').is_file()]
    if not need:
        return dout
    wavs = M.load_base(base, need)
    din = OUT / 'stems_in' / base
    shutil.rmtree(din, ignore_errors=True)
    din.mkdir(parents=True)
    for i in need:
        np.save(din / f'{i}.npy', wavs[i].astype(np.float32))
    del wavs
    env = {k: v for k, v in os.environ.items() if k != 'MKL_NUM_THREADS'}
    env.update(OMP_NUM_THREADS='2')
    t0 = time.time()
    for att in range(4):   # worker skips stems already written, so a retry resumes
        r = subprocess.run([str(KJ_PY), '-I', str(SELF), '--demucs-stems-worker', str(din), str(dout)], cwd=REPO, env=env)
        if not r.returncode:
            break
        log(f'demucs stems {base}: worker rc={r.returncode} attempt {att}, retry in {300 * (att + 1)}s')
        time.sleep(300 * (att + 1))
    wall = time.time() - t0
    shutil.rmtree(din, ignore_errors=True)
    if r.returncode or any(not (dout / f'{i}.npy').is_file() for i in need):
        raise RuntimeError(f'demucs stems worker failed for {base} rc={r.returncode}')
    cp = OUT / 'demucs_cost.json'
    c = json.loads(cp.read_text()) if cp.is_file() else {}
    c[f'{base}_{len(c)}'] = {'n': len(need), 'wall_s': wall}
    P.jdump(cp, c)
    log(f'demucs stems {base}: {len(need)} clips in {wall:.1f}s')
    return dout


def d_builders(base, wavs, stems_dir, recipes, which):
    """which: cell name -> builder(i)."""
    def mix(rec):
        return lambda i: mix_wave(np.load(stems_dir / f'{i}.npy'), rec)

    def arp(r, ctrl):
        def f(i):
            if (base, i) not in KEYC:
                KEYC[(base, i)] = key_of(wavs[i])
            return arp_wave(wavs[i], KEYC[(base, i)], r, ctrl)
        return f
    out = {'base': lambda i: wavs[i]}
    for c in which:
        if c in ('mixA', 'mixB', 'mixC', 'mixABC'):
            out[c] = mix(recipes[c])
        elif c == 'mix_star':
            out[c] = mix(recipes[recipes['_sel']['mix_star']])
        elif c == 'mix_ctrl':
            out[c] = mix(ctrl_recipe(recipes[recipes['_sel']['mix_star']]))
        elif c in ('arp12', 'arp18'):
            out[c] = arp(int(c[3:]), False)
        elif c == 'arp_star':
            out[c] = arp(int(recipes['_sel']['arp_star'][3:]), False)
        elif c == 'arp_ctrl':
            out[c] = arp(int(recipes['_sel']['arp_star'][3:]), True)
    return out


def stage_D_dev(smoke, caps_all, top3):
    mc, _ = get_ids(smoke)
    sp = get_split(mc)
    dev = sp['dev_mc']
    recipes = recipes_from_top3(top3['top3'])
    for base in listen_bases(smoke):
        keys = {c: f'Ddev_{base}_{c}' for c in D_DEV}
        if all(cell_done(k, dev, 2, True, True) for k in keys.values()):
            continue
        wavs = M.load_base(base, dev)
        sd = demucs_stems(base, dev)
        bl = d_builders(base, wavs, sd, recipes, D_DEV[1:])
        caps = {i: caps_all[i] for i in dev}
        for c in D_DEV:
            run_cell(keys[c], dev, bl[c], caps, 2, True, 'caption')
        del wavs


def select_D(smoke):
    p = OUT / 'selection.json'
    if p.is_file():
        return json.loads(p.read_text())
    mc, _ = get_ids(smoke)
    dev = get_split(mc)['dev_mc']
    lb = listen_bases(smoke)
    dm = {}
    for c in D_DEV[1:]:
        v = []
        for b in lb:
            a = M.tsv_read(OUT / 'per_clip' / f'Ddev_{b}_{c}.tsv')
            r = M.tsv_read(OUT / 'per_clip' / f'Ddev_{b}_base.tsv')
            v.append(float(np.nanmean(M.arr(a, dev, 'CE') - M.arr(r, dev, 'CE'))))
        dm[c] = float(np.mean(v))
    mix = max(['mixA', 'mixB', 'mixC', 'mixABC'], key=lambda c: dm[c])
    arp = max(['arp12', 'arp18'], key=lambda c: dm[c])
    sel = {'kind': SEL_KIND, 'seed': SEED, 'n_dev': len(dev), 'bases_used': lb, 'mix_star': mix, 'arp_star': arp,
           'dev_mean_dCE': dm, 'written_before_test_aggregates': True}
    P.jdump(p, sel)
    log(f'selection frozen: mix_star={mix} arp_star={arp} dev dCE={dm}')
    return sel


def stage_D_test(smoke, caps_all, top3, sel):
    mc, _ = get_ids(smoke)
    test = get_split(mc)['test_mc']
    recipes = recipes_from_top3(top3['top3'])
    recipes['_sel'] = sel
    for base in d_bases(smoke):
        keys = {c: f'D_{base}_{c}' for c in D_TEST}
        if all(cell_done(k, test, 2, True, True) for k in keys.values()):
            shutil.rmtree(OUT / 'stems' / base, ignore_errors=True)
            continue
        wavs = M.load_base(base, test)
        sd = demucs_stems(base, test)
        bl = d_builders(base, wavs, sd, recipes, D_TEST[1:])
        caps = {i: caps_all[i] for i in test}
        for c in D_TEST:
            run_cell(keys[c], test, bl[c], caps, 2, True, 'caption')
        del wavs
        shutil.rmtree(OUT / 'stems' / base, ignore_errors=True)


# ── Part E ──
def es_params(smoke):
    lam = 20 if smoke else 40
    budget = 200 if smoke else 20000
    R = 3
    return lam, R, budget // (R * lam), budget


def stage_E(smoke, pool):
    lam, R, gens, budget = es_params(smoke)
    (OUT / 'es').mkdir(parents=True, exist_ok=True)
    sp, ep = OUT / 'es/state.json', OUT / 'es/evals.jsonl'
    state = json.loads(sp.read_text()) if sp.is_file() else {'restarts': {}}
    lines = []
    if ep.is_file():
        for ln in ep.read_text().splitlines():
            rec = json.loads(ln)
            if rec['g'] < state['restarts'].get(str(rec['r']), {'gen': 0})['gen']:
                lines.append(ln)
    P.atomic(ep, '\n'.join(lines) + ('\n' if lines else ''))
    mu = lam // 4
    wr = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wr = wr / wr.sum()
    t_es0 = time.time()
    n_new = 0
    for r in range(R):
        st = state['restarts'].get(str(r))
        if st is None:
            st = {'gen': 0, 'mean': np.random.default_rng([SEED, 195, r]).uniform(0.2, 0.8, NP).tolist(),
                  'sigma': [0.25] * NP}
        mean, sig = np.array(st['mean']), np.array(st['sigma'])
        for g in range(st['gen'], gens):
            rng = np.random.default_rng([SEED, 195, r, g])
            pop = np.clip(mean + sig * rng.standard_normal((lam, NP)), 0, 1)
            wavs = pool.map(_job_E, list(pop), chunksize=2)
            aes = X.aes4(wavs, SR)
            ce = np.array(aes['CE'])
            ce = np.where(np.isfinite(ce), ce, -9.0)
            with open(ep, 'a') as f:
                for k in range(lam):
                    f.write(json.dumps({'r': r, 'g': g, 'p': pop[k].tolist(), **{c: float(aes[c][k]) for c in Q.AES_COLS}}) + '\n')
            top = np.argsort(-ce)[:mu]
            xs = pop[top]
            nm = (wr[:, None] * xs).sum(0)
            ns = np.sqrt((wr[:, None] * (xs - nm) ** 2).sum(0) + 1e-12)
            sig = np.maximum(0.7 * sig + 0.3 * ns, 0.02)
            mean = nm
            n_new += lam
            P.jdump(sp, {'restarts': {**state['restarts'], str(r): {'gen': g + 1, 'mean': mean.tolist(), 'sigma': sig.tolist()}},
                         'lam': lam, 'R': R, 'gens': gens})
            state = json.loads(sp.read_text())
            P.progress('E', f'r{r}g{g}', n_new, budget)
        state['restarts'][str(r)] = {'gen': gens, 'mean': mean.tolist(), 'sigma': sig.tolist()}
        P.jdump(sp, {'restarts': state['restarts'], 'lam': lam, 'R': R, 'gens': gens})
    if n_new:
        P.add_timing('E_es', n_new, time.time() - t_es0, {})
    ev = [json.loads(ln) for ln in ep.read_text().splitlines()]
    tp = OUT / 'e_top10.json'
    if not tp.is_file():
        order = sorted(range(len(ev)), key=lambda k: -ev[k]['CE'])
        seen, top = set(), []
        for k in order:
            key = tuple(round(x, 6) for x in ev[k]['p'])
            if key in seen:
                continue
            seen.add(key)
            top.append({'id': f'e{len(top):02d}', 'p': ev[k]['p'], 'r': ev[k]['r'], 'g': ev[k]['g'],
                        **{c: ev[k][c] for c in Q.AES_COLS}})
            if len(top) == 10:
                break
        P.jdump(tp, {'kind': 'ce_weakness_195_e_top10_v1', 'n_evals': len(ev), 'top10': top})
    top = json.loads(tp.read_text())['top10']
    ids = [t['id'] for t in top]
    arr = {t['id']: synth_E(t['p']) for t in top}
    caps = {i: 'music' for i in ids}
    run_cell('E_top10', ids, lambda i: arr[i], caps, 2, True, 'caption')


# ═════════ run all ═════════
def run_all(smoke):
    global X, EXC
    setup(smoke)
    EXC = json.loads(EXCERPTS.read_text())
    mc, win = get_ids(smoke)
    get_split(mc)
    caps_all = P.caps_all()
    P.progress('start', '', 0, 0)
    pool = mp.get_context('fork').Pool(P.NTHREAD)         # forked before CUDA / scorers exist
    build_arps(pool)
    X = Z()
    X.load_all()
    log(f'scorers loaded; torch max alloc {P.SC.torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    stage_B(smoke, pool)
    stage_C(smoke)
    top3 = json.loads(c_top3_path().read_text())
    stage_D_dev(smoke, caps_all, top3)
    sel = select_D(smoke)
    stage_D_test(smoke, caps_all, top3, sel)
    stage_E(smoke, pool)
    pool.close()
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
    cp = OUT / 'demucs_cost.json'
    dem = json.loads(cp.read_text()) if cp.is_file() else {}

    def rate(prefix):
        v = [x for k, x in t['keys'].items() if k.startswith(prefix) and x['n'] and k != 'E_es']
        n = sum(x['n'] for x in v)
        return (sum(x['wall_s'] for x in v) / n) if n else None
    r = {'B': rate('B_'), 'C': rate('C_'), 'Ddev': rate('Ddev_'), 'D': rate('D_')}
    dn, dw = sum(x['n'] for x in dem.values()), sum(x['wall_s'] for x in dem.values())
    rd = dw / dn if dn else None
    mc, win = get_ids(smoke)
    act = c_active(win)
    frac = (sum(len(v) for v in act.values()) / (4 * len(win))) if win else 0.6
    n_c = 551 + 4 * 7 * frac * 551
    est = {'B': r['B'] * 4800 if r['B'] else None,
           'C': r['C'] * n_c if r['C'] else None,
           'D_dev': (r['Ddev'] * 6 * 200 * 7 if r['Ddev'] else None),
           'D_test': (r['D'] * 9 * 800 * 5 if r['D'] else None),
           'demucs': (rd * (6 * 200 + 9 * 800) if rd else None)}
    es = t['keys'].get('E_es')
    if es:
        est['E'] = es['wall_s'] / es['n'] * 20000
    tot = sum(v for v in est.values() if v) / 3600
    rec = {'per_clip_s': r, 'demucs_s_per_clip': rd, 'est_stage_hours': {k: (v / 3600 if v else None) for k, v in est.items()},
           'projected_total_hours': tot,
           'note': 'smoke timings include warm-up / small-chunk overhead, so this over-estimates; C scaled by measured active fraction'}
    P.jdump(OUT / 'projection.json', rec)
    log('PROJECTION stage hours: ' + json.dumps({k: (round(v / 3600, 2) if v else None) for k, v in est.items()}))
    log(f'PROJECTION full run: {tot:.1f} h')
    return rec


# ═════════ summary ═════════
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


def rd_cell(key):
    p = OUT / 'per_clip' / f'{key}.tsv'
    return M.tsv_read(p) if p.is_file() else None


def dci(a, r, ids, bt, cols=DELTA):
    out = {}
    for c in cols:
        ci = bt.ci(M.arr(a, ids, c) - M.arr(r, ids, c))
        if ci is not None:
            out[c] = ci
    return out


def summarize_B(smoke):
    ids = b_ids(smoke)
    ex = {e['id']: e for e in (EXC or json.loads(EXCERPTS.read_text()))}
    bt = Boot([ex[i.rsplit('_', 1)[0]]['midi'] for i in ids])
    base = rd_cell('B_base')
    cells = {c: rd_cell(f'B_{c}') for c in B_CELLS}
    d = {c: dci(cells[c], base, ids, bt) for c in B_CELLS if c != 'base' and cells[c]}
    rep = d.get('extra_harm_30', {}).get('CE')
    replication = ('replicates' if (rep and rep[0] > 0 and rep[1] > 0) else 'not_replicated') if rep else 'missing'
    diffs = {}
    for dose in (30, 60, 100):
        h, r = cells.get(f'extra_harm_{dose}'), cells.get(f'extra_rand_{dose}')
        if h and r:
            diffs[dose] = bt.ci(M.arr(h, ids, 'CE') - M.arr(r, ids, 'CE'))
    nh = sum(1 for v in diffs.values() if v and v[1] > 0)
    nd = sum(1 for v in diffs.values() if v and v[2] < 0)
    pref = 'CE prefers in-harmony added notes' if nh >= 2 else ('CE prefers dissonant added notes' if nd >= 2 else '不分和聲')
    sat = {}
    for fam in ('extra_harm', 'extra_rand', 'wrong'):
        mv = {dose: d.get(f'{fam}_{dose}', {}).get('CE', [math.nan])[0] for dose in DOSES}
        mx = max((v for v in mv.values() if math.isfinite(v)), default=math.nan)
        sat[fam] = {'mean_dCE_by_dose': mv, 'max': mx,
                    'saturation_dose_pct': next((dose for dose in DOSES if math.isfinite(mv[dose]) and mx > 0 and mv[dose] >= 0.9 * mx), None)}
    return {'n_ids': len(ids), 'n_clusters': bt.K, 'cells': d, 'replication_extra_harm_30': replication,
            'replication_dCE_extra_harm_30': rep, 'harm_minus_rand_dCE': diffs, 'harmony_preference': pref,
            'n_harm_gt_rand_ci': nh, 'n_rand_gt_harm_ci': nd, 'saturation': sat,
            'ref_123_dCE_range': [0.15, 0.17]}


def summarize_C(smoke):
    mc, win = get_ids(smoke)
    act = c_active(win)
    tmap = M.track_map()
    base = rd_cell('C_base')
    out = {'n_windows': len(win), 'n_active': {s: len(v) for s, v in act.items()}, 'cells': {}}
    for s in CSTEMS:
        for t in C_TAGS:
            cell = rd_cell(f'C_{s}_{t}')
            ids = act[s]
            if cell is None or not ids:
                continue
            bt = Boot([tmap[i] for i in ids])
            out['cells'][f'{s}_{t}'] = {'n': len(ids), 'delta': dci(cell, base, ids, bt)}
    top3 = json.loads(c_top3_path().read_text())
    out['top3'] = top3['top3']
    return out


def success_D(e):
    ce, cd = e['delta'].get('CE'), (e.get('vs_control') or {}).get('CE_diff')
    m_ce = M_CE_FRAC * H.S['CE_lvl30']
    return bool(ce and cd and ce[1] > m_ce and cd[1] > 0), m_ce


def head(n):
    return 'established' if n >= 5 else ('base_dependent' if n >= 2 else 'not_established')


def summarize_D(smoke, sel):
    mc, _ = get_ids(smoke)
    test = get_split(mc)['test_mc']
    S_CE = H.S['CE_lvl30']
    lb = listen_bases(smoke)
    cells = {}
    for b in d_bases(smoke):
        for c in D_TEST:
            r = rd_cell(f'D_{b}_{c}')
            if r:
                cells[(b, c)] = r

    def bmean(b, c):
        v = M.arr(cells[(b, 'base')], test, c)
        return float(np.nanmean(v)) if np.isfinite(v).any() else math.nan
    S = {}
    have = [b for b in d_bases(smoke) if (b, 'base') in cells]
    for c in ('PQ', 'PAM', 'CMI_q', 'CE'):
        g = [bmean(b, c) for b in have]
        S[c] = float(max(g) - min(g)) if g and all(math.isfinite(x) for x in g) else None
    out = {'S_CE': S_CE, 'm_CE': M_CE_FRAC * S_CE, 'S_tie_from_bases': S, 'selection': sel, 'n_test': len(test),
           'listenable_bases': lb, 'bases': {}}
    bt = Boot(list(range(len(test))))
    for b in d_bases(smoke):
        if (b, 'base') not in cells:
            continue
        rec = {}
        for ov, ctrl in (('mix_star', 'mix_ctrl'), ('arp_star', 'arp_ctrl')):
            if (b, ov) not in cells:
                continue
            e = Q.analyse(cells, b, test, bt, S, S_CE, ov, 'base', ctrl)
            e['success'], _ = success_D(e)
            g = e.get('guards', {})
            ce_up = bool(e['delta'].get('CE') and e['delta']['CE'][1] > 0)
            e['ce_only_gain'] = bool(e['success'] and (g.get('PAM', {}).get('ci_below_zero') or g.get('CMI_q', {}).get('ci_below_zero')))
            rec[ov] = e
        for ov in ('mix_ctrl', 'arp_ctrl'):
            if (b, ov) in cells:
                rec[ov] = {'delta': dci(cells[(b, ov)], cells[(b, 'base')], test, bt)}
        out['bases'][b] = rec
    for fam, ov in (('mix', 'mix_star'), ('arp', 'arp_star')):
        who = [b for b in lb if out['bases'].get(b, {}).get(ov, {}).get('success')]
        out[f'headline_{fam}'] = {'cell': sel[ov], 'n_success_of_listenable': len(who), 'n_listenable': len(lb), 'bases': who,
                                  'reading': head(len(who)),
                                  'n_ce_only_gain': sum(bool(out['bases'].get(b, {}).get(ov, {}).get('ce_only_gain')) for b in lb)}
    return out


def summarize_E(smoke):
    lam, R, gens, budget = es_params(smoke)
    ev = [json.loads(ln) for ln in (OUT / 'es/evals.jsonl').read_text().splitlines()]
    top = json.loads((OUT / 'e_top10.json').read_text())['top10']
    rows = rd_cell('E_top10')
    ids = [t['id'] for t in top]
    realce = []
    for b in ('real',):
        for key in ([f'D_{b}_base'] + [f'Ddev_{b}_base']):
            r = rd_cell(key)
            if r:
                realce += [v['CE'] for v in r.values()]
    realce = np.array(realce, float)
    med = float(np.median(realce)) if len(realce) else math.nan
    ce10 = [rows[i]['CE'] for i in ids]
    pct = [float((realce < c).mean() * 100) if len(realce) else None for c in ce10]
    any_hi = bool(len(realce) and any(c > med for c in ce10))
    best = {r_: max(e['CE'] for e in ev if e['r'] == r_) for r_ in range(R) if any(e['r'] == r_ for e in ev)}
    return {'n_evals': len(ev), 'budget': budget, 'restarts': R, 'lam': lam, 'gens_per_restart': gens, 'best_CE_by_restart': best,
            'best_CE': max(e['CE'] for e in ev), 'real_CE_n': int(len(realce)), 'real_CE_median': med,
            'real_CE_pctiles': {str(q): float(np.percentile(realce, q)) for q in (5, 25, 50, 75, 95)} if len(realce) else {},
            'top10': [{'id': i, 'CE_search': t['CE'], 'CE_rescored': rows[i]['CE'], 'pct_in_real': p_,
                       **{c: rows[i][c] for c in ('PQ', 'CU', 'PC', 'CLAP', 'PAM', 'CMI_q', 'A1_MI', 'A3a_MI')}}
                      for i, t, p_ in zip(ids, top, pct)],
            'verdict': '存在高 CE 簡單聲音' if any_hi else '找不到高 CE 簡單聲音'}


def summarize(smoke):
    global EXC
    setup(smoke)
    EXC = EXC or json.loads(EXCERPTS.read_text())
    mc, win = get_ids(smoke)
    sel = select_D(smoke)
    out = {'kind': KIND, 'smoke': smoke, 'boot': {'NB': NB, 'seed': SEED}, 'S_CE': H.S['CE_lvl30'],
           'B': summarize_B(smoke), 'C': summarize_C(smoke), 'D': summarize_D(smoke, sel), 'E': summarize_E(smoke),
           'verdicts': {}}
    out['verdicts'] = {'B_replication_extra_harm_30': out['B']['replication_extra_harm_30'],
                       'B_harmony_preference': out['B']['harmony_preference'],
                       'C_top3': [f'{e["stem"]}_{e["tag"]}' for e in out['C']['top3']],
                       'D_mix': out['D']['headline_mix']['reading'], 'D_arp': out['D']['headline_arp']['reading'],
                       'E': out['E']['verdict']}
    P.jdump(OUT / 'summary.json', out)
    P.atomic(OUT / 'summary_table.md', table_md(out))
    log('summary written; verdicts: ' + json.dumps(out['verdicts'], ensure_ascii=False))
    return out


def fmt(ci):
    return 'n/a' if not ci else f'{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]'


def table_md(o):
    B, C, D, E = o['B'], o['C'], o['D'], o['E']
    L = [f'# 195 summary ({"smoke" if o["smoke"] else "full"}; S_CE = {o["S_CE"]:.4f}, m_CE = {D["m_CE"]:.4f}; bootstrap {NB}, seed {SEED})', '',
         '## Verdicts', '']
    L += [f'- {k}: **{v}**' for k, v in o['verdicts'].items()]
    L += ['', f'## B: MIDI density ladder ({B["n_ids"]} ids, {B["n_clusters"]} performances); dCE vs base', '',
          '| cell | dCE | dPQ | dA1_MI | dPAM | dCMI_q |', '|---|---|---|---|---|---|']
    for c, d in B['cells'].items():
        L.append(f'| {c} | {fmt(d.get("CE"))} | {fmt(d.get("PQ"))} | {fmt(d.get("A1_MI"))} | {fmt(d.get("PAM"))} | {fmt(d.get("CMI_q"))} |')
    L += ['', 'harm - rand dCE: ' + ', '.join(f'{k}%: {fmt(v)}' for k, v in B['harm_minus_rand_dCE'].items()),
          '', 'saturation dose (%): ' + ', '.join(f'{k} {v["saturation_dose_pct"]}' for k, v in B['saturation'].items()), '',
          f'## C: MUSDB stem gains (active windows: {C["n_active"]} of {C["n_windows"]}); top-3 frozen', '',
          '| cell | n | dCE | dPQ | dA1_MI | dPAM |', '|---|---|---|---|---|---|']
    for c, v in C['cells'].items():
        d = v['delta']
        L.append(f'| {c} | {v["n"]} | {fmt(d.get("CE"))} | {fmt(d.get("PQ"))} | {fmt(d.get("A1_MI"))} | {fmt(d.get("PAM"))} |')
    L += ['', f'## D: recipes (selection {D["selection"]["mix_star"]}, {D["selection"]["arp_star"]}; test n = {D["n_test"]})', '',
          '| base | cell | success | dCE | dCE vs control | dA1 | dPAM | dCMI_q | CE-only gain |', '|---|---|---|---|---|---|---|---|---|']
    for b, rec in D['bases'].items():
        for ov in ('mix_star', 'arp_star'):
            e = rec.get(ov)
            if not e:
                continue
            d = e['delta']
            L.append(f'| {b} | {ov} | {e["success"]} | {fmt(d.get("CE"))} | {fmt((e.get("vs_control") or {}).get("CE_diff"))} | '
                     f'{fmt(d.get("A1_MI"))} | {fmt(d.get("PAM"))} | {fmt(d.get("CMI_q"))} | {e["ce_only_gain"]} |')
    L += ['', f'headline mix: {D["headline_mix"]["n_success_of_listenable"]}/{D["headline_mix"]["n_listenable"]} -> '
              f'{D["headline_mix"]["reading"]}; arp: {D["headline_arp"]["n_success_of_listenable"]}/{D["headline_arp"]["n_listenable"]} -> '
              f'{D["headline_arp"]["reading"]}', '',
          f'## E: black-box search ({E["n_evals"]} evals; best CE {E["best_CE"]:.3f}; real median {E["real_CE_median"]:.3f}, n = {E["real_CE_n"]})', '',
          '| id | CE (search) | CE (rescored) | pct in real | PQ | CLAP | PAM | A1_MI |', '|---|---|---|---|---|---|---|---|']
    for t in E['top10']:
        L.append(f'| {t["id"]} | {t["CE_search"]:.3f} | {t["CE_rescored"]:.3f} | '
                 f'{"n/a" if t["pct_in_real"] is None else f"{t['pct_in_real']:.1f}"} | {t["PQ"]:.3f} | {t["CLAP"]:.3f} | {t["PAM"]:.3f} | {t["A1_MI"]:.3f} |')
    return '\n'.join(L) + '\n'


# ═════════ validate / preflight ═════════
def _finite_json(o, path=''):
    if isinstance(o, float):
        return [] if math.isfinite(o) else [path]
    if isinstance(o, dict):
        return [e for k, v in o.items() for e in _finite_json(v, f'{path}/{k}')]
    if isinstance(o, list):
        return [e for k, v in enumerate(o) for e in _finite_json(v, f'{path}[{k}]')]
    return []


def validate(smoke):
    errs = []
    try:
        global EXC
        setup(smoke)
        EXC = EXC or json.loads(EXCERPTS.read_text())
        mc, win = get_ids(smoke)
        sp = get_split(mc)
        act = c_active(win)

        def chk(key, ids, tier, clap, caps):
            p = OUT / 'per_clip' / f'{key}.tsv'
            if not p.is_file():
                errs.append(f'missing cell {p.name}')
            elif not M.rows_ok(M.tsv_read(p), ids, Q.req_cols(tier, caps, clap)):
                errs.append(f'cell {p.name}: id set or required columns not finite')
        bi = b_ids(smoke)
        for c in B_CELLS:
            chk(f'B_{c}', bi, 2, False, False)
        chk('C_base', win, 2, False, False)
        for s in CSTEMS:
            for t in C_TAGS:
                if act[s]:
                    chk(f'C_{s}_{t}', act[s], 2, False, False)
        for b in listen_bases(smoke):
            for c in D_DEV:
                chk(f'Ddev_{b}_{c}', sp['dev_mc'], 2, True, True)
        for b in d_bases(smoke):
            for c in D_TEST:
                chk(f'D_{b}_{c}', sp['test_mc'], 2, True, True)
        chk('E_top10', [f'e{k:02d}' for k in range(10)], 2, True, True)
        lam, R, gens, budget = es_params(smoke)
        stp = OUT / 'es/state.json'
        if not stp.is_file():
            errs.append('missing es/state.json')
        else:
            st = json.loads(stp.read_text())['restarts']
            if len(st) != R or any(v['gen'] != gens for v in st.values()):
                errs.append('ES not complete')
            n = sum(1 for _ in open(OUT / 'es/evals.jsonl'))
            if n != R * gens * lam:
                errs.append(f'es/evals.jsonl has {n} lines, expected {R * gens * lam}')
        for f in ('split.json', 'c_top3.json', 'selection.json', 'e_top10.json', 'summary.json', 'summary_table.md'):
            if not (OUT / f).is_file():
                errs.append(f'missing {f}')
        if (OUT / 'summary.json').is_file():
            s = json.loads((OUT / 'summary.json').read_text())
            if s.get('kind') != KIND:
                errs.append('summary.json wrong kind')
            for k in ('B_replication_extra_harm_30', 'B_harmony_preference', 'C_top3', 'D_mix', 'D_arp', 'E'):
                if not s.get('verdicts', {}).get(k):
                    errs.append(f'verdict {k} missing/empty')
            bad = _finite_json(s)
            if bad:
                errs.append(f'non-finite numbers in summary.json, e.g. {bad[:3]}')
            for f in ('selection.json', 'c_top3.json'):
                if (OUT / f).is_file() and (OUT / f).stat().st_mtime > (OUT / 'summary.json').stat().st_mtime:
                    errs.append(f'{f} newer than summary.json')
            if (OUT / 'selection.json').is_file() and (OUT / 'c_top3.json').is_file() and \
                    (OUT / 'selection.json').stat().st_mtime < (OUT / 'c_top3.json').stat().st_mtime:
                errs.append('selection.json older than c_top3.json')
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
    for p in (H.PARTB / 'part_b_selection.json', H.TTM115, H.MC_TSV, P.REF_DIR, M.MUSDB_DIR / 'windows.json', M.MUSDB_DIR / 'cache',
              H.PY, KJ_PY, DEMUCS_CKPT, EXCERPTS, MS.BANKS['generaluser'], MS.BANKS['fluidr3']):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    try:
        mc, win = get_ids(smoke)
        want = 16 if smoke else 1000
        if len(mc) != want:
            errs.append(f'{len(mc)} Part B ids with reference audio, expected {want}')
        for b in d_bases(smoke):
            if b == 'real':
                continue
            miss = [i for i in mc if not M.base_path(b, i).is_file()]
            if miss:
                errs.append(f'base {b}: {len(miss)} missing audio files, e.g. {miss[0]}')
        miss = [i for i in win if not (M.MUSDB_DIR / 'cache' / f'{i}.npz').is_file()]
        if miss:
            errs.append(f'musdb: {len(miss)} missing cache files, e.g. {miss[0]}')
        if not smoke and len(win) != 551:
            errs.append(f'{len(win)} musdb windows, expected 551')
        ex = json.loads(EXCERPTS.read_text())
        if len(ex) != 150:
            errs.append(f'{len(ex)} excerpts, expected 150')
        import importlib
        for mod in ('tinysoundfont', 'soxr', 'librosa', 'pyloudnorm'):
            importlib.import_module(mod)
        r = subprocess.run([str(KJ_PY), '-I', '-c', 'import demucs, torch'], capture_output=True)
        if r.returncode:
            errs.append('demucs not importable in karaoke-jp-align venv')
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if errs:
        return 1
    log(f'PREFLIGHT OK (smoke={smoke})')
    if not (OUT / 'summary.json').is_file():
        need_mem = 8 if smoke else 25
        if mem_available_gb() < need_mem:
            log(f'PREFLIGHT WAIT MemAvailable {mem_available_gb():.1f} GiB < {need_mem} GiB')
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
        OUT = H.EVAL_ROOT / 'ce_weakness_195_smoke'
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
