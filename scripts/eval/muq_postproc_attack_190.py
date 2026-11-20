#!/usr/bin/env python
"""190: MuQ A1 post-processing attacks (eval-only driver).

Design: docs/experiments/muq_postproc_attack_190_20261112.md (pre-registered; this file only implements it).

Four cell families, no new musical content is added to any signal:
  F1  8-12 kHz band noise at 24 kHz (hf_snr{30,20,10}) vs 2-4 kHz control (mf_snr*), reference up24
  F2  20 s stimulus, noise only outside the official 10 s MuQ centre crop (side_snr{20,10,0}); control ctr_snr10; ref ext
  F4  dither in the quietest 50 ms frames (qd_m{40,30,20}) vs same level on the loudest frames (ld_m*)
  F5  htdemucs separate-and-re-sum (demucs); worker runs in ~/venvs/karaoke-jp-align (demucs is not in the dac venv)
F3 (anti-phase) is not run (all scorers down-mix to mono before the model).

Imports the 189 module (and through it 188) and reuses loaders / scorers / tsv helpers unchanged; neither is edited.
Stages are resumable through per-cell TSVs under <root>/per_clip/.
"""
import os
import sys

# ── demucs worker: must run before any heavy import (karaoke-jp-align venv has no pyloudnorm / MuQ stack) ──
if len(sys.argv) > 1 and sys.argv[1] == '--demucs-worker':
    def _demucs_worker(din, dout):
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
        t0 = _t.time()
        for k in range(0, len(files), 4):
            fs = files[k:k + 4]
            xs, stats = [], []
            for p in fs:
                w = _np.load(p).astype(_np.float64)
                up = resample_poly(w, 441, 160).astype(_np.float32)
                st = torch.from_numpy(_np.stack([up, up]))          # mono -> stereo copy
                ref = st.mean(0)
                m, s = ref.mean(), ref.std()
                xs.append((st - m) / s)
                stats.append((float(m), float(s)))
            T = max(x.shape[-1] for x in xs)
            batch = torch.stack([torch.nn.functional.pad(x, (0, T - x.shape[-1])) for x in xs])
            with torch.no_grad():
                src = apply_model(model, batch, shifts=0, split=True, overlap=0.25, device='cuda', progress=False)
            for j, p in enumerate(fs):
                n = xs[j].shape[-1]
                mix = src[j, :, :, :n].sum(0).float().cpu().numpy()        # re-sum the 4 stems (normalised domain)
                mono = mix.mean(0) * stats[j][1] + stats[j][0]
                down = resample_poly(mono.astype(_np.float64), 160, 441).astype(_np.float32)
                tmp = dout / (p.name + '.tmp.npy')
                _np.save(tmp, down)
                tmp.replace(dout / p.name)
        print(f'demucs worker: {len(files)} clips in {_t.time() - t0:.1f}s', flush=True)
    _demucs_worker(sys.argv[2], sys.argv[3])
    sys.exit(0)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
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

P = M.P
H = M.H
REPO = P.REPO
SELF = Path(__file__).resolve()
SR, NS = P.SR, P.NS
OUT = H.EVAL_ROOT / 'muq_postproc_attack_190'
KJ_PY = Path('/home/kojiek/venvs/karaoke-jp-align/bin/python')
DEMUCS_CKPT = Path('/home/kojiek/.cache/torch/hub/checkpoints/955717e8-8726e21a.th')
KIND = 'muq_postproc_attack_190_summary_v1'
IS_SMOKE = False
SEED_BOOT = 20261112
NB = 5000
M_A1 = M.M_A1
BASES, MC_BASES, SEEDS = M.BASES, M.MC_BASES, M.SEEDS
SMOKE_BASES = ['real', 'musicgen', 'musdb']
MIN_FREE = 20 * 1024**3

SCORE_COLS, DELTA_COLS, MUQ_COLS, AES_COLS = M.SCORE_COLS, M.DELTA_COLS, M.MUQ_COLS, M.AES_COLS
GUARD_COLS, SE_COLS, TIER_COLS = M.GUARD_COLS, M.SE_COLS, M.TIER_COLS

HOLE = 79000                  # F2 noise covers [0, HOLE) and [L-HOLE, L) of the 319,200-sample stimulus
F4_MIN_FRAMES = 10            # 0.5 s of quiet 50 ms frames
F4_MIN_SUBSET = 100

# name -> (tier, sr, clap)
CELLS = {'base': (3, 16000, True),
         'up24': (3, 24000, True),
         'hf_snr30': (2, 24000, True), 'hf_snr20': (3, 24000, True), 'hf_snr10': (2, 24000, True),
         'mf_snr30': (2, 24000, True), 'mf_snr20': (3, 24000, True), 'mf_snr10': (2, 24000, True),
         'ext': (3, 16000, False),
         'side_snr20': (2, 16000, False), 'side_snr10': (3, 16000, False), 'side_snr0': (2, 16000, False),
         'ctr_snr10': (3, 16000, False),
         'qd_m40': (2, 16000, True), 'qd_m30': (3, 16000, True), 'qd_m20': (2, 16000, True),
         'ld_m40': (2, 16000, True), 'ld_m30': (3, 16000, True), 'ld_m20': (2, 16000, True),
         'demucs': (3, 16000, True)}
FAMILIES = {
    'F1': {'ref': 'up24', 'treat': ['hf_snr30', 'hf_snr20', 'hf_snr10'], 'ctrl': ['mf_snr30', 'mf_snr20', 'mf_snr10'],
           'main': 'hf_snr20', 'weak_to_strong': ['hf_snr30', 'hf_snr20', 'hf_snr10']},
    'F2': {'ref': 'ext', 'treat': ['side_snr20', 'side_snr10', 'side_snr0'], 'ctrl': [None, 'ctr_snr10', None],
           'main': 'side_snr10', 'weak_to_strong': ['side_snr20', 'side_snr10', 'side_snr0']},
    'F4': {'ref': 'base', 'treat': ['qd_m40', 'qd_m30', 'qd_m20'], 'ctrl': ['ld_m40', 'ld_m30', 'ld_m20'],
           'main': 'qd_m30', 'weak_to_strong': ['qd_m40', 'qd_m30', 'qd_m20']},
    'F5': {'ref': 'base', 'treat': ['demucs'], 'ctrl': [None], 'main': 'demucs', 'weak_to_strong': ['demucs']},
}


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def cell_key(base, ov):
    return f'T_{base}_{ov}'


def req_cols(tier, has_caps, clap):
    return [c for c in TIER_COLS[tier] if (has_caps and clap) or c != 'CLAP']


def get_ids(smoke):
    mc, win = M.get_ids(smoke)
    return (mc[:16] if smoke else mc), win


def bases_of(smoke):
    return SMOKE_BASES if smoke else BASES


# ── audio helpers ────────────────────────────────────────
def lvl_sr(w, sr, target=-30.0, cap=0.999):
    """P.to_lufs with the pyloudnorm meter at the stimulus rate (P.to_lufs hard-codes 16 kHz)."""
    import pyloudnorm
    w = np.asarray(w, np.float64)
    lufs = pyloudnorm.Meter(sr).integrated_loudness(w[:, None])
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    if not math.isfinite(lufs) or peak == 0.0:
        g = 0.0
    else:
        g = target - lufs
        if peak * 10 ** (g / 20) > cap:
            g = 20 * math.log10(cap / peak)
    return np.clip(w * 10 ** (g / 20), -1.0, 1.0).astype(np.float32)


def up24_of(w):
    from scipy.signal import resample_poly
    return lvl_sr(resample_poly(np.asarray(w, np.float64), 3, 2), 24000)


def band_noise(n, kind, rng):
    """Unit-power noise, 8 kHz-Nyquist (12 kHz) high-pass for 'hf' (a 12 kHz upper edge IS Nyquist at 24 kHz), 2-4 kHz
    band-pass for 'mf'.  8th-order Butterworth, sos."""
    from scipy.signal import butter, sosfilt
    if kind == 'hf':
        sos = butter(8, 8000, btype='highpass', fs=24000, output='sos')
    else:
        sos = butter(8, [2000, 4000], btype='bandpass', fs=24000, output='sos')
    x = sosfilt(sos, rng.standard_normal(n + 2000))[2000:]
    return x / np.sqrt(np.mean(x ** 2))


def build_f1(w, kind, snr, base, i, ov):
    up = up24_of(w).astype(np.float64)
    pw = float(np.mean(up ** 2))
    n = band_noise(len(up), kind, P.rng_for(base, i, ov)) * math.sqrt(pw / 10 ** (snr / 10))
    return lvl_sr(up + n, 24000)


def ext_of(w):
    five = 80000
    return np.concatenate([w[five:], w, w[:five]]).astype(np.float32)


def build_f2(w, snr, region, base, i, ov):
    x = ext_of(w).astype(np.float64)
    L = len(x)
    rng = P.rng_for(base, i, ov)
    segs = [(0, HOLE), (L - HOLE, L)] if region == 'side' else [(len(w) - len(w) + 79600, 79600 + len(w))]
    for a, b in segs:
        pw = float(np.mean(x[a:b] ** 2))
        x[a:b] += rng.standard_normal(b - a) * math.sqrt(pw / 10 ** (snr / 10))
    return x.astype(np.float32)      # deliberately NOT renormalised (would change the MuQ window)


def f4_frames(w):
    """-> (rms per 50 ms frame, quiet frame indices, loudest frame indices (same count))."""
    fl = 800
    nf = len(w) // fl
    fr = np.asarray(w[:nf * fl], np.float64).reshape(nf, fl)
    rms = np.sqrt((fr ** 2).mean(1))
    med = float(np.median(rms))
    thr = med * 10 ** (-20 / 20)
    quiet = np.where(rms < thr)[0] if med > 0 else np.array([], int)
    loud = np.argsort(-rms)[:len(quiet)]
    return rms, quiet, np.sort(loud)


def build_f4(w, level_db, which, base, i, ov):
    fl, fade = 800, 80
    x = np.asarray(w, np.float64).copy()
    _, quiet, loud = f4_frames(w)
    idx = quiet if which == 'qd' else loud
    if len(idx) == 0:
        return lvl_sr(x, SR)
    tgt = float(np.sqrt(np.mean(x ** 2))) * 10 ** (level_db / 20)
    env = np.zeros(len(x))
    mark = np.zeros(len(x) // fl, bool)
    mark[idx] = True
    k = 0
    while k < len(mark):
        if mark[k]:
            e = k
            while e < len(mark) and mark[e]:
                e += 1
            a, b = k * fl, e * fl
            env[a:b] = 1.0
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
            env[a:a + fade] = ramp
            env[b - fade:b] = np.minimum(env[b - fade:b], ramp[::-1])
            k = e
        else:
            k += 1
    noise = P.rng_for(base, i, ov).standard_normal(len(x)) * tgt * env
    return lvl_sr(x + noise, SR)


def builder(ov, wavs, base, demucs_dir):
    """-> build(id) for one cell (runs in the thread pool)."""
    if ov == 'base':
        return wavs.__getitem__
    if ov == 'up24':
        return lambda i: up24_of(wavs[i])
    if ov.startswith(('hf_snr', 'mf_snr')):
        kind, snr = ov[:2], int(ov.split('snr')[1])
        return lambda i: build_f1(wavs[i], kind, snr, base, i, ov)
    if ov == 'ext':
        return lambda i: ext_of(wavs[i])
    if ov.startswith('side_snr'):
        return lambda i: build_f2(wavs[i], int(ov.split('snr')[1]), 'side', base, i, ov)
    if ov.startswith('ctr_snr'):
        return lambda i: build_f2(wavs[i], int(ov.split('snr')[1]), 'ctr', base, i, ov)
    if ov.startswith(('qd_m', 'ld_m')):
        return lambda i: build_f4(wavs[i], -int(ov[4:]), ov[:2], base, i, ov)
    if ov == 'demucs':
        return lambda i: P.lvl30(P.fit(np.load(demucs_dir / f'{i}.npy')))[0]
    raise KeyError(ov)


# ── scorer (adds sample-rate / variable-length handling to 189's XScorer) ─
class Y(M.XScorer):
    def aes4(self, wavs, sr=SR):
        torch = P.SC.torch
        out = {c: [] for c in AES_COLS}
        for s in range(0, len(wavs), 32):
            items = [{'path': torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': sr}
                     for w in wavs[s:s + 32]]
            try:
                rows = P.SC.aes.forward(items)
            except Exception:  # noqa: BLE001
                rows = [P.SC.aes.forward([it])[0] for it in items]
            for c in AES_COLS:
                out[c] += [float(r[c]) for r in rows]
        return out

    def muq24(self, wavs):
        """MuQ-Eval on natively 24 kHz stimuli: official 10 s centre crop, no resampling (zero-pad if shorter)."""
        torch = P.SC.torch
        res = {f'{v}_{h}': [] for v in ('A1', 'A3a') for h in ('MI', 'TA')}
        with torch.no_grad():
            for s in range(0, len(wavs), 16):
                xs = []
                for w in wavs[s:s + 16]:
                    x = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).cuda()
                    L = 240000
                    if x.shape[-1] >= L:
                        st = (x.shape[-1] - L) // 2
                        x = x[st:st + L]
                    else:
                        x = torch.nn.functional.pad(x, (0, L - x.shape[-1]))
                    xs.append(x)
                x = torch.stack(xs)
                for v in ('A1', 'A3a'):
                    o = P.SC.muq[v](x)
                    for h in ('MI', 'TA'):
                        res[f'{v}_{h}'].extend(P.SC.head_val(o[h]).tolist())
        return res

    def clap_sr(self, wavs, caps, sr):
        import librosa
        torch = P.SC.torch
        out = []
        with torch.no_grad():
            for w, c in zip(wavs, caps):
                x48 = librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=48000)
                ae = P.SC.clap.get_audio_embedding_from_data(x48[None], use_tensor=False)
                if c not in P.SC.txt:
                    P.SC.txt[c] = P.SC.clap.get_text_embedding([c], use_tensor=False)
                te = P.SC.txt[c]
                out.append(float((ae * te).sum() / (np.linalg.norm(ae) * np.linalg.norm(te))))
        return out

    def songeval_sr(self, wavs, sr):
        if sr != 24000:
            return self.songeval(wavs)
        SE, muq, gen = self.se
        res = [SE.score_songeval_batch([np.ascontiguousarray(w, dtype=np.float32) for w in wavs[s:s + 16]], muq, gen,
                                       'cuda') for s in range(0, len(wavs), 16)]
        return np.concatenate(res)

    def score(self, wavs, caps, tier, cmi_mode, sr=SR, clap=True):
        n = len(wavs)
        cols = {}

        def clock(name, fn):
            t0 = time.time()
            r = fn()
            P.SC.torch.cuda.synchronize()
            self.t[name] += time.time() - t0
            self.n[name] += n
            return r

        cols.update(clock('muq', lambda: (self.muq24(wavs) if sr == 24000 else P.SC.muq_scores(wavs)[0])))
        if tier >= 1:
            cols.update(clock('aes', lambda: self.aes4(wavs, sr)))
            cols['CLAP'] = (clock('clap', lambda: self.clap_sr(wavs, caps, sr)) if (caps and clap) else [math.nan] * n)
        use_caps = bool(caps) and cmi_mode == 'caption'
        if tier >= 2:
            import newmetrics_pam
            import newmetrics_cmirm
            cols['PAM'] = clock('pam', lambda: [newmetrics_pam.score_pam(w, sr=sr)['pam'] for w in wavs])
            if use_caps:
                cols['CMI_q'] = clock('cmi', lambda: [newmetrics_cmirm.score_cmirm(
                    w, sr=sr, prompt_mode='caption', caption=c)['quality'] for w, c in zip(wavs, caps)])
            else:
                cols['CMI_q'] = clock('cmi', lambda: [newmetrics_cmirm.score_cmirm(
                    w, sr=sr, prompt_mode='empty')['quality'] for w in wavs])
        if tier >= 3:
            if use_caps:
                cols['CMI_q_empty'] = clock('cmi_empty', lambda: [newmetrics_cmirm.score_cmirm(
                    w, sr=sr, prompt_mode='empty')['quality'] for w in wavs])
            else:
                cols['CMI_q_empty'] = list(cols['CMI_q'])
            se = clock('songeval', lambda: self.songeval_sr(wavs, sr))
            for k, c in enumerate(SE_COLS):
                cols[c] = [float(v) for v in se[:, k]]
        self.smi_peak()
        return [{c: float(cols[c][i]) for c in cols} for i in range(n)]


X = Y()
STATS = {}      # cell key -> {'n_over_1': int, 'max_abs': float}


def run_cell(key, ids, build, caps, ov, cmi_mode):
    tier, sr, clap = CELLS[ov]
    path = OUT / 'per_clip' / f'{key}.tsv'
    req = req_cols(tier, bool(caps), clap)
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
    log(f'{key}: tier {tier}, sr {sr}, {len(ids)} stimuli in {wall:.1f}s ({len(ids) / wall:.2f}/s); '
        f'{over} stimuli with |x|>1 (max {mx:.3f})')
    return res


# ── F2 harness check: MuQ input of side_* must equal ext's, bit for bit ──
def f2_input_check(base, wavs, ids, n=8):
    torch = P.SC.torch
    ok, worst = True, 0.0
    for i in ids[:n]:
        e = P.SC.to24([ext_of(wavs[i])])
        for ov in FAMILIES['F2']['treat']:
            s = P.SC.to24([build_f2(wavs[i], int(ov.split('snr')[1]), 'side', base, i, ov)])
            worst = max(worst, float((e - s).abs().max()))
            ok = ok and bool(torch.equal(e, s))
    return {'checked_ids': min(n, len(ids)), 'inputs_bitwise_equal': ok, 'max_abs_diff': worst}


# ── demucs precompute (before scorers are loaded, so the GPU is free) ──
def demucs_precompute(smoke, ids_of):
    cost_p = OUT / 'demucs_cost.json'
    cost = json.loads(cost_p.read_text()) if cost_p.is_file() else {}
    for base in bases_of(smoke):
        ids = ids_of(base)
        key = cell_key(base, 'demucs')
        p = OUT / 'per_clip' / f'{key}.tsv'
        caps = base != 'musdb'
        if p.is_file() and M.rows_ok(M.tsv_read(p), ids, req_cols(3, caps, True)):
            continue
        dout = OUT / 'demucs' / base
        need = [i for i in ids if not (dout / f'{i}.npy').is_file()]
        if not need:
            continue
        wavs = M.load_base(base, need)
        din = OUT / 'demucs_in' / base
        shutil.rmtree(din, ignore_errors=True)
        din.mkdir(parents=True)
        for i in need:
            np.save(din / f'{i}.npy', wavs[i].astype(np.float32))
        del wavs
        env = {k: v for k, v in os.environ.items() if k != 'MKL_NUM_THREADS'}
        env.update(OMP_NUM_THREADS='2')
        t0 = time.time()
        r = subprocess.run([str(KJ_PY), '-I', str(SELF), '--demucs-worker', str(din), str(dout)], cwd=REPO, env=env)
        wall = time.time() - t0
        shutil.rmtree(din, ignore_errors=True)
        if r.returncode or any(not (dout / f'{i}.npy').is_file() for i in need):
            raise RuntimeError(f'demucs worker failed for {base} rc={r.returncode}')
        cost[base] = {'n': len(need), 'wall_s': wall}
        P.jdump(cost_p, cost)
        log(f'demucs {base}: {len(need)} clips in {wall:.1f}s')


# ── main run ─────────────────────────────────────────────
def setup(smoke):
    M.OUT = OUT
    P.OUT = OUT
    P.IS_SMOKE = smoke
    P.CFG['chunk'] = 16 if smoke else 64
    P.POOL = ThreadPoolExecutor(P.NTHREAD)
    H.load_scales()
    OUT.mkdir(parents=True, exist_ok=True)


def run_all(smoke):
    setup(smoke)
    mc, win = get_ids(smoke)
    caps_all = P.caps_all()

    def ids_of(b):
        return win if b == 'musdb' else mc

    P.progress('start', '', 0, 0)
    demucs_precompute(smoke, ids_of)
    X.load_all()
    log(f'all scorers loaded; torch max alloc {P.SC.torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    fq_p = OUT / 'f4_quiet_frames.json'
    fq = json.loads(fq_p.read_text()) if fq_p.is_file() else {}
    chk_p = OUT / 'f2_input_check.json'
    chk = json.loads(chk_p.read_text()) if chk_p.is_file() else {}
    for base in bases_of(smoke):
        ids = ids_of(base)
        wavs = M.load_base(base, ids)
        if base not in fq:
            fq[base] = {i: int(len(f4_frames(wavs[i])[1])) for i in ids}
            P.jdump(fq_p, fq)
        if base not in chk:
            chk[base] = f2_input_check(base, wavs, ids)
            P.jdump(chk_p, chk)
            log(f'F2 input check {base}: {chk[base]}')
        caps = None if base == 'musdb' else {i: caps_all[i] for i in ids}
        cmi_mode = 'empty' if base == 'musdb' else 'caption'
        for ov in CELLS:
            run_cell(cell_key(base, ov), ids, builder(ov, wavs, base, OUT / 'demucs' / base), caps, ov, cmi_mode)
        shutil.rmtree(OUT / 'demucs' / base, ignore_errors=True)
        del wavs
    P.jdump(OUT / 'gpu_peak.json', {'torch_max_alloc_gib': P.SC.torch.cuda.max_memory_allocated() / 2**30,
                                    'torch_max_reserved_gib': P.SC.torch.cuda.max_memory_reserved() / 2**30,
                                    'nvidia_smi_used_max_mib': X.peak_smi})
    P.progress('scored', '', 0, 0)
    project(smoke)


# ── cost projection ──────────────────────────────────────
def project(smoke):
    tp, cp = OUT / 'timing.json', OUT / 'cost.json'
    if not tp.is_file():
        return None
    t = json.loads(tp.read_text())
    per_ov = {}
    for k, v in t['keys'].items():
        if not k.startswith('T_'):
            continue
        base, ov = k[2:].split('_', 1) if k[2:].split('_', 1)[0] != 'ma' else (None, None)
        for ov_name in CELLS:
            if k.endswith('_' + ov_name):
                per_ov.setdefault(ov_name, []).append(v['wall_s'] / v['n'])
    wall = sum(float(np.mean(per_ov[o])) * (9 * 1000 + 551) for o in CELLS if o in per_ov)
    dem = json.loads((OUT / 'demucs_cost.json').read_text()) if (OUT / 'demucs_cost.json').is_file() else {}
    dn = sum(v['n'] for v in dem.values())
    dw = sum(v['wall_s'] for v in dem.values())
    dem_h = (dw / dn * (9 * 1000 + 551) / 3600) if dn else None
    cost = json.loads(cp.read_text()) if cp.is_file() else {'t': {}, 'n': {}}
    per_scorer = {k: cost['t'][k] / cost['n'][k] for k in cost['t'] if cost['n'].get(k)}
    rec = {'per_cell_type_s_per_clip': {o: float(np.mean(v)) for o, v in per_ov.items()},
           'per_scorer_s_per_clip': per_scorer, 'demucs_worker_s_per_clip': (dw / dn if dn else None),
           'demucs_hours_full': dem_h, 'scoring_hours_full': wall / 3600,
           'projected_total_hours': wall / 3600 + (dem_h or 0),
           'note': 'full = 9 MusicCaps bases x 1000 + 551 musdb windows per cell; per-cell-type s/clip averaged over the '
                   'bases scored here (smoke includes first-chunk warm-up, so it over-estimates)'}
    P.jdump(OUT / 'projection.json', rec)
    log('PROJECTION per-scorer: ' + ', '.join(f'{k} {v * 1000:.1f}ms' for k, v in per_scorer.items()))
    log(f'PROJECTION full run: scoring {wall / 3600:.1f} h + demucs {dem_h} h = {rec["projected_total_hours"]:.1f} h')
    return rec


# ── summary ──────────────────────────────────────────────
class Boot:
    def __init__(self, clusters):
        _, self.k = np.unique(np.asarray(clusters), return_inverse=True)
        self.K = int(self.k.max()) + 1
        self.idx = np.random.default_rng(SEED_BOOT).integers(0, self.K, (NB, self.K))

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


def analyse(cells, base, ids, bt, S, S_CE, ov, ref, ctrl):
    """One treatment cell vs its reference (+ matched control).  Pure scorer movements, no better/worse words."""
    a = cells[(base, ov)]
    r = cells[(base, ref)]

    def dl(x, col):
        return M.arr(x, ids, col) - M.arr(r, ids, col)

    e = {'delta': {}, 'guards': {}}
    for c in DELTA_COLS:
        ci = bt.ci(dl(a, c))
        if ci is None:
            continue
        e['delta'][c] = ci
        if c in GUARD_COLS:
            tie = 0.10 * S[c] if S.get(c) else None
            e['guards'][c] = {'ci_below_zero': ci[2] < 0, 'beyond_tie_down': (ci[0] <= -tie) if tie else None,
                              'beyond_tie_up': (ci[0] >= tie) if tie else None, 'tie': tie}
    a1, ce = e['delta'].get('A1_MI'), e['delta'].get('CE')
    if a1 is None or ce is None:
        return e
    e['A1_down'] = bool(a1[2] <= -M_A1)
    e['CE_beyond_tie_down'] = bool(ce[0] <= -0.10 * S_CE)
    e['verdict_v3'] = ('no_effect' if not a1[1] >= M_A1 else ('success' if ce[2] < 0 else 'not_success_CE_not_down'))
    gp, gc = e['guards'].get('PAM'), e['guards'].get('CMI_q')
    e['independent_unseen'] = (None if (gp is None or gc is None)
                               else bool(not gp['ci_below_zero'] and not gc['ci_below_zero']))
    # blind-spot reading (section 2.3): A1 CI inside +-m and >=2 of PAM / CMI caption / PQ / CE clearly down
    movers = []
    for c, tie in (('PAM', 0.10 * S['PAM'] if S.get('PAM') else None), ('CMI_q', 0.10 * S['CMI_q'] if S.get('CMI_q') else None),
                   ('PQ', 0.10 * S['PQ'] if S.get('PQ') else None), ('CE', 0.10 * S_CE)):
        ci = e['delta'].get(c)
        if ci is not None and tie is not None and ci[2] < -tie:
            movers.append(c)
    a1_flat = bool(a1[1] > -M_A1 and a1[2] < M_A1)
    e['blind'] = {'A1_inside_pm_m': a1_flat, 'movers_down_beyond_tie': movers,
                  'label': ('muq_moves' if not a1_flat else 'blind_spot' if len(movers) >= 2
                            else 'nobody_sees' if not movers else 'single_scorer_sees')}
    if ctrl is not None and (base, ctrl) in cells:
        c_ = cells[(base, ctrl)]
        vc = {'A1_diff': bt.ci(dl(a, 'A1_MI') - dl(c_, 'A1_MI')), 'CE_diff': bt.ci(dl(a, 'CE') - dl(c_, 'CE')),
              'control_A1': bt.ci(dl(c_, 'A1_MI')), 'control_CE': bt.ci(dl(c_, 'CE'))}
        e['vs_control'] = vc
        a1_ok, ce_ok = vc['A1_diff'][1] >= M_A1, vc['CE_diff'][2] < 0
        e['a1_ok'], e['ce_ok'] = a1_ok, ce_ok
        if e['verdict_v3'] != 'success':
            e['label'] = e['verdict_v3']
        else:
            e['label'] = ('success_specific' if (a1_ok and ce_ok) else 'A1_specific_CE_nonspecific' if a1_ok
                          else 'CE_specific_A1_nonspecific' if ce_ok else 'nonspecific')
    else:
        e['label'] = e['verdict_v3']
    return e


def reading(n_ok, musdb_ok):
    if n_ok >= 7:
        return 'general' if musdb_ok else ('general_musdb_not_evaluable' if musdb_ok is None else 'seven_plus_but_musdb_disagrees')
    return 'base_dependent' if n_ok >= 3 else 'not_established'


def summarize(smoke):
    setup(smoke)
    mc, win = get_ids(smoke)
    bases = bases_of(smoke)
    tmap = M.track_map()
    S_CE = H.S['CE_lvl30']
    cells = {}
    for b in bases:
        for ov in CELLS:
            p = OUT / 'per_clip' / f'{cell_key(b, ov)}.tsv'
            if p.is_file():
                cells[(b, ov)] = M.tsv_read(p)
    fq = json.loads((OUT / 'f4_quiet_frames.json').read_text()) if (OUT / 'f4_quiet_frames.json').is_file() else {}

    def bmean(b, c):
        v = M.arr(cells[(b, 'base')], mc, c)
        return float(np.nanmean(v)) if np.isfinite(v).any() else math.nan

    S = {}
    if all((b, 'base') in cells for b in ['real', 'musicgen']):
        have = [b for b in (['real', 'musicgen', 'musicldm'] + [f'ma_cfg0_s{s}' for s in SEEDS]
                            + [f'ma_cfg3neg_s{s}' for s in SEEDS]) if (b, 'base') in cells]
        for c in DELTA_COLS:
            groups = [bmean('real', c), bmean('musicgen', c)] if not smoke else [bmean(b, c) for b in have]
            if not smoke:
                groups += [bmean('musicldm', c),
                           float(np.mean([bmean(f'ma_cfg0_s{s}', c) for s in SEEDS])),
                           float(np.mean([bmean(f'ma_cfg3neg_s{s}', c) for s in SEEDS])), ]
            S[c] = float(max(groups) - min(groups)) if all(math.isfinite(x) for x in groups) else None
    out = {'kind': KIND, 'smoke': smoke, 'M_A1': M_A1, 'S_CE': S_CE, 'S_tie': S, 'boot': {'NB': NB, 'seed': SEED_BOOT},
           'families': {f: {k: v for k, v in d.items() if k != 'weak_to_strong'} for f, d in FAMILIES.items()},
           'bases': {}, 'note': 'scorer movements only; no better/worse labels'}
    # F2 zero assertion (side vs ext, all four MuQ columns, every base)
    f2 = {'threshold': 1e-6, 'max_abs': {}, 'pass': True}
    chk = json.loads((OUT / 'f2_input_check.json').read_text()) if (OUT / 'f2_input_check.json').is_file() else {}
    f2['input_check'] = chk
    for b in bases:
        ids = win if b == 'musdb' else mc
        if (b, 'ext') not in cells:
            continue
        for ov in FAMILIES['F2']['treat']:
            if (b, ov) not in cells:
                continue
            mx = max(float(np.nanmax(np.abs(M.arr(cells[(b, ov)], ids, c) - M.arr(cells[(b, 'ext')], ids, c))))
                     for c in MUQ_COLS)
            f2['max_abs'][f'{b}/{ov}'] = mx
            f2['pass'] = f2['pass'] and mx < 1e-6
    f2['pass'] = bool(f2['pass'] and f2['max_abs'] and all(v.get('inputs_bitwise_equal') for v in chk.values()))
    out['f2_zero_assertion'] = f2
    f4_subset = {}
    for b in bases:
        ids = win if b == 'musdb' else mc
        if (b, 'base') not in cells:
            continue
        bt = Boot([tmap[i] for i in ids] if b == 'musdb' else list(range(len(ids))))
        rec = {'n': len(ids), 'base_means': {c: (float(np.nanmean(M.arr(cells[(b, 'base')], ids, c)))
                                                 if np.isfinite(M.arr(cells[(b, 'base')], ids, c)).any() else None)
                                             for c in SCORE_COLS}, 'fam': {}}
        sub = [i for i in ids if fq.get(b, {}).get(i, 0) >= F4_MIN_FRAMES]
        f4_subset[b] = {'n_subset': len(sub), 'n': len(ids), 'in_main_endpoint': len(sub) >= F4_MIN_SUBSET}
        # sanity: up24 vs the 16 kHz base (resampling alone)
        if (b, 'up24') in cells:
            rec['sanity_up24_vs_base16k'] = {c: bt.ci(M.arr(cells[(b, 'up24')], ids, c) - M.arr(cells[(b, 'base')], ids, c))
                                             for c in MUQ_COLS + AES_COLS + ['CLAP', 'PAM', 'CMI_q']
                                             if np.isfinite(M.arr(cells[(b, 'up24')], ids, c)).any()}
        for fam, fd in FAMILIES.items():
            frec = {}
            for ov, ctrl in zip(fd['treat'], fd['ctrl']):
                if (b, ov) not in cells or (b, fd['ref']) not in cells:
                    continue
                frec[ov] = analyse(cells, b, ids, bt, S, S_CE, ov, fd['ref'], ctrl)
                if fam == 'F4' and len(sub) >= 2:
                    bs = Boot([tmap[i] for i in sub] if b == 'musdb' else list(range(len(sub))))
                    frec[ov]['subset_quiet_ge_0.5s'] = analyse(cells, b, sub, bs, S, S_CE, ov, fd['ref'], ctrl)
                    frec[ov]['subset_quiet_ge_0.5s']['n'] = len(sub)
            if fam == 'F2' and 'side_snr10' in frec and (b, 'ctr_snr10') in cells:
                d_ctr = M.arr(cells[(b, 'ctr_snr10')], ids, 'A1_MI') - M.arr(cells[(b, 'ext')], ids, 'A1_MI')
                d_side = M.arr(cells[(b, 'side_snr10')], ids, 'A1_MI') - M.arr(cells[(b, 'ext')], ids, 'A1_MI')
                frec['ctr_snr10_vs_side_snr10_A1'] = {'ctr_dA1': bt.ci(d_ctr), 'side_dA1': bt.ci(d_side),
                                                      'ctr_minus_side': bt.ci(d_ctr - d_side)}
                frec['ctr_snr10_dA1_ci'] = bt.ci(d_ctr)
            # weakest dose (lowest strength) whose dA1 CI lower >= m
            ok = [ov for ov in fd['weak_to_strong'] if ov in frec and (frec[ov].get('delta', {}).get('A1_MI') or [0, -9e9])[1] >= M_A1]
            frec['min_dose_dA1_ge_m'] = ok[0] if ok else None
            rec['fam'][fam] = frec
        out['bases'][b] = rec
    out['f4_subset'] = f4_subset
    # headline (section 2.4)
    head = {}
    for fam, fd in FAMILIES.items():
        ov = fd['main']
        who, excl = [], []
        for b in MC_BASES:
            fr = out['bases'].get(b, {}).get('fam', {}).get(fam, {}).get(ov)
            if fr is None:
                continue
            if fam == 'F4':
                if not f4_subset.get(b, {}).get('in_main_endpoint'):
                    excl.append(b)
                    continue
                fr = fr.get('subset_quiet_ge_0.5s', {})
            lab = fr.get('blind', {}).get('label') if fam == 'F2' else fr.get('label')
            if (fam == 'F2' and lab == 'blind_spot') or (fam == 'F1' and lab == 'success_specific') \
                    or (fam == 'F4' and lab == 'success_specific') or (fam == 'F5' and lab == 'success'):
                who.append(b)
        mfr = out['bases'].get('musdb', {}).get('fam', {}).get(fam, {}).get(ov)
        musdb_ok = None
        if mfr is not None:
            if fam == 'F4':
                mfr = mfr.get('subset_quiet_ge_0.5s') if f4_subset.get('musdb', {}).get('in_main_endpoint') else None
            if mfr is not None:
                d = mfr.get('delta', {})
                if fam == 'F2':
                    musdb_ok = bool(mfr.get('blind', {}).get('label') == 'blind_spot')
                else:
                    a1, ce = d.get('A1_MI'), d.get('CE')
                    musdb_ok = bool(a1 and ce and a1[1] > 0 and ce[2] < 0)
        head[fam] = {'main_cell': ov, 'n_pass_of_9': len(who), 'bases': who, 'excluded_small_subset': excl,
                     'musdb_consistent': musdb_ok, 'reading': reading(len(who), musdb_ok),
                     'criterion': ('blind_spot' if fam == 'F2' else 'success' if fam == 'F5' else 'success_specific')}
    out['headline'] = head
    P.jdump(OUT / 'summary.json', out)
    P.atomic(OUT / 'summary_table.md', table_md(out))
    log('summary written; headline: ' + json.dumps({f: (v['n_pass_of_9'], v['reading']) for f, v in head.items()}))
    return out


def fmt(ci):
    return 'n/a' if not ci else f'{ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}]'


def table_md(out):
    L = [f'# 190 summary ({"smoke" if out["smoke"] else "full"}; m = {M_A1}, S_CE = {out["S_CE"]:.4f}; bootstrap {NB}, '
         f'seed {SEED_BOOT}, musdb clustered by track)', '',
         '## Headline (section 2.4)', '', '| family | main cell | criterion | bases passing (of 9) | musdb | reading |', '|---|---|---|---|---|---|']
    for f, v in out['headline'].items():
        L.append(f'| {f} | {v["main_cell"]} | {v["criterion"]} | {v["n_pass_of_9"]} | {v["musdb_consistent"]} | {v["reading"]} |')
    z = out['f2_zero_assertion']
    L += ['', f'F2 zero-delta assertion (side vs ext, MuQ 4 columns, < 1e-6): **{"PASS" if z["pass"] else "FAIL"}**; '
          f'max |d| = {max(z["max_abs"].values()) if z["max_abs"] else "n/a"}', '']
    for f, fd in FAMILIES.items():
        L += [f'## {f}: per-base deltas vs {fd["ref"]} (A1_MI, CE, PQ; label)', '',
              '| base | cell | dA1_MI | dCE | dPQ | dPAM | dCMI_q | label | blind |', '|---|---|---|---|---|---|---|---|---|']
        for b, rec in out['bases'].items():
            for ov in fd['treat']:
                e = rec['fam'].get(f, {}).get(ov)
                if not e or 'delta' not in e:
                    continue
                dd = e['delta']
                L.append(f'| {b} | {ov} | {fmt(dd.get("A1_MI"))} | {fmt(dd.get("CE"))} | {fmt(dd.get("PQ"))} | '
                         f'{fmt(dd.get("PAM"))} | {fmt(dd.get("CMI_q"))} | {e.get("label")} | {e.get("blind", {}).get("label")} |')
        L.append('')
    L += ['## F4 quiet-subset sizes (>= 0.5 s of quiet frames)', '', '| base | n subset | n | in main endpoint |', '|---|---|---|---|']
    for b, v in out['f4_subset'].items():
        L.append(f'| {b} | {v["n_subset"]} | {v["n"]} | {v["in_main_endpoint"]} |')
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
                elif not M.rows_ok(M.tsv_read(p), ids, req_cols(tier, b != 'musdb', clap)):
                    errs.append(f'cell {p.name}: id set or required columns not finite (tier {tier})')
        for f in ('f4_quiet_frames.json', 'f2_input_check.json'):
            if not (OUT / f).is_file():
                errs.append(f'missing {f}')
        sp = OUT / 'summary.json'
        if not sp.is_file():
            errs.append('missing summary.json')
        else:
            s = json.loads(sp.read_text())
            if s.get('kind') != KIND:
                errs.append('summary.json wrong kind')
            if not s.get('f2_zero_assertion', {}).get('pass'):
                errs.append('F2 zero-delta assertion failed (harness error): '
                            + json.dumps(s.get('f2_zero_assertion', {}).get('max_abs')))
        if not (OUT / 'summary_table.md').is_file():
            errs.append('missing summary_table.md')
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


def mem_available_gb():
    for ln in open('/proc/meminfo'):
        if ln.startswith('MemAvailable:'):
            return int(ln.split()[1]) / 1048576
    return 0.0


def preflight(smoke):
    errs = []
    if shutil.disk_usage(H.EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    for p in (H.PARTB / 'part_b_selection.json', H.TTM115, H.MC_TSV, P.REF_DIR, M.MUSDB_DIR / 'windows.json',
              M.MUSDB_DIR / 'cache', M.SE_CKPT, H.PY, KJ_PY, DEMUCS_CKPT):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    try:
        mc, win = get_ids(smoke)
        want = 16 if smoke else 1000
        if len(mc) != want:
            errs.append(f'{len(mc)} Part B ids with reference audio, expected {want}')
        for b in MC_BASES:
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
        need = 20 * 1024 if smoke else 20 * 1024  # smoke peak measured 14.1 GiB alloc / 17.8 GiB reserved
        if f < need:
            log(f'PREFLIGHT WAIT GPU free {f} MiB < {need} MiB')
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
        OUT = H.EVAL_ROOT / 'muq_postproc_attack_190_smoke'
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
