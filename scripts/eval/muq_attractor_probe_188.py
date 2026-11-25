#!/usr/bin/env python
"""188: what MuQ-Eval A1 likes -- take apart, hand-synthesise and overlay the 183 R-arm attractor.

Design: docs/experiments/muq_attractor_probe_188_20261110.md (pre-registered; this file only implements it).

Stages (each resumable through per-key TSV caches under <root>/per_clip/):
  A   describe     A-1 spectrum/period of the 64 R clips (attractor.json); A-2 regenerate 17 checkpoints
                   (torchrun worker, float32) and score them; A-3 embedding-space distance
  B   decompose    52 versions x the R clips (+ the matching control), retention rho
  C   synthesise   ~190 formula stimuli
  D   dev          pick O_synth and r* on 200 dev real clips ONLY, freeze selection.json (timestamp + sha256)
      test         500 x {real, gen} base, 18 treatment/control overlay cells; refuses without selection.json
      fad          gen base at r* (base, +O_synth, +O_R)
  F   real music   2000 real clips: A1/A3a vs the attractor features
  summarize        summary.json (all pre-registered verdict strings) + listening pack

Scoring uses the 183 harness functions unchanged (muq_train_exploit_183 is imported, never edited); MuQ is batched
with GPU resampling, AES and CLAP run in memory (CLAP strictly one clip per call).  See design doc section 6.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path('/home/kojiek/MeanAudio')
sys.path[:0] = [str(REPO), str(REPO / 'scripts/eval'), str(REPO / 'scripts/training')]
import muq_train_exploit_183 as H  # noqa: E402

SELF = Path(__file__).resolve()
TORCHRUN = '/home/kojiek/venvs/dac/bin/torchrun'
DESIGN = 'docs/experiments/muq_attractor_probe_188_20261110.md'
KIND = 'muq_attractor_probe_188_summary_v1'
LISTEN_R = REPO / 'deliverables/listen_R183'
REF_DIR = Path('/mnt/HDD/kojiek/musiccaps_reference')
FT = Path('/home/kojiek/exps_nvme/muq_reward_ft_183')
D187 = Path('/home/kojiek/exps_nvme/muq_reward_dose_187')
SR = 16000
NS = 159600                       # 9.975 s, the length of the 183/187 generations and the listen_R183 clips
COLS = ['A1_MI', 'A1_TA', 'A3a_MI', 'A3a_TA', 'PQ', 'CE', 'CLAP']
R_LIST = [18, 12, 6, 0, -6]
M_A1, DEV_STEP = 0.055, 0.11     # m_A1 and 2m (design 2.D selection)
SEED = 188
NEED_FREE_MIB = 16 * 1024
MIN_FREE = 20 * 1024**3
NTHREAD = 8
CH = 64

FULL = dict(n_test=500, n_dev=200, n_f=2000, n_r=64, n_regen=64, chunk=64, regen_arms=None, gen_only_rstar=False)
SMOKE = dict(n_test=24, n_dev=8, n_f=32, n_r=6, n_regen=8, chunk=16, gen_only_rstar=False,
             regen_arms=['control', 'R_s14159265_3000', 'pilot_l0.1_500', 'ladder_s14159265_160'])
CFG = dict(FULL)
OUT = H.EVAL_ROOT / 'muq_attractor_probe_188'
DELIV = REPO / 'deliverables/listen_attractor_188'
IS_SMOKE = False


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def jdump(path, obj):
    atomic(path, json.dumps(obj, indent=1, default=float))


def rng_for(*parts):
    return np.random.default_rng([SEED] + [zlib.crc32(str(p).encode()) for p in parts])


# ── audio helpers ────────────────────────────────────────
def read_wav(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if sr != SR:
        raise RuntimeError(f'{path}: sr {sr}')
    return x.mean(axis=1).astype(np.float32)


def lufs_of(w):
    import pyloudnorm
    return pyloudnorm.Meter(SR).integrated_loudness(np.asarray(w, np.float64)[:, None])


def to_lufs(w, target=-30.0, cap=0.999):
    """Same rule as level_match_rescore._gain_one (float64 in memory). Returns (float32, lufs_before)."""
    w = np.asarray(w, np.float64)
    lufs = lufs_of(w)
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    if not math.isfinite(lufs) or peak == 0.0:
        g = 0.0
    else:
        g = target - lufs
        if peak * 10 ** (g / 20) > cap:
            g = 20 * math.log10(cap / peak)
    return np.clip(w * 10 ** (g / 20), -1.0, 1.0).astype(np.float32), lufs


def lvl30(w):
    return to_lufs(w, -30.0)


def fit(x, n=NS):
    x = np.asarray(x, np.float32)
    return x[:n] if len(x) >= n else np.resize(x, n).astype(np.float32)


# ── progress / timing ────────────────────────────────────
PROG = {'stage': '', 'key': '', 'done': 0, 'total': 0}


def progress(stage=None, key=None, done=None, total=None):
    for k, v in (('stage', stage), ('key', key), ('done', done), ('total', total)):
        if v is not None:
            PROG[k] = v
    jdump(OUT / 'progress.json', {**PROG, 'updated': time.strftime('%FT%T%z'), 'smoke': IS_SMOKE})


def add_timing(key, n, wall, comp):
    p = OUT / 'timing.json'
    t = json.loads(p.read_text()) if p.is_file() else {'keys': {}, 'components': {}}
    t['keys'][key] = {'n': n, 'wall_s': wall, 'rate': n / wall if wall > 0 else None}
    for k, v in comp.items():
        t['components'][k] = t['components'].get(k, 0.0) + v
    t['total_n'] = sum(v['n'] for v in t['keys'].values())
    t['total_wall_s'] = sum(v['wall_s'] for v in t['keys'].values())
    jdump(p, t)


# ── scorer ───────────────────────────────────────────────
class Scorer:
    def __init__(self):
        self.muq, self.aes, self.clap, self.rs = {}, None, None, None
        self.txt = {}
        self.t = Counter()

    def _muq(self):
        if not self.muq:
            import torch
            import torchaudio
            from muq_gen_steer_179_score import muq_hf_cache
            from newmetrics_muqeval import _load
            with muq_hf_cache():
                for v in ('A1', 'A3a'):
                    self.muq[v] = _load(v, 'cuda')[0]
            self.rs = torchaudio.transforms.Resample(SR, 24000).cuda()
            self.torch = torch

    def _aes(self):
        if self.aes is None:
            from muq_gen_steer_179_score import standard_hf_cache
            with standard_hf_cache():
                import audiobox_aesthetics.infer as aes_infer
                import eval_metrics as EM
                aes_infer.read_wav = EM._read_wav_sf
                self.aes = aes_infer.AesPredictor(checkpoint_pth=None, batch_size=32)

    def _clap(self):
        if self.clap is None:
            from muq_gen_steer_179_score import standard_hf_cache
            import eval_metrics as EM
            with standard_hf_cache():
                self.clap = EM.load_clap()

    def to24(self, wavs):
        torch = self.torch
        out = []
        for w in wavs:
            x = self.rs(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).cuda())
            L = 240000
            if x.shape[-1] >= L:
                s = (x.shape[-1] - L) // 2
                x = x[s:s + L]
            else:
                x = torch.nn.functional.pad(x, (0, L - x.shape[-1]))
            out.append(x)
        return torch.stack(out)

    @staticmethod
    def head_val(v):
        import torch
        v = v.float()
        if v.dim() == 2 and v.shape[-1] == 5:
            v = (torch.softmax(v, -1) * torch.linspace(1, 5, 5, device=v.device)).sum(-1)
        return v.reshape(-1).cpu().numpy()

    def muq_scores(self, wavs, embed=False):
        self._muq()
        torch = self.torch
        res = {f'{v}_{h}': [] for v in ('A1', 'A3a') for h in ('MI', 'TA')}
        emb = []
        with torch.no_grad():
            for s in range(0, len(wavs), 16):
                x = self.to24(wavs[s:s + 16])
                for v in ('A1', 'A3a'):
                    o = self.muq[v](x)
                    for h in ('MI', 'TA'):
                        res[f'{v}_{h}'].extend(self.head_val(o[h]).tolist())
                if embed:
                    emb.append(self.muq['A1'].encoder(x).float().cpu().numpy())
        return res, (np.concatenate(emb) if embed else None)

    def aes_scores(self, wavs):
        self._aes()
        torch = self.torch
        pq, ce = [], []
        for s in range(0, len(wavs), 32):
            chunk = wavs[s:s + 32]
            try:
                rows = self.aes.forward([{'path': torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': SR}
                                         for w in chunk])
            except Exception:  # noqa: BLE001
                rows = [self.aes.forward([{'path': torch.from_numpy(np.ascontiguousarray(w))[None],
                                           'sample_rate': SR}])[0] for w in chunk]
            pq += [float(r['PQ']) for r in rows]
            ce += [float(r['CE']) for r in rows]
        return pq, ce

    def clap_scores(self, wavs, caps):
        import librosa
        self._clap()
        torch = self.torch
        out = []
        with torch.no_grad():
            for w, c in zip(wavs, caps):
                x48 = librosa.resample(np.asarray(w, np.float32), orig_sr=SR, target_sr=48000)
                ae = self.clap.get_audio_embedding_from_data(x48[None], use_tensor=False)
                if c not in self.txt:
                    self.txt[c] = self.clap.get_text_embedding([c], use_tensor=False)
                te = self.txt[c]
                out.append(float((ae * te).sum() / (np.linalg.norm(ae) * np.linalg.norm(te))))
        return out

    def score(self, wavs, caps=None, embed=False):
        t0 = time.time()
        m, emb = self.muq_scores(wavs, embed)
        self.torch.cuda.synchronize()
        t1 = time.time()
        pq, ce = self.aes_scores(wavs)
        t2 = time.time()
        clap = self.clap_scores(wavs, caps) if caps else [math.nan] * len(wavs)
        t3 = time.time()
        self.t['muq'] += t1 - t0
        self.t['aes'] += t2 - t1
        self.t['clap'] += t3 - t2
        rows = [{**{k: m[k][i] for k in m}, 'PQ': pq[i], 'CE': ce[i], 'CLAP': clap[i]} for i in range(len(wavs))]
        return rows, emb


SC = Scorer()
POOL = None                       # ThreadPoolExecutor for stimulus builders


def tsv_read(path):
    with open(path, newline='') as f:
        return {r['id']: {k: float(v) for k, v in r.items() if k != 'id'} for r in csv.DictReader(f, delimiter='\t')}


def tsv_write(path, rows, ids):
    lines = ['id\t' + '\t'.join(COLS)] + [i + '\t' + '\t'.join(repr(float(rows[i][c])) for c in COLS) for i in ids]
    atomic(path, '\n'.join(lines) + '\n')


def run_set(key, items, build, caps=None, stage=''):
    """Score build(item) for every item; per-key TSV cache. caps: {item: caption} for CLAP, else CLAP = nan."""
    items = [str(x) for x in items]
    path = OUT / 'per_clip' / f'{key}.tsv'
    if path.is_file():
        rows = tsv_read(path)
        if set(rows) == set(items):
            return rows
    t0, tm0 = time.time(), dict(SC.t)
    ch = CFG['chunk']
    chunks = [items[i:i + ch] for i in range(0, len(items), ch)]
    res = {}
    nxt = [POOL.submit(build, x) for x in chunks[0]]
    for k, c in enumerate(chunks):
        cur = nxt
        if k + 1 < len(chunks):
            nxt = [POOL.submit(build, x) for x in chunks[k + 1]]
        wavs = [f.result() for f in cur]
        for w in wavs:
            if not np.all(np.isfinite(w)):
                raise RuntimeError(f'{key}: non-finite stimulus')
        rows, _ = SC.score(wavs, [caps[x] for x in c] if caps else None)
        for x, r in zip(c, rows):
            if not all(math.isfinite(r[q]) for q in COLS[:6]) or (caps and not math.isfinite(r['CLAP'])):
                raise RuntimeError(f'{key}: non-finite score for {x}: {r}')
            res[x] = r
        progress(stage, key, len(res), len(items))
    tsv_write(path, res, items)
    wall = time.time() - t0
    add_timing(key, len(items), wall, {k: SC.t[k] - tm0.get(k, 0.0) for k in SC.t})
    log(f'{key}: {len(items)} stimuli in {wall:.1f}s ({len(items) / wall:.1f}/s)')
    return res


def arr(rows, ids, col):
    return np.array([rows[i][col] for i in ids], float)


# ── ids / data ───────────────────────────────────────────
_CAP = {}


def caps_all():
    if not _CAP:
        import eval_metrics as EM
        _CAP.update(dict(EM.load_rows(H.MC_TSV)))
    return _CAP


def ref_path(i):
    return REF_DIR / f'{i}.wav'


def pools():
    """dev/pool/test ids (design section 1). Deterministic from the files on disk."""
    caps = caps_all()
    partb = H.part_b_ids()
    pb = set(partb)
    cand = sorted(i for i in caps if i not in pb and ref_path(i).is_file())
    perm = np.random.default_rng(SEED).permutation(len(cand))
    pool = [cand[j] for j in perm[:CFG['n_f']]]
    test = [i for i in partb if ref_path(i).is_file()][:CFG['n_test']]
    return {'dev': pool[:CFG['n_dev']], 'pool_rest': pool[CFG['n_dev']:], 'test': test, 'n_cand': len(cand)}


@lru_cache(maxsize=None)
def real30(i):
    return lvl30(read_wav(ref_path(i)))[0]


def r_clip(i, which):
    return read_wav(LISTEN_R / 'wav' / f'{i:02d}_{which}.wav')


_RC = {}


def RW(i):
    if ('R', i) not in _RC:
        _RC[('R', i)] = fit(r_clip(i, 'R'))
    return _RC[('R', i)]


def CW(i):
    if ('c', i) not in _RC:
        _RC[('c', i)] = fit(r_clip(i, 'control'))
    return _RC[('c', i)]


def rids():
    return [f'{i:02d}' for i in range(CFG['n_r'])]


# ── features ─────────────────────────────────────────────
def _pool_init():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ.pop('MKL_NUM_THREADS', None)


FPOOL = None


def acf_period(flux, fps):
    n = len(flux)
    f = flux - flux.mean()
    sp = np.fft.rfft(f, 2 * n)
    ac = np.fft.irfft(np.abs(sp) ** 2)[:n]
    if ac[0] <= 0:
        return 0.0, 0.0, 0.0
    ac = ac / ac[0] * n / (n - np.arange(n))
    lo, hi = int(0.05 * fps), min(int(1.0 * fps), n - 2)
    seg = ac[lo:hi]
    pk = [k for k in range(1, len(seg) - 1) if seg[k] > seg[k - 1] and seg[k] >= seg[k + 1]]
    if pk:
        best = max(pk, key=lambda k: seg[k])
        per, pmax = (lo + best) / fps, float(seg[best])
    else:
        per, pmax = 0.0, 0.0
    l16 = float(ac[int(round(0.145 * fps)):int(round(0.175 * fps)) + 1].max())
    return per, pmax, l16


def feats(w, partials=None):
    """Attractor features of one 16 kHz clip (already -30 LUFS). All defined in design section 2.A / 2.F."""
    import librosa
    from scipy.ndimage import uniform_filter1d
    w = np.asarray(w, np.float32)
    S = np.abs(librosa.stft(w, n_fft=1024, hop_length=80, center=True)) ** 2         # 200 fps
    freqs = np.fft.rfftfreq(1024, 1 / SR)
    M = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=64, n_fft=1024)
    L = np.log10(M + 1e-8)
    flux = np.maximum(0.0, np.diff(L, axis=1)).sum(0)
    per, pmax, p016 = acf_period(flux, 200.0)
    ps = S.mean(1)
    tot = ps.sum() + 1e-20
    out = {'period': per, 'per_max': pmax, 'per016': p016,
           'band200_500': float(ps[(freqs >= 200) & (freqs < 500)].sum() / tot),
           'centroid': float((freqs * ps).sum() / tot)}
    if partials:
        msk = np.zeros_like(freqs, bool)
        for f0 in partials:
            msk |= np.abs(freqs - f0) <= 0.03 * f0
        out['partial3'] = float(ps[msk].sum() / tot)
    e = M.sum(0)
    act = e > e.max() * 1e-4
    mean_v = M[:, act].mean(1) if act.any() else M.mean(1)
    mv = M[:, act] if act.any() else M
    cos = (mv * mean_v[:, None]).sum(0) / (np.linalg.norm(mv, axis=0) * np.linalg.norm(mean_v) + 1e-20)
    out['frame_stab'] = float(cos.mean())
    contour = M.mean(1)
    contour = contour / (np.linalg.norm(contour) + 1e-20)
    mag = np.abs(librosa.stft(w, n_fft=1024, hop_length=256, center=True))
    Hm, Pm = librosa.decompose.hpss(mag)
    out['harm_ratio'] = float((Hm ** 2).sum() / ((Hm ** 2).sum() + (Pm ** 2).sum() + 1e-20))
    thr = flux.mean() + flux.std()
    pk = [k for k in range(1, len(flux) - 1) if flux[k] > thr and flux[k] >= flux[k - 1] and flux[k] > flux[k + 1]]
    last, cnt = -99, 0
    for k in pk:
        if k - last >= 5:
            cnt += 1
            last = k
    out['onset_rate'] = cnt / (len(w) / SR)
    fr = uniform_filter1d(w.astype(np.float64) ** 2, 800)[::400]
    rms = np.sqrt(fr)
    out['rms_cv'] = float(rms.std() / (rms.mean() + 1e-20))
    return out, contour


def _feat_job(args):
    w, partials = args
    f, c = feats(w, partials)
    f['_contour'] = c.tolist()
    return f


def feat_many(wavs, partials=None):
    global FPOOL
    if FPOOL is None:
        import multiprocessing as mp
        FPOOL = mp.get_context('fork').Pool(4, initializer=_pool_init)
    return FPOOL.map(_feat_job, [(w, partials) for w in wavs], chunksize=4)


def env_fold(w):
    from scipy.ndimage import uniform_filter1d
    return np.sqrt(uniform_filter1d(np.asarray(w, np.float64) ** 2, 160, mode='nearest'))[::16]   # 1 kHz envelope


def fold_period(w, ts=None):
    """Best fold period in 0.140-0.170 s (variance of the folded 1 kHz envelope) + 64-bin profile rolled to the max rise."""
    e = env_fold(w)
    t = np.arange(len(e)) / 1000.0
    ts = np.arange(0.140, 0.1705, 0.0005) if ts is None else ts
    best, bv = None, -1.0
    ev = e.var() + 1e-20
    for T in ts:
        ph = ((t % T) / T * 32).astype(int)
        cnt = np.bincount(ph, minlength=32)
        m = np.bincount(ph, weights=e, minlength=32) / np.maximum(cnt, 1)
        v = m.var() / ev
        if v > bv:
            best, bv = T, v
    ph = ((t % best) / best * 64).astype(int)
    cnt = np.bincount(ph, minlength=64)
    prof = np.bincount(ph, weights=e, minlength=64) / np.maximum(cnt, 1)
    k0 = int(np.argmax(np.diff(np.concatenate([prof, prof[:1]]))))
    prof = np.roll(prof, -k0)
    prof = prof / (prof.max() + 1e-20)
    return float(best), float(bv), prof, k0 / 64 * best


# ── Part A-1: the attractor ──────────────────────────────
ATT = {}


def attractor():
    p = OUT / 'attractor.json'
    if p.is_file():
        return json.loads(p.read_text())
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, welch
    n = CFG['n_r']
    log(f'A-1 attractor from {n} R clips')
    acc, per = None, []
    for i in range(n):
        f, pxx = welch(RW(i).astype(np.float64), fs=SR, nperseg=65536, nfft=131072)
        per.append(pxx)
        acc = pxx if acc is None else acc + pxx
    mean = acc / n
    df = f[1] - f[0]
    sm = gaussian_filter1d(mean, 4.0 / df)
    sel = f <= 4000
    idx, _ = find_peaks(sm * sel, distance=int(25 / df))
    idx = sorted(idx, key=lambda k: -sm[k])[:24]
    peaks = []
    for k in idx:
        lo, hi = max(0, k - int(12 / df)), k + int(12 / df) + 1
        fc = float((f[lo:hi] * mean[lo:hi]).sum() / mean[lo:hi].sum())
        amp = float(math.sqrt(mean[lo:hi].sum()))
        pk = [float(f[lo:hi][int(np.argmax(p_[lo:hi]))]) for p_ in per]
        peaks.append({'f': fc, 'amp': amp, 'spread': {'median': float(np.median(pk)), 'p10': float(np.percentile(pk, 10)),
                                                      'p90': float(np.percentile(pk, 90))}})
    top = max(p_['amp'] for p_ in peaks)
    for p_ in peaks:
        p_['amp_rel'] = p_['amp'] / top
    peaks.sort(key=lambda p_: p_['f'])
    mid = sorted([p_ for p_ in peaks if 150 <= p_['f'] <= 600], key=lambda p_: -p_['amp'])[:3]
    mid.sort(key=lambda p_: p_['f'])
    low = sorted([p_ for p_ in peaks if 40 <= p_['f'] <= 100], key=lambda p_: -p_['amp'])
    if len(mid) < 3 or not low:
        raise SystemExit(f'A-1: could not find 3 partials in 150-600 Hz and one in 40-100 Hz ({len(mid)}, {len(low)})')
    Ts, profs = [], []
    for i in range(n):
        T, v, prof, _ = fold_period(RW(i))
        Ts.append(T)
        profs.append(prof)
    T_med = float(np.median(Ts))
    prof = np.mean(profs, axis=0)
    prof = prof / prof.max()
    f3 = mid[0]['f']
    att = {'n_clips': n, 'freq_res_hz': df, 'peaks': peaks, 'main3': [p_['f'] for p_ in mid],
           'main3_amp': [p_['amp_rel'] for p_ in mid], 'main3_spread': [p_['spread'] for p_ in mid],
           'f_low': low[0]['f'], 'f_low_amp': low[0]['amp_rel'], 'f_low_spread': low[0]['spread'],
           'ratio_to_f3_over_3': [p_['f'] / (f3 / 3) for p_ in mid],
           'T_med': T_med, 'T_p10': float(np.percentile(Ts, 10)), 'T_p90': float(np.percentile(Ts, 90)),
           'period_profile': prof.tolist(),
           'between_clip_wave_corr_note': 'see a2/a3 for between-clip contour cos'}
    jdump(p, att)
    log(f'A-1: main partials {[round(x, 1) for x in att["main3"]]} Hz, f_low {att["f_low"]:.1f} Hz, T {T_med * 1000:.1f} ms')
    return att


# ── Part A-2: regenerate checkpoints ─────────────────────
def regen_arm_paths():
    arms = {'control': None}
    for s in ('14159265', '16180339'):
        arms[f'R_s{s}_3000'] = FT / f'seeds/s{s}/net_step003000.pth'
    for st in (500, 1000, 1500, 2000, 2500, 3000):
        arms[f'pilot_l0.1_{st}'] = FT / f'pilot/lam0.1/net_step{st:06d}.pth'
    for lam in ('0.3', '1.0'):
        arms[f'pilot_l{lam}_3000'] = FT / f'pilot/lam{lam}/net_step003000.pth'
    for st in (10, 20, 40, 80, 160):
        arms[f'ladder_s14159265_{st}'] = D187 / f'ladder/s14159265/net_step{st:06d}.pth'
    arms['Z160_s14159265'] = D187 / 'zero/s14159265/net_step000160.pth'
    keep = CFG['regen_arms']
    return {k: v for k, v in arms.items() if keep is None or k in keep}


def regen_worker(root, n, arms):
    """torchrun entry: regenerate each arm in float32 with the listen_R183 recipe (same noise, 1-NFE CFG0)."""
    sys.path[:0] = [str(REPO / 'scripts/training')]
    root = Path(root)
    od = root / 'regen'
    od.mkdir(parents=True, exist_ok=True)
    import types
    import torch
    sys.argv = [sys.argv[0]]
    import muq_reward_ft_183 as R
    args = types.SimpleNamespace(seed=14159265, lr=1e-5, exp_id='attractor_188_regen', init='',
                                 out_dir=str(root / 'regen_run'))
    cfg, trainer, *_, init = R.setup(args)
    dec = R.Decoder(trainer)
    rows = R.build_val_tsv(root / 'regen_val.tsv', 1000)[:n]
    caps = [r['caption'] for r in rows]
    jdump(od / 'ids.json', [{'id': r['id'], 'caption': r['caption']} for r in rows])
    net = trainer.network.module
    net.eval()
    ctrl = {k: v.clone() for k, v in net.state_dict().items()}
    g = torch.Generator(device='cuda').manual_seed(183)
    noise = torch.randn(n, net.latent_seq_len, net.latent_dim, device='cuda', generator=g)
    paths = regen_arm_paths()
    for arm in arms:
        out = od / f'{arm}.npy'
        if out.is_file():
            continue
        ck = paths[arm]
        if ck is not None and not Path(ck).is_file():
            atomic(od / f'{arm}.missing', str(ck))
            log(f'regen {arm}: checkpoint missing {ck}')
            continue
        net.load_state_dict(ctrl if ck is None else torch.load(ck, map_location='cuda'))
        ws = []
        with torch.no_grad():
            for s in range(0, n, 8):
                tf, tfc = trainer.features.encode_text(caps[s:s + 8])
                x = R.sample_1nfe(net, noise[s:s + 8], tf.float(), tfc.float())
                wav = dec.latent_to_wav(x).cpu().numpy()
                ws += [R.lvl30(w)[0].astype(np.float32)[:NS] for w in wav]
        np.save(str(out) + '.tmp.npy', np.stack(ws))
        os.replace(str(out) + '.tmp.npy', out)
        log(f'regen {arm}: {len(ws)} clips')
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


def regen_all():
    arms = list(regen_arm_paths())
    od = OUT / 'regen'
    todo = [a for a in arms if not (od / f'{a}.npy').is_file() and not (od / f'{a}.missing').is_file()]
    if not todo:
        return True
    log(f'A-2 regen worker for {len(todo)} arms')
    env = {k: v for k, v in os.environ.items() if k != 'MKL_NUM_THREADS'}
    cmd = [TORCHRUN, '--standalone', '--nproc_per_node=1', str(SELF), '--regen-worker', '--root', str(OUT),
           '--n-regen', str(CFG['n_regen']), '--regen-arms', ','.join(todo)]
    with open(OUT / 'regen_worker.log', 'a') as g:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=g, stderr=subprocess.STDOUT).returncode
    log(f'A-2 regen worker rc={rc}')
    return rc == 0


# ── Part A: score regen arms, embedding distances ────────
def cos_pairs(C):
    C = np.asarray(C)
    if len(C) < 2:
        return float('nan')
    G = C @ C.T
    return float(np.median(G[~np.eye(len(C), dtype=bool)]))


def part_a2(att):
    ok = regen_all()
    od = OUT / 'regen'
    arms = {}
    ref_c = None
    if (OUT / 'a2_ref_contour.json').is_file():
        ref_c = np.array(json.loads((OUT / 'a2_ref_contour.json').read_text()))
    else:
        cs = np.array([f['_contour'] for f in feat_many([RW(i) for i in range(CFG['n_r'])], att['main3'])])
        ref_c = cs.mean(0)
        ref_c = ref_c / np.linalg.norm(ref_c)
        jdump(OUT / 'a2_ref_contour.json', ref_c.tolist())
    for arm in regen_arm_paths():
        if (od / f'{arm}.missing').is_file():
            arms[arm] = {'status': 'missing', 'path': (od / f'{arm}.missing').read_text()}
            continue
        if not (od / f'{arm}.npy').is_file():
            arms[arm] = {'status': 'not_generated'}
            continue
        w = np.load(od / f'{arm}.npy')
        rows = run_set(f'A2_{arm}', [f'{i:02d}' for i in range(len(w))], lambda x, w=w: w[int(x)], stage='A2')
        ids = sorted(rows)
        fs = feat_many(list(w), att['main3'])
        C = np.array([f['_contour'] for f in fs])
        arms[arm] = {'status': 'ok', 'n': len(ids), **{c: float(arr(rows, ids, c).mean()) for c in COLS[:6]},
                     'frac_A1_gt5': float((arr(rows, ids, 'A1_MI') > 5).mean()),
                     'period_median': float(np.median([f['period'] for f in fs])),
                     'per_max_mean': float(np.mean([f['per_max'] for f in fs])),
                     'per016_mean': float(np.mean([f['per016'] for f in fs])),
                     'partial3_share_mean': float(np.mean([f['partial3'] for f in fs])),
                     'between_clip_contour_cos_median': cos_pairs(C),
                     'cos_to_R_ref_contour_mean': float((C @ ref_c).mean())}
    return {'worker_ok': ok, 'arms': arms}


def part_a3():
    p = OUT / 'a3.json'
    if p.is_file():
        return json.loads(p.read_text())
    pl = pools()
    n_r = CFG['n_r']
    groups = {'dev_real': [real30(i) for i in pl['dev']], 'control': [lvl30(CW(i))[0] for i in range(n_r)],
              'R': [lvl30(RW(i))[0] for i in range(n_r)]}
    emb, mi = {}, {}
    for g, ws in groups.items():
        rows, e = SC.score(ws, embed=True)
        emb[g], mi[g] = e, np.array([r['A1_MI'] for r in rows])
    real = emb['dev_real']
    mu = real.mean(0)
    k = min(32, len(real) - 1)
    U, Sv, Vt = np.linalg.svd(real - mu, full_matrices=False)
    sig = Sv[:k] / math.sqrt(max(len(real) - 1, 1)) + 1e-12
    out = {'n_pca': k}
    for g, e in emb.items():
        z = ((e - mu) @ Vt[:k].T) / sig
        zd = np.sqrt((z ** 2).sum(1))
        cs = (e @ mu) / (np.linalg.norm(e, axis=1) * np.linalg.norm(mu))
        out[g] = {'n': len(e), 'A1_MI_mean': float(mi[g].mean()), 'norm_median': float(np.median(np.linalg.norm(e, axis=1))),
                  'cos_to_real_mean_median': float(np.median(cs)), 'z_dist_median': float(np.median(zd)),
                  'z_dist_p90': float(np.percentile(zd, 90))}
    rz = out['dev_real']['z_dist_p90']
    out['R_z_dist_over_real_p90'] = out['R']['z_dist_median'] / rz
    jdump(p, out)
    return out


# ── Part B: decompose ────────────────────────────────────
@lru_cache(maxsize=None)
def sos_f(fc, kind, order=8):
    from scipy.signal import butter
    return butter(order, fc, kind, fs=SR, output='sos')


def filt(x, fc, kind, order=8):
    from scipy.signal import sosfiltfilt
    return sosfiltfilt(sos_f(fc, kind, order), np.asarray(x, np.float64))


OCT = [(0, 125), (125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def band_op(x, lo, hi, keep):
    if lo == 0:
        return filt(x, hi, 'low' if keep else 'high')
    if hi >= 8000:
        return filt(x, lo, 'high' if keep else 'low')
    return filt(x, (lo, hi), 'bandpass' if keep else 'bandstop', 4)


def pink(n, rng):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X), dtype=float)
    f[0] = 1.0
    X = X / np.sqrt(f)
    X[0] = 0
    return np.fft.irfft(X, n)


def env20(x):
    from scipy.ndimage import uniform_filter1d
    return np.sqrt(uniform_filter1d(np.asarray(x, np.float64) ** 2, 320, mode='nearest'))


def _speed(rate):
    import librosa
    return lambda r, c, i, nm: librosa.effects.time_stretch(r, rate=rate)


def _shift(k):
    import librosa
    return lambda r, c, i, nm: librosa.effects.pitch_shift(r, sr=SR, n_steps=k)


def _shuffle(sec):
    def f(r, c, i, nm):
        seg = int(round(sec * SR))
        parts = [r[s:s + seg] for s in range(0, len(r), seg)]
        order = rng_for(nm, i).permutation(len(parts))
        return np.concatenate([parts[k] for k in order])
    return f


def _tile(r, c, i, nm):
    T, _, _, ph0 = fold_period(r)
    s0 = int(ph0 * SR)
    seg = r[s0:s0 + int(round(T * SR))]
    return np.resize(seg, NS)


def _randphase(r, c, i, nm):
    X = np.fft.rfft(np.asarray(r, np.float64))
    ph = rng_for(nm, i).uniform(0, 2 * np.pi, len(X))
    ph[0] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), len(r))


def _envpink(r, c, i, nm):
    e = env20(r)
    return pink(len(r), rng_for(nm, i)) * (e / e.mean())


def _envctrl(r, c, i, nm):
    e = env20(r)
    return c * (e / e.mean())


def _flat(r, c, i, nm):
    e = env20(r)
    return r / (e + 1e-3 * e.max())


def _eq(r, c, i, nm):
    from scipy.signal import welch
    f, pr = welch(np.asarray(r, np.float64), fs=SR, nperseg=8192)
    _, pc = welch(np.asarray(c, np.float64), fs=SR, nperseg=8192)
    g = np.clip(np.sqrt((pr + 1e-20) / (pc + 1e-20)), 10 ** (-60 / 20), 10 ** (60 / 20))
    X = np.fft.rfft(np.asarray(c, np.float64))
    return np.fft.irfft(X * np.interp(np.fft.rfftfreq(len(c), 1 / SR), f, g), len(c))


def _frac(p):
    def f(r, c, i, nm):
        k = int(p * len(r))
        a = np.zeros(len(r))
        a[:max(k - 160, 0)] = 1.0
        lo, hi = max(k - 160, 0), min(k + 160, len(r))
        a[lo:hi] = np.linspace(1, 0, hi - lo)
        return a * r + (1 - a) * c
    return f


def b_versions():
    v = {'ref_R': lambda r, c, i, nm: r, 'ref_ctrl': lambda r, c, i, nm: c,
         'shuf016': _shuffle(0.16), 'shuf005': _shuffle(0.05), 'reverse': lambda r, c, i, nm: r[::-1].copy()}
    for s in (0.5, 0.71, 1.41, 2.0):
        v[f'speed{s}'] = _speed(s)
    v.update({'flat20ms': _flat, 'tile_period': _tile, 'randphase': _randphase, 'env_pink': _envpink,
              'env_ctrl': _envctrl})
    for k in (0, 2, -2, 4, -4, 7, -7, 12, -12):
        v[f'transpose{k:+d}'] = _shift(k)
    for fc in (500, 1000, 2000, 4000):
        v[f'lp{fc}'] = (lambda fc: lambda r, c, i, nm: filt(r, fc, 'low'))(fc)
    for fc in (150, 300, 500, 1000):
        v[f'hp{fc}'] = (lambda fc: lambda r, c, i, nm: filt(r, fc, 'high'))(fc)
    for lo, hi in OCT:
        v[f'notch{lo}-{hi}'] = (lambda lo, hi: lambda r, c, i, nm: band_op(r, lo, hi, False))(lo, hi)
    for lo, hi in OCT:
        v[f'keep{lo}-{hi}'] = (lambda lo, hi: lambda r, c, i, nm: band_op(r, lo, hi, True))(lo, hi)
    v['eq_ctrl'] = _eq
    for p in (10, 25, 50, 75):
        v[f'frac{p}'] = _frac(p / 100)
    v['loud-40'] = lambda r, c, i, nm: r
    v['loud-20'] = lambda r, c, i, nm: r
    return v


B_TARGET = {'loud-40': -40.0, 'loud-20': -20.0}


def build_B(name, i):
    i = int(i)
    out = b_versions()[name](RW(i), CW(i), i, name)
    out = fit(np.asarray(out, np.float32))
    return to_lufs(out, B_TARGET.get(name, -30.0))[0]


def rho_ci(v, c, r, rng, nb=2000):
    v, c, r = (np.asarray(x, float) for x in (v, c, r))
    idx = rng.integers(0, len(v), size=(nb, len(v)))
    den = r[idx].mean(1) - c[idx].mean(1)
    b = (v[idx].mean(1) - c[idx].mean(1)) / den
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def part_b():
    ids = rids()
    for name in b_versions():
        run_set(f'B_{name}', ids, lambda x, name=name: build_B(name, x), stage='B')


def part_b_summary(rng):
    ids = rids()
    ref = {k: tsv_read(OUT / 'per_clip' / f'B_{k}.tsv') for k in ('ref_R', 'ref_ctrl')}
    a_r, a_c = arr(ref['ref_R'], ids, 'A1_MI'), arr(ref['ref_ctrl'], ids, 'A1_MI')
    table = []
    for name in b_versions():
        rows = tsv_read(OUT / 'per_clip' / f'B_{name}.tsv')
        a = arr(rows, ids, 'A1_MI')
        den = a_r.mean() - a_c.mean()
        table.append({'name': name, 'A1_MI': float(a.mean()), 'A3a_MI': float(arr(rows, ids, 'A3a_MI').mean()),
                      'PQ': float(arr(rows, ids, 'PQ').mean()), 'CE': float(arr(rows, ids, 'CE').mean()),
                      'rho': float((a.mean() - a_c.mean()) / den), 'rho_ci': rho_ci(a, a_c, a_r, rng)})
    return {'n_clips': len(ids), 'n_versions': len(table), 'A1_ref_R': float(a_r.mean()), 'A1_ref_ctrl': float(a_c.mean()),
            'table': table}


# ── Part C: hand synthesis ───────────────────────────────
T_ENV = {'const': None, 'am0.08': ('am', .08), 'am0.12': ('am', .12), 'am0.16': ('am', .16), 'am0.24': ('am', .24),
         'am0.32': ('am', .32), 'am0.64': ('am', .64), 'pluck0.16': ('pluck', .16), 'pluck0.32': ('pluck', .32),
         'meas': ('meas', None)}
TIMBRES = ('sine', 'saw', 'sawlp')
GROUPS = ('P1', 'P2', 'P3', 'P4', 'P5', 'P6')
NOISE_CTRL = ['white', 'pink', 'pink_am0.16', 'click0.16', 'kick0.16', 'kick0.5']


def c_names():
    tonal = [f'{g}_{t}_{e}' for g in GROUPS for t in TIMBRES for e in T_ENV] + [f'bridge_N{n}' for n in (3, 6, 12, 24)]
    return tonal, NOISE_CTRL


def group_notes(g, att):
    m, am = att['main3'], att['main3_amp']
    if g == 'P1':
        return [m[1]], [1.0]
    if g == 'P2':
        return list(m), list(am)
    if g == 'P3':
        return list(m) + [att['f_low']], list(am) + [att['f_low_amp']]
    if g == 'P4':
        return [261.63, 329.63, 392.00], [1.0, 1.0, 1.0]
    if g == 'P5':
        return [1000.0], [1.0]
    return [f * 2 ** (7 / 12) for f in m], list(am)


def synth_tone(notes, amps, timbre, n=NS):
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, a in zip(notes, amps):
        if timbre == 'sine':
            y += a * np.sin(2 * np.pi * f * t)
        else:
            for k in range(1, min(int(7000 // f), 200) + 1):
                y += a / k * np.sin(2 * np.pi * k * f * t)
    if timbre == 'sawlp':
        from scipy.signal import butter, sosfilt
        y = sosfilt(butter(4, 1500, 'low', fs=SR, output='sos'), y)
    return y


def synth_env(spec, att, n=NS):
    t = np.arange(n) / SR
    if spec is None:
        return np.ones(n)
    kind, T = spec
    if kind == 'am':
        return (1 - np.cos(2 * np.pi * t / T)) / 2
    if kind == 'pluck':
        ph = t % T
        return np.minimum(ph / 0.005, 1.0) * np.exp(-ph / (T / 3))
    prof = np.array(att['period_profile'])
    Tm = att['T_med']
    ph = (t % Tm) / Tm * len(prof)
    return np.interp(ph, np.arange(len(prof) + 1), np.concatenate([prof, prof[:1]]))


def build_C(name):
    att = ATT
    n = NS
    if name in NOISE_CTRL:
        rng = rng_for('C', name)
        t = np.arange(n) / SR
        if name == 'white':
            y = rng.standard_normal(n)
        elif name == 'pink':
            y = pink(n, rng)
        elif name == 'pink_am0.16':
            y = pink(n, rng) * (1 - np.cos(2 * np.pi * t / 0.16)) / 2
        elif name == 'click0.16':
            y = np.zeros(n)
            k = np.hanning(9)
            for s in range(0, n - 9, int(0.16 * SR)):
                y[s:s + 9] += k
        else:
            T = float(name[4:])
            ph = t % T
            y = np.sin(2 * np.pi * np.cumsum(45 + 75 * np.exp(-ph / 0.03)) / SR) * np.exp(-ph / 0.06)
        return lvl30(y)[0]
    if name.startswith('bridge_N'):
        k = int(name[8:])
        top = sorted(att['peaks'], key=lambda p: -p['amp_rel'])[:k]
        y = synth_tone([p['f'] for p in top], [p['amp_rel'] for p in top], 'sine') * synth_env(T_ENV['meas'], att)
        return lvl30(y)[0]
    g, tim, e = name.split('_', 2)
    notes, amps = group_notes(g, att)
    y = synth_tone(notes, amps, tim) * synth_env(T_ENV[e], att)
    return lvl30(y)[0]


def part_c():
    tonal, ctrl = c_names()
    names = tonal + ctrl
    return run_set('C_all', names, build_C, stage='C')


def part_c_summary():
    tonal, ctrl = c_names()
    rows = tsv_read(OUT / 'per_clip' / 'C_all.tsv')
    ref = {k: tsv_read(OUT / 'per_clip' / f'B_{k}.tsv') for k in ('ref_R', 'ref_ctrl')}
    ids = rids()
    a_r, a_c = arr(ref['ref_R'], ids, 'A1_MI').mean(), arr(ref['ref_ctrl'], ids, 'A1_MI').mean()
    dev = tsv_read(OUT / 'per_clip' / 'D_dev_base.tsv')
    p90 = float(np.percentile(arr(dev, sorted(dev), 'A1_MI'), 90))
    rho = lambda a: (a - a_c) / (a_r - a_c)  # noqa: E731
    table = [{'name': k, 'kind': 'control' if k in ctrl else 'tonal', **{c: rows[k][c] for c in COLS[:6]},
              'rho': rho(rows[k]['A1_MI'])} for k in tonal + ctrl]
    best = max((t for t in table if t['kind'] == 'tonal'), key=lambda t: t['A1_MI'])
    if best['rho'] >= 0.5:
        verdict = 'simple_recipe_reproduces'
    elif best['A1_MI'] > p90:
        verdict = 'simple_recipe_partial'
    else:
        verdict = 'not_reproduced'
    main = {'group': {}, 'timbre': {}, 'envelope': {}}
    for t in table:
        if t['kind'] != 'tonal' or t['name'].startswith('bridge'):
            continue
        g, tim, e = t['name'].split('_', 2)
        for k, v in (('group', g), ('timbre', tim), ('envelope', e)):
            main[k].setdefault(v, []).append(t['A1_MI'])
    main = {k: {v: float(np.mean(x)) for v, x in d.items()} for k, d in main.items()}
    top10 = sorted(table, key=lambda t: -t['A1_MI'])[:10]
    return {'verdict': verdict, 'n_stimuli': len(table), 'best_tonal': best, 'dev_real_A1_p90': p90,
            'A1_ref_R': float(a_r), 'A1_ref_ctrl': float(a_c), 'main_effects': main, 'top10': top10, 'table': table}


# ── Part D: overlay ──────────────────────────────────────
OV_CTRL = ['pink', 'sine1k', 'click', 'music']
C_NAME = {'pink': 'pink', 'sine1k': 'P5_sine_const', 'click': 'click0.16'}


def mix(base30, ov, r):
    ov = fit(ov, len(base30)) if len(ov) != len(base30) else ov
    ol = lufs_of(ov)
    o = np.asarray(ov, np.float64) * 10 ** ((-30.0 - r - ol) / 20)
    return lvl30(np.asarray(base30, np.float64) + o)[0]


_OV = {}


def overlay(name, sel=None):
    key = (name, sel['O_synth'] if sel else None)
    if key in _OV:
        return _OV[key]
    if name == 'osynth':
        w = build_C(sel['O_synth'])
    elif name == 'oR':
        w = lvl30(RW(sel['O_R']['idx']))[0]
    else:
        w = build_C(C_NAME[name])
    _OV[key] = w
    return w


def dev_ids():
    return pools()['dev']


def sel_path():
    return OUT / 'selection.json'


def load_selection():
    p = sel_path()
    if not p.is_file():
        raise SystemExit('D-test refuses to run: selection.json missing (run D-dev first)')
    s = json.loads(p.read_text())
    body = {k: v for k, v in s.items() if k not in ('sha256',)}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != s.get('sha256'):
        raise SystemExit('D-test refuses to run: selection.json sha256 mismatch (edited after freezing?)')
    return s


def d_dev():
    if sel_path().is_file():
        return load_selection()
    ids = dev_ids()
    base = run_set('D_dev_base', ids, real30, stage='D-dev')
    tonal, _ = c_names()
    Cr = tsv_read(OUT / 'per_clip' / 'C_all.tsv')
    cand = sorted(tonal, key=lambda k: -Cr[k]['A1_MI'])[:6]
    b_a1 = arr(base, ids, 'A1_MI')
    tab = {}
    for c in cand:
        ov = build_C(c)
        rows = run_set(f'D_dev_cand_{c}_r6', ids, lambda i, ov=ov: mix(real30(i), ov, 6), stage='D-dev')
        tab[c] = float((arr(rows, ids, 'A1_MI') - b_a1).mean())
    osy = max(tab, key=lambda k: tab[k])
    ov = build_C(osy)
    dose = {}
    for r in R_LIST:
        rows = run_set(f'D_dev_os_r{r}' if r != 6 else f'D_dev_cand_{osy}_r6', ids,
                       lambda i, r=r: mix(real30(i), ov, r), stage='D-dev')
        dose[str(r)] = float((arr(rows, ids, 'A1_MI') - b_a1).mean())
    okr = [r for r in R_LIST if dose[str(r)] >= DEV_STEP]
    rstar = max(okr) if okr else 0
    rref = tsv_read(OUT / 'per_clip' / 'B_ref_R.tsv')
    order = sorted(rids(), key=lambda i: rref[i]['A1_MI'])
    ior = int(order[len(order) // 2])
    body = {'document_kind': 'muq_attractor_probe_188_selection_v1', 'created': time.strftime('%FT%T%z'),
            'created_epoch': time.time(), 'smoke': IS_SMOKE, 'n_dev': len(ids), 'candidates_dev_dA1_r6': tab,
            'O_synth': osy, 'dev_dA1_by_r': dose, 'dev_step_2m': DEV_STEP, 'r_star': rstar,
            'O_R': {'idx': ior, 'A1_on_disk_clip': rref[f'{ior:02d}']['A1_MI'], 'rule': 'median R_A1 of the R clips'},
            'dev_ids_sha256': hashlib.sha256(','.join(ids).encode()).hexdigest()}
    body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    # sha256 covers the body without the sha256 key itself
    chk = {k: v for k, v in body.items() if k != 'sha256'}
    body['sha256'] = hashlib.sha256(json.dumps(chk, sort_keys=True).encode()).hexdigest()
    jdump(sel_path(), body)
    log(f'D-dev: O_synth {osy}  dev dA1 by r {dose}  r* {rstar}  O_R clip {ior}')
    return body


def gen_base_dir():
    return OUT / 'gen_base/audio'


def gen_base(test):
    """Regenerate the 500 test prompts with the 183 control (cfg0, 25 steps) through eval.py."""
    ad = gen_base_dir()
    if ad.is_dir() and sorted(p.stem for p in ad.glob('*.flac')) == sorted(test):
        return
    import eval_metrics as EM  # noqa: F401
    caps = caps_all()
    tsv = OUT / 'gen_base/test.tsv'
    tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(tsv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['id', 'caption'])
        for i in test:
            w.writerow([i, caps[i]])
    shutil.rmtree(ad, ignore_errors=True)
    H.P = H.Paths(OUT / 'h183')
    a = dict(H.cell_spec('C_cfg0_s14159265'))
    a['tsv'] = str(tsv)
    cmd = H.eval_cmd(a, ad)
    log(f'gen base: eval.py on {len(test)} prompts')
    with open(OUT / 'gen_base/gen.log', 'w') as g:
        subprocess.run(cmd, cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    got = sorted(p.stem for p in ad.glob('*.flac'))
    if got != sorted(test):
        raise RuntimeError(f'gen base: {len(got)} flac, expected {len(test)}')


_G = {}


def gen30(i):
    if i not in _G:
        _G[i] = lvl30(read_wav(gen_base_dir() / f'{i}.flac'))[0]
    return _G[i]


def test_cells(rstar):
    cells = [('base', None)] + [(o, r) for o in ('osynth', 'oR') for r in R_LIST]
    rc = [rstar, 0] if rstar != 0 else [0, 6]
    cells += [(o, r) for o in OV_CTRL for r in rc]
    return cells


def cell_key(base, ov, r):
    return f'T_{base}_base' if ov == 'base' else f'T_{base}_{ov}_r{r}'


def d_test(sel):
    pl = pools()
    test = pl['test']
    caps = caps_all()
    gen_base(test)
    dev = pl['dev']
    for base, getter in (('real', real30), ('gen', gen30)):
        for ov, r in test_cells(sel['r_star']):
            if base == 'gen' and CFG['gen_only_rstar'] and ov != 'base' and r not in (sel['r_star'], 0):
                continue
            key = cell_key(base, ov, r)
            if ov == 'base':
                run_set(key, test, getter, caps, stage='D-test')
                continue
            if ov == 'music':
                f = lambda i, getter=getter, r=r: mix(getter(i), real30(dev[test.index(i) % len(dev)]), r)  # noqa: E731
            else:
                o = overlay(ov, sel)
                f = lambda i, getter=getter, o=o, r=r: mix(getter(i), o, r)  # noqa: E731
            run_set(key, test, f, caps, stage='D-test')


def fad_cell(name):
    """Subprocess entry: FAD of <root>/fad/<name>/ against the MusicCaps reference."""
    import eval_metrics as EM
    d = OUT / 'fad' / name
    rows = EM.load_rows(OUT / 'gen_base/test.tsv')
    t0 = time.time()
    fad, pairs = EM.score_fad(rows, d, H.FAD_REF, num_samples=0)
    jdump(OUT / 'fad' / f'{name}.json', {'fad': fad, 'pairs': pairs, 'n': len(rows), 'seconds': time.time() - t0})
    return 0


def d_fad(sel):
    import soundfile as sf
    pl = pools()
    test, rs = pl['test'], sel['r_star']
    for name, ov in (('base', None), ('osynth', 'osynth'), ('oR', 'oR')):
        out = OUT / 'fad' / f'{name}.json'
        if out.is_file():
            continue
        d = OUT / 'fad' / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        o = overlay(ov, sel) if ov else None
        for i in test:
            w = gen30(i) if o is None else mix(gen30(i), o, rs)
            sf.write(str(d / f'{i}.flac'), w, SR, subtype='PCM_24')
        env = {k: v for k, v in os.environ.items() if k != 'MKL_NUM_THREADS'}
        cmd = [str(H.PY), '-I', str(SELF), '--root', str(OUT), '--fad-cell', name]
        if IS_SMOKE:
            cmd.append('--smoke')
        with open(OUT / 'fad' / f'{name}.log', 'w') as g:
            rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=g, stderr=subprocess.STDOUT).returncode
        if rc:
            if IS_SMOKE:
                jdump(out, {'fad': None, 'error': f'rc {rc} (smoke n too small?)'})
            else:
                raise RuntimeError(f'FAD {name} rc={rc}, see {OUT / "fad" / f"{name}.log"}')
        shutil.rmtree(d, ignore_errors=True)
        log(f'FAD {name}: {json.loads(out.read_text())}')


def d_summary(sel, rng):
    pl = pools()
    test = pl['test']
    S = H.S
    sc = {'A3a': S['A3a_MI_lvl30'], 'PQ': S['PQ_lvl30'], 'CE': S['CE_lvl30'], 'CLAP': S['CLAP']}
    out = {'selection': {k: sel[k] for k in ('O_synth', 'r_star', 'O_R', 'dev_dA1_by_r', 'candidates_dev_dA1_r6', 'sha256')},
           'scales': {'A1': S['A1_MI_lvl30'], **sc}, 'bases': {}}
    for base in ('real', 'gen'):
        bp = OUT / 'per_clip' / f'T_{base}_base.tsv'
        bt = tsv_read(bp)
        cells = {}
        for ov, r in test_cells(sel['r_star'])[1:]:
            p = OUT / 'per_clip' / f'{cell_key(base, ov, r)}.tsv'
            if not p.is_file():
                continue
            rows = tsv_read(p)
            d = {c: arr(rows, test, c) - arr(bt, test, c) for c in ('A1_MI', 'A3a_MI', 'PQ', 'CE', 'CLAP')}
            cells[f'{ov}_r{r}'] = {
                'ov': ov, 'r': r, 'dA1': float(d['A1_MI'].mean()), 'dA1_ci': H.boot_ci(d['A1_MI'], rng),
                'dA3a': float(d['A3a_MI'].mean()), 'dPQ': float(d['PQ'].mean()), 'dCE': float(d['CE'].mean()),
                'dCLAP': float(d['CLAP'].mean()),
                'dA3a_ci': H.boot_ci(d['A3a_MI'], rng), 'dPQ_ci': H.boot_ci(d['PQ'], rng),
                'dCE_ci': H.boot_ci(d['CE'], rng), 'dCLAP_ci': H.boot_ci(d['CLAP'], rng),
                'A1_mean': float(arr(rows, test, 'A1_MI').mean()), 'frac_gt5': float((arr(rows, test, 'A1_MI') > 5).mean()),
                '_d': d['A1_MI']}
        verdicts = {}
        rs = sel['r_star']
        for ov in ('osynth', 'oR'):
            k = f'{ov}_r{rs}'
            if k not in cells:
                continue
            c = cells[k]
            ctr = [(f'{o}_r{rs}', cells[f'{o}_r{rs}']) for o in OV_CTRL if f'{o}_r{rs}' in cells]
            cmax_k, cmax = max(ctr, key=lambda kv: kv[1]['dA1'])
            diff = c['_d'] - cmax['_d']
            diff_ci = H.boot_ci(diff, rng)
            T = {x: (c[f'd{x}'] / sc[x]) / (c['dA1'] / S['A1_MI_lvl30']) if c['dA1'] != 0 else None for x in sc}
            if c['dA1_ci'][0] < M_A1:
                v, why = 'no_effect', 'dA1 CI lower bound < m'
            elif diff_ci[0] <= 0:
                v, why = 'nonspecific', f'dA1 minus best control ({cmax_k}) CI lower <= 0'
            elif any(t is not None and t >= 0.5 for t in T.values()):
                v, why = 'general_shift', 'a transfer ratio >= 0.5'
            else:
                v, why = 'exploit', 'all transfer ratios < 0.5'
            verdicts[ov] = {'verdict': v, 'reason': why, 'r': rs, 'dA1': c['dA1'], 'dA1_ci': c['dA1_ci'],
                            'best_control': cmax_k, 'dA1_minus_best_control': float(diff.mean()),
                            'dA1_minus_best_control_ci': diff_ci, 'transfer_T': T}
        for c in cells.values():
            c.pop('_d')
        out['bases'][base] = {'n': len(test), 'base_means': {c: float(arr(bt, test, c).mean()) for c in COLS},
                              'cells': cells, 'verdicts': verdicts}
    out['fad_gen_base_at_rstar'] = {n: json.loads((OUT / 'fad' / f'{n}.json').read_text())
                                    for n in ('base', 'osynth', 'oR') if (OUT / 'fad' / f'{n}.json').is_file()}
    out['main_verdict'] = out['bases']['real']['verdicts']['osynth']['verdict']
    return out


# ── Part F ───────────────────────────────────────────────
F_FEATS = ['per016', 'per_max', 'band200_500', 'frame_stab', 'harm_ratio', 'onset_rate', 'rms_cv', 'centroid', 'partial3']
F_VERDICT_FEATS = ['per_max', 'frame_stab', 'harm_ratio', 'band200_500']


def part_f(att):
    pl = pools()
    ids = pl['dev'] + pl['pool_rest']
    run_set('D_dev_base', pl['dev'], real30, stage='F')
    if pl['pool_rest']:
        run_set('F_rest', pl['pool_rest'], real30, stage='F')
    fp = OUT / 'F_feats.json'
    feats_d = json.loads(fp.read_text()) if fp.is_file() else {}
    todo = [i for i in ids if i not in feats_d]
    for s in range(0, len(todo), 64):
        c = todo[s:s + 64]
        fs = feat_many([real30(i) for i in c], att['main3'])
        for i, f in zip(c, fs):
            f.pop('_contour')
            feats_d[i] = f
        jdump(fp, feats_d)
        progress('F', 'F_feats', len(feats_d), len(ids))
    return ids, feats_d


def part_f_summary(att, rng):
    from scipy.stats import spearmanr
    pl = pools()
    ids = pl['dev'] + pl['pool_rest']
    rows = tsv_read(OUT / 'per_clip' / 'D_dev_base.tsv')
    if pl['pool_rest']:
        rows.update(tsv_read(OUT / 'per_clip' / 'F_rest.tsv'))
    fd = json.loads((OUT / 'F_feats.json').read_text())
    a1 = arr(rows, ids, 'A1_MI')
    a3 = arr(rows, ids, 'A3a_MI')
    X = np.array([[fd[i][k] for k in F_FEATS] for i in ids], float)
    sp = {k: {'A1': float(spearmanr(X[:, j], a1)[0]), 'A3a': float(spearmanr(X[:, j], a3)[0])} for j, k in enumerate(F_FEATS)}
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    perm = np.random.default_rng(SEED).permutation(len(ids))
    folds = np.array_split(perm, 5)
    pred = np.zeros(len(ids))
    for f in folds:
        tr = np.setdiff1d(perm, f)
        A = np.c_[np.ones(len(tr)), Z[tr]]
        beta = np.linalg.lstsq(A, a1[tr], rcond=None)[0]
        pred[f] = np.c_[np.ones(len(f)), Z[f]] @ beta
    r2 = float(1 - ((a1 - pred) ** 2).sum() / ((a1 - a1.mean()) ** 2).sum())
    caps = caps_all()
    order = np.argsort(a1)

    def desc(js):
        return [{'id': ids[j], 'A1_MI': float(a1[j]), 'caption': caps[ids[j]][:200],
                 **{k: float(X[j, q]) for q, k in enumerate(F_FEATS)}} for j in js]
    ref = {k: tsv_read(OUT / 'per_clip' / f'B_{k}.tsv') for k in ('ref_R',)}
    rr = rids()
    rf = feat_many([lvl30(RW(i))[0] for i in range(CFG['n_r'])], att['main3'])
    pct = {}
    for j, k in enumerate(F_FEATS):
        v = np.array([f[k] for f in rf], float)
        pct[k] = {'R_mean': float(v.mean()), 'real_median': float(np.median(X[:, j])),
                  'R_median_percentile_in_real': float(np.median([(X[:, j] < x).mean() for x in v]) * 100)}
    a1R = arr(ref['ref_R'], rr, 'A1_MI')
    pct['A1_MI'] = {'R_mean': float(a1R.mean()), 'real_max': float(a1.max()),
                    'R_median_percentile_in_real': float(np.median([(a1 < x).mean() for x in a1R]) * 100)}
    nhit = sum(sp[k]['A1'] >= 0.10 for k in F_VERDICT_FEATS)
    return {'n': len(ids), 'A1_real_mean': float(a1.mean()), 'A1_real_p90': float(np.percentile(a1, 90)),
            'A1_real_max': float(a1.max()), 'spearman': sp, 'linear_r2_5fold': r2,
            'verdict_features': F_VERDICT_FEATS, 'n_features_spearman_ge_0.10': int(nhit),
            'verdict': 'exaggerated_preference' if nhit >= 3 else 'off_manifold',
            'top20': desc(order[::-1][:20]), 'bottom20': desc(order[:20]), 'R_percentiles': pct}


# ── scorer gate ──────────────────────────────────────────
def scorer_gate():
    p = OUT / 'scorer_gate.json'
    if p.is_file() and json.loads(p.read_text()).get('passed'):
        return True
    import soundfile as sf
    import eval_metrics as EM
    from muq_gen_steer_179_score import muq_hf_cache, standard_hf_cache
    from newmetrics_muqeval import score_muqeval
    sj = json.loads((LISTEN_R / 'scores.json').read_text())
    n = min(8, CFG['n_r'])
    wavs = [r_clip(i, w) for i in range(n) for w in ('control', 'R')]
    rows, _ = SC.score(wavs)
    d183 = []
    for i in range(n):
        for j, w in enumerate(('control', 'R')):
            d183.append({'A1': rows[2 * i + j]['A1_MI'] - sj[i][f'{w}_A1'], 'A3a': rows[2 * i + j]['A3a_MI'] - sj[i][f'{w}_A3a']})
    with muq_hf_cache():
        dh = [abs(rows[2 * i + 1]['A1_MI'] - score_muqeval(wavs[2 * i + 1], SR, variant='A1')['MI']) for i in range(min(3, n))]
    tmp = OUT / 'gate_tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    caps = caps_all()
    gids = pools()['dev'][:3]
    for i in gids:
        sf.write(str(tmp / f'{i}.flac'), real30(i), SR, subtype='PCM_24')
    gw = [real30(i) for i in gids]
    pq, ce = SC.aes_scores(gw)
    with standard_hf_cache():
        ref_aes, failed = EM.score_aes([str(tmp / f'{i}.flac') for i in gids], progress=False)
        ref_clap = EM.score_clap([(i, caps[i]) for i in gids], tmp, model=SC.clap if SC.clap is not None else None, progress=False) \
            if SC.clap is not None else None
    if ref_clap is None:
        SC._clap()
        ref_clap = EM.score_clap([(i, caps[i]) for i in gids], tmp, model=SC.clap, progress=False)
    my_clap = SC.clap_scores(gw, [caps[i] for i in gids])
    da = [abs(pq[k] - ref_aes[str(tmp / f'{i}.flac')]['PQ']) for k, i in enumerate(gids)]
    dc = [abs(my_clap[k] - ref_clap[i]) for k, i in enumerate(gids)]
    rec = {'vs_scores_json_max_abs': {'A1': max(abs(x['A1']) for x in d183), 'A3a': max(abs(x['A3a']) for x in d183)},
           'vs_score_muqeval_max_abs': max(dh), 'vs_score_aes_PQ_max_abs': max(da), 'vs_score_clap_max_abs': max(dc),
           'tol': {'scores_json': 0.05, 'muqeval': 0.02, 'aes': 0.02, 'clap': 0.002}}
    rec['passed'] = bool(not failed and rec['vs_scores_json_max_abs']['A1'] <= 0.05 and rec['vs_scores_json_max_abs']['A3a'] <= 0.05
                         and max(dh) <= 0.02 and max(da) <= 0.02 and max(dc) <= 0.002)
    shutil.rmtree(tmp, ignore_errors=True)
    jdump(p, rec)
    log(f'scorer gate {"PASS" if rec["passed"] else "FAIL"}: {rec}')
    return rec['passed']


# ── summary ──────────────────────────────────────────────
def summarize(att):
    rng = np.random.default_rng(SEED)
    sel = load_selection()
    a2 = part_a2(att)
    s = {'document_kind': KIND, 'created': time.strftime('%FT%T%z'), 'smoke': IS_SMOKE, 'design': DESIGN, 'config': CFG,
         'partA': {'attractor': att, 'a2': a2, 'a3': part_a3()},
         'partB': part_b_summary(rng), 'partC': part_c_summary(), 'partD': d_summary(sel, rng),
         'partF': part_f_summary(att, rng)}
    s['verdicts'] = {'partC': s['partC']['verdict'], 'partD_main': s['partD']['main_verdict'],
                     'partD_oR_real': s['partD']['bases']['real']['verdicts'].get('oR', {}).get('verdict'),
                     'partD_gen_osynth': s['partD']['bases']['gen']['verdicts'].get('osynth', {}).get('verdict'),
                     'partF': s['partF']['verdict']}
    tp = OUT / 'timing.json'
    s['timing'] = json.loads(tp.read_text()) if tp.is_file() else {}
    jdump(OUT / 'summary.json', s)
    log(f'wrote summary.json verdicts {s["verdicts"]}')
    return s


# ── listening pack ───────────────────────────────────────
def listening_pack(att, sel):
    import soundfile as sf
    d = DELIV
    if d.exists():
        shutil.rmtree(d)
    (d / 'wav').mkdir(parents=True)
    items = []

    def add(group, name, w, score):
        fn = f'{len(items):03d}_{name.replace("/", "_")}.wav'
        sf.write(str(d / 'wav' / fn), np.clip(w, -1, 1), SR, subtype='PCM_16')
        items.append({'group': group, 'name': name, 'file': fn, **score})

    Cr = tsv_read(OUT / 'per_clip' / 'C_all.tsv')
    tonal, ctrl = c_names()
    for k in sorted(tonal, key=lambda k: -Cr[k]['A1_MI'])[:6] + ctrl:
        add('Part C: top 6 tonal + 6 controls', k, build_C(k), Cr[k])
    b = {k: tsv_read(OUT / 'per_clip' / f'B_{k}.tsv') for k in b_versions()}
    ior = sel['O_R']['idx']
    n_r = CFG['n_r']
    for i in (ior, (ior + 7) % n_r):
        for k in ('ref_R', 'ref_ctrl', 'shuf016', 'reverse', 'randphase', 'env_pink', 'env_ctrl', 'eq_ctrl', 'lp500', 'hp500',
                  'tile_period', 'flat20ms', 'frac25', 'transpose+0'):
            add(f'Part B: R clip {i:02d}', f'{k}', build_B(k, i), b[k][f'{i:02d}'])
    pl = pools()
    test = pl['test'][:6]
    for i in test:
        for base, getter in (('real', real30),):
            bt = tsv_read(OUT / 'per_clip' / f'T_{base}_base.tsv')
            add(f'Part D: test clip {i}', f'{i}__base', getter(i), bt[i])
            for ov, r in test_cells(sel['r_star'])[1:]:
                if ov in ('osynth', 'oR') or r == sel['r_star']:
                    k = cell_key(base, ov, r)
                    o = overlay(ov, sel) if ov != 'music' else real30(pl['dev'][pl['test'].index(i) % len(pl['dev'])])
                    add(f'Part D: test clip {i}', f'{i}__{ov}_r{r:+d}', mix(getter(i), o, r), tsv_read(OUT / 'per_clip' / f'{k}.tsv')[i])
    rows = []
    cur = None
    for it in items:
        if it['group'] != cur:
            cur = it['group']
            rows.append(f'<tr><th colspan="7" style="text-align:left;padding-top:14px">{cur}</th></tr>')
        rows.append(f'<tr><td>{it["name"]}</td><td><audio controls preload="none" src="wav/{it["file"]}"></audio></td>'
                    f'<td>{it["A1_MI"]:.2f}</td><td>{it["A3a_MI"]:.2f}</td><td>{it["PQ"]:.2f}</td><td>{it["CE"]:.2f}</td>'
                    f'<td>{"" if math.isnan(it["CLAP"]) else f"{it[chr(67) + chr(76) + chr(65) + chr(80)]:.3f}"}</td></tr>')
    html = ('<!doctype html><meta charset="utf-8"><title>188 listening pack</title>'
            '<style>body{font:14px system-ui;margin:16px;max-width:1000px}td,th{padding:3px 8px;border-bottom:1px solid #8884}</style>'
            f'<h2>188: MuQ A1 attractor listening pack{" (SMOKE)" if IS_SMOKE else ""}</h2>'
            f'<p>O_synth = <b>{sel["O_synth"]}</b>, r* = {sel["r_star"]} LU, O_R = R clip {sel["O_R"]["idx"]:02d}. '
            'Scores: A1 MI, A3a MI, AES PQ, CE (all -30 LUFS), CLAP where a caption exists. Listening aid only, no human ratings.</p>'
            '<table><tr><th>stimulus</th><th></th><th>A1</th><th>A3a</th><th>PQ</th><th>CE</th><th>CLAP</th></tr>'
            + ''.join(rows) + '</table>')
    (d / 'listen.html').write_text(html)
    jdump(d / 'items.json', items)
    log(f'listening pack: {len(items)} wavs -> {d}')


# ── modes ────────────────────────────────────────────────
def free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def preflight():
    errs = []
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(H.EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    for p in (H.PARTB / 'part_b_selection.json', H.TTM115, H.MC_TSV, Path(H.FAD_REF), REF_DIR, LISTEN_R / 'scores.json',
              Path(TORCHRUN), REPO / 'scripts/training/muq_reward_ft_183.py', FT / 'selection.json'):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    try:
        n = len(json.loads((LISTEN_R / 'scores.json').read_text()))
        if n < CFG['n_r']:
            errs.append(f'listen_R183 scores.json has {n} < {CFG["n_r"]} clips')
        for i in range(CFG['n_r']):
            for w in ('R', 'control'):
                if not (LISTEN_R / 'wav' / f'{i:02d}_{w}.wav').is_file():
                    errs.append(f'missing listen_R183 wav {i:02d}_{w}')
                    break
        H.P = H.Paths(OUT / 'h183')
        a = H.cell_spec('C_cfg0_s14159265')
        H.eval_cmd(a, Path('/dev/null'))
        if not Path(a['model_path']).is_file():
            errs.append(f'control checkpoint missing {a["model_path"]}')
        pl = pools()
        if len(pl['test']) < CFG['n_test']:
            errs.append(f'only {len(pl["test"])} test ids with reference audio, expected {CFG["n_test"]}')
        if len(pl['dev'] + pl['pool_rest']) < CFG['n_f']:
            errs.append(f'only {pl["n_cand"]} dev/pool candidates, expected >= {CFG["n_f"]}')
        miss = [k for k, v in regen_arm_paths().items() if v is not None and not Path(v).is_file()]
        if miss:
            log(f'PREFLIGHT note: {len(miss)} regen checkpoints missing (recorded as missing arms): {miss}')
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if errs:
        return 1
    log(f'PREFLIGHT OK (smoke={IS_SMOKE})')
    if not (OUT / 'summary.json').is_file():
        f = free_mib()
        need = 10 * 1024 if IS_SMOKE else NEED_FREE_MIB
        if f < need:
            log(f'PREFLIGHT WAIT GPU free {f} MiB < {need} MiB')
            return 75
    return 0


VERDICT_SETS = {'partC': {'simple_recipe_reproduces', 'simple_recipe_partial', 'not_reproduced'},
                'partD_main': {'no_effect', 'nonspecific', 'general_shift', 'exploit'},
                'partF': {'exaggerated_preference', 'off_manifold'}}


def validate():
    errs = []
    sp = OUT / 'summary.json'
    if not sp.is_file():
        log('VALIDATE FAIL: summary.json missing')
        return 1
    s = json.loads(sp.read_text())
    if s.get('document_kind') != KIND:
        errs.append('summary document_kind')
    g = OUT / 'scorer_gate.json'
    if not g.is_file() or not json.loads(g.read_text()).get('passed'):
        errs.append('scorer gate missing or failed')
    try:
        sel = load_selection()
        t0 = sel['created_epoch']
        for p in (OUT / 'per_clip').glob('T_*.tsv'):
            if p.stat().st_mtime < t0:
                errs.append(f'{p.name} older than selection.json (test scored before selection frozen)')
    except SystemExit as e:
        errs.append(str(e))
    v = s.get('verdicts', {})
    for k, allowed in VERDICT_SETS.items():
        if v.get(k) not in allowed:
            errs.append(f'verdict {k}={v.get(k)!r}')
    pb = s.get('partB', {})
    if pb.get('n_versions') != 52 or not all(math.isfinite(t['rho']) for t in pb.get('table', [])):
        errs.append('partB table incomplete')
    pc = s.get('partC', {})
    if pc.get('n_stimuli') != 190 or len(pc.get('top10', [])) != 10 or not pc.get('main_effects'):
        errs.append('partC incomplete')
    pd = s.get('partD', {})
    for base in ('real', 'gen'):
        b = pd.get('bases', {}).get(base, {})
        if not b.get('cells') or 'osynth' not in b.get('verdicts', {}) or 'oR' not in b.get('verdicts', {}):
            errs.append(f'partD {base} incomplete')
            continue
        for c in b['cells'].values():
            if not all(k in c for k in ('dA1', 'dA1_ci', 'dA3a', 'dPQ', 'dCE', 'dCLAP', 'frac_gt5')):
                errs.append(f'partD {base} cell fields')
                break
    pf = s.get('partF', {})
    if not pf.get('spearman') or 'linear_r2_5fold' not in pf or len(pf.get('top20', [])) < min(20, pf.get('n', 0)) \
            or not pf.get('R_percentiles'):
        errs.append('partF incomplete')
    a = s.get('partA', {})
    if not a.get('attractor', {}).get('main3') or not a.get('a2', {}).get('arms') or not a.get('a3', {}).get('R'):
        errs.append('partA incomplete')
    for e in errs:
        log(f'VALIDATE FAIL: {e}')
    if errs:
        return 1
    log(f'VALIDATE OK verdicts={v}')
    return 0


def run_all(smoke):
    global POOL
    POOL = ThreadPoolExecutor(NTHREAD)
    OUT.mkdir(parents=True, exist_ok=True)
    H.load_scales()
    H.P = H.Paths(OUT / 'h183')
    progress('start', '', 0, 0)
    att = None
    # A-1 needs only the R wavs; ATT is also what Part C / D synthesise from
    att = attractor()
    ATT.update(att)
    if not feat_many([RW(0)], att['main3']):   # forks the feature pool before CUDA is touched
        raise SystemExit('feature pool failed')
    regen_all()                                 # GPU worker first, as a separate process
    if not scorer_gate():
        log('scorer gate failed: stop (design: stop on harness mismatch)')
        return 1
    part_b()
    part_c()
    sel = d_dev()
    part_a2(att)
    part_a3()
    d_test(sel)
    d_fad(sel)
    part_f(att)
    summarize(att)
    listening_pack(att, sel)
    progress('done', '', 0, 0)
    return validate()


def main():
    global OUT, DELIV, IS_SMOKE, CFG
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--regen-worker', action='store_true')
    ap.add_argument('--fad-cell')
    ap.add_argument('--root', help=argparse.SUPPRESS)
    ap.add_argument('--n-regen', type=int)
    ap.add_argument('--regen-arms')
    ap.add_argument('--n-test', type=int, help='reduction step 1 (design section 4)')
    ap.add_argument('--n-f', type=int, help='reduction step 2')
    ap.add_argument('--gen-only-rstar', action='store_true', help='reduction step 3')
    a = ap.parse_args()
    os.chdir(REPO)
    if a.smoke:
        IS_SMOKE = True
        CFG = dict(SMOKE)
        OUT = H.EVAL_ROOT / 'muq_attractor_probe_188_smoke'
        DELIV = REPO / 'deliverables/listen_attractor_188_smoke'
    if a.root:
        OUT = Path(a.root)
    for k, v in (('n_test', a.n_test), ('n_f', a.n_f)):
        if v:
            CFG[k] = v
    if a.gen_only_rstar:
        CFG['gen_only_rstar'] = True
    if a.regen_worker:
        return regen_worker(OUT, a.n_regen, a.regen_arms.split(','))
    if a.fad_cell:
        return fad_cell(a.fad_cell)
    if a.preflight:
        return preflight()
    if a.validate_only:
        return validate()
    if a.summary_only:
        global POOL
        POOL = ThreadPoolExecutor(NTHREAD)
        H.load_scales()
        H.P = H.Paths(OUT / 'h183')
        att = json.loads((OUT / 'attractor.json').read_text())
        ATT.update(att)
        feat_many([RW(0)], att['main3'])
        summarize(att)
        listening_pack(att, load_selection())
        return validate()
    rc = preflight()
    if rc:
        return rc
    return run_all(a.smoke)


if __name__ == '__main__':
    sys.exit(main())
