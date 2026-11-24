#!/usr/bin/env python
"""200: Arena 後果實驗（eval-only）。設計：docs/experiments/paper_closing_evidence_200_202_20261125.md §2、§4（預登錄；本檔只實作）。

問題：Music Arena 3,998 段、15 個系統、1,412 場有人類選擇的對戰。把八種處理（lp6k／ps_m2／ts_090／eq250_p10／osynth_r18／
side_snr10／rb_id／gain_m6）套在 F0（中央 30 秒、單聲道、原取樣率）上，用六個評分器重評；看「只處理人類判輸的一段」會不會讓
評分器與人類的一致率下降（L），只處理判贏的一段（W），兩段都處理（B）。

  python arena_consequence_200.py --smoke        # 約 40 段、>=20 場完整對戰、全處理 × 全評分器 → ~/eval_output_nvme/arena_consequence_200_smoke/
  python arena_consequence_200.py                # 全量，續跑（每個 處理×shard 一份 per_clip/*.tsv，tmp 寫完 rename，已存在就跳過）
  python arena_consequence_200.py --analyze      # CPU：一致率、flip、rank shift、判定 → docs/experiments/results/arena_consequence_200_summary.json
  python arena_consequence_200.py --preflight / --validate-only

不留任何音檔（刺激在記憶體中建好、評完即丟；AES 以記憶體 tensor 送入）。F0 的 A1 MI 必須與 r2 score 逐段一致（中位數 |差| <= 1e-3），否則停。
"""
import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent.parent
EVAL_ROOT = Path.home() / 'eval_output_nvme'
OUT = EVAL_ROOT / 'arena_consequence_200'
SUMMARY = REPO / 'docs/experiments/results/arena_consequence_200_summary.json'
KIND = 'arena_consequence_200_summary_v1'
IS_SMOKE = False

R2 = Path('/mnt/seagate/benchmarks_r2')
PICK = R2 / 'meta/music_arena_pick.json'
MANIFEST = R2 / 'manifests/arena.tsv'
INDEX = R2 / 'derived/arena/index.tsv'
DER = R2 / 'derived/arena'
R2_A1 = R2 / 'scores/arena/muqeval_A1__native_raw.tsv'
OVERLAY_SEL = EVAL_ROOT / 'muq_attractor_probe_188/selection.json'
OVERLAY_ATT = EVAL_ROOT / 'muq_attractor_probe_188/attractor.json'

SEED = 20261125
NBOOT = 5000
SHARD = 250
MIN_FREE = 20 * 1024**3
N30 = 30
TREATS = ['lp6k', 'ps_m2', 'ts_090', 'eq250_p10', 'osynth_r18', 'side_snr10', 'rb_id', 'gain_m6']
CONTROLS = ['rb_id', 'gain_m6']
CONDS = ['F0'] + TREATS
NO_RELOCK = ('gain_m6', 'side_snr10')
SCORERS = ['A1_MI', 'A3a_MI', 'PQ', 'CE', 'CMI_q', 'PAM']
COLS = ['id', 'lufs_f0', 'lufs_out', 'n_samples', 'sr'] + SCORERS
MARGIN = 0.03
SIDE_MARGIN_S = 0.25      # side_snr10: noise only outside (central 10 s +- 0.25 s), so that the MuQ window stays untouched after resampling


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


# ═════════ data ═════════
def read_index():
    with open(INDEX, newline='') as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r['loud'] == 'raw' and r['variant'] == 'native' and not r['invalid']]
    return {r['unit_id'].split('|')[1]: dict(path=DER / r['files'], sr=int(r['sr'])) for r in rows}


def read_manifest():
    csv.field_size_limit(10**9)
    with open(MANIFEST, newline='') as f:
        return {r['item_id']: dict(caption=r['caption'], system=r['group']) for r in csv.DictReader(f, delimiter='\t')}


def read_battles(clips):
    pick = json.loads(PICK.read_text())
    out = []
    for b in pick['battles']:
        if b['audio_a'] in clips and b['audio_b'] in clips:
            out.append(dict(a=b['audio_a'], b=b['audio_b'], pref=b['preference'], uuid=b['battle_uuid']))
    return out


def all_ids():
    return sorted(read_index())


def smoke_ids():
    """~40 clips = 20+ complete battles, covering 32/44.1/48 kHz; deterministic."""
    idx = read_index()
    bat = [b for b in read_battles(idx) if b['pref'] in ('A', 'B')]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(bat))
    pick, ids, srs = [], [], set()
    for k in order:
        b = bat[k]
        if len(pick) >= 20 and len(srs) == 3:
            break
        new = {idx[b['a']]['sr'], idx[b['b']]['sr']} - srs
        if len(pick) < 20 or new:
            pick.append(b)
            ids += [b['a'], b['b']]
            srs |= {idx[b['a']]['sr'], idx[b['b']]['sr']}
    return sorted(set(ids))


def shards(ids):
    return [ids[i:i + SHARD] for i in range(0, len(ids), SHARD)]


def load_center30(path):
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        sr, n = f.samplerate, f.frames
        L = N30 * sr
        if n > L:
            s = (n - L) // 2
            f.seek(s)
            y = f.read(L, dtype='float32', always_2d=True)
        else:
            y = f.read(dtype='float32', always_2d=True)
    return y.mean(axis=1).astype(np.float32), sr


# ═════════ treatments (native sr; every one starts from F0) ═════════
def lufs_of(x, sr):
    import pyloudnorm
    return float(pyloudnorm.Meter(sr).integrated_loudness(np.asarray(x, np.float64)[:, None]))


def relock(y, target, sr, cap=0.999):
    y = np.asarray(y, np.float64)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    cur = lufs_of(y, sr)
    if not math.isfinite(target) or not math.isfinite(cur) or peak == 0.0:
        g = 0.0
    else:
        g = target - cur
        if peak * 10 ** (g / 20) > cap:
            g = 20 * math.log10(cap / peak)
    return np.clip(y * 10 ** (g / 20), -1.0, 1.0).astype(np.float32)


def rubberband(x, sr, tempo=1.0, pitch=1.0):
    x = np.ascontiguousarray(x, np.float32)
    q = ':pitchq=quality' if pitch != 1.0 else ''
    p = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'f32le', '-ar', str(sr), '-ac', '1', '-i', '-',
                        '-af', f'rubberband=tempo={tempo:.6f}:pitch={pitch:.8f}{q}', '-f', 'f32le', '-'],
                       input=x.tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).astype(np.float64)


def fit_len(y, n):
    y = np.asarray(y)
    return y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), y.dtype)])


def eq_sos(f0, gain_db, sr, q=math.sqrt(2)):
    from scipy.signal import tf2sos
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / sr
    al = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    return tf2sos(np.array([1 + al * A, -2 * cw, 1 - al * A]), np.array([1 + al / A, -2 * cw, 1 - al / A]))


_OV = {}


def overlay_loop(sr):
    """188 frozen O_synth (16 kHz, 9.975 s, -30 LUFS) -> native sr (soxr HQ)."""
    if sr not in _OV:
        import soxr
        if 'w16' not in _OV:
            import muq_attractor_probe_188 as P188
            sel = json.loads(OVERLAY_SEL.read_text())
            P188.ATT.update(json.loads(OVERLAY_ATT.read_text()))
            _OV['w16'] = np.asarray(P188.build_C(sel['O_synth']), np.float32)
            _OV['name'] = sel['O_synth']
        _OV[sr] = soxr.resample(_OV['w16'], 16000, sr, quality='HQ').astype(np.float32)
    return _OV[sr]


def build(cond, x, sr, item, lufs0):
    """-> float32 stimulus for `cond`; x = F0 (float32 mono)."""
    n = len(x)
    xd = np.asarray(x, np.float64)
    if cond == 'F0':
        return x
    if cond == 'gain_m6':
        return (xd * 10 ** (-6 / 20)).astype(np.float32)
    if cond == 'side_snr10':
        y = xd.copy()
        hole = int((5.0 + SIDE_MARGIN_S) * sr)
        c = n // 2
        rng = np.random.default_rng([SEED, zlib.crc32(item.encode()), zlib.crc32(cond.encode())])
        for a, b in ((0, max(0, c - hole)), (min(n, c + hole), n)):
            if b > a:
                pw = float(np.mean(y[a:b] ** 2))
                y[a:b] += rng.standard_normal(b - a) * math.sqrt(pw / 10)
        return y.astype(np.float32)
    if cond == 'lp6k':
        from muq_lowpass_probe_20261028 import lp
        y = lp(xd, sr, 6000.0)
    elif cond == 'ps_m2':
        y = fit_len(rubberband(x, sr, 1.0, 2 ** (-2 / 12)), n)
    elif cond == 'ts_090':
        y = rubberband(x, sr, 0.9, 1.0)
    elif cond == 'rb_id':
        y = fit_len(rubberband(x, sr, 1.0, 1.0), n)
    elif cond == 'eq250_p10':
        from scipy.signal import sosfilt
        y = sosfilt(eq_sos(250.0, 10.0, sr), xd)
    elif cond == 'osynth_r18':
        o = np.resize(overlay_loop(sr), n).astype(np.float64)
        ol = lufs_of(o, sr)
        if math.isfinite(lufs0) and math.isfinite(ol):
            y = xd + o * 10 ** ((lufs0 - 18.0 - ol) / 20)
        else:
            y = xd
    else:
        raise KeyError(cond)
    return relock(y, lufs0, sr)


# ═════════ scorers ═════════
class Scorer:
    def __init__(self):
        self.ready = False
        self.t = {}

    def load(self):
        import torch
        from muq_gen_steer_179_score import muq_hf_cache, standard_hf_cache
        import newmetrics_muqeval as NMQ
        import newmetrics_pam
        import newmetrics_cmirm
        self.torch = torch
        with muq_hf_cache():
            NMQ._load('A1', 'cuda')
            NMQ._load('A3a', 'cuda')
            newmetrics_pam._load('cuda')
            pip_muq = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'muq' or k.startswith('muq.')}
            try:
                newmetrics_cmirm._load('cuda', False)
            finally:
                for k in [k for k in sys.modules if k == 'muq' or k.startswith('muq.')]:
                    del sys.modules[k]
                sys.modules.update(pip_muq)
        with standard_hf_cache():
            import audiobox_aesthetics.infer as aes_infer
            import eval_metrics as EM
            aes_infer.read_wav = EM._read_wav_sf
            self.aes = aes_infer.AesPredictor(checkpoint_pth=None, batch_size=32)
        self.NMQ, self.PAM, self.CMI = NMQ, newmetrics_pam, newmetrics_cmirm
        self.ready = True

    def clock(self, name, fn):
        t0 = time.time()
        r = fn()
        self.torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + time.time() - t0
        return r

    def score(self, wavs, srs, caps):
        """list of float32 mono at native sr -> list of dict(SCORERS)."""
        if not self.ready:
            self.load()
        torch = self.torch
        n = len(wavs)
        cols = {k: [] for k in SCORERS}

        def muq():
            for w, sr in zip(wavs, srs):
                a = self.NMQ.score_muqeval(w, sr, 'A1')
                b = self.NMQ.score_muqeval(w, sr, 'A3a')
                cols['A1_MI'].append(a['MI'])
                cols['A3a_MI'].append(b['MI'])
        self.clock('muq', muq)

        def aes():
            for s in range(0, n, 32):
                items = [{'path': torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': sr}
                         for w, sr in zip(wavs[s:s + 32], srs[s:s + 32])]
                try:
                    rows = self.aes.forward(items)
                except Exception:  # noqa: BLE001
                    rows = [self.aes.forward([it])[0] for it in items]
                for r in rows:
                    cols['PQ'].append(float(r['PQ']))
                    cols['CE'].append(float(r['CE']))
        self.clock('aes', aes)
        self.clock('pam', lambda: [cols['PAM'].append(float(self.PAM.score_pam(w, sr=sr)['pam'])) for w, sr in zip(wavs, srs)])
        self.clock('cmi', lambda: [cols['CMI_q'].append(float(self.CMI.score_cmirm(w, sr=sr, prompt_mode='caption', caption=c)['quality']))
                                   for w, sr, c in zip(wavs, srs, caps)])
        rows = [{k: cols[k][i] for k in SCORERS} for i in range(n)]
        for r in rows:
            if not all(math.isfinite(v) for v in r.values()):
                raise RuntimeError(f'non-finite score {r}')
        return rows


SC = Scorer()


# ═════════ run ═════════
def tsv_path(cond, k):
    return OUT / 'per_clip' / f'{cond}_{k:02d}.tsv'


def tsv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        f.write('\t'.join(COLS) + '\n')
        for r in rows:
            f.write('\t'.join(r[c] if isinstance(r[c], str) else repr(float(r[c])) for c in COLS) + '\n')
    os.replace(tmp, path)


def tsv_read(path):
    with open(path, newline='') as f:
        return {r['id']: {k: float(v) for k, v in r.items() if k != 'id'} for r in csv.DictReader(f, delimiter='\t')}


def cell_ok(cond, k, ids):
    p = tsv_path(cond, k)
    if not p.is_file():
        return False
    try:
        r = tsv_read(p)
    except Exception:  # noqa: BLE001
        return False
    return set(r) == set(ids) and all(math.isfinite(v) for x in r.values() for v in x.values())


def progress(**kw):
    jdump(OUT / 'progress.json', {**kw, 'updated': time.strftime('%FT%T%z'), 'smoke': IS_SMOKE})


def add_timing(cond, k, n, wall):
    p = OUT / 'timing.json'
    t = json.loads(p.read_text()) if p.is_file() else {'cells': {}}
    t['cells'][f'{cond}_{k:02d}'] = {'n': n, 'wall_s': wall, 'per_clip_s': wall / max(n, 1)}
    t['components_s'] = {k2: v for k2, v in SC.t.items()}
    t['total_wall_s'] = sum(c['wall_s'] for c in t['cells'].values())
    t['total_n'] = sum(c['n'] for c in t['cells'].values())
    jdump(p, t)


def identity_gate(f0_rows):
    """F0 A1 MI vs r2 whole-file score (centre 10 s of central 30 s == centre 10 s of the file)."""
    ref = {}
    with open(R2_A1, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            ref[r['unit_id'].split('|')[1]] = float(r['muqeval_A1_MI'])
    d = np.array([abs(v['A1_MI'] - ref[i]) for i, v in f0_rows.items() if i in ref])
    g = {'n': int(len(d)), 'median_abs_diff': float(np.median(d)), 'max_abs_diff': float(d.max()),
         'p95_abs_diff': float(np.percentile(d, 95)), 'ok': bool(np.median(d) <= 1e-3)}
    jdump(OUT / 'gate.json', g)
    return g


def run_shard(k, ids, idx, man):
    todo = [c for c in CONDS if not cell_ok(c, k, ids)]
    if not todo:
        return
    t0 = time.time()
    data = []
    for i in ids:
        x, sr = load_center30(idx[i]['path'])
        if sr != idx[i]['sr']:
            raise RuntimeError(f'{i}: sr {sr} != index {idx[i]["sr"]}')
        data.append((i, x, sr, lufs_of(x, sr)))
    log(f'shard {k}: loaded {len(ids)} clips in {time.time() - t0:.1f}s; todo {todo}')
    caps = [man[i]['caption'] for i in ids]
    for c in todo:
        t1 = time.time()
        SC.t.clear()
        bt = time.time()
        wavs = [build(c, x, sr, i, l0) for i, x, sr, l0 in data]
        build_s = time.time() - bt
        for w in wavs:
            if not np.all(np.isfinite(w)):
                raise RuntimeError(f'{c}: non-finite stimulus')
        rows = SC.score(wavs, [sr for _, _, sr, _ in data], caps)
        out = []
        for (i, x, sr, l0), w, r in zip(data, wavs, rows):
            lo = lufs_of(w, sr) if c != 'F0' else l0
            out.append({'id': i, 'lufs_f0': l0, 'lufs_out': lo, 'n_samples': len(w), 'sr': sr, **r})
        tsv_write(tsv_path(c, k), out)
        wall = time.time() - t1
        SC.t['build'] = build_s
        add_timing(c, k, len(ids), wall)
        log(f'{c}_{k:02d}: {len(ids)} clips {wall:.1f}s ({wall / len(ids):.2f}s/clip; build {build_s:.1f}s {dict((a, round(b, 1)) for a, b in SC.t.items())})')
        progress(stage='run', cell=f'{c}_{k:02d}', done=len(list((OUT / 'per_clip').glob('*.tsv'))))
        if c == 'F0':
            g = identity_gate(tsv_read(tsv_path('F0', k)))
            log(f'identity gate: {g}')
            if not g['ok']:
                raise SystemExit(f'IDENTITY GATE FAILED: median |dA1| {g["median_abs_diff"]:.2e} > 1e-3 (see {OUT / "gate.json"})')
    del data


def run_all(smoke):
    ids = smoke_ids() if smoke else all_ids()
    idx, man = read_index(), read_manifest()
    jdump(OUT / 'ids.json', {'n': len(ids), 'ids': ids, 'shard': SHARD})
    sh = [ids] if smoke else shards(ids)
    log(f'{len(ids)} clips in {len(sh)} shard(s); smoke={smoke}')
    for k, s in enumerate(sh):
        run_shard(k, s, idx, man)
    progress(stage='done', done=len(sh) * len(CONDS))


# ═════════ analysis ═════════
def load_all(ids_shards):
    v = {c: {} for c in CONDS}
    for c in CONDS:
        for k, s in enumerate(ids_shards):
            v[c].update(tsv_read(tsv_path(c, k)))
    return v


def agree(d, h):
    return np.where(d == 0, 0.5, (np.sign(d) == h).astype(float))


def analyze(smoke):
    ids = json.loads((OUT / 'ids.json').read_text())['ids']
    sh = [ids] if smoke else shards(ids)
    v = load_all(sh)
    man = read_manifest()
    idx = read_index()
    bats = [b for b in read_battles(idx) if b['pref'] in ('A', 'B') and b['a'] in v['F0'] and b['b'] in v['F0']]
    allb = [b for b in read_battles(idx)]
    h = np.array([1.0 if b['pref'] == 'A' else -1.0 for b in bats])
    ia = [b['a'] for b in bats]
    ib = [b['b'] for b in bats]
    los = [b['b'] if b['pref'] == 'A' else b['a'] for b in bats]
    is_a_loser = np.array([b['pref'] == 'B' for b in bats])      # loser is A
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, max(len(bats), 1), (NBOOT, len(bats)))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    res = {'kind': KIND, 'smoke': smoke, 'created': time.strftime('%FT%T%z'), 'n_clips': len(ids), 'n_battles_all': len(allb),
           'n_battles_decided': len(bats), 'seed': SEED, 'nboot': NBOOT, 'margin': MARGIN, 'treatments': TREATS, 'scorers': SCORERS}
    gp = OUT / 'gate.json'
    res['identity_gate'] = json.loads(gp.read_text()) if gp.is_file() else None
    # level / side checks
    lv = {}
    for c in TREATS:
        if c in NO_RELOCK:
            continue
        e = np.array([v[c][i]['lufs_out'] - v[c][i]['lufs_f0'] for i in ids if math.isfinite(v[c][i]['lufs_f0'])])
        lv[c] = {'mean_err_lu': float(e.mean()), 'max_abs_err_lu': float(np.abs(e).max()), 'n_over_0.1': int((np.abs(e) > 0.1).sum())}
    res['lufs_error'] = lv
    res['side_snr10_vs_F0_maxabs'] = {s: float(max(abs(v['side_snr10'][i][s] - v['F0'][i][s]) for i in ids)) for s in ('A1_MI', 'A3a_MI')}
    sysof = {i: man[i]['system'] for i in ids}
    systems = sorted(set(sysof.values()))
    res['cells'] = {}
    for s in SCORERS:
        f0 = {i: v['F0'][i][s] for i in ids}
        d0 = np.array([f0[a] - f0[b] for a, b in zip(ia, ib)])
        ag0 = agree(d0, h)
        base_m = {sy: float(np.mean([f0[i] for i in ids if sysof[i] == sy])) for sy in systems}
        base_rank = {sy: 1 + sum(base_m[o] > base_m[sy] for o in systems) for sy in systems}
        for c in TREATS:
            t = {i: v[c][i][s] for i in ids}
            # Δ overall and per system (clip bootstrap)
            dl = np.array([t[i] - f0[i] for i in ids])
            ci_ids = rng.integers(0, len(ids), (2000, len(ids)))
            cell = {'mean_delta': float(dl.mean()), 'mean_delta_ci': ci(dl[ci_ids].mean(1)), 'per_system': {}}
            for sy in systems:
                m = np.array([sysof[i] == sy for i in ids])
                x = dl[m]
                bb = rng.integers(0, len(x), (2000, len(x)))
                cell['per_system'][sy] = {'n': int(m.sum()), 'mean': float(x.mean()), 'ci': ci(x[bb].mean(1))}
            # battle-level agreement
            ta, tb = np.array([t[a] for a in ia]), np.array([t[b] for b in ib])
            fa, fb = np.array([f0[a] for a in ia]), np.array([f0[b] for b in ib])
            dL = np.where(is_a_loser, ta - fb, fa - tb)       # treat only the human loser
            dW = np.where(is_a_loser, fa - tb, ta - fb)       # treat only the human winner
            dB = ta - tb
            agL, agW, agB = agree(dL, h), agree(dW, h), agree(dB, h)
            boot = lambda a: a[bidx].mean(1) if len(a) else np.zeros(NBOOT)
            b0, bL, bW, bB = boot(ag0), boot(agL), boot(agW), boot(agB)
            ag = {'F0': float(ag0.mean()) if len(ag0) else None, 'L': float(agL.mean()), 'W': float(agW.mean()), 'B': float(agB.mean())}
            diff = {'L-F0': (bL - b0), 'W-F0': (bW - b0), 'B-F0': (bB - b0), 'W-L': (bW - bL)}
            cell['agreement'] = ag
            cell['agreement_ci'] = {'F0': ci(b0), 'L': ci(bL), 'W': ci(bW), 'B': ci(bB)}
            cell['diff'] = {k: {'est': float(np.mean(x)), 'ci': ci(x)} for k, x in diff.items()}
            # flip: agreed at F0 (strict) -> picks the loser after L
            strict0 = (d0 != 0) & (np.sign(d0) == h)
            flipped = (dL != 0) & (np.sign(dL) == -h)
            cell['flip_rate'] = float((strict0 & flipped).sum() / max(strict0.sum(), 1))
            fb_ = (strict0 & flipped)[bidx].sum(1) / np.maximum(strict0[bidx].sum(1), 1)
            cell['flip_rate_ci'] = ci(fb_)
            # rank shift of system sy when only its clips are treated
            shifts = {}
            for sy in systems:
                nm = float(np.mean([t[i] for i in ids if sysof[i] == sy]))
                nr = 1 + sum((base_m[o] > nm) for o in systems if o != sy)
                shifts[sy] = {'rank0': base_rank[sy], 'rank_t': int(nr), 'moved_up': int(base_rank[sy] - nr)}
            cell['rank_shift'] = shifts
            cell['rank_max_abs'] = int(max(abs(x['moved_up']) for x in shifts.values()))
            cell['rank_n_up'] = int(sum(x['moved_up'] >= 1 for x in shifts.values()))
            cell['rank_n_down'] = int(sum(x['moved_up'] <= -1 for x in shifts.values()))
            # verdict (pre-registered)
            dl_, wl = cell['diff']['L-F0'], cell['diff']['W-L']
            if dl_['est'] <= -MARGIN and dl_['ci'][1] < 0 and wl['ci'][0] > 0:
                verdict = '有後果'
            elif dl_['ci'][0] > -MARGIN and dl_['ci'][1] < MARGIN:
                verdict = '無後果'
            else:
                verdict = '未定'
            cell['verdict'] = verdict
            if c in CONTROLS and verdict == '有後果':
                cell['control_flag'] = '對照處理被判有後果，疑似評分器噪音或流程問題'
            res['cells'][f'{c}|{s}'] = cell
    res['verdict_table'] = {c: {s: res['cells'][f'{c}|{s}']['verdict'] for s in SCORERS} for c in TREATS}
    res['mean_delta_table'] = {c: {s: res['cells'][f'{c}|{s}']['mean_delta'] for s in SCORERS} for c in TREATS}
    res['summary_sha_of_inputs'] = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
    path = (OUT / 'summary_smoke.json') if smoke else SUMMARY
    jdump(path, res)
    log(f'wrote {path}')
    return res


# ═════════ validate / preflight ═════════
def validate(smoke):
    errs = []
    try:
        ids_p = OUT / 'ids.json'
        if not ids_p.is_file():
            errs.append('missing ids.json')
        else:
            ids = json.loads(ids_p.read_text())['ids']
            sh = [ids] if smoke else shards(ids)
            for c in CONDS:
                for k, s in enumerate(sh):
                    if not cell_ok(c, k, s):
                        errs.append(f'bad/missing cell {c}_{k:02d}')
        g = OUT / 'gate.json'
        if not g.is_file() or not json.loads(g.read_text()).get('ok'):
            errs.append('identity gate missing or failed')
        sp = (OUT / 'summary_smoke.json') if smoke else SUMMARY
        if not sp.is_file():
            errs.append(f'missing {sp}')
        else:
            s = json.loads(sp.read_text())
            if s.get('kind') != KIND:
                errs.append('summary wrong kind')
            if 'NaN' in sp.read_text() or 'Infinity' in sp.read_text():
                errs.append('non-finite numbers in summary')
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs[:30]:
        log(f'VALIDATE FAIL {e}')
    if errs:
        return 1
    log('VALIDATE OK')
    return 0


def gpu_free_mib():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        return int(r.stdout.split()[0])
    except Exception:  # noqa: BLE001
        return 0


def mem_available_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1024**2
    return 0.0


def preflight(smoke):
    errs = []
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    for p in (PICK, MANIFEST, INDEX, R2_A1, OVERLAY_SEL, OVERLAY_ATT):
        if not Path(p).is_file():
            errs.append(f'missing {p}')
    try:
        idx = read_index()
        ids = smoke_ids() if smoke else sorted(idx)
        if not smoke and len(ids) != 3998:
            errs.append(f'{len(ids)} clips in r2 index, expected 3998')
        miss = [i for i in ids if not idx[i]['path'].is_file()]
        if miss:
            errs.append(f'{len(miss)} cached wavs missing, e.g. {miss[0]}')
        import importlib
        for mod in ('soundfile', 'soxr', 'pyloudnorm', 'scipy'):
            importlib.import_module(mod)
        if shutil.which('ffmpeg') is None:
            errs.append('ffmpeg missing')
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if errs:
        return 1
    log(f'PREFLIGHT OK (smoke={smoke})')
    if not SUMMARY.is_file() or smoke:
        need = 8 if smoke else 14
        if mem_available_gb() < need:
            log(f'PREFLIGHT WAIT MemAvailable {mem_available_gb():.1f} GiB < {need} GiB')
            return 75
        if gpu_free_mib() < 8 * 1024:
            log(f'PREFLIGHT WAIT GPU free {gpu_free_mib()} MiB < {8 * 1024} MiB')
            return 75
    return 0


def main():
    global OUT, IS_SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    a = ap.parse_args()
    os.chdir(REPO)
    if a.smoke:
        IS_SMOKE = True
        OUT = EVAL_ROOT / 'arena_consequence_200_smoke'
    if a.preflight:
        return preflight(a.smoke)
    if a.validate_only:
        return validate(a.smoke)
    if a.analyze:
        analyze(a.smoke)
        return 0
    rc = preflight(a.smoke)
    if rc:
        return rc
    run_all(a.smoke)
    analyze(a.smoke)
    return validate(a.smoke)


if __name__ == '__main__':
    sys.exit(main())
