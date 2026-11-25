#!/usr/bin/env python
"""189: MuQ A1 overlay attack, full matrix (eval-only driver).

Design: docs/experiments/muq_overlay_matrix_189_20261111.md (pre-registered; this file only implements it).

Question: does mixing a quiet synthetic loop sound (188's O_synth, plus two timbre variants frozen on the 188 dev
clips) into ten different base signals move MuQ-Eval A1 up while AES CE moves down, and does it beat the control
sounds (1 kHz sine, another music clip) at the same level?  Ten bases: real MusicCaps recordings, MeanAudio CFG0 /
CFG3+neg x 3 seeds, MusicGen, MusicLDM (1,000 Part B prompts each) and 551 MUSDB18 windows.  Overlay r LU
quieter than the base, r in {24, 18, 12, 6, 0}; mix renormalised to -30 LUFS.

Stages (each resumable through per-cell TSVs under <root>/per_clip/):
  dev       freeze osynth2 / osynth3 on the 188 dev clips (A1 only, r = 6); selection.json (sha256 + epoch)
  test      9,551 ids x 36 cells, tiered scorer set (tier 1 MuQ+AES+CLAP, 2 +PAM+CMI-RM caption, 3 +CMI-RM empty
            +SongEval); MEva on base and osynth r18; FAD (generated/real base cells) on base / osynth r18 / r0
  summary   summary.json (paired deltas, bootstrap CIs, verdict strings, guards); never says better/worse
  validate  everything on disk is complete and ordered after the selection freeze

Imports the 188 module and reuses its synthesis / mix / scoring code unchanged (188 is sha-bound and never edited).
"""
import os

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
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import muq_attractor_probe_188 as P  # noqa: E402

H = P.H
REPO = P.REPO
SELF = Path(__file__).resolve()
SR, NS = P.SR, P.NS
OUT = H.EVAL_ROOT / 'muq_overlay_matrix_189'
D188 = H.EVAL_ROOT / 'muq_attractor_probe_188'
KIND = 'muq_overlay_matrix_189_summary_v1'
SEL_KIND = 'muq_overlay_matrix_189_selection_v1'
IS_SMOKE = False

LEVELS = [24, 18, 12, 6, 0]
OVS = ['osynth', 'osynth2', 'osynth3', 'sine1k', 'music', 'osynth_tr2', 'specnoise']
TREATMENTS = ['osynth', 'osynth2', 'osynth3']
CONTROLS = ('sine1k', 'music')
SEEDS = ['14159265', '16180339', '27182818']
BASES = (['real'] + [f'ma_cfg0_s{s}' for s in SEEDS] + [f'ma_cfg3neg_s{s}' for s in SEEDS]
         + ['musicgen', 'musicldm', 'musdb'])
MC_BASES = [b for b in BASES if b != 'musdb']
MA_DIR = ('/home/kojiek/eval_output_nvme/phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full_s{seed}'
          '_mc_mf25_{proto}/audio')
TTM_DIR = H.EVAL_ROOT / 'ttm_metrics_115_20261009'
MUSDB_DIR = H.EVAL_ROOT / 'metric_win_tie_126c_20261019/stimuli'
MEVA_PY = REPO / 'runtime/meva_20261003/venv/bin/python'
MEVA_SCRIPT = REPO / 'scripts/eval/meva_score_dir.py'
SE_CKPT = REPO / 'scripts/eval/songeval_src/ckpt/model.safetensors'
MIN_FREE = 20 * 1024**3

M_A1 = H.M_A1
NB = 5000
SE_COLS = ['se_coh', 'se_mem', 'se_nat', 'se_cla', 'se_mus']
MUQ_COLS = ['A1_MI', 'A1_TA', 'A3a_MI', 'A3a_TA']
AES_COLS = ['PQ', 'CE', 'CU', 'PC']
SCORE_COLS = MUQ_COLS + AES_COLS + ['CLAP', 'PAM', 'CMI_q', 'CMI_q_empty'] + SE_COLS
GUARD_COLS = ['PAM', 'CMI_q', 'CMI_q_empty'] + SE_COLS
DELTA_COLS = MUQ_COLS + AES_COLS + ['CLAP'] + GUARD_COLS
TIER_COLS = {0: MUQ_COLS,
             1: MUQ_COLS + AES_COLS + ['CLAP'],
             2: MUQ_COLS + AES_COLS + ['CLAP', 'PAM', 'CMI_q'],
             3: SCORE_COLS}


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def tier_of(ov, r):
    if ov == 'base':
        return 3
    if ov in ('osynth', 'sine1k', 'music'):
        return 3 if r in (18, 0) else 2
    return 2 if r in (18, 0) else 1


def cell_key(base, ov, r):
    return f'T_{base}_base' if ov == 'base' else f'T_{base}_{ov}_r{r}'


def meva_cells():
    return [('base', None), ('osynth', 18)]


def fad_cells():
    return [('base', None), ('osynth', 18), ('osynth', 0)]


# ── tsv ──────────────────────────────────────────────────
def tsv_write(path, rows, ids):
    lines = ['id\t' + '\t'.join(SCORE_COLS)]
    for i in ids:
        lines.append(i + '\t' + '\t'.join(repr(float(rows[i].get(c, math.nan))) for c in SCORE_COLS))
    P.atomic(path, '\n'.join(lines) + '\n')


def tsv_read(path):
    with open(path, newline='') as f:
        return {r['id']: {k: float(v) for k, v in r.items() if k != 'id'} for r in csv.DictReader(f, delimiter='\t')}


def need_cols(tier, has_caps):
    return [c for c in TIER_COLS[tier] if has_caps or c != 'CLAP']


def rows_ok(rows, ids, cols):
    return set(rows) == set(ids) and all(math.isfinite(rows[i].get(c, math.nan)) for i in ids for c in cols)


# ── ids / bases ──────────────────────────────────────────
_WIN = []


def windows():
    if not _WIN:
        _WIN.extend(json.loads((MUSDB_DIR / 'windows.json').read_text())['windows'])
    return _WIN


def get_ids(smoke):
    mc = [i for i in H.part_b_ids() if P.ref_path(i).is_file()]
    ws = windows()
    if smoke:
        mc = mc[:24]
        tracks = []
        for w in ws:
            if w['track'] not in tracks:
                tracks.append(w['track'])
        keep, cnt, win = set(tracks[:4]), Counter(), []
        for w in ws:
            if w['track'] in keep and cnt[w['track']] < 2:
                win.append(w['win'])
                cnt[w['track']] += 1
    else:
        win = [w['win'] for w in ws]
    return mc, win


def track_map():
    return {w['win']: w['track'] for w in windows()}


def base_path(base, i):
    if base.startswith('ma_'):
        proto, seed = base[3:].split('_s')
        return Path(MA_DIR.format(seed=seed, proto='cfg0' if proto == 'cfg0' else 'cfg3_neg')) / f'{i}.flac'
    if base in ('musicgen', 'musicldm'):
        m = 'musicgen_medium' if base == 'musicgen' else 'musicldm'
        return TTM_DIR / m / 'lvl30/audio' / f'{i}.flac'
    raise KeyError(base)


def read_any(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    x = x.mean(axis=1).astype(np.float32)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR).astype(np.float32)
    return x


def load_one(base, i):
    if base == 'real':
        w = P.read_wav(P.ref_path(i))
    elif base == 'musdb':
        w = np.load(MUSDB_DIR / 'cache' / f'{i}.npz')['mixture'].astype(np.float32)
    else:
        w = read_any(base_path(base, i))
    return P.lvl30(P.fit(w[:NS]))[0]


def load_base(base, ids):
    t0 = time.time()
    ws = list(P.POOL.map(lambda i: load_one(base, i), ids))
    log(f'loaded base {base}: {len(ids)} ids in {time.time() - t0:.1f}s')
    return dict(zip(ids, ws))


# ── scorers ──────────────────────────────────────────────
class XScorer:
    """188's Scorer (MuQ / AES / CLAP) plus AES CU/PC, PAM, CMI-RM and SongEval, with per-scorer cost counters."""

    def __init__(self):
        self.t, self.n = Counter(), Counter()
        self.se = None
        self.peak_smi = 0

    def load_all(self):
        from muq_gen_steer_179_score import muq_hf_cache
        P.SC._muq()
        P.SC._aes()
        P.SC._clap()
        with muq_hf_cache():
            import newmetrics_pam
            import newmetrics_cmirm
            import metric_win_tie_126d as SE
            newmetrics_pam._load('cuda')
            # CMI-RM ships a forked `muq` package (adds muq_mulan.utils / models.mymodel) that collides with the pip
            # `muq` used by MuQ-Eval and SongEval.  Swap sys.modules only while CMI-RM is constructed (the model
            # object keeps its own classes); afterwards the pip package is restored.
            pip_muq = {k: sys.modules.pop(k) for k in list(sys.modules) if k == 'muq' or k.startswith('muq.')}
            try:
                newmetrics_cmirm._load('cuda', False)
            finally:
                for k in [k for k in sys.modules if k == 'muq' or k.startswith('muq.')]:
                    del sys.modules[k]
                sys.modules.update(pip_muq)
            self.se = (SE,) + tuple(SE.load_songeval('cuda'))
        w = (np.random.default_rng(0).standard_normal(NS) * 0.05).astype(np.float32)
        self.score([w, w], ['warm up', 'warm up'], 3, 'caption')
        self.t.clear()
        self.n.clear()
        P.SC.torch.cuda.reset_peak_memory_stats()

    def aes4(self, wavs):
        torch = P.SC.torch
        out = {c: [] for c in AES_COLS}
        for s in range(0, len(wavs), 32):
            items = [{'path': torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': SR}
                     for w in wavs[s:s + 32]]
            try:
                rows = P.SC.aes.forward(items)
            except Exception:  # noqa: BLE001
                rows = [P.SC.aes.forward([it])[0] for it in items]
            for c in AES_COLS:
                out[c] += [float(r[c]) for r in rows]
        return out

    def songeval(self, wavs):
        SE, muq, gen = self.se
        torch = P.SC.torch
        res = []
        for s in range(0, len(wavs), 16):
            w24 = [P.SC.rs(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).cuda()).float().cpu().numpy()
                   for w in wavs[s:s + 16]]
            res.append(SE.score_songeval_batch(w24, muq, gen, 'cuda'))
        return np.concatenate(res)

    def smi_peak(self):
        try:
            self.peak_smi = max(self.peak_smi, int(subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True).stdout.split()[0]))
        except Exception:  # noqa: BLE001
            pass

    def score(self, wavs, caps, tier, cmi_mode):
        """-> list of row dicts.  caps: list of captions or None (musdb)."""
        n = len(wavs)
        cols = {}

        def clock(name, fn):
            t0 = time.time()
            r = fn()
            P.SC.torch.cuda.synchronize()
            self.t[name] += time.time() - t0
            self.n[name] += n
            return r

        cols.update(clock('muq', lambda: P.SC.muq_scores(wavs)[0]))
        if tier >= 1:
            cols.update(clock('aes', lambda: self.aes4(wavs)))
            cols['CLAP'] = clock('clap', lambda: P.SC.clap_scores(wavs, caps)) if caps else [math.nan] * n
        use_caps = bool(caps) and cmi_mode == 'caption'
        if tier >= 2:
            import newmetrics_pam
            import newmetrics_cmirm
            cols['PAM'] = clock('pam', lambda: [newmetrics_pam.score_pam(w, sr=SR)['pam'] for w in wavs])
            if use_caps:
                cols['CMI_q'] = clock('cmi', lambda: [newmetrics_cmirm.score_cmirm(
                    w, sr=SR, prompt_mode='caption', caption=c)['quality'] for w, c in zip(wavs, caps)])
            else:
                cols['CMI_q'] = clock('cmi', lambda: [newmetrics_cmirm.score_cmirm(
                    w, sr=SR, prompt_mode='empty')['quality'] for w in wavs])
        if tier >= 3:
            if use_caps:
                cols['CMI_q_empty'] = clock('cmi_empty', lambda: [newmetrics_cmirm.score_cmirm(
                    w, sr=SR, prompt_mode='empty')['quality'] for w in wavs])
            else:
                cols['CMI_q_empty'] = list(cols['CMI_q'])
            se = clock('songeval', lambda: self.songeval(wavs))
            for k, c in enumerate(SE_COLS):
                cols[c] = [float(v) for v in se[:, k]]
        self.smi_peak()
        return [{c: float(cols[c][i]) for c in cols} for i in range(n)]


X = XScorer()


def cost_add(dt, dn):
    p = OUT / 'cost.json'
    c = json.loads(p.read_text()) if p.is_file() else {'t': {}, 'n': {}}
    for k, v in dt.items():
        c['t'][k] = c['t'].get(k, 0.0) + v
    for k, v in dn.items():
        c['n'][k] = c['n'].get(k, 0) + v
    P.jdump(p, c)


def run_cell(key, ids, build, caps, tier, cmi_mode, stage):
    """Score build(id) for every id; per-key TSV cache valid only if all columns of the tier are finite."""
    path = OUT / 'per_clip' / f'{key}.tsv'
    req = need_cols(tier, bool(caps))
    if path.is_file():
        rows = tsv_read(path)
        if rows_ok(rows, ids, req):
            return rows
    t0, t_before, n_before = time.time(), Counter(X.t), Counter(X.n)
    ch = P.CFG['chunk']
    chunks = [ids[k:k + ch] for k in range(0, len(ids), ch)]
    res = {}
    nxt = [P.POOL.submit(build, x) for x in chunks[0]]
    for k, c in enumerate(chunks):
        cur = nxt
        if k + 1 < len(chunks):
            nxt = [P.POOL.submit(build, x) for x in chunks[k + 1]]
        wavs = [f.result() for f in cur]
        for w in wavs:
            if not np.all(np.isfinite(w)):
                raise RuntimeError(f'{key}: non-finite stimulus')
        rows = X.score(wavs, [caps[x] for x in c] if caps else None, tier, cmi_mode)
        for x, r in zip(c, rows):
            bad = [q for q in req if not math.isfinite(r.get(q, math.nan))]
            if bad:
                raise RuntimeError(f'{key}: non-finite {bad} for {x}: {r}')
            res[x] = r
        P.progress(stage, key, len(res), len(ids))
    tsv_write(path, res, ids)
    wall = time.time() - t0
    dt = {k: X.t[k] - t_before.get(k, 0.0) for k in X.t}
    P.add_timing(key, len(ids), wall, dt)
    cost_add(dt, {k: X.n[k] - n_before.get(k, 0) for k in X.n})
    log(f'{key}: tier {tier}, {len(ids)} stimuli in {wall:.1f}s ({len(ids) / wall:.2f}/s)')
    return res


# ── selection / dev ──────────────────────────────────────
def sel_path():
    return OUT / 'selection.json'


def sha_of(body):
    return hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != 'sha256'},
                                     sort_keys=True).encode()).hexdigest()


def load_sel():
    p = sel_path()
    if not p.is_file():
        raise SystemExit('test scoring refuses to run: selection.json missing (run the dev step first)')
    s = json.loads(p.read_text())
    if sha_of(s) != s.get('sha256'):
        raise SystemExit('test scoring refuses to run: selection.json sha256 mismatch (edited after freezing?)')
    return s


def dev_step(c_all, dev, dev_base, sel188, n_cand):
    if sel_path().is_file():
        return load_sel()
    assert not (set(dev) & set(H.part_b_ids())), 'dev ids overlap Part B'
    missing = [i for i in dev if i not in dev_base]
    assert not missing, f'dev ids not in 188 D_dev_base: {missing[:3]}'
    b_a1 = np.array([dev_base[i]['A1_MI'] for i in dev])
    cands, pick = {}, {}
    for fam in ('saw', 'sine'):
        names = [n for n in c_all if not n.startswith('P5_') and not n.startswith('bridge')
                 and len(n.split('_', 2)) == 3 and n.split('_', 2)[1] == fam]
        names = sorted(names, key=lambda n: -c_all[n]['A1_MI'])[:n_cand]
        best, bv = None, -1e9
        for n in names:
            ov = P.build_C(n)
            rows = run_cell(f'DEV_{n}', dev, lambda i, ov=ov: P.mix(P.real30(i), ov, 6), None, 0, 'caption', 'dev')
            dd = float(np.mean(np.array([rows[i]['A1_MI'] for i in dev]) - b_a1))
            cands[n] = {'standalone_A1': float(c_all[n]['A1_MI']), 'dev_dA1': dd, 'family': fam}
            if dd > bv:
                best, bv = n, dd
        pick[fam] = best
    body = {'document_kind': SEL_KIND, 'created': time.strftime('%FT%T%z'), 'created_epoch': time.time(),
            'smoke': IS_SMOKE, 'osynth': sel188['O_synth'], 'osynth2': pick['saw'], 'osynth3': pick['sine'],
            'candidates': cands, 'dev_n': len(dev), 'level': 6,
            'dev_ids_sha256': hashlib.sha256(','.join(dev).encode()).hexdigest()}
    body['sha256'] = sha_of(body)
    P.jdump(sel_path(), body)
    log(f'dev: osynth2 {pick["saw"]}  osynth3 {pick["sine"]}  candidates {json.dumps(cands)}')
    return body


# ── overlays ─────────────────────────────────────────────
def specnoise_from(w, rng):
    n = len(w)
    sp = np.abs(np.fft.rfft(np.asarray(w, np.float64))) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    cs = np.concatenate([[0.0], np.cumsum(sp)])
    lo = np.searchsorted(f, f * 2 ** (-1 / 6), 'left')
    hi = np.searchsorted(f, f * 2 ** (1 / 6), 'right')
    sm = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    ph = np.exp(2j * np.pi * rng.random(len(f)))
    ph[0] = 1.0
    if n % 2 == 0:
        ph[-1] = 1.0
    return P.lvl30(np.fft.irfft(np.sqrt(sm) * ph, n=n))[0]


def build_overlays(sel):
    ov = {n: P.build_C(sel[n]) for n in TREATMENTS}
    assert P.C_NAME['sine1k'] == 'P5_sine_const'
    ov['sine1k'] = P.build_C(P.C_NAME['sine1k'])
    g, tim, e = sel['osynth'].split('_', 2)
    notes, amps = P.group_notes(g, P.ATT)
    y = P.synth_tone([f * 2 ** (2 / 12) for f in notes], amps, tim) * P.synth_env(P.T_ENV[e], P.ATT)
    ov['osynth_tr2'] = P.lvl30(y)[0]
    ov['specnoise'] = specnoise_from(ov['osynth'], P.rng_for('specnoise'))
    for k, v in ov.items():
        assert len(v) == NS and np.all(np.isfinite(v)), k
    return ov


def top_peaks(w, k=4):
    sp = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    f = np.fft.rfftfreq(len(w), 1 / SR)
    out = []
    for j in np.argsort(sp)[::-1]:
        if all(abs(f[j] - q) > 20 for q in out):
            out.append(float(f[j]))
        if len(out) == k:
            break
    return sorted(out), float(sp.max() / (np.median(sp) + 1e-12))


def sanity(ov, base_wavs):
    rec = {}
    for n in ('osynth', 'osynth_tr2', 'specnoise', 'sine1k'):
        pk, ratio = top_peaks(ov[n])
        rec[n] = {'peaks_hz': pk, 'peak_over_median': ratio, 'lufs': P.lufs_of(ov[n])}
    i = next(iter(base_wavs))
    rec['mix_lufs_real_osynth_r18'] = P.lufs_of(P.mix(base_wavs[i], ov['osynth'], 18))
    rec['mix_lufs_real_osynth_r0'] = P.lufs_of(P.mix(base_wavs[i], ov['osynth'], 0))
    P.jdump(OUT / 'sanity.json', rec)
    log('sanity: ' + json.dumps(rec))
    return rec


# ── MEva / FAD ───────────────────────────────────────────
def export_flacs(d, ids, build):
    import soundfile as sf
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for i, w in zip(ids, P.POOL.map(build, ids)):
        sf.write(str(d / f'{i}.flac'), w, SR, subtype='PCM_24')


def meva_cell(key, ids, build):
    out = OUT / 'meva' / f'{key}.tsv'
    if out.is_file():
        return
    d, o = OUT / 'tmp' / f'meva_{key}', OUT / 'tmp' / f'meva_{key}_out'
    export_flacs(d, ids, build)
    shutil.rmtree(o, ignore_errors=True)
    o.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([str(MEVA_PY), str(MEVA_SCRIPT), '--audio_dir', str(d), '--out_dir', str(o)],
                       cwd=REPO, capture_output=True, text=True)
    wall = time.time() - t0
    if r.returncode or not (o / 'meva_per_clip.tsv').is_file():
        raise RuntimeError(f'MEva {key} rc={r.returncode}: {r.stderr[-800:]}')
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(o / 'meva_per_clip.tsv', out)
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(o, ignore_errors=True)
    mc = OUT / 'meva_cost.json'
    c = json.loads(mc.read_text()) if mc.is_file() else {}
    c[key] = {'n': len(ids), 'wall_s': wall}
    P.jdump(mc, c)
    log(f'MEva {key}: {len(ids)} clips in {wall:.1f}s')


def fad_cell_key(base, ov, r):
    return f'{base}_base' if ov == 'base' else f'{base}_{ov}_r{r}'


def fad_run(key, ids, caps, build):
    out = OUT / 'fad' / f'{key}.json'
    if out.is_file():
        return
    (OUT / 'fad').mkdir(parents=True, exist_ok=True)
    d = OUT / 'tmp' / f'fad_{key}'
    export_flacs(d, ids, build)
    with open(OUT / 'fad' / f'{key}.ids.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['id', 'caption'])
        for i in ids:
            w.writerow([i, caps[i]])
    env = {k: v for k, v in os.environ.items() if k != 'MKL_NUM_THREADS'}
    cmd = [str(H.PY), '-I', str(SELF), '--root', str(OUT), '--fad-cell', key]
    if IS_SMOKE:
        cmd.append('--smoke')
    t0 = time.time()
    with open(OUT / 'fad' / f'{key}.log', 'w') as g:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=g, stderr=subprocess.STDOUT).returncode
    shutil.rmtree(d, ignore_errors=True)
    if rc:
        if IS_SMOKE:
            P.jdump(out, {'fad': None, 'error': f'rc {rc} (smoke n too small?)', 'n': len(ids),
                          'seconds': time.time() - t0})
            log(f'FAD {key}: failed rc={rc} (tolerated in smoke)')
            return
        raise RuntimeError(f'FAD {key} rc={rc}, see {OUT / "fad" / f"{key}.log"}')
    log(f'FAD {key}: {json.loads(out.read_text()).get("fad")}')


def fad_cell_main(key):
    import eval_metrics as EM
    d = OUT / 'tmp' / f'fad_{key}'
    rows = EM.load_rows(OUT / 'fad' / f'{key}.ids.tsv')
    t0 = time.time()
    fad, pairs = EM.score_fad(rows, d, H.FAD_REF, num_samples=0)
    P.jdump(OUT / 'fad' / f'{key}.json', {'fad': fad, 'pairs': pairs, 'n': len(rows), 'seconds': time.time() - t0})
    shutil.rmtree(d, ignore_errors=True)
    return 0


# ── preflight ────────────────────────────────────────────
def preflight(smoke):
    errs = []
    if shutil.disk_usage(H.EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    for p in (D188 / 'attractor.json', D188 / 'selection.json', D188 / 'per_clip/C_all.tsv',
              D188 / 'per_clip/D_dev_base.tsv', H.PARTB / 'part_b_selection.json', H.TTM115, H.MC_TSV,
              Path(H.FAD_REF), P.REF_DIR, MUSDB_DIR / 'windows.json', MUSDB_DIR / 'cache', MEVA_PY, MEVA_SCRIPT,
              SE_CKPT, H.PY):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    try:
        mc, win = get_ids(smoke)
        want = 24 if smoke else 1000
        if len(mc) != want:
            errs.append(f'{len(mc)} Part B ids with reference audio, expected {want}')
        for b in MC_BASES:
            if b == 'real':
                continue
            miss = [i for i in mc if not base_path(b, i).is_file()]
            if miss:
                errs.append(f'base {b}: {len(miss)} missing audio files, e.g. {miss[0]}')
        miss = [i for i in win if not (MUSDB_DIR / 'cache' / f'{i}.npz').is_file()]
        if miss:
            errs.append(f'musdb: {len(miss)} missing cache files, e.g. {miss[0]}')
        if not smoke and len(win) != 551:
            errs.append(f'{len(win)} musdb windows, expected 551')
        if len({track_map()[i] for i in win}) < (4 if smoke else 100):
            errs.append('too few musdb tracks')
        if len(P.pools()['dev']) < 200:
            errs.append('188 dev pool smaller than 200')
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if errs:
        return 1
    log(f'PREFLIGHT OK (smoke={smoke})')
    if not (OUT / 'summary.json').is_file():
        f = P.free_mib()
        need = 10 * 1024 if smoke else 16 * 1024
        if f < need:
            log(f'PREFLIGHT WAIT GPU free {f} MiB < {need} MiB')
            return 75
    return 0


# ── main run ─────────────────────────────────────────────
def setup(smoke):
    P.ATT.update(json.loads((D188 / 'attractor.json').read_text()))
    sel188 = P.load_selection()            # P.OUT is still D188 here
    dev = P.pools()['dev']                 # FULL cfg: the 200 dev ids (disjoint from Part B)
    if smoke:
        dev = dev[:16]                     # smoke: n=16 dev clips
    c_all = P.tsv_read(D188 / 'per_clip/C_all.tsv')
    dev_base = P.tsv_read(D188 / 'per_clip/D_dev_base.tsv')
    P.OUT = OUT
    P.IS_SMOKE = smoke
    P.CFG['chunk'] = 16 if smoke else 64
    P.POOL = ThreadPoolExecutor(P.NTHREAD)
    H.load_scales()
    OUT.mkdir(parents=True, exist_ok=True)
    return sel188, dev, c_all, dev_base


def run_all(smoke):
    sel188, dev, c_all, dev_base = setup(smoke)
    mc, win = get_ids(smoke)
    caps_all = P.caps_all()
    P.progress('start', '', 0, 0)
    X.load_all()
    log(f'all scorers loaded; torch max alloc {P.SC.torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    sel = dev_step(c_all, dev, dev_base, sel188, 2 if smoke else 3)
    ovs = build_overlays(sel)
    levels = [18, 0] if smoke else LEVELS
    for base in BASES:
        ids = win if base == 'musdb' else mc
        wavs = load_base(base, ids)
        if base == 'real':
            sanity(ovs, wavs)
        caps = None if base == 'musdb' else {i: caps_all[i] for i in ids}
        cmi_mode = 'empty' if base == 'musdb' else 'caption'
        pos = {i: k for k, i in enumerate(ids)}
        for ov, r in [('base', None)] + [(o, r) for o in OVS for r in levels]:
            if ov == 'base':
                build = wavs.__getitem__
            elif ov == 'music':
                build = (lambda i, r=r: P.mix(wavs[i], P.real30(dev[pos[i] % len(dev)]), r))
            else:
                build = (lambda i, o=ovs[ov], r=r: P.mix(wavs[i], o, r))
            run_cell(cell_key(base, ov, r), ids, build, caps, tier_of(ov, r), cmi_mode, 'test')
            if (ov, r) in meva_cells():
                meva_cell(cell_key(base, ov, r), ids, build)
            if base != 'musdb' and (ov, r) in fad_cells():
                fad_run(fad_cell_key(base, ov, r), ids, caps, build)
        del wavs
    P.jdump(OUT / 'gpu_peak.json', {'torch_max_alloc_gib': P.SC.torch.cuda.max_memory_allocated() / 2**30,
                                    'torch_max_reserved_gib': P.SC.torch.cuda.max_memory_reserved() / 2**30,
                                    'nvidia_smi_used_max_mib': X.peak_smi})
    P.progress('scored', '', 0, 0)
    project(smoke)


# ── cost projection ──────────────────────────────────────
TIER_SCORERS = {1: ['muq', 'aes', 'clap'], 2: ['muq', 'aes', 'clap', 'pam', 'cmi'],
                3: ['muq', 'aes', 'clap', 'pam', 'cmi', 'cmi_empty', 'songeval']}


def project(smoke):
    cp, tp = OUT / 'cost.json', OUT / 'timing.json'
    if not cp.is_file() or not tp.is_file():
        return None
    c, t = json.loads(cp.read_text()), json.loads(tp.read_text())
    per = {k: c['t'][k] / c['n'][k] for k in c['t'] if c['n'].get(k)}
    tk = {k: v for k, v in t['keys'].items() if k.startswith('T_')}
    sum_n, sum_w = sum(v['n'] for v in tk.values()), sum(v['wall_s'] for v in tk.values())
    comp_t = sum(c['t'].values())      # includes DEV cells (muq only); small
    overhead = max(0.0, sum_w - comp_t) / max(sum_n, 1)
    counts = Counter(tier_of(o, r) for o, r in [('base', None)] + [(o, r) for o in OVS for r in LEVELS])
    wall, detail = 0.0, {}
    for tier in (1, 2, 3):
        sc = sum(per.get(k, 0.0) for k in TIER_SCORERS[tier])
        clips = counts[tier] * (9 * 1000 + 551)
        detail[f'tier{tier}'] = {'cells_per_base': counts[tier], 'clips': clips, 's_per_clip': sc + overhead,
                                 'hours': clips * (sc + overhead) / 3600}
        wall += clips * (sc + overhead)
    mcost = json.loads((OUT / 'meva_cost.json').read_text()) if (OUT / 'meva_cost.json').is_file() else {}
    mc_w = [v['wall_s'] for k, v in mcost.items() if 'musdb' not in k]
    ms_w = [v['wall_s'] for k, v in mcost.items() if 'musdb' in k]
    meva_h = None
    if mc_w and ms_w and smoke:
        a, b = float(np.mean(mc_w)), float(np.mean(ms_w))
        pc = (a - b) / 16
        if pc <= 0:
            pc = a / 24
        fixed = max(0.0, b - 8 * pc)
        meva_h = (18 * (fixed + 1000 * pc) + 2 * (fixed + 551 * pc)) / 3600
    elif mcost:
        meva_h = sum(v['wall_s'] for v in mcost.values()) / 3600
    fad_s = []
    for p in (D188 / 'fad').glob('*.json'):
        try:
            v = json.loads(p.read_text())
            if v.get('seconds') and v.get('n'):
                fad_s.append(v['seconds'] * 1000 / v['n'])
        except Exception:  # noqa: BLE001
            pass
    fad_h = 27 * float(np.mean(fad_s)) / 3600 if fad_s else None
    total_h = wall / 3600 + (meva_h or 0) + (fad_h or 0)
    rec = {'per_clip_s': per, 'overhead_s_per_clip': overhead, 'tiers': detail, 'meva_hours': meva_h,
           'fad_hours_from_188_rate': fad_h, 'scoring_hours': wall / 3600, 'projected_total_hours': total_h,
           'note': 'scorer cost per clip measured in this run; MEva two-point fit from smoke n=24 vs n=8; '
                   'FAD from the 188 per-clip rate x 1000 clips x 27 cells; musdb treated like MusicCaps'}
    P.jdump(OUT / 'projection.json', rec)
    log('PROJECTION per-clip: ' + ', '.join(f'{k} {v * 1000:.1f}ms' for k, v in per.items())
        + f'; overhead {overhead * 1000:.1f}ms')
    log(f'PROJECTION full run: scoring {wall / 3600:.1f} h + MEva {meva_h} h + FAD {fad_h} h = {total_h:.1f} h')
    return rec


# ── summary ──────────────────────────────────────────────
class Boot:
    def __init__(self, clusters):
        _, self.k = np.unique(np.asarray(clusters), return_inverse=True)
        self.K = int(self.k.max()) + 1
        self.idx = np.random.default_rng(20261111).integers(0, self.K, (NB, self.K))

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


def arr(rows, ids, col):
    return np.array([rows[i].get(col, math.nan) for i in ids], float)


def read_meva(path):
    with open(path, newline='') as f:
        return {r['id']: float(r['meva_raw']) for r in csv.DictReader(f, delimiter='\t')}


def summarize(smoke):
    levels = [18, 0] if smoke else LEVELS
    mc, win = get_ids(smoke)
    tmap = track_map()
    sel = load_sel()
    H.load_scales()
    S_CE = H.S['CE_lvl30']
    cells = {}
    for base in BASES:
        for ov, r in [('base', None)] + [(o, r) for o in OVS for r in levels]:
            p = OUT / 'per_clip' / f'{cell_key(base, ov, r)}.tsv'
            if p.is_file():
                cells[(base, ov, r)] = tsv_read(p)

    def bmean(b, c):
        v = arr(cells[(b, 'base', None)], mc, c)
        return float(np.nanmean(v)) if np.isfinite(v).any() else math.nan

    sysm = {}
    for c in DELTA_COLS:
        sysm[c] = [bmean('real', c), float(np.mean([bmean(f'ma_cfg0_s{s}', c) for s in SEEDS])),
                   float(np.mean([bmean(f'ma_cfg3neg_s{s}', c) for s in SEEDS])), bmean('musicgen', c),
                   bmean('musicldm', c)]
    S = {c: (float(max(v) - min(v)) if all(math.isfinite(x) for x in v) else None) for c, v in sysm.items()}
    meva = {}
    for b in BASES:
        p = OUT / 'meva' / f'{cell_key(b, "base", None)}.tsv'
        if p.is_file():
            meva[b] = {'base': read_meva(p)}
            q = OUT / 'meva' / f'{cell_key(b, "osynth", 18)}.tsv'
            if q.is_file():
                meva[b]['osynth_r18'] = read_meva(q)

    def mm(b):
        return float(np.mean(list(meva[b]['base'].values())))

    S_meva = None
    if all(b in meva for b in ('real', 'musicgen', 'musicldm')) and all(
            f'ma_cfg0_s{s}' in meva and f'ma_cfg3neg_s{s}' in meva for s in SEEDS):
        v = [mm('real'), np.mean([mm(f'ma_cfg0_s{s}') for s in SEEDS]), np.mean([mm(f'ma_cfg3neg_s{s}') for s in SEEDS]),
             mm('musicgen'), mm('musicldm')]
        S_meva = float(max(v) - min(v))

    out = {'kind': KIND, 'smoke': smoke, 'M_A1': M_A1, 'S_CE': S_CE, 'S_tie': S, 'S_meva': S_meva,
           'system_means_for_S': sysm,
           'selection': {k: sel[k] for k in ('osynth', 'osynth2', 'osynth3', 'candidates', 'dev_n', 'created_epoch',
                                              'sha256')},
           'levels': levels, 'bases': {}, 'base_means': {}, 'note': 'scorer movements only; no better/worse labels'}
    for base in BASES:
        ids = win if base == 'musdb' else mc
        if (base, 'base', None) not in cells:
            continue
        bt = Boot([tmap[i] for i in ids] if base == 'musdb' else list(range(len(ids))))
        b = cells[(base, 'base', None)]
        out['base_means'][base] = {c: (float(np.nanmean(arr(b, ids, c))) if np.isfinite(arr(b, ids, c)).any() else None)
                                   for c in SCORE_COLS}

        def d(ov, r, col):
            x = cells.get((base, ov, r))
            return None if x is None else arr(x, ids, col) - arr(b, ids, col)

        rec = {'n': len(ids), 'treat': {}, 'pitch_specific': {}, 'dose': {}}
        for tr in TREATMENTS:
            rec['treat'][tr] = {}
            for r in levels:
                if (base, tr, r) not in cells:
                    continue
                e = {'delta': {}, 'guards': {}}
                for c in DELTA_COLS:
                    dd = d(tr, r, c)
                    ci = bt.ci(dd) if dd is not None else None
                    if ci is None:
                        continue
                    e['delta'][c] = ci
                    if c in GUARD_COLS:
                        tie = 0.10 * S[c] if S.get(c) else None
                        e['guards'][c] = {'ci_below_zero': ci[2] < 0, 'ci_above_zero': ci[1] > 0,
                                          'beyond_tie_down': (ci[0] <= -tie) if tie else None,
                                          'beyond_tie_up': (ci[0] >= tie) if tie else None, 'tie': tie}
                gp, gc = e['guards'].get('PAM'), e['guards'].get('CMI_q')
                e['independent_unseen'] = (None if (gp is None or gc is None)
                                           else bool(not gp['ci_below_zero'] and not gc['ci_below_zero']))
                a1, ce = e['delta'].get('A1_MI'), e['delta'].get('CE')
                if a1 is None or ce is None:
                    continue
                e['CE_beyond_tie'] = bool(ce[0] <= -0.10 * S_CE)
                if not (a1[1] >= M_A1):
                    v3 = 'no_effect'
                elif ce[2] < 0:
                    v3 = 'success'
                else:
                    v3 = 'not_success_CE_not_down'
                e['verdict_v3'] = v3
                vc = {}
                for c_ov in CONTROLS:
                    ya, yc = d(c_ov, r, 'A1_MI'), d(c_ov, r, 'CE')
                    if ya is None or yc is None:
                        continue
                    vc[c_ov] = {'A1_diff': bt.ci(d(tr, r, 'A1_MI') - ya), 'CE_diff': bt.ci(d(tr, r, 'CE') - yc)}
                e['vs_controls'] = vc
                if len(vc) == len(CONTROLS):
                    a1_ok = all(v['A1_diff'][1] >= M_A1 for v in vc.values())
                    ce_ok = all(v['CE_diff'][2] < 0 for v in vc.values())
                    e['a1_ok'], e['ce_ok'] = a1_ok, ce_ok
                    if v3 != 'success':
                        e['label'] = v3
                    elif a1_ok and ce_ok:
                        e['label'] = 'success_specific'
                    elif a1_ok:
                        e['label'] = 'A1_specific_CE_nonspecific'
                    elif ce_ok:
                        e['label'] = 'CE_specific_A1_nonspecific'
                    else:
                        e['label'] = 'nonspecific'
                rec['treat'][tr][str(r)] = e
            ok_r = [r for r in levels if (rec['treat'][tr].get(str(r), {}).get('delta', {}).get('A1_MI') or [0, -9e9])[1] >= M_A1]
            rec['dose'][tr] = {'min_dose_r': (max(ok_r) if ok_r else None), 'M_A1': M_A1,
                               'note': 'largest r (LU, lower = louder overlay) whose dA1 CI lower >= M_A1'}
        for r in levels:
            ps = {}
            for alt in ('osynth_tr2', 'specnoise'):
                x, y = d('osynth', r, 'A1_MI'), d(alt, r, 'A1_MI')
                if x is not None and y is not None:
                    ci = bt.ci(x - y)
                    ps[alt] = {'A1_diff': ci, 'specific': bool(ci[1] >= M_A1)}
            if ps:
                rec['pitch_specific'][str(r)] = ps
        if base in meva and 'osynth_r18' in meva[base]:
            mb, mo = meva[base]['base'], meva[base]['osynth_r18']
            common = [i for i in ids if i in mb and i in mo]
            ci = Boot([tmap[i] for i in common] if base == 'musdb' else list(range(len(common)))).ci(
                np.array([mo[i] - mb[i] for i in common]))
            tie = 0.10 * S_meva if S_meva else None
            rec['meva_r18'] = {'delta': ci, 'base_mean': float(np.mean([mb[i] for i in common])),
                               'ci_below_zero': ci[2] < 0, 'beyond_tie_down': (ci[0] <= -tie) if tie else None,
                               'tie': tie}
        out['bases'][base] = rec
    fad = {}
    for b in MC_BASES:
        row = {}
        for ov, r in fad_cells():
            p = OUT / 'fad' / f'{fad_cell_key(b, ov, r)}.json'
            if p.is_file():
                row['base' if ov == 'base' else f'{ov}_r{r}'] = json.loads(p.read_text()).get('fad')
        if row:
            f0 = row.get('base')
            row['delta'] = {k: (v - f0 if (v is not None and f0 is not None) else None)
                            for k, v in row.items() if k != 'base'}
            fad[b] = row
    out['fad'] = fad
    head = {}
    for tr in TREATMENTS:
        head[tr] = {}
        for r in levels:
            def lab(b):
                return out['bases'].get(b, {}).get('treat', {}).get(tr, {}).get(str(r), {})
            who = [b for b in MC_BASES if lab(b).get('label') == 'success_specific']
            who_v3 = [b for b in MC_BASES if lab(b).get('verdict_v3') == 'success']
            uns = [b for b in MC_BASES if lab(b).get('independent_unseen') is True]
            md = lab('musdb').get('delta', {})
            ms, mce = md.get('A1_MI'), md.get('CE')
            musdb_pos = bool(ms and ms[1] > 0)
            musdb_ce_down = bool(mce and mce[2] < 0)
            n_ok = len(who)
            if n_ok >= 7 and musdb_pos and musdb_ce_down:
                reading = 'general'
            elif n_ok >= 3:
                reading = 'base_dependent'
            else:
                reading = 'only_188_base'
            head[tr][str(r)] = {'n_success_specific_of_9': n_ok, 'bases': who,
                                'n_success_v3_of_9': len(who_v3), 'bases_success_v3': who_v3,
                                'n_independent_unseen_of_9': len(uns),
                                'musdb_dA1_ci_lower_gt_0': musdb_pos, 'musdb_dA1': ms,
                                'musdb_dCE_ci_upper_lt_0': musdb_ce_down, 'musdb_dCE': mce, 'reading': reading}
    out['recipe_replication_r18'] = None
    if all(t in head and '18' in head[t] for t in ('osynth2', 'osynth3')):
        n2 = head['osynth2']['18']['n_success_v3_of_9']
        n3 = head['osynth3']['18']['n_success_v3_of_9']
        out['recipe_replication_r18'] = {'osynth2': n2, 'osynth3': n3, 'single_recipe': bool(n2 <= 2 and n3 <= 2)}
    out['headline'] = head
    out['headline_osynth_r18'] = head['osynth'].get('18')
    P.jdump(OUT / 'summary.json', out)
    log('summary written; headline osynth r18: ' + json.dumps(out['headline_osynth_r18']))
    return out


# ── validate ─────────────────────────────────────────────
def validate(smoke):
    errs = []
    try:
        mc, win = get_ids(smoke)
        levels = [18, 0] if smoke else LEVELS
        sel = load_sel()
        for base in BASES:
            ids = win if base == 'musdb' else mc
            for ov, r in [('base', None)] + [(o, r) for o in OVS for r in levels]:
                key = cell_key(base, ov, r)
                p = OUT / 'per_clip' / f'{key}.tsv'
                if not p.is_file():
                    errs.append(f'missing cell {key}')
                    continue
                if not rows_ok(tsv_read(p), ids, need_cols(tier_of(ov, r), base != 'musdb')):
                    errs.append(f'cell {key}: id set or required columns not finite (tier {tier_of(ov, r)})')
                if p.stat().st_mtime < sel['created_epoch']:
                    errs.append(f'cell {key} older than selection.json')
            for ov, r in meva_cells():
                if not (OUT / 'meva' / f'{cell_key(base, ov, r)}.tsv').is_file():
                    errs.append(f'missing MEva {cell_key(base, ov, r)}')
            if base != 'musdb':
                for ov, r in fad_cells():
                    p = OUT / 'fad' / f'{fad_cell_key(base, ov, r)}.json'
                    if not p.is_file():
                        errs.append(f'missing FAD {p.name}')
                    elif not smoke and json.loads(p.read_text()).get('fad') is None:
                        errs.append(f'FAD {p.name} has no value')
        sp = OUT / 'summary.json'
        if not sp.is_file():
            errs.append('missing summary.json')
        elif json.loads(sp.read_text()).get('kind') != KIND:
            errs.append('summary.json wrong kind')
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


def main():
    global OUT, IS_SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--fad-cell')
    ap.add_argument('--root', help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.chdir(REPO)
    if a.smoke:
        IS_SMOKE = True
        OUT = H.EVAL_ROOT / 'muq_overlay_matrix_189_smoke'
    if a.root:
        OUT = Path(a.root)
    if a.fad_cell:
        return fad_cell_main(a.fad_cell)
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
