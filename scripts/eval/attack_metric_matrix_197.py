#!/usr/bin/env python
"""197: attack x metric matrix (eval-only driver).

Design: docs/experiments/attack_metric_matrix_197_20261117.md (pre-registered; this file only implements it).
All parts are tier 2 (MuQ A1/A3a, AES PQ/CE/CU/PC, CLAP, PAM, CMI-RM).  Stimuli are built on the fly, scored, then dropped;
only the first 5 clips of every cell/base are kept in examples/.

  --part T1    9 MusicCaps bases x 1,000 Part B ids x 15 cells (captions; CLAP on; CMI-RM caption)
  --part T2A   MAESTRO 150 excerpts x 2 SoundFonts x 13 cells (empty prompt, CLAP empty)
  --part T2B   384 synthetic chord-progression segments (12 tonics x maj/min x o4/o5 x 4 voicings x 2 SoundFonts)
  --part all   (default) T2B, T2A, T1 in that order

Statistics (bootstrap, markers, verdicts) live in scripts/analysis/attack_metric_matrix_tables.py, which also reshapes the
existing 189/190/193 per_clip rows; this driver only scores and writes per_clip/*.tsv, summary.json, REPORT.json, projection.json.
Reuses 195 (Z scorer, run_cell, key_of, ARPS, op_extra), 190 (Y, lvl_sr, req_cols) and 188/189 (mix, loaders, tsv) unchanged.
Resume: a cell whose per_clip tsv exists and passes rows_ok is skipped.
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
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ce_weakness_195 as C  # noqa: E402

M, P, H, Q, MS = C.M, C.P, C.H, C.Q, C.MS
SR, NS = C.SR, C.NS
REPO = C.REPO
OUT = H.EVAL_ROOT / 'attack_metric_matrix_197'
KIND = 'attack_metric_matrix_197_summary_v1'
SEED = 20261117
NB = 5000
MIN_FREE = 20 * 1024**3
SEEDS = M.SEEDS
BASES = M.MC_BASES
SMOKE_BASES = ['real', 'ma_cfg3neg_s14159265', 'musicgen']
LISTEN = ['real'] + [f'ma_cfg3neg_s{s}' for s in SEEDS] + ['musicgen', 'musicldm']
KJ_PY = Path('/home/kojiek/venvs/karaoke-jp/bin/python')
IS_SMOKE = False

# ── T1 cells ─────────────────────────────────────────────
GAIN = {'gain_p6': 6.0, 'gain_m6': -6.0}
PS = {'ps_p1': 1, 'ps_m1': -1, 'ps_p2': 2, 'ps_m2': -2}
TS = {'ts_090': 0.9, 'ts_110': 1.1}
OVL = {'kick05_r12': 12, 'kickrand_r12': 12, 'click05_r12': 12, 'sq220_r18': 18, 'nburst05_r12': 12}
T1_CELLS = ['base'] + list(GAIN) + ['sil05', 'rb_id'] + list(PS) + list(TS) + list(OVL) + ['pno0']
# ── T2A cells ────────────────────────────────────────────
T2A_CELLS = ['base', 'wrong_10', 'wrong_30', 'wrong_100', 'extra_harm_30', 'extra_rand_30', 'jitter_50', 'drop_30',
             'tr_p1', 'tr_m1', 'oct_p1', 'tempo_090', 'tempo_110']
BANKS = ['generaluser', 'fluidr3']
# ── T2B grid ─────────────────────────────────────────────
TONICS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
MODES = ['maj', 'min']
OCTS = ['o4', 'o5']
VOICES = ['block', 'up', 'down', 'rep']
PROG = {'maj': [(0, 4), (7, 4), (9, 3), (5, 4)],       # I V vi IV : (root semitone above tonic, third interval)
        'min': [(0, 3), (8, 4), (3, 4), (10, 4)]}      # i VI III VII

X = None


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


# ═════════ stimuli: T1 ═════════
def rubberband(x, tempo=1.0, pitch=1.0):
    """ffmpeg rubberband (same recipe as 134)."""
    x = np.ascontiguousarray(x, np.float32)
    q = ':pitchq=quality' if pitch != 1.0 else ''
    p = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-',
                        '-af', f'rubberband=tempo={tempo:.6f}:pitch={pitch:.8f}{q}', '-f', 'f32le', '-'],
                       input=x.tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).astype(np.float64)


def kick_template():
    t = np.arange(int(0.4 * SR)) / SR
    return np.sin(2 * np.pi * 55.0 * t) * np.exp(-t / 0.1) * np.minimum(1.0, t / 0.002)


def place(times, tpl):
    y = np.zeros(NS)
    for t in times:
        s = int(round(t * SR))
        n = min(len(tpl), NS - s)
        if n > 0:
            y[s:s + n] += tpl[:n]
    return y


def grid_times():
    return [0.5 * k for k in range(20)]


def rand_times(i):
    r = np.random.default_rng(MS.h('197kr', i, SEED))
    return sorted(r.uniform(0.0, 9.6, 20).tolist())


_OV = {}


def overlay(cell, i):
    if cell == 'kickrand_r12':
        return place(rand_times(i), kick_template())
    if cell not in _OV:
        if cell == 'kick05_r12':
            _OV[cell] = place(grid_times(), kick_template())
        elif cell == 'click05_r12':
            _OV[cell] = place(grid_times(), np.ones(16))
        elif cell == 'sq220_r18':
            _OV[cell] = np.sign(np.sin(2 * np.pi * 220.0 * np.arange(NS) / SR))
        elif cell == 'nburst05_r12':
            r = np.random.default_rng(MS.h('197nb', SEED))
            y = np.zeros(NS)
            for t in grid_times():
                s = int(round(t * SR))
                y[s:s + 800] += r.standard_normal(800)
            _OV[cell] = y
    return _OV[cell]


def t1_stim(cell, w, i, base=None):
    w = np.asarray(w, np.float64)
    if cell == 'base':
        return np.asarray(w, np.float32)
    if cell in GAIN:
        return np.asarray(w * 10 ** (GAIN[cell] / 20), np.float32)      # not re-locked, float, not clipped
    if cell == 'sil05':
        y = np.concatenate([np.zeros(int(0.5 * SR)), w])[:len(w)]
    elif cell == 'rb_id':
        y = P.fit(rubberband(w, 1.0, 1.0), len(w))
    elif cell in PS:
        y = P.fit(rubberband(w, 1.0, 2 ** (PS[cell] / 12)), len(w))
    elif cell in TS:
        y = rubberband(w, TS[cell], 1.0)
    elif cell in OVL:
        return P.mix(w, overlay(cell, i), OVL[cell])
    elif cell == 'pno0':
        key = key_for(base, i, w)
        return C.lock(w + C.ARPS[(key[0], key[1], 0)])
    else:
        raise KeyError(cell)
    return Q.lvl_sr(y, SR)


def key_for(base, i, w):
    if (base, i) not in C.KEYC:
        C.KEYC[(base, i)] = C.key_of(w)
    return C.KEYC[(base, i)]


def build_pno(pool):
    keys = [(t, m) for m in (0, 1) for t in range(12)]
    raw = pool.map(C._job_arp, keys)
    for k, x in zip(keys, raw):
        C.ARPS[(k[0], k[1], 0)] = P.fit(P.to_lufs(x, -30.0)[0])


# ═════════ stimuli: T2A (MAESTRO) ═════════
def a_op(cell):
    if cell == 'base':
        return None
    if cell.startswith('wrong_'):
        return MS.wrong_notes(int(cell.split('_')[-1]) / 100)
    if cell == 'extra_harm_30':
        return C.op_extra(0.30, True)
    if cell == 'extra_rand_30':
        return C.op_extra(0.30, False)
    if cell == 'jitter_50':
        return MS.jitter(0.050, False)
    if cell == 'drop_30':
        return MS.drop(0.3)
    if cell == 'tr_p1':
        return MS.transpose(1)
    if cell == 'tr_m1':
        return MS.transpose(-1)
    if cell == 'oct_p1':
        return MS.transpose(12)
    if cell == 'tempo_090':
        return MS.tempo(0.9)
    if cell == 'tempo_110':
        return MS.tempo(1.1)
    raise KeyError(cell)


def a_notes(exi, cell):
    ex = C.EXC[exi]
    thr = 11.0 if cell.startswith('tempo_') else 10.0      # tempo_110 compresses: take the extra second of source so the 10 s window stays full
    notes = sorted(tuple(n) for n in ex['notes'] if n[0] < thr)
    fn = a_op(cell)
    if fn is not None:
        notes = sorted(fn(notes, np.random.default_rng(MS.h('197A', ex['id'], cell, SEED))))
    return [n for n in notes if n[0] < 10.0]


def _job_A(job):
    exi, bank, cell = job
    x = MS.synth(a_notes(exi, cell), bank, 0, 0)
    return C.lock(x[int(MS.LEAD * SR):int(MS.LEAD * SR) + 160000])


# ═════════ stimuli: T2B (synthetic progressions) ═════════
def b_ids():
    smoke = IS_SMOKE
    return [f'{t}_{m}_{o}_{v}_{b}' for t in TONICS[:1 if smoke else 12] for m in (['maj'] if smoke else MODES)
            for o in OCTS for v in VOICES for b in BANKS]


def b_notes(tonic, mode, octv, voicing):
    """10 chords x 1 s (2 beats at 120 BPM), progression repeated 2.5 times; velocity 80; tonic root in C4-B4 (o4) or +12 (o5)."""
    t = TONICS.index(tonic)
    base = 60 + t + (12 if octv == 'o5' else 0)
    notes = []
    for c in range(10):
        r, third = PROG[mode][c % 4]
        tri = [base + r, base + r + third, base + r + 7]
        t0 = float(c)
        if voicing == 'block':
            for b in range(2):
                notes += [(t0 + 0.5 * b, t0 + 0.5 * b + 0.47, p, 80) for p in tri]
        else:
            seq = {'up': [0, 1, 2, 0], 'down': [2, 1, 0, 2], 'rep': [0, 0, 0, 0]}[voicing]
            for k, s in enumerate(seq):
                notes.append((t0 + 0.25 * k, t0 + 0.25 * k + 0.23, tri[s], 80))
    return notes


def _job_B(job):
    sid = job
    tonic, mode, octv, voicing, bank = sid.split('_')
    x = MS.synth(b_notes(tonic, mode, octv, voicing), bank, 0, 0)
    return C.lock(x[int(MS.LEAD * SR):int(MS.LEAD * SR) + 160000])


# ═════════ run machinery ═════════
def setup(smoke):
    C.OUT = OUT
    C.IS_SMOKE = smoke
    C.setup(smoke)
    P.POOL = P.POOL      # ThreadPoolExecutor created by C.setup


def get_ids(smoke):
    mc, _ = C.get_ids(smoke)
    return mc[:8] if smoke else mc


def bases_of(smoke):
    return SMOKE_BASES if smoke else BASES


def run_cell(key, ids, build, caps, clap, cmi_mode):
    """C.run_cell (tier 2) with CUDA-OOM retry/backoff; a partial cell is rescored from scratch."""
    for att in range(8):
        try:
            return C.run_cell(key, ids, build, caps, 2, clap, cmi_mode)
        except Exception as e:  # noqa: BLE001
            oom = 'out of memory' in str(e).lower() or type(e).__name__ == 'OutOfMemoryError'
            if not oom or att == 7:
                raise
            P.SC.torch.cuda.empty_cache()
            log(f'{key}: CUDA OOM attempt {att}, sleep {60 * (att + 1)}s')
            time.sleep(60 * (att + 1))


def cell_done(key, ids, clap, has_caps):
    return C.cell_done(key, ids, 2, clap, has_caps)


def stage_T2A(smoke, pool):
    ids = C.b_ids(smoke)
    idx = {i: k for k, i in enumerate(ids)}
    todo = [c for c in T2A_CELLS if not cell_done(f'T2A_{c}', ids, False, False)]

    def submit(cell):
        return pool.map_async(_job_A, [(k // 2, BANKS[k % 2], cell) for k in range(len(ids))], chunksize=2)
    pend = submit(todo[0]) if todo else None
    for n, cell in enumerate(todo):
        t0 = time.time()
        arr = pend.get()
        rs = time.time() - t0
        pend = submit(todo[n + 1]) if n + 1 < len(todo) else None
        arr = {i: arr[idx[i]] for i in ids}
        run_cell(f'T2A_{cell}', ids, lambda i, a=arr: a[i], None, False, 'empty')
        log(f'T2A {cell}: wait for render {rs:.1f}s')
    log(f'T2A done ({len(ids)} ids)')


def stage_T2B(smoke, pool):
    ids = b_ids()
    if cell_done('T2B_all', ids, False, False):
        return
    arr = dict(zip(ids, pool.map(_job_B, ids, chunksize=2)))
    run_cell('T2B_all', ids, lambda i: arr[i], None, False, 'empty')
    log(f'T2B done ({len(ids)} segments)')


def stage_T1(smoke, caps_all):
    ids = get_ids(smoke)
    for base in bases_of(smoke):
        todo = [c for c in T1_CELLS if not cell_done(f'T1_{base}_{c}', ids, True, True)]
        if not todo:
            continue
        wavs = M.load_base(base, ids)
        caps = {i: caps_all[i] for i in ids}
        for c in todo:
            run_cell(f'T1_{base}_{c}', ids, (lambda c_: lambda i: t1_stim(c_, wavs[i], i, base))(c), caps, True, 'caption')
        del wavs


def run_all(smoke, part):
    global X
    setup(smoke)
    with open(C.EXCERPTS) as f:
        C.EXC = json.load(f)
    caps_all = P.caps_all()
    pool = mp.get_context('fork').Pool(P.NTHREAD)       # fork BEFORE CUDA / scorers exist
    if part in ('all', 'T1'):
        build_pno(pool)
    P.progress('start', '', 0, 0)
    X = C.Z()
    X.load_all()
    C.X = X
    log(f'all scorers loaded; torch max alloc {P.SC.torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    if part in ('all', 'T2B'):
        stage_T2B(smoke, pool)
    if part in ('all', 'T2A'):
        stage_T2A(smoke, pool)
    pool.close()
    if part in ('all', 'T1'):
        stage_T1(smoke, caps_all)
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
    per, hours = {}, {}
    for k, v in t['keys'].items():
        if k.startswith('T1_'):
            ov = next(c for c in T1_CELLS if k.endswith('_' + c))
            per.setdefault(('T1', ov), []).append(v['wall_s'] / v['n'])
        elif k.startswith('T2A_'):
            per[('T2A', k[4:])] = [v['wall_s'] / v['n']]
        elif k.startswith('T2B_'):
            per[('T2B', 'all')] = [v['wall_s'] / v['n']]
    n_t1, n_t2a, n_t2b = 9 * 1000, 300, 384
    hours['T1'] = sum(float(np.mean(v)) * n_t1 for (p, _), v in per.items() if p == 'T1') / 3600
    hours['T2A'] = sum(float(np.mean(v)) * n_t2a for (p, _), v in per.items() if p == 'T2A') / 3600
    hours['T2B'] = sum(float(np.mean(v)) * n_t2b for (p, _), v in per.items() if p == 'T2B') / 3600
    # a part that was not smoked contributes 0 here; the caller combines
    cost = json.loads(cp.read_text()) if cp.is_file() else {'t': {}, 'n': {}}
    rec = {'per_cell_s_per_clip': {f'{p}:{c}': float(np.mean(v)) for (p, c), v in per.items()},
           'per_scorer_s_per_clip': {k: cost['t'][k] / cost['n'][k] for k in cost['t'] if cost['n'].get(k)},
           'projected_hours_by_part': hours, 'projected_total_hours': sum(hours.values()), 'hard_cap_hours': 16,
           'note': 'full = T1 9 bases x 1000 x 15 cells, T2A 300 x 13, T2B 384; smoke includes warm-up and tiny batches, so over-estimates'}
    P.jdump(OUT / 'projection.json', rec)
    log('PROJECTION by part (h): ' + json.dumps({k: round(v, 2) for k, v in hours.items()}) +
        f'; total {rec["projected_total_hours"]:.1f} h')
    return rec


# ═════════ summary / report ═════════
def expected_cells(smoke):
    ids_mc = get_ids(smoke)
    out = {}
    for b in bases_of(smoke):
        for c in T1_CELLS:
            out[f'T1_{b}_{c}'] = (ids_mc, True, True)
    with open(C.EXCERPTS) as f:
        C.EXC = json.load(f)
    a_ids = C.b_ids(smoke)
    for c in T2A_CELLS:
        out[f'T2A_{c}'] = (a_ids, False, False)
    out['T2B_all'] = (b_ids(), False, False)
    return out


def summarize(smoke, part='all'):
    setup(smoke)
    exp = expected_cells(smoke)
    cells = {}
    for k, (ids, clap, caps) in exp.items():
        if part != 'all' and not k.startswith(part):
            continue
        p = OUT / 'per_clip' / f'{k}.tsv'
        if p.is_file():
            rows = M.tsv_read(p)
            if M.rows_ok(rows, ids, Q.req_cols(2, caps, clap)):
                cells[k] = {c: float(np.nanmean(M.arr(rows, ids, c))) for c in
                            ('A1_MI', 'A3a_MI', 'PQ', 'CE', 'CU', 'PC', 'CLAP', 'PAM', 'CMI_q') if np.isfinite(M.arr(rows, ids, c)).any()}
                cells[k]['n'] = len(ids)
    rec = {'kind': KIND, 'smoke': smoke, 'seed': SEED, 'NB': NB, 'part': part, 'n_cells_expected': len(exp),
           'n_cells_scored': len(cells), 'cell_means': cells,
           'note': 'quick means only; statistics (bootstrap, markers, verdicts) are in attack_metric_matrix_tables.py'}
    P.jdump(OUT / 'summary.json', rec)
    status = 'complete' if len(cells) == len(exp) else 'partial'
    P.jdump(OUT / 'REPORT.json', {'kind': 'attack_metric_matrix_197_report_v1', 'status': status, 'smoke': smoke,
                                  'cells_scored': len(cells), 'cells_expected': len(exp),
                                  'finished': time.strftime('%FT%T%z')})
    log(f'summary written: {len(cells)}/{len(exp)} cells ({status})')
    return rec


def validate(smoke, part='all'):
    errs = []
    try:
        setup(smoke)
        exp = expected_cells(smoke)
        for k, (ids, clap, caps) in exp.items():
            if part != 'all' and not k.startswith(part):
                continue
            p = OUT / 'per_clip' / f'{k}.tsv'
            if not p.is_file():
                errs.append(f'missing cell {p.name}')
            elif not M.rows_ok(M.tsv_read(p), ids, Q.req_cols(2, caps, clap)):
                errs.append(f'cell {p.name}: id set or required columns not finite')
        for f in ('summary.json', 'REPORT.json'):
            if not (OUT / f).is_file():
                errs.append(f'missing {f}')
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs[:30]:
        log(f'VALIDATE FAIL {e}')
    if errs:
        log(f'VALIDATE FAIL {len(errs)} problem(s)')
        return 1
    log('VALIDATE OK')
    return 0


# ═════════ stimulus sanity check (no GPU) ═════════
def lufs(w, sr=SR):
    import pyloudnorm
    return float(pyloudnorm.Meter(sr).integrated_loudness(np.asarray(w, np.float64)[:, None]))


def write_smf(path, notes):
    """minimal format-0 SMF, 480 tpq, 120 BPM (1 s = 960 ticks)."""
    def vlq(n):
        b = [n & 0x7F]
        n >>= 7
        while n:
            b.append((n & 0x7F) | 0x80)
            n >>= 7
        return bytes(reversed(b))
    ev = []
    for on, off, p, v in notes:
        ev += [(int(round(on * 960)), 1, p, v), (int(round(off * 960)), 0, p, 0)]
    ev.sort()
    tr, cur = bytes([0, 0xFF, 0x51, 3, 0x07, 0xA1, 0x20]), 0
    for t, on, p, v in ev:
        tr += vlq(t - cur) + bytes([0x90 if on else 0x80, p, v])
        cur = t
    tr += bytes([0, 0xFF, 0x2F, 0])
    Path(path).write_bytes(b'MThd' + (6).to_bytes(4, 'big') + (0).to_bytes(2, 'big') + (1).to_bytes(2, 'big') +
                           (480).to_bytes(2, 'big') + b'MTrk' + len(tr).to_bytes(4, 'big') + tr)


MIDI_READ = """
import sys, json, pretty_midi
pm = pretty_midi.PrettyMIDI(sys.argv[1])
n = sorted(pm.instruments[0].notes, key=lambda x: (x.start, x.pitch))
print(json.dumps([[round(x.start, 3), round(x.end, 3), x.pitch, x.velocity] for x in n]))
"""


def readback(notes, tag):
    p = OUT / 'midi_check' / f'{tag}.mid'
    p.parent.mkdir(parents=True, exist_ok=True)
    write_smf(p, notes)
    r = subprocess.run([str(KJ_PY), '-c', MIDI_READ, str(p)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def stim_check(smoke):
    """rebuild one id of every T1 cell and verify LUFS / length / structure; render T2B probes and read them back with pretty_midi."""
    setup(smoke)
    with open(C.EXCERPTS) as f:
        C.EXC = json.load(f)
    ids = get_ids(smoke)
    pool = mp.get_context('fork').Pool(4)
    build_pno(pool)
    rec = {'t1': {}, 't2b': {}, 't2a': {}}
    i0 = ids[0]
    for base in bases_of(smoke)[:2]:
        wavs = M.load_base(base, ids[:2])
        w = wavs[i0]
        for c in T1_CELLS:
            y = t1_stim(c, w, i0, base)
            r = {'len_ratio': len(y) / len(w), 'lufs': lufs(y), 'peak': float(np.max(np.abs(y))), 'finite': bool(np.isfinite(y).all()),
                 'dtype': str(y.dtype)}
            if c == 'sil05':
                r['first_0.5s_max_abs'] = float(np.max(np.abs(y[:8000])))
            if c in OVL:
                ov = overlay(c, i0)
                thr = 0.2 * np.max(np.abs(ov))
                on = [k for k in range(1, len(ov)) if abs(ov[k]) >= thr > abs(ov[k - 1])]
                gaps = np.diff([k / SR for k in on]) if len(on) > 1 else []
                r['overlay_onsets_n'] = len(on)
                r['overlay_gap_s_unique'] = sorted({round(float(g), 3) for g in gaps})[:6] if len(gaps) else []
            rec['t1'][f'{base}:{c}'] = r
        if base == 'real':
            for c in OVL:
                ov = overlay(c, i0)
                o = np.asarray(ov, np.float64) * 10 ** ((-30.0 - OVL[c] - lufs(ov)) / 20)
                rec['t1'][f'{base}:{c}']['realised_overlay_lufs'] = lufs(o)
                rec['t1'][f'{base}:{c}']['realised_LU_diff'] = lufs(w) - lufs(o)
    pool.close()
    # T2B read-back
    for sid in ['C_maj_o4_up', 'C_maj_o4_rep', 'C_maj_o4_down', 'C_maj_o4_block', 'C_maj_o5_up', 'C_min_o4_up']:
        t, m, o, v = sid.split('_')
        nts = b_notes(t, m, o, v)
        back = readback(nts, sid)
        rec['t2b'][sid] = {'first_chord_notes': [(x[0], x[2]) for x in back if x[0] < 1.0][:8], 'n_notes': len(back),
                           'last_end_s': max(x[1] for x in back)}
    o4 = readback(b_notes('C', 'maj', 'o4', 'up'), 'o4u')
    o5 = readback(b_notes('C', 'maj', 'o5', 'up'), 'o5u')
    rec['t2b']['o5_minus_o4_semitones_unique'] = sorted({b[2] - a[2] for a, b in zip(o4, o5)})
    full = [f'{t}_{m}_{o}_{v}_{b}' for t in TONICS for m in MODES for o in OCTS for v in VOICES for b in BANKS]
    rec['t2b']['full_n_segments'] = len(full)
    rec['t2b']['full_n_unique'] = len(set(full))
    sigs = {(t, m, o, v): tuple(b_notes(t, m, o, v)) for t in TONICS for m in MODES for o in OCTS for v in VOICES}
    rec['t2b']['n_unique_note_sequences_of_192'] = len(set(sigs.values()))
    # T2A tempo length / window fill
    for cell in ('base', 'tempo_090', 'tempo_110', 'tr_p1', 'oct_p1', 'drop_30'):
        ns = a_notes(0, cell)
        rec['t2a'][cell] = {'n_notes': len(ns), 'last_onset_s': max(n[0] for n in ns), 'first_pitch': ns[0][2]}
    P.jdump(OUT / 'stim_check.json', rec)
    log('stim_check written')
    print(json.dumps(rec, indent=1))
    return 0


def save_examples(smoke):
    """first 5 clips of every T1 cell for the first 3 bases -> examples/ (wav float32; cheap, no GPU)."""
    import soundfile as sf
    setup(smoke)
    with open(C.EXCERPTS) as f:
        C.EXC = json.load(f)
    pool = mp.get_context('fork').Pool(4)
    build_pno(pool)
    pool.close()
    ids = get_ids(smoke)[:5]
    for base in bases_of(smoke)[:3]:
        wavs = M.load_base(base, ids)
        for c in T1_CELLS:
            for i in ids:
                p = OUT / 'examples' / f'T1_{base}_{c}_{i}.wav'
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), t1_stim(c, wavs[i], i, base), SR, subtype='FLOAT')
    log('examples written')
    return 0


# ═════════ preflight ═════════
def preflight(smoke, part):
    errs = []
    if shutil.disk_usage(H.EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    for p in (H.PARTB / 'part_b_selection.json', H.TTM115, H.MC_TSV, P.REF_DIR, M.SE_CKPT, H.PY, C.EXCERPTS, KJ_PY):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    if shutil.which('ffmpeg') is None:
        errs.append('ffmpeg not found')
    try:
        mc = get_ids(smoke)
        want = 8 if smoke else 1000
        if len(mc) != want:
            errs.append(f'{len(mc)} Part B ids with reference audio, expected {want}')
        for b in (bases_of(smoke) if smoke else BASES):
            if b == 'real':
                continue
            miss = [i for i in mc if not M.base_path(b, i).is_file()]
            if miss:
                errs.append(f'base {b}: {len(miss)} missing audio files, e.g. {miss[0]}')
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
        need_gpu = (14 if smoke else 17) * 1024        # full: same scorer stack as 195/196, nvidia-smi peak ~15.6 GiB (D14)
        if f < need_gpu:
            log(f'PREFLIGHT WAIT GPU free {f} MiB < {need_gpu} MiB')
            return 75
    return 0


def main():
    global OUT, IS_SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--part', default='all', choices=['all', 'T1', 'T2A', 'T2B'])
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--stim-check', action='store_true')
    ap.add_argument('--save-examples', action='store_true')
    ap.add_argument('--root', help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.chdir(REPO)
    if a.smoke:
        IS_SMOKE = True
        OUT = H.EVAL_ROOT / 'attack_metric_matrix_197_smoke'
    if a.root:
        OUT = Path(a.root)
    if a.preflight:
        return preflight(a.smoke, a.part)
    if a.stim_check:
        return stim_check(a.smoke)
    if a.save_examples:
        return save_examples(a.smoke)
    if a.validate_only:
        return validate(a.smoke, a.part)
    if a.summary_only:
        summarize(a.smoke, a.part)
        return validate(a.smoke, a.part)
    rc = preflight(a.smoke, a.part)
    if rc:
        return rc
    run_all(a.smoke, a.part)
    summarize(a.smoke, a.part)
    return validate(a.smoke, a.part)


if __name__ == '__main__':
    sys.exit(main())
