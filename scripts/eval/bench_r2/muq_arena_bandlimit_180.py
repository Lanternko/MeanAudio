#!/usr/bin/env python3
"""180: Music Arena re-scored by MuQ-Eval with the band above 8 kHz removed (docs/experiments/muq_arena_bandlimit_180_20261105.md).

  python muq_arena_bandlimit_180.py score     # GPU, resumable (rows keyed by unit_id × cond)
  python muq_arena_bandlimit_180.py analyze   # CPU: agreement vs human votes, paired bootstrap

Input = r2 arena native_raw decoded wavs (same audio as the r2 primary cell).
Conditions: F0 identity / sr16k soxr HQ (= 11-03 F1) / lp8k Butterworth-8 sosfiltfilt at native sr.
"""
import csv
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # scripts/eval
_sp = importlib.util.spec_from_file_location('analyze_r2', HERE / 'analyze_r2.py')
A = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(A)
D = A.D

OUT = Path.home() / 'nvme_experiment_artifacts/meanaudio/muq_arena_bandlimit_180'
CONDS = ('F0', 'sr16k', 'lp8k')
COLS = ('A1_MI', 'A1_TA', 'A3a_MI', 'A3a_TA')
SEED, NBOOT = 20261105, 2000


def log(*a):
    print(time.strftime('%T'), *a, flush=True)


def units():
    with open(D.DER / 'arena/index.tsv', newline='') as f:
        rows = [r for r in csv.DictReader(f, delimiter='\t') if r['loud'] == 'raw' and r['variant'] == 'native']
    return [(r['unit_id'].split('|')[1], D.DER / 'arena' / r['files']) for r in rows if not r['invalid']]


def band_fracs(m, sr):
    """whole-excerpt energy fractions (dB) of 4-8k, 8-12k, >8k over the first 30 s."""
    import librosa
    x = m[:int(30 * sr)]
    S = np.abs(librosa.stft(x, n_fft=4096, hop_length=2048)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=4096)
    tot = S.sum() + 1e-20
    db = lambda lo, hi: float(10 * np.log10(S[(f >= lo) & (f < hi)].sum() / tot + 1e-12))
    return {'b4_8k': db(4000, 8000), 'b8_12k': db(8000, 12000), 'b8k_up': db(8000, sr / 2 + 1)}


def load(path):
    import soundfile as sf
    import soxr
    from muq_lowpass_probe_20261028 import lp
    y, sr = sf.read(str(path), dtype='float32', always_2d=True)
    m = y.mean(axis=1)
    out = {'F0': (m, sr), 'sr16k': (soxr.resample(m, sr, 16000, quality='HQ').astype(np.float32), 16000),
           'lp8k': (lp(m, sr, 8000.0), sr)}
    return out, dict(sr=sr, **band_fracs(m, sr))


def score():
    from newmetrics_muqeval import score_muqeval
    OUT.mkdir(parents=True, exist_ok=True)
    f_sc, f_bd = OUT / 'scores.tsv', OUT / 'band.tsv'
    done = set()
    if f_sc.exists():
        with open(f_sc, newline='') as f:
            done = {r['item_id'] for r in csv.DictReader(f, delimiter='\t')}
    us = [u for u in units() if u[0] not in done]
    log(f'{len(us)} todo, {len(done)} done')
    fs, fb = open(f_sc, 'a', newline=''), open(f_bd, 'a', newline='')
    ws = csv.writer(fs, delimiter='\t')
    wb = csv.writer(fb, delimiter='\t')
    if not done:
        ws.writerow(['item_id', 'cond', *COLS])
        wb.writerow(['item_id', 'sr', 'b4_8k', 'b8_12k', 'b8k_up'])
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        for k, ((item, _), (conds, bd)) in enumerate(zip(us, ex.map(lambda u: load(u[1]), us))):
            # one item = all conds written together, so resume is item-granular
            rows = []
            for c in CONDS:
                y, sr = conds[c]
                s = {f'{v}_{h}': x for v in ('A1', 'A3a') for h, x in score_muqeval(y, sr, variant=v).items()}
                rows.append([item, c, *(f'{s[k2]:.6f}' for k2 in COLS)])
            ws.writerows(rows)
            wb.writerow([item, bd['sr'], *(f'{bd[k2]:.3f}' for k2 in ('b4_8k', 'b8_12k', 'b8k_up'))])
            if k % 100 == 0:
                fs.flush(); fb.flush()
                log(f'{k}/{len(us)} {time.time() - t0:.0f}s')
    fs.close(); fb.close()
    log(f'done {time.time() - t0:.0f}s')


def analyze():
    from scipy import stats
    sc = {}
    with open(OUT / 'scores.tsv', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            sc.setdefault(r['cond'], {}).setdefault(r['item_id'], {c: float(r[c]) for c in COLS})
    with open(OUT / 'band.tsv', newline='') as f:
        band = {r['item_id']: {k: float(v) for k, v in r.items() if k != 'item_id'} for r in csv.DictReader(f, delimiter='\t')}
    isys = {it['item_id']: it['group'] for it in D.read_manifest('arena')}
    pairs = A.arena_pairs()
    dec = [p for p in pairs if p['win'] in (0, 1) and all(p['a'] in sc[c] and p['b'] in sc[c] for c in CONDS)]
    h = np.array([1.0 if p['win'] == 0 else -1.0 for p in dec])
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, len(dec), (NBOOT, len(dec)))  # one battle = one decided vote = one cluster
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    # r2 BT on human votes (same as analyze_r2)
    bt = A.bradley_terry(sorted({isys[p['a']] for p in dec} | {isys[p['b']] for p in dec}),
                         [((p['sys_a'], p['sys_b']) if p['win'] == 0 else (p['sys_b'], p['sys_a'])) for p in dec])
    wide = np.array([abs(band[p['a']]['b8k_up'] - band[p['b']]['b8k_up']) >= 6 for p in dec])
    # +1 if A is the wider-band side
    a_wider = np.array([1.0 if band[p['a']]['b8k_up'] > band[p['b']]['b8k_up'] else -1.0 for p in dec])
    res = {'n_decided': len(dec), 'n_wide_gap': int(wide.sum()),
           'human_prefers_wider_in_wide_gap': float((a_wider[wide] == h[wide]).mean()), 'cols': {}}
    for col in COLS:
        per = {}
        forced = {}
        for c in CONDS:
            d = np.array([sc[c][p['a']][col] - sc[c][p['b']][col] for p in dec])
            vals = [v[col] for v in sc[c].values()]
            m = 0.10 * (max(vals) - min(vals))
            fz = np.where(d == 0, 0.5, (np.sign(d) == h).astype(float))
            am = np.where(np.abs(d) < m, 0.5, (np.sign(d) == h).astype(float))
            forced[c] = fz
            sysm = {}
            for i, v in sc[c].items():
                sysm.setdefault(isys[i], []).append(v[col])
            sysm = {k: float(np.mean(v)) for k, v in sysm.items()}
            common = sorted(set(sysm) & set(bt))
            per[c] = dict(forced=float(fz.mean()), forced_ci=ci(fz[bidx].mean(1)), agree_m=float(am.mean()), m=float(m),
                          forced_wide=float(fz[wide].mean()), forced_narrow=float(fz[~wide].mean()),
                          bt_tau=float(stats.kendalltau([sysm[s] for s in common], [bt[s] for s in common])[0]),
                          sys_mean=sysm)
        for c in ('sr16k', 'lp8k'):
            dd = forced[c] - forced['F0']
            per[c]['delta_forced'] = float(dd.mean())
            per[c]['delta_forced_ci'] = ci(dd[bidx].mean(1))
            per[c]['delta_forced_wide'] = float(dd[wide].mean())
            per[c]['delta_forced_wide_ci'] = ci(dd[wide][rng.integers(0, wide.sum(), (NBOOT, wide.sum()))].mean(1))
            per[c]['n_flipped'] = int((forced[c] != forced['F0']).sum())
            # per-system mean Δ and its relation to the system's >8k energy
            dsys, esys = {}, {}
            for i in sc['F0']:
                dsys.setdefault(isys[i], []).append(sc[c][i][col] - sc['F0'][i][col])
                esys.setdefault(isys[i], []).append(band[i]['b8k_up'])
            dsys = {k: float(np.mean(v)) for k, v in dsys.items()}
            esys = {k: float(np.median(v)) for k, v in esys.items()}
            ks = sorted(dsys)
            per[c]['sys_delta'] = dsys
            per[c]['sys_delta_vs_8kup_spearman'] = float(stats.spearmanr([dsys[k] for k in ks], [esys[k] for k in ks])[0])
            per[c]['sys_rank_spearman_vs_F0'] = float(stats.spearmanr([per['F0']['sys_mean'][k] for k in ks],
                                                                     [per[c]['sys_mean'][k] for k in ks])[0])
        res['cols'][col] = per
    res['sys_8kup_median_db'] = {k: float(np.median([band[i]['b8k_up'] for i in sc['F0'] if isys[i] == k]))
                                 for k in sorted(set(isys.values()))}
    # sanity: F0 must reproduce r2 stored scores
    r2 = A.load_scores('arena')[('muqeval_A1', 'native', 'raw')]
    diffs = [abs(sc['F0'][i]['A1_MI'] - r2[i]['muqeval_A1_MI']) for i in sc['F0'] if i in r2]
    res['F0_vs_r2_A1_MI_maxabs'] = float(max(diffs))
    res['F0_vs_r2_n'] = len(diffs)
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for col in COLS:
        p = res['cols'][col]
        log(col, ' '.join(f"{c}:{p[c]['forced']:.3f}" for c in CONDS),
            ' '.join(f"Δ{c}={p[c]['delta_forced']:+.4f}{[round(x, 4) for x in p[c]['delta_forced_ci']]}"
                     f" wide={p[c]['delta_forced_wide']:+.4f} flip={p[c]['n_flipped']}" for c in ('sr16k', 'lp8k')),
            f"tau {p['F0']['bt_tau']:.3f}/{p['sr16k']['bt_tau']:.3f}/{p['lp8k']['bt_tau']:.3f}")
    log('F0 vs r2 max|Δ|', res['F0_vs_r2_A1_MI_maxabs'], 'n', res['F0_vs_r2_n'], 'wide pairs', res['n_wide_gap'])


if __name__ == '__main__':
    {'score': score, 'analyze': analyze}[sys.argv[1]]()
