#!/usr/bin/env python3
"""191 Part A: what does AES CE reward beyond human CE? (2026-11-11, post hoc, descriptive, CPU)

Residual e = AES CE - (a + b * human CE)  (OLS on each set), regressed on simple audio features:
onset rate, loudness, bandwidth, percussive share (HPSS) and htdemucs stem energy shares
(from ce_human_residual_stems_20261111.py).  Sets: MusicCaps 522 real clips, PAM generated 400
(pooled and within system), PAM real 100.  PQ is run alongside as a contrast axis.
Reads existing AES scores and public human ratings; scores nothing.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_v] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
EV = REPO / 'research/eval'
CACHE = Path.home() / 'eval_output_nvme/ce_human_residual_20261111'
OUT = REPO / 'docs/experiments/results/ce_human_residual_20261111_summary.json'
RNG = np.random.default_rng(20261111)
NB = 2000
FEATS = ['onset_rate', 'lufs', 'rolloff95', 'hf_db', 'perc_share', 'flatness',
         'frac_drums', 'frac_bass', 'frac_vocals']


def feat(a):
    import librosa
    import soundfile as sf
    k, f = a
    x, sr = sf.read(f, dtype='float32', always_2d=True)
    x = x.mean(1)
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512)) ** 2
    fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    roll = float(np.median(librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=sr, roll_percent=0.95)))
    p = S.sum(1)
    hf = float(10 * np.log10((p[fr >= 4000].sum() + 1e-12) / (p.sum() + 1e-12)))
    flat = float(np.median(librosa.feature.spectral_flatness(S=np.sqrt(S))))
    y = librosa.resample(x, orig_sr=sr, target_sr=22050)
    on = librosa.onset.onset_detect(y=y, sr=22050, hop_length=512)
    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(D)
    ps = float((np.abs(P) ** 2).sum() / ((np.abs(P) ** 2).sum() + (np.abs(H) ** 2).sum() + 1e-12))
    return k, len(on) / (len(y) / 22050), roll, hf, ps, flat, len(x) / sr


def features(name, paths):
    dst = CACHE / f'feat_{name}.tsv'
    if not dst.exists():
        with Pool(4) as pool:
            r = pool.map(feat, sorted(paths.items()), chunksize=8)
        pd.DataFrame(r, columns=['key', 'onset_rate', 'rolloff95', 'hf_db', 'perc_share', 'flatness', 'dur']
                     ).to_csv(dst, sep='\t', index=False)
    f = pd.read_csv(dst, sep='\t').set_index('key')
    s = pd.read_csv(CACHE / f'stems_{name}.tsv', sep='\t').set_index('key')
    return f.join(s)


def z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-12)


def ols(X, y):
    X1 = np.c_[np.ones(len(X)), X]
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    return b, y - X1 @ b


def analyse(d, axis, group=None):
    """d: frame with human_<axis>, raw_<axis>, FEATS; group: column to demean within (system)"""
    d = d.dropna(subset=FEATS + [f'human_{axis}', f'raw_{axis}'])
    h, a = d[f'human_{axis}'].values, d[f'raw_{axis}'].values
    X = d[FEATS].values.astype(float)
    if group is not None:
        g = d[group].values
        dm = lambda v: v - pd.Series(v).groupby(g).transform('mean').values
        h, a = dm(h), dm(a)
        X = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    (_, slope), e = ols(h[:, None], a)
    n = len(d)
    idx = RNG.integers(0, n, (NB, n))
    o = {'n': int(n), 'r_aes_human': float(np.corrcoef(a, h)[0, 1]), 'slope_aes_on_human': float(slope),
         'resid_sd': float(e.std()), 'features': {}}
    Xz, ez = np.column_stack([z(X[:, j]) for j in range(X.shape[1])]), z(e)
    beta, r = ols(Xz, ez)
    o['R2_resid_on_features'] = float(1 - r.var() / ez.var())
    bb = np.array([ols(Xz[i], ez[i])[0][1:] for i in idx[:500]])
    # 5-fold CV R2 (no leakage of the feature fit)
    perm, pred = RNG.permutation(n), np.zeros(n)
    for k in range(5):
        te = perm[k::5]; tr = np.setdiff1d(perm, te)
        b, _ = ols(Xz[tr], ez[tr]); pred[te] = np.c_[np.ones(len(te)), Xz[te]] @ b
    o['R2_cv5'] = float(1 - ((ez - pred) ** 2).mean() / ez.var())
    for j, f in enumerate(FEATS):
        rho = lambda i: stats.spearmanr(X[i, j], e[i])[0]
        bs = np.array([rho(i) for i in idx])
        o['features'][f] = {
            'rho_resid': float(stats.spearmanr(X[:, j], e)[0]),
            'rho_resid_ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            'rho_human': float(stats.spearmanr(X[:, j], h)[0]),
            'rho_aes': float(stats.spearmanr(X[:, j], a)[0]),
            'beta': float(beta[j + 1]),
            'beta_ci': [float(np.percentile(bb[:, j], 2.5)), float(np.percentile(bb[:, j], 97.5))],
            # AES-score change for a 1 SD move in the feature, at fixed human score
            'aes_per_sd': float(beta[j + 1] * e.std())}
    # onset-rate quintiles: mean residual, to see shape
    q = pd.qcut(X[:, 0], 5, labels=False, duplicates='drop')
    o['onset_quintile_resid'] = [round(float(e[q == k].mean()), 3) for k in sorted(set(q))]
    o['onset_quintile_edges'] = [round(float(v), 2) for v in np.quantile(X[:, 0], [0, .2, .4, .6, .8, 1])]
    return o


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ce_human_residual_stems_20261111 import items  # noqa: F401  (path resolution only)
    it = items()
    mc = pd.read_csv(EV / 'output/aes_human_corr/per_clip.tsv', sep='\t').rename(columns={'ytid': 'key'}).set_index('key')
    pam = pd.read_csv(EV / 'output/aes_human_corr_pam/per_clip.tsv', sep='\t').set_index('key')
    mc = mc.join(features('mc', it['mc'])); pam = pam.join(features('pam', it['pam']))
    gen, real = pam[pam.system != 'real'], pam[pam.system == 'real']
    res = {'features': FEATS, 'note': 'e = AES - OLS(human); rho = Spearman; beta = standardised multiple regression of e'}
    for ax in ('CE', 'PQ'):
        res[ax] = {'musiccaps522': analyse(mc, ax), 'pam_gen400': analyse(gen, ax),
                   'pam_gen400_within_system': analyse(gen, ax, 'system'), 'pam_real100': analyse(real, ax)}
    res['feature_means'] = {'musiccaps522': mc[FEATS].mean().round(4).to_dict(),
                            **{s: g[FEATS].mean().round(4).to_dict() for s, g in pam.groupby('system')}}
    res['system_resid_CE'] = {}
    g = gen.dropna(subset=['human_CE', 'raw_CE'])
    _, e = ols(g.human_CE.values[:, None], g.raw_CE.values)
    res['system_resid_CE'] = pd.Series(e, index=g.index).groupby(g.system).mean().round(3).to_dict()
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for ax in ('CE', 'PQ'):
        for s, o in res[ax].items():
            print(f"== {ax} {s} n={o['n']} r={o['r_aes_human']:.3f} resid_sd={o['resid_sd']:.3f} "
                  f"R2={o['R2_resid_on_features']:.3f} cvR2={o['R2_cv5']:.3f} onsetQ={o['onset_quintile_resid']}")
            for f, v in o['features'].items():
                print(f"   {f:12s} rho_e {v['rho_resid']:+.3f} [{v['rho_resid_ci'][0]:+.3f},{v['rho_resid_ci'][1]:+.3f}] "
                      f"rho_h {v['rho_human']:+.3f} rho_aes {v['rho_aes']:+.3f} beta {v['beta']:+.3f} "
                      f"[{v['beta_ci'][0]:+.3f},{v['beta_ci'][1]:+.3f}] aes/sd {v['aes_per_sd']:+.3f}")
    print(json.dumps(res['system_resid_CE']), json.dumps(res['feature_means'], indent=0))


if __name__ == '__main__':
    main()
