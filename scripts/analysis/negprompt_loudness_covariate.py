"""Cell-level regression of negative-prompt PQ gain on the loudness confound.

The negprompt ablation flagged `silence` and `fidelity_short` as loudness-
confounded: they raise PQ but also push RMS up several dB and crest below the
level at which pure CFG @ 4.5 was judged distorted. That flag is qualitative --
it says a confound is present, not how many PQ points it is worth.

This estimates the worth. Across every scored cell we regress the paired PQ
delta on the signal-level shifts (RMS, crest, spectral centroid) that the same
cell produced, and read off:

  * the slope -- how many PQ points ride on one dB of loudness shift;
  * R^2 -- how much of the between-cell PQ spread the confound alone explains;
  * per-cell residuals -- the PQ gain that survives after the loudness path is
    subtracted, which is the quantity the `fidelity` claim actually needs.

WHAT THIS IS NOT. It does not cut the loudness path; it prices it. And it is an
association across cells, not a causal estimate: whatever makes a cell louder
may also make it better by some route this model cannot see, in which case the
slope absorbs that route too and the residuals are conservative (too small).
That direction of error is the safe one for a "the gain is not loudness" claim
and the unsafe one for the reverse, so read a large positive residual as
evidence and a small one as inconclusive.

LIMITS baked into the data, not fixable here:
  * n is the number of cells (~36), not clips. negprompt_ablation_matrix.py
    deletes the audio after scoring, so per-clip loudness is unrecoverable for
    cells already run. Cells are also nested (shared subset, a cfg ladder within
    each arm), so they are not independent and the p-values are optimistic.
  * signal stats come from the first 256 of 1024 clips, sorted by filename.
  * Audiobox PQ is not level-invariant in practice; that premise is what makes
    the whole question live.

Usage:  python scripts/analysis/negprompt_loudness_covariate.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ABL = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation')

PREDICTORS = [
    ('d_rms_db', 'RMS shift (dB)'),
    ('d_crest', 'crest_mean shift'),
    ('d_centroid', 'centroid shift (Hz)'),
]


def load_cells():
    """Return baseline-relative records, one per scored non-cfg0 cell."""
    raw = {}
    for path in sorted(ABL.glob('*.json')):
        d = json.loads(path.read_text())
        raw[d['label']] = d

    base = {}
    for label, d in raw.items():
        if d['cfg_strength'] == 0.0:
            base[d['arm']] = d

    cells = []
    for label, d in sorted(raw.items()):
        if d['cfg_strength'] == 0.0:
            continue
        b = base.get(d['arm'])
        if b is None or 'paired_delta_vs_cfg0' not in d:
            continue
        s, sb = d['signal_stats'], b['signal_stats']
        cells.append({
            'label': label,
            'arm': d['arm'],
            'cfg': d['cfg_strength'],
            'neg': d['negative_key'],
            'd_pq': d['paired_delta_vs_cfg0']['PQ']['mean_delta'],
            'd_clap': d['paired_delta_vs_cfg0']['clap']['mean_delta'],
            'd_rms_db': s['rms_db_mean'] - sb['rms_db_mean'],
            'd_crest': s['crest_mean'] - sb['crest_mean'],
            'd_centroid': s['spectral_centroid_hz_mean'] - sb['spectral_centroid_hz_mean'],
            'crest_min': s['crest_min'],
        })
    return cells


def ols(y, X, names):
    """Plain least squares with an intercept. Returns coefs, SEs, t, p, R^2."""
    n = len(y)
    A = np.column_stack([np.ones(n)] + [np.asarray(x, float) for x in X])
    k = A.shape[1]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = n - k
    if dof <= 0:
        return coef, None, None, None, None, resid
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    t = coef / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot if ss_tot > 0 else float('nan')
    return coef, se, t, p, r2, resid


def fit_report(cells, keys, title):
    y = np.array([c['d_pq'] for c in cells])
    X = [[c[k] for c in cells] for k in keys]
    names = ['intercept'] + keys
    coef, se, t, p, r2, resid = ols(y, X, names)
    print(f'\n  {title}   (n={len(y)})')
    if se is None:
        print('    too few cells for inference')
        return resid
    print(f'    {"term":<14}{"coef":>10}{"se":>10}{"t":>8}{"p":>10}')
    for i, nm in enumerate(names):
        print(f'    {nm:<14}{coef[i]:>10.4f}{se[i]:>10.4f}{t[i]:>8.2f}{p[i]:>10.4f}')
    print(f'    R^2 = {r2:.3f}    residual sd = {resid.std(ddof=1):.4f}')
    return resid


def main():
    cells = load_cells()
    print('=' * 78)
    print('Negative-prompt PQ gain vs loudness confound -- cell-level regression')
    print('=' * 78)
    print(f'\ncells: {len(cells)}   arms: {sorted({c["arm"] for c in cells})}')

    # ---- univariate association, pooled and per arm ----------------------
    print('\n' + '-' * 78)
    print('1. Univariate association of dPQ with each signal shift')
    print('-' * 78)
    groups = [('pooled', cells)]
    for arm in sorted({c['arm'] for c in cells}):
        groups.append((arm, [c for c in cells if c['arm'] == arm]))
    for gname, g in groups:
        y = np.array([c['d_pq'] for c in g])
        print(f'\n  {gname}  (n={len(g)})')
        for key, lab in PREDICTORS:
            x = np.array([c[key] for c in g])
            lr = stats.linregress(x, y)
            rho, prho = stats.spearmanr(x, y)
            print(f'    {lab:<22} slope={lr.slope:+8.4f}  r={lr.rvalue:+.3f}'
                  f'  R^2={lr.rvalue ** 2:.3f}  p={lr.pvalue:.4f}'
                  f'  |  spearman={rho:+.3f} (p={prho:.4f})')

    # ---- multivariate ----------------------------------------------------
    print('\n' + '-' * 78)
    print('2. Multivariate models (pooled)')
    print('-' * 78)
    fit_report(cells, ['d_rms_db'], 'dPQ ~ dRMS')
    fit_report(cells, ['d_rms_db', 'd_crest'], 'dPQ ~ dRMS + dcrest')
    resid_full = fit_report(cells, ['d_rms_db', 'd_crest', 'd_centroid'],
                            'dPQ ~ dRMS + dcrest + dcentroid')

    # ---- per-cell residuals after pricing out loudness -------------------
    print('\n' + '-' * 78)
    print('3. Per-cell PQ gain, loudness-predicted part, and residual')
    print('   (model: dPQ ~ dRMS + dcrest, pooled -- the two confound axes)')
    print('-' * 78)
    y = np.array([c['d_pq'] for c in cells])
    X = [[c['d_rms_db'] for c in cells], [c['d_crest'] for c in cells]]
    coef, se, t, p, r2, resid = ols(y, X, ['d_rms_db', 'd_crest'])
    fitted = y - resid
    order = np.argsort(-y)
    print(f'\n  {"cell":<40}{"dPQ":>8}{"pred":>8}{"resid":>8}{"dRMS":>8}{"crestmin":>9}')
    for i in order:
        c = cells[i]
        print(f'  {c["label"]:<40}{c["d_pq"]:>8.3f}{fitted[i]:>8.3f}'
              f'{resid[i]:>8.3f}{c["d_rms_db"]:>8.2f}{c["crest_min"]:>9.2f}')

    # ---- the cfg 3.0 content ladder, which is what the doc claims on -----
    print('\n' + '-' * 78)
    print('4. cfg 3.0 content ladder on c2p0_slot0 (the doc\'s main table)')
    print('-' * 78)
    ladder = [(i, c) for i, c in enumerate(cells)
              if c['arm'] == 'c2p0_slot0' and c['cfg'] == 3.0]
    ladder.sort(key=lambda ic: -ic[1]['d_pq'])
    print(f'\n  {"negative":<18}{"dPQ":>8}{"pred":>8}{"resid":>8}'
          f'{"resid%":>9}{"dRMS":>8}{"crestmin":>9}')
    for i, c in ladder:
        pct = 100 * resid[i] / c['d_pq'] if c['d_pq'] else float('nan')
        print(f'  {c["neg"]:<18}{c["d_pq"]:>8.3f}{fitted[i]:>8.3f}'
              f'{resid[i]:>8.3f}{pct:>8.0f}%{c["d_rms_db"]:>8.2f}'
              f'{c["crest_min"]:>9.2f}')

    # ---- 5. is the slope causal? the `none` cells are the test -----------
    # Every `none` cell shifts loudness and crest with nothing in the negative
    # slot. If the dRMS slope were a real loudness->PQ path, those cells would
    # collect their predicted gain. Fit the confound axes on `none` cells ALONE,
    # where "has negative text" is held at zero by construction.
    print('\n' + '-' * 78)
    print('5. Pure-CFG cells only: does a loudness shift buy PQ with no text?')
    print('-' * 78)
    nones = [c for c in cells if c['neg'] == 'none']
    print(f'\n  {"cell":<36}{"dPQ":>8}{"dRMS":>8}{"dcrest":>9}{"crestmin":>9}')
    for c in sorted(nones, key=lambda c: -c['d_rms_db']):
        print(f'  {c["label"]:<36}{c["d_pq"]:>8.3f}{c["d_rms_db"]:>8.2f}'
              f'{c["d_crest"]:>9.3f}{c["crest_min"]:>9.2f}')
    yn = np.array([c['d_pq'] for c in nones])
    xn = np.array([c['d_rms_db'] for c in nones])
    lrn = stats.linregress(xn, yn)
    print(f'\n  dPQ ~ dRMS within `none` only (n={len(nones)}): '
          f'slope={lrn.slope:+.4f}  r={lrn.rvalue:+.3f}  p={lrn.pvalue:.4f}')
    print(f'  dRMS range covered here: {xn.min():+.2f} .. {xn.max():+.2f} dB '
          f'(vs silence at +4.77 dB -- extrapolation beyond this is unsupported)')

    # ---- 6. collinearity and the has-text indicator ----------------------
    print('\n' + '-' * 78)
    print('6. Collinearity, and what survives adding a has-negative-text term')
    print('-' * 78)
    rms = np.array([c['d_rms_db'] for c in cells])
    cre = np.array([c['d_crest'] for c in cells])
    r_rc = stats.pearsonr(rms, cre)
    print(f'\n  corr(dRMS, dcrest) = {r_rc[0]:+.3f}  (p={r_rc[1]:.4f})'
          '   -- strong negative => suppression, each masks the other alone')
    has_text = np.array([0.0 if c['neg'] == 'none' else 1.0 for c in cells])
    r_rt = stats.pearsonr(rms, has_text)
    print(f'  corr(dRMS, has_text) = {r_rt[0]:+.3f}  (p={r_rt[1]:.4f})'
          '   -- how far loudness and the thing we want are entangled')
    for c, h in zip(cells, has_text):
        c['has_text_x'] = h
    fit_report(cells, ['d_rms_db', 'd_crest', 'has_text_x'],
               'dPQ ~ dRMS + dcrest + has_negative_text')

    print('\n' + '-' * 78)
    print('Read the residual column as: PQ gain not attributable to the cell\'s')
    print('own loudness/crest shift. Positive and large => the gain survives')
    print('pricing out the confound. Near zero => the cell is explained by it.')
    print('Association across ~36 nested cells, not a causal estimate.')
    print('-' * 78)


if __name__ == '__main__':
    main()
