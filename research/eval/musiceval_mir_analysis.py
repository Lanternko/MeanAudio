"""Analysis stage of musiceval_mir_incremental.py (pre-registered in
docs/experiments/mir_incremental_validity_musiceval_20260930.md).

Main set = MusicEval shared set (25 systems x 100 prompts = 2,500 clips); the 248 demo
clips (own prompts) only enter the all-clip correlations.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from musiceval_mir_incremental import AXES, MIR, NUIS, ME_ROOT, HERE, MC_ARMS

B_BOOT = int(__import__("os").environ.get("B_BOOT", 2000))
# 'all' = every non-tied same-prompt pair (the gate); the |dhuman| bins are descriptive
BINS = [('all', 0.1, 99), ('<=0.4', 0.1, 0.5), ('0.6-0.8', 0.5, 0.9), ('>=1.0', 0.9, 99)]
MODELS = {'B': AXES + NUIS, 'F': AXES + NUIS + MIR, 'M': MIR + NUIS}
LINES = []


def say(s=''):
    print(s)
    LINES.append(s)


def ci(a):
    a = np.asarray(a)
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def fmt(v, c):
    return f'{v:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]'


# ── data ────────────────────────────────────────────────
def load(out):
    lab = pd.read_csv(ME_ROOT / 'sets' / 'total_mos_list.txt', header=None, names=['f', 'ovl', 'rel'])
    lab['key'] = lab.f.str[:-4]
    lab['sys'] = lab.f.str.extract(r'-(S\d+)_')[0]
    lab['prompt'] = lab.f.str.extract(r'_(P\d+)\.wav')[0]
    aes = pd.read_csv(out / 'musiceval_aes.tsv', sep='\t')
    mm = pd.read_csv(out / 'feat_madmom.tsv', sep='\t')
    es = pd.read_csv(out / 'feat_essentia.tsv', sep='\t')
    feat = mm.merge(es, on=['set', 'key'], how='outer')
    df = lab.merge(aes, on='key').merge(feat[feat.set == 'musiceval'].drop(columns='set'), on='key')
    assert len(df) == len(lab) == 2748, (len(df), len(lab))
    df['log_dur'] = np.log(df.dur)
    df['lufs'] = df.lufs.replace([np.inf, -np.inf], np.nan)
    n_per_sys = df.sys.value_counts()
    df['shared'] = df.sys.map(n_per_sys).eq(100)
    person = pd.read_csv(ME_ROOT / 'person_mos' / 'total_person_mos.txt', header=None,
                         names=['f', 'rater', 'ovl', 'rel'])
    return df, feat, person


def pipe(kind):
    if kind == 'ridge':
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 3, 13)))
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=20, random_state=0)


def oof(df, cols, y, groups, n_splits, kind='ridge'):
    X = df[cols].to_numpy(float)
    pred = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pred[te] = pipe(kind).fit(X[tr], y[tr]).predict(X[te])
    return pred


# ── noise ceiling ───────────────────────────────────────
def noise_ceiling(sh, person, target, rng):
    p = person[person.f.isin(sh.f)]
    wide = p.groupby('f')[target].apply(list)
    assert wide.map(len).eq(5).all(), wide.map(len).value_counts()
    R = np.stack(wide.loc[sh.f].to_numpy()).astype(float)  # (n, 5) in sh order
    full = R.mean(1)
    r23, agree = [], {b[0]: [] for b in BINS}
    pairs = same_prompt_pairs(sh)
    for _ in range(200):
        perm = rng.permuted(np.tile(np.arange(5), (len(R), 1)), axis=1)
        a = np.take_along_axis(R, perm[:, :2], 1).mean(1)
        b = np.take_along_axis(R, perm[:, 2:], 1).mean(1)
        r23.append(pearsonr(a, b)[0])
        for name, lo, hi in BINS:
            i, j = pairs[name]
            da, db = a[i] - a[j], b[i] - b[j]
            ok = (da != 0) & (db != 0)
            agree[name].append(np.mean(np.sign(da[ok]) == np.sign(db[ok])))
    r = float(np.mean(r23))
    # single-rater reliability rho with sqrt(R(2) R(3)) = r, R(k) = k rho / (1 + (k-1) rho)
    Rk = lambda k, rho: k * rho / (1 + (k - 1) * rho)
    grid = np.linspace(1e-4, 0.9999, 100000)
    rho = grid[np.argmin(np.abs(np.sqrt(Rk(2, grid) * Rk(3, grid)) - r))]
    return {'r_2v3': r, 'rho1': float(rho), 'R5': float(Rk(5, rho)), 'r_ceiling': float(np.sqrt(Rk(5, rho))),
            'pair_half_agree': {k: float(np.mean(v)) for k, v in agree.items()}}


def same_prompt_pairs(sh):
    """Index pairs (within prompt) binned by |human OVL diff| using the 5-rater mean."""
    y = sh.ovl.to_numpy()
    out = {b[0]: ([], []) for b in BINS}
    for _, idx in sh.groupby('prompt').indices.items():
        ii, jj = np.triu_indices(len(idx), 1)
        i, j = idx[ii], idx[jj]
        d = np.abs(y[i] - y[j])
        for name, lo, hi in BINS:
            m = (d > lo) & (d < hi)
            out[name][0].extend(i[m])
            out[name][1].extend(j[m])
    return {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}


# ── per-cluster sufficient statistics for bootstrap ─────
def cluster_stats(y, p, cl):
    d = pd.DataFrame({'c': cl, 'y': y, 'p': p, 'yy': y * y, 'pp': p * p, 'py': p * y,
                      'se': (y - p) ** 2, 'n': 1})
    return d.groupby('c')[['y', 'p', 'yy', 'pp', 'py', 'se', 'n']].sum()


def r2_r(s):
    n = s['n']
    sst = s['yy'] - s['y'] ** 2 / n
    cov = s['py'] - s['p'] * s['y'] / n
    vp = s['pp'] - s['p'] ** 2 / n
    return 1 - s['se'] / sst, cov / np.sqrt(sst * vp)


def pair_concord(sh, pred, pairs, prompt_idx):
    """Per-prompt concordant counts and totals for each bin."""
    y = sh.ovl.to_numpy()
    res = {}
    for name, (i, j) in pairs.items():
        c = (np.sign(pred[i] - pred[j]) == np.sign(y[i] - y[j])).astype(float)
        pr = prompt_idx[i]
        res[name] = (np.bincount(pr, c, minlength=prompt_idx.max() + 1),
                     np.bincount(pr, minlength=prompt_idx.max() + 1).astype(float))
    return res


def evaluate(sh, preds, target, cv_name, rng):
    y = sh[target].to_numpy()
    cl = sh.prompt.to_numpy() if cv_name == 'prompt' else sh.sys.to_numpy()
    stats = {m: cluster_stats(y, p, cl) for m, p in preds.items()}
    keys = stats['B'].index.to_numpy()
    point = {}
    for m, p in preds.items():
        sysm = pd.DataFrame({'s': sh.sys, 'y': y, 'p': p}).groupby('s').mean()
        point[m] = {'R2': float(r2_r(stats[m].sum())[0]), 'r': float(pearsonr(y, p)[0]),
                    'sys_spearman': float(spearmanr(sysm.y, sysm.p)[0])}
    pairs = same_prompt_pairs(sh) if target == 'ovl' else None
    pidx = pd.factorize(sh.prompt, sort=True)[0]
    conc = {m: pair_concord(sh, p, pairs, pidx) for m, p in preds.items()} if pairs else {}
    for m in preds:
        if conc:
            point[m]['pair_agree'] = {k: float(c.sum() / t.sum()) for k, (c, t) in conc[m].items()}
            point[m]['pair_n'] = {k: int(t.sum()) for k, (c, t) in conc[m].items()}
    # paired bootstrap F-B over clusters (prompts or systems)
    boot = {'dR2': [], 'dr': [], 'dsys': []}
    boot.update({f'dpair_{b[0]}': [] for b in BINS} if conc else {})
    sysm_all = pd.DataFrame({'s': sh.sys, 'y': y, 'B': preds['B'], 'F': preds['F']}).groupby('s').mean()
    for _ in range(B_BOOT):
        pick = rng.choice(len(keys), len(keys))
        sB, sF = stats['B'].iloc[pick].sum(), stats['F'].iloc[pick].sum()
        (r2B, rB), (r2F, rF) = r2_r(sB), r2_r(sF)
        boot['dR2'].append(r2F - r2B)
        boot['dr'].append(rF - rB)
        if cv_name == 'system':
            sm = sysm_all.loc[keys[pick]]
            boot['dsys'].append(spearmanr(sm.y, sm.F)[0] - spearmanr(sm.y, sm.B)[0])
        if conc:
            for name in conc['B']:
                cB, tB = conc['B'][name]
                cF, _ = conc['F'][name]
                # prompts are the resampling unit for pair agreement in both CV schemes
                pp = rng.choice(len(cB), len(cB)) if cv_name == 'system' else pick
                boot[f'dpair_{name}'].append(cF[pp].sum() / tB[pp].sum() - cB[pp].sum() / tB[pp].sum())
    delta = {'dR2': point['F']['R2'] - point['B']['R2'], 'dr': point['F']['r'] - point['B']['r'],
             'dsys': point['F']['sys_spearman'] - point['B']['sys_spearman']}
    if conc:
        for name in conc['B']:
            delta[f'dpair_{name}'] = point['F']['pair_agree'][name] - point['B']['pair_agree'][name]
    cis = {k: ci(v) for k, v in boot.items() if len(v)}
    return point, delta, cis


# ── per-feature ─────────────────────────────────────────
def resid(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    return y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]


def partial_r(d, f, target, ctrl, within_sys=False):
    d = d.dropna(subset=[f, target] + ctrl)
    cols = [f, target] + ctrl
    X = d[cols].to_numpy(float)
    if within_sys:
        X = X - d.groupby('sys')[cols].transform('mean').to_numpy(float)
    return pearsonr(resid(X[:, 0], X[:, 2:]), resid(X[:, 1], X[:, 2:]))[0]


def per_feature(df, sh, rng):
    ctrl = AXES + NUIS
    res = {}
    prompts = sh.prompt.unique()
    by_p = {p: g for p, g in sh.groupby('prompt')}
    boots = [pd.concat([by_p[p] for p in rng.choice(prompts, len(prompts))]) for _ in range(500)]
    for f in MIR:
        d = df.dropna(subset=[f])
        r_all = pearsonr(d[f], d.ovl)[0]
        pr = partial_r(sh, f, 'ovl', ctrl)
        pw = partial_r(sh, f, 'ovl', ctrl, within_sys=True)
        bp = [partial_r(b, f, 'ovl', ctrl) for b in boots]
        bw = [partial_r(b, f, 'ovl', ctrl, within_sys=True) for b in boots]
        res[f] = {'n_missing': int(df[f].isna().sum()), 'r_all': float(r_all),
                  'r_shared': float(pearsonr(sh.dropna(subset=[f])[f], sh.dropna(subset=[f]).ovl)[0]),
                  'partial': float(pr), 'partial_ci': ci(bp), 'within_sys_partial': float(pw),
                  'within_sys_partial_ci': ci(bw)}
    for a in AXES:
        res[a] = {'r_all': float(pearsonr(df[a], df.ovl)[0]),
                  'r_all_rel': float(pearsonr(df[a], df.rel)[0]),
                  'within_sys_r': float(pearsonr(df[a] - df.groupby('sys')[a].transform('mean'),
                                                 df.ovl - df.groupby('sys').ovl.transform('mean'))[0])}
    return res


# ── external sets ───────────────────────────────────────
def external(feat, me_partial):
    res = {}
    pam = pd.read_csv(HERE / 'output' / 'aes_human_corr_pam' / 'per_clip.tsv', sep='\t')
    pam = pam.rename(columns={f'raw_{a}': a for a in AXES}).merge(
        feat[feat.set == 'pam'].drop(columns='set'), on='key')
    pam = pam[pam.system != 'real']
    nat = pd.read_csv(HERE / 'output' / 'aes_human_corr' / 'per_clip.tsv', sep='\t')
    nat = nat.rename(columns={f'raw_{a}': a for a in AXES}).merge(
        feat[feat.set == 'aesnat'].drop(columns='set').rename(columns={'key': 'ytid'}), on='ytid')
    for name, d, targets in [('pam_gen', pam, ['pam_OVL', 'human_PQ', 'human_CE']),
                             ('aes_natural', nat, ['human_PQ', 'human_CE'])]:
        res[name] = {'n': len(d)}
        for t in targets:
            res[name][t] = {}
            for f in MIR:
                if d[f].notna().sum() < 50 or d[f].nunique() < 5:
                    continue
                dd = d.dropna(subset=[f])
                pr = pearsonr(resid(dd[f].to_numpy(float), dd[AXES + ['lufs']].to_numpy(float)),
                              resid(dd[t].to_numpy(float), dd[AXES + ['lufs']].to_numpy(float)))[0]
                res[name][t][f] = {'partial': float(pr), 'n': len(dd),
                                   'same_sign_as_musiceval': bool(np.sign(pr) == np.sign(me_partial[f]))}
    return res


# ── exploratory: arms vs real ───────────────────────────
def arms(feat):
    res = {}
    base = None
    for s, (_, _, mdir) in MC_ARMS.items():
        aes = pd.read_csv(mdir / 'per_clip.tsv', sep='\t').rename(columns={'id': 'key'})
        d = feat[feat.set == s].merge(aes[['key', 'PQ', 'CE', 'clap']], on='key').set_index('key')
        if base is None:
            base = d
        res[s] = {}
        for c in ['PQ', 'CE', 'clap'] + MIR:
            m = pd.concat([d[c], base[c]], axis=1, keys=['a', 'r']).dropna()
            res[s][c] = {'mean': float(d[c].mean()), 'n': int(d[c].notna().sum()),
                         'diff_vs_real': float((m.a - m.r).mean()),
                         'win_vs_real': float((m.a > m.r).mean())}
    return res


def analyze(out):
    rng = np.random.default_rng(0)
    df, feat, person = load(out)
    sh = df[df.shared].reset_index(drop=True)
    assert sh.sys.nunique() == 25 and sh.prompt.nunique() == 100 and len(sh) == 2500
    S = {'n_all': len(df), 'n_shared': len(sh), 'missing': {f: int(df[f].isna().sum()) for f in MIR}}
    say(f'MusicEval: {len(df)} clips, shared {len(sh)} (25 sys x 100 prompts)')
    say('missing: ' + ', '.join(f'{f}={n}' for f, n in S['missing'].items() if n))

    S['ceiling'] = {t: noise_ceiling(sh, person, t, rng) for t in ['ovl', 'rel']}
    for t, c in S['ceiling'].items():
        say(f"[ceiling {t}] r(2v3) {c['r_2v3']:.3f}  rho1 {c['rho1']:.3f}  R(5) {c['R5']:.3f}  "
            f"r-ceiling {c['r_ceiling']:.3f}  pair half-agree {c['pair_half_agree']}")

    S['per_feature'] = pf = per_feature(df, sh, rng)
    say('\n[AES axes vs OVL, all 2748]  r / within-system r / r vs REL')
    for a in AXES:
        say(f"  {a}: {pf[a]['r_all']:+.3f} / {pf[a]['within_sys_r']:+.3f} / {pf[a]['r_all_rel']:+.3f}")
    say('\n[MIR per-feature vs OVL]  r_all | partial|AES+nuis [CI] | within-sys partial [CI] | missing')
    for f in MIR:
        v = pf[f]
        say(f"  {f:15s} {v['r_all']:+.3f} | {fmt(v['partial'], v['partial_ci'])} | "
            f"{fmt(v['within_sys_partial'], v['within_sys_partial_ci'])} | {v['n_missing']}")

    S['cv'] = {}
    for target in ['ovl', 'rel']:
        y = sh[target].to_numpy()
        for cv_name, groups, k in [('prompt', sh.prompt, 10), ('system', sh.sys, 25)]:
            for kind in (['ridge', 'hgb'] if target == 'ovl' else ['ridge']):
                preds = {m: oof(sh, cols, y, groups.to_numpy(), k, kind) for m, cols in MODELS.items()}
                point, delta, cis = evaluate(sh, preds, target, cv_name, rng)
                tag = f'{target}/{cv_name}/{kind}'
                S['cv'][tag] = {'point': point, 'delta_FminusB': delta, 'ci': cis}
                say(f'\n[CV {tag}]')
                for m in MODELS:
                    p = point[m]
                    extra = ('  pair ' + ' '.join(f"{b}:{p['pair_agree'][b]:.3f}" for b in p['pair_agree'])
                             if 'pair_agree' in p else '')
                    say(f"  {m}: R2 {p['R2']:.4f}  r {p['r']:.4f}  sysSpearman {p['sys_spearman']:.3f}{extra}")
                for k2, v in delta.items():
                    say(f'  F-B {k2}: ' + (fmt(v, cis[k2]) if k2 in cis else f'{v:+.4f}'))
                if 'pair_n' in point['B']:
                    say(f"  pairs n: {point['B']['pair_n']}")

    # pre-registered gates (ridge, OVL)
    P, L = S['cv']['ovl/prompt/ridge'], S['cv']['ovl/system/ridge']
    g1 = P['delta_FminusB']['dR2'] >= 0.02 and P['ci']['dR2'][0] > 0
    g2 = P['ci']['dpair_all'][0] > 0
    g2_all = {b[0]: P['ci'][f'dpair_{b[0]}'][0] > 0 for b in BINS}
    g3 = L['ci']['dR2'][0] > 0 and L['delta_FminusB']['dsys'] >= 0
    S['gates'] = {'incremental_dR2': bool(g1), 'pair_agree_ci_by_bin': g2_all,
                  'incremental_validity': bool(g1 and g2), 'model_comparison': bool(g1 and g2 and g3),
                  'loso_dR2_ci_gt0': bool(L['ci']['dR2'][0] > 0), 'loso_dsys': L['delta_FminusB']['dsys']}
    say(f"\n[GATES] {json.dumps(S['gates'])}")

    S['external'] = ext = external(feat, {f: pf[f]['partial'] for f in MIR})
    say('\n[external] partial r given raw AES4 + LUFS (sign same as MusicEval?)')
    for name, d in ext.items():
        for t, fs in d.items():
            if t == 'n':
                continue
            say(f"  {name} (n={d['n']}) {t}: " + ', '.join(
                f"{f} {v['partial']:+.3f}{'' if v['same_sign_as_musiceval'] else '*'}" for f, v in fs.items()))

    S['arms'] = ar = arms(feat)
    say('\n[exploratory arms, 1000 MusicCaps prompts] mean (diff vs real, win vs real)')
    for c in ['PQ', 'CE', 'clap'] + MIR:
        say(f'  {c:15s} ' + '  '.join(
            f"{s}: {ar[s][c]['mean']:.3f} ({ar[s][c]['diff_vs_real']:+.3f}, {ar[s][c]['win_vs_real']:.2f})"
            for s in ar))

    (out / 'summary.txt').write_text('\n'.join(LINES) + '\n')
    (out / 'summary.json').write_text(json.dumps(S, indent=1))
    df.to_csv(out / 'per_clip_musiceval.tsv', sep='\t', index=False)
