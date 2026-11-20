#!/usr/bin/env python3
"""p2 138 action: seed-robustness analysis for table B (H1-H4 verdicts, variance, MDD, tie floor).

Spec (pre-registered, frozen): docs/experiments/seed_robustness_table_b_135_20261021.md §1.
Contract: GPU_QUEUE_CONTRACT (job "138"). Frozen before any new-seed result exists: after the
shares of 135-137 are scored this file must not change (a change is a deviation, recorded in
the spec with the original verdicts reported next to it; its sha256 is bound in the contract).

Inputs  seed 42: the 115 / 118 lvl30 per-clip tables and raw FAD of the existing runs;
        new seeds: /mnt/seagate/meanaudio/seed_robustness_135/<system>_<seed>/lvl30/ (jobs 135-137).
        Seeds: ours/MusicGen 42,101,202,303,404 (ours x 3 training seeds), MusicLDM 42,101,202.
Steps   (no flag) raw FAD of the 30 new shares (eval_metrics.py --fad, --ref_dir
        /mnt/HDD/kojiek/musiccaps_reference, nothing else scored), then the analysis.
        --preflight / --validate-only  as in the other actions.
        --fake OUT [--n 256]  dry run on the first N prompts of the *existing* seed-42 data,
                  replicated into fake seeds with synthetic jitter; writes only under OUT and
                  stamps every file FAKE. This is how the program was verified before freezing.

Definitions (spec words in quotes, the rest is the frozen implementation choice)
  system mean = mean of the per-prompt lvl30 score over all prompts (silent clips stay in, as in 115).
  differences "all 75": every (ours training seed, ours inference seed) x (MusicGen inference seed)
      = 3*5*5; versus MusicLDM 3*5*3 = 45.
  hierarchical CI "2,000 bootstraps, 95% percentile": each replicate (1) resamples the 5,521 prompts
      with replacement, one index set for every system (pairing kept), (2) resamples ours' training
      seeds 3 of 3, and within each drawn training seed its inference seeds 5 of 5 (nested),
      (3) resamples the comparator's inference seeds (5 of 5; MusicLDM 3 of 3) independently.
      numpy default_rng(20261021).
  verdicts for H1/H2 (spec order of the rules; what no rule covers is seed_dependent):
      robust            >=95% of the differences > m0 AND CI lower bound > m0
      holds_mean_only   mean > m0 but not robust
      reversed          mean < 0 and CI upper bound < 0
      seed_dependent    everything else (mean in (0, m0], >5% of differences < 0 without the
                        above, mean <= 0 with a CI that crosses 0)
      A case with mean > m0 and >5% of differences < 0 is seed_dependent (the spec's ">5% below 0"
      clause wins over holds_mean_only; fixed 2026-10-21 before any result); still flagged
      `more_than_5pct_below_zero`.
  H3 (tie, |delta| < m0): robust = >=95% of |differences| < m0 and CI inside (-m0, m0);
      holds_mean_only = |mean| < m0 only; not_tie = CI entirely outside the band; else seed_dependent.
  H4 (expected negative, bound 0): H1 logic on the negated differences with m0 = 0.
  Holm over H1-H4 on one-sided bootstrap p = (1 + #replicates violating the claim) / (B + 1).
  variance: sigma_inf^2 = within-training-seed variance (ddof 1) of single-inference-seed system
      means, averaged over training seeds; sigma_train^2 = max(0, var of the training-seed means
      - sigma_inf^2 / K); prompt SD = SD over prompts of the seed-averaged per-prompt score.
  MDD: 2 * sqrt(sum of the two sides' sigma_inf^2 + sigma_train^2) for a single run per side;
      `_all_seeds` divides each term by its number of seeds (K inference, T training).
  tie floor: |same prompt, same system and training seed, two different inference seeds| pooled
      over prompts and seed pairs: median and 90th percentile.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seed_robustness_135 as sr  # noqa: E402
import ttm_metrics_115 as p115  # noqa: E402

METRICS = ('PQ', 'CE', 'CU', 'PC', 'clap', 'meva')
M0 = {'PQ': 0.0919, 'clap': 0.0104}
N_BOOT = 2000
BOOT_SEED = 20261021
SEEDS_FULL = (42, 101, 202, 303, 404)
SEEDS_LDM = (42, 101, 202)
TRAINS = tuple(sr.TRAIN_SEEDS)
SYSTEMS = ('ours_cfg3_neg', 'ours_cfg0', 'musicgen_medium', 'musicldm')
PAIRS = [(a, b) for a in ('ours_cfg3_neg', 'ours_cfg0') for b in ('musicgen_medium', 'musicldm')]
OUT_DIR = sr.ROOT / 'analysis_138'


# ── loading ──────────────────────────────────────────────
def seeds_of(system):
    return SEEDS_LDM if system == 'musicldm' else SEEDS_FULL


def shares_of(system, seed):
    """[(train or None, share dict)] for one system at one inference seed."""
    if system.startswith('ours_'):
        cfg = system[len('ours_'):]
        return [(t, sr.ours_share(t, cfg, seed)) for t in TRAINS]
    return [(None, sr.ext_share(system, seed))]


def table_paths(sh):
    """(per_clip.tsv, meva_per_clip.tsv) of a share's lvl30 block; seed 42 = the 115/118 originals."""
    if sh['seed'] == 42:
        d, n = sr.existing_lvl30_dir(sh)
        return d / 'scores' / f'{n}_lvl30' / 'per_clip.tsv', d / 'meva' / 'meva_per_clip.tsv'
    d = sr.ROOT / sh['name'] / 'lvl30'
    return d / 'scores' / f'{sh["name"]}_lvl30' / 'per_clip.tsv', d / 'meva' / 'meva_per_clip.tsv'


def read_vectors(sh, ids):
    pc, mv = table_paths(sh)
    a, b = sr.read_tsv(pc), sr.read_tsv(mv)
    out = np.empty((len(METRICS), len(ids)))
    for j, i in enumerate(ids):
        for k, m in enumerate(METRICS):
            out[k, j] = float(b[i]['meva_raw'] if m == 'meva' else a[i][m])
    if not np.isfinite(out).all():
        raise SystemExit(f'{sh["name"]}: non-finite per-clip scores')
    return out


def load_real(ids):
    D = {}
    for system in SYSTEMS:
        seeds = seeds_of(system)
        if system.startswith('ours_'):
            D[system] = np.stack([np.stack([read_vectors(dict(shares_of(system, s))[t], ids) for s in seeds])
                                  for t in TRAINS])  # T,K,M,P
        else:
            D[system] = np.stack([read_vectors(shares_of(system, s)[0][1], ids) for s in seeds])  # K,M,P
    return D


def existing_fad(sh):
    d, n = sr.existing_lvl30_dir(sh)
    summ = sr.M118 / 'summary.json' if (sh['kind'] == 'ours' and sh['train'] != 's14159265') else sr.M115 / 'summary.json'
    return json.loads(summ.read_text())['systems'][n]['blocks']['raw']['fad']


def fad_json(sh):
    return sr.ROOT / sh['name'] / 'fad' / f'{sh["name"]}_raw_fad' / 'metrics.json'


def share_fad(sh):
    if sh['seed'] == 42:
        return existing_fad(sh)
    return json.loads(fad_json(sh).read_text())['metrics']['fad']


def load_fad():
    F = {}
    for system in SYSTEMS:
        seeds = seeds_of(system)
        if system.startswith('ours_'):
            F[system] = np.array([[share_fad(dict(shares_of(system, s))[t]) for s in seeds] for t in TRAINS])
        else:
            F[system] = np.array([share_fad(shares_of(system, s)[0][1]) for s in seeds])
    return F


def load_fake(n, rng):
    """First n prompts of the existing seed-42 tables, replicated into jittered fake seeds."""
    ids = list(sr.read_tsv(sr.TSV))[:n]
    D, F = {}, {}
    for system in SYSTEMS:
        seeds = seeds_of(system)
        trains = TRAINS if system.startswith('ours_') else (None,)
        arr = np.stack([read_vectors(shares_of(system, 42)[min(ti, len(shares_of(system, 42)) - 1)][1], ids)
                        for ti in range(len(trains))])  # (T,M,P)
        sd = arr.std(axis=2, keepdims=True)
        out = []
        for t in range(len(trains)):
            row = []
            for k, _ in enumerate(seeds):
                if k == 0:
                    row.append(arr[t])
                else:
                    row.append(arr[t] + rng.normal(0, 0.3, arr[t].shape) * sd[t] + rng.normal(0, 0.02, (len(METRICS), 1)) * sd[t])
            out.append(np.stack(row))
        out = np.stack(out)  # T,K,M,P
        D[system] = out if system.startswith('ours_') else out[0]
        F[system] = (rng.normal(5.0, 0.3, (len(trains), len(seeds))) if system.startswith('ours_')
                     else rng.normal(5.0, 0.3, len(seeds)))
    return ids, D, F


# ── statistics ───────────────────────────────────────────
def as_ours(a):
    return a if a.ndim == 4 else a[None]  # T,K,M,P


def system_means(a, mi):
    return a[..., mi, :].mean(axis=-1)  # ours: T,K ; external: K


def point_diffs(a, b, mi):
    ma, mb = system_means(a, mi), system_means(b, mi)
    ma = ma.reshape(-1)
    return (ma[:, None] - mb[None, :]).reshape(-1)  # one entry per (train, inf_a, inf_b)


def boot_diff(a, b, mi, W, rng):
    """Hierarchical bootstrap of mean(ours) - mean(comparator); returns B replicates."""
    B = W.shape[1]
    MA = np.einsum('tkp,pb->tkb', a[:, :, mi, :], W) if a.ndim == 4 else None
    MB = np.einsum('kp,pb->kb', b[:, mi, :], W) if b.ndim == 3 else None
    if MA is None or MB is None:
        raise ValueError('ours must be 4-d and the comparator 3-d')
    T, K = MA.shape[:2]
    K2 = MB.shape[0]
    tr = rng.integers(0, T, (B, T))
    ka = rng.integers(0, K, (B, T, K))
    kb = rng.integers(0, K2, (B, K2))
    r = np.arange(B)
    ours = MA[tr[:, :, None], ka, r[:, None, None]].mean(axis=(1, 2))
    comp = MB[kb, r[:, None]].mean(axis=1)
    return ours - comp


def prompt_weights(P, rng):
    return rng.multinomial(P, np.full(P, 1.0 / P), size=N_BOOT).T / P  # P,B


def judge_greater(d, lo, hi, m0):
    mean = float(d.mean())
    frac = float((d > m0).mean())
    below0 = float((d < 0).mean())
    a, b = frac >= 0.95, lo > m0
    if a and b:
        v = 'robust'
    elif mean > m0 and below0 <= 0.05:
        v = 'holds_mean_only'
    elif mean < 0 and hi < 0:
        v = 'reversed'
    else:
        v = 'seed_dependent'
    return {'verdict': v, 'criterion_a_ge95pct_above_m0': bool(a), 'criterion_b_ci_lower_above_m0': bool(b),
            'frac_above_m0': frac, 'frac_below_zero': below0, 'more_than_5pct_below_zero': below0 > 0.05}


def judge_tie(d, lo, hi, m0):
    mean = float(d.mean())
    frac = float((np.abs(d) < m0).mean())
    a, b = frac >= 0.95, (lo > -m0 and hi < m0)
    if a and b:
        v = 'robust'
    elif abs(mean) < m0:
        v = 'holds_mean_only'
    elif lo > m0 or hi < -m0:
        v = 'not_tie'
    else:
        v = 'seed_dependent'
    return {'verdict': v, 'criterion_a_ge95pct_inside_band': bool(a), 'criterion_b_ci_inside_band': bool(b),
            'frac_inside_band': frac}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def dist(d):
    return {'n': int(d.size), 'mean': float(d.mean()), 'min': float(d.min()), 'p5': float(np.percentile(d, 5)),
            'median': float(np.median(d)), 'p95': float(np.percentile(d, 95)), 'max': float(d.max()),
            'frac_positive': float((d > 0).mean())}


def ci(rep):
    return float(np.percentile(rep, 2.5)), float(np.percentile(rep, 97.5))


def hypotheses(D, W, rng):
    spec = [('H1', 'ours_cfg3_neg', 'musicgen_medium', 'PQ', 'greater', M0['PQ']),
            ('H2', 'ours_cfg3_neg', 'musicgen_medium', 'clap', 'greater', M0['clap']),
            ('H3', 'ours_cfg0', 'musicldm', 'PQ', 'tie', M0['PQ']),
            ('H4', 'ours_cfg3_neg', 'musicldm', 'clap', 'less', 0.0)]
    out, ps = {}, []
    for name, sa, sb, m, kind, m0 in spec:
        mi = METRICS.index(m)
        d = point_diffs(as_ours(D[sa]), D[sb], mi)
        rep = boot_diff(as_ours(D[sa]), D[sb], mi, W, rng)
        lo, hi = ci(rep)
        if kind == 'greater':
            j, p = judge_greater(d, lo, hi, m0), (1 + int((rep <= m0).sum())) / (N_BOOT + 1)
        elif kind == 'less':
            j = judge_greater(-d, -hi, -lo, m0)
            p = (1 + int((rep >= m0).sum())) / (N_BOOT + 1)
        else:
            j = judge_tie(d, lo, hi, m0)
            p = (1 + max(int((rep <= -m0).sum()), int((rep >= m0).sum()))) / (N_BOOT + 1)
        ps.append(p)
        out[name] = {'ours': sa, 'comparator': sb, 'metric': m, 'kind': kind, 'bound': m0,
                     'mean_diff': float(d.mean()), 'ci95': [lo, hi], 'bootstrap_p_one_sided': p,
                     'differences': dist(d), **j}
    for (name, *_), padj in zip(spec, holm(ps)):
        out[name]['p_holm'] = padj
    return out


def variance_block(D, F):
    res = {}
    for system in SYSTEMS:
        a = D[system]
        ours = system.startswith('ours_')
        arr = as_ours(a) if ours else a[None]  # T,K,M,P
        T, K = arr.shape[:2]
        res[system] = {'n_training_seeds': T, 'n_inference_seeds': K}
        for mi, m in enumerate(METRICS):
            mm = arr[:, :, mi, :].mean(axis=-1)  # T,K
            s2_inf = float(np.mean(mm.var(axis=1, ddof=1)))
            s2_train = float(max(0.0, np.var(mm.mean(axis=1), ddof=1) - s2_inf / K)) if T > 1 else 0.0
            prompt_sd = float(arr[:, :, mi, :].mean(axis=(0, 1)).std(ddof=1))
            res[system][m] = {'sigma_inf': s2_inf ** 0.5, 'sigma_train': s2_train ** 0.5,
                              'sd_over_inference_seeds_of_system_mean': float(np.std(mm.mean(axis=0), ddof=1)),
                              'sd_over_prompts': prompt_sd}
        f = as_ours(F[system][..., None])[..., 0] if ours else F[system][None]
        fi = float(np.mean(f.var(axis=1, ddof=1)))
        res[system]['fad_raw'] = {'sigma_inf': fi ** 0.5, 'range': [float(f.min()), float(f.max())]}
    return res


def mdd_block(var):
    out = {}
    for sa, sb in PAIRS:
        row = {}
        for m in METRICS:
            va, vb = var[sa][m], var[sb][m]
            Ta, Ka, Kb = var[sa]['n_training_seeds'], var[sa]['n_inference_seeds'], var[sb]['n_inference_seeds']
            single = (va['sigma_inf'] ** 2 + va['sigma_train'] ** 2 + vb['sigma_inf'] ** 2 + vb['sigma_train'] ** 2) ** 0.5
            allseeds = (va['sigma_inf'] ** 2 / Ka + va['sigma_train'] ** 2 / Ta + vb['sigma_inf'] ** 2 / Kb) ** 0.5
            row[m] = {'mdd_single_run': 2 * single, 'mdd_all_seeds': 2 * allseeds, 'm0': M0.get(m)}
        out[f'{sa}__vs__{sb}'] = row
    return out


def tie_floor(D):
    out = {}
    for system in SYSTEMS:
        arr = as_ours(D[system]) if system.startswith('ours_') else D[system][None]
        T, K = arr.shape[:2]
        out[system] = {}
        for mi, m in enumerate(METRICS):
            d = np.concatenate([np.abs(arr[t, k1, mi] - arr[t, k2, mi]) for t in range(T)
                                for k1 in range(K) for k2 in range(k1 + 1, K)])
            out[system][m] = {'median_abs': float(np.median(d)), 'p90_abs': float(np.percentile(d, 90)), 'n': int(d.size)}
    return out


def pair_tables(D, F, W, rng):
    out = {}
    for sa, sb in PAIRS:
        row = {}
        for mi, m in enumerate(METRICS):
            a, b = as_ours(D[sa]), D[sb]
            d = point_diffs(a, b, mi)
            lo, hi = ci(boot_diff(a, b, mi, W, rng))
            ma, mb = a[:, :, mi, :].reshape(-1, a.shape[-1]), b[:, mi, :]
            better = np.array([(x > y).mean() for x in ma for y in mb])
            row[m] = {'differences': dist(d), 'ci95_hierarchical': [lo, hi],
                      'frac_prompts_ours_better': {'min': float(better.min()), 'median': float(np.median(better)),
                                                   'max': float(better.max())}}
        fa = as_ours(F[sa][..., None])[..., 0].reshape(-1)
        fb = np.asarray(F[sb]).reshape(-1)
        fd = (fa[:, None] - fb[None, :]).reshape(-1)
        row['fad_raw'] = {'differences': dist(fd), 'note': 'set-level; no prompt bootstrap'}
        out[f'{sa}__vs__{sb}'] = row
    return out


def analyse(ids, D, F, fake=False):
    rng = np.random.default_rng(BOOT_SEED)
    P = len(ids)
    W = prompt_weights(P, rng)
    var = variance_block(D, F)
    result = {'document_kind': 'seed_robustness_138_summary_v1', 'fake': fake, 'n_prompts': P, 'n_boot': N_BOOT,
              'boot_seed': BOOT_SEED, 'seeds': {s: list(seeds_of(s)) for s in SYSTEMS}, 'training_seeds': list(TRAINS),
              'hypotheses': hypotheses(D, W, rng), 'pairs': pair_tables(D, F, W, rng),
              'variance': var, 'mdd': mdd_block(var), 'tie_floor': tie_floor(D),
              'system_means_lvl30': {s: {m: (D[s][..., mi, :].mean(axis=-1)).tolist() for mi, m in enumerate(METRICS)}
                                     for s in SYSTEMS},
              'fad_raw': {s: np.asarray(F[s]).tolist() for s in SYSTEMS}}
    return result


def diffs_csv(D, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ours', 'comparator', 'metric', 'train_idx', 'ours_seed', 'comp_seed', 'diff'])
        for sa, sb in PAIRS:
            a = as_ours(D[sa])
            for mi, m in enumerate(METRICS):
                ma, mb = system_means(a, mi), system_means(D[sb], mi)
                for t in range(ma.shape[0]):
                    for ki, k in enumerate(seeds_of(sa)):
                        for ji, j in enumerate(seeds_of(sb)):
                            w.writerow([sa, sb, m, t, k, j, f'{ma[t, ki] - mb[ji]:.6f}'])


def f(x, nd=4):
    return f'{x:+.{nd}f}'


def markdown(r):
    t = ['# 138 seed robustness tables' + (' (FAKE: dry-run data, not a result)' if r['fake'] else ''), '',
         f'n_prompts {r["n_prompts"]}, bootstrap {r["n_boot"]} (seed {r["boot_seed"]}).', '',
         '## Verdicts', '', '| H | comparison | metric | bound | mean diff | 95% CI | verdict | p (Holm) | share above bound | note |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for h, v in r['hypotheses'].items():
        share = v.get('frac_above_m0', v.get('frac_inside_band'))
        note = 'more than 5% of differences below 0' if v.get('more_than_5pct_below_zero') else ''
        t.append(f'| {h} | {v["ours"]} - {v["comparator"]} | {v["metric"]} | {v["kind"]} {v["bound"]} | {f(v["mean_diff"])} '
                 f'| [{f(v["ci95"][0])}, {f(v["ci95"][1])}] | **{v["verdict"]}** | {v["p_holm"]:.4f} | {share:.3f} | {note} |')
    t += ['', '## Differences (all seed combinations) per pair and metric', '',
          '| pair | metric | n | mean | min | p5 | median | p95 | max | hier. CI | better-prompt share min/med/max |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for pair, row in r['pairs'].items():
        for m in METRICS:
            d, c, b = row[m]['differences'], row[m]['ci95_hierarchical'], row[m]['frac_prompts_ours_better']
            t.append(f'| {pair} | {m} | {d["n"]} | {f(d["mean"])} | {f(d["min"])} | {f(d["p5"])} | {f(d["median"])} | '
                     f'{f(d["p95"])} | {f(d["max"])} | [{f(c[0])}, {f(c[1])}] | {b["min"]:.3f}/{b["median"]:.3f}/{b["max"]:.3f} |')
        d = row['fad_raw']['differences']
        t.append(f'| {pair} | FAD raw | {d["n"]} | {f(d["mean"], 3)} | {f(d["min"], 3)} | {f(d["p5"], 3)} | {f(d["median"], 3)} | '
                 f'{f(d["p95"], 3)} | {f(d["max"], 3)} | n/a | n/a |')
    t += ['', '## Variance decomposition (SD)', '', '| system | metric | sigma_inf | sigma_train | SD of system mean over inference seeds | SD over prompts |',
          '|---|---|---|---|---|---|']
    for s, row in r['variance'].items():
        for m in METRICS:
            v = row[m]
            t.append(f'| {s} | {m} | {v["sigma_inf"]:.4f} | {v["sigma_train"]:.4f} | {v["sd_over_inference_seeds_of_system_mean"]:.4f} | {v["sd_over_prompts"]:.4f} |')
        t.append(f'| {s} | FAD raw | {row["fad_raw"]["sigma_inf"]:.3f} | | range {row["fad_raw"]["range"][0]:.2f}-{row["fad_raw"]["range"][1]:.2f} | |')
    t += ['', '## Minimum detectable difference (2 sigma of the difference of two system means)', '',
          '| pair | metric | MDD single run | MDD all seeds averaged | m0 |', '|---|---|---|---|---|']
    for pair, row in r['mdd'].items():
        for m in METRICS:
            v = row[m]
            t.append(f'| {pair} | {m} | {v["mdd_single_run"]:.4f} | {v["mdd_all_seeds"]:.4f} | {v["m0"] if v["m0"] is not None else ""} |')
    t += ['', '## Same prompt, same system, two inference seeds: |difference| floor', '',
          '| system | metric | median | p90 |', '|---|---|---|---|']
    for s, row in r['tie_floor'].items():
        for m in METRICS:
            t.append(f'| {s} | {m} | {row[m]["median_abs"]:.4f} | {row[m]["p90_abs"]:.4f} |')
    return '\n'.join(t) + '\n'


def write_outputs(r, D, out_dir, json_path=None, md_path=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = Path(json_path) if json_path else out_dir / 'summary.json'
    mp = Path(md_path) if md_path else out_dir / 'tables.md'
    jp.write_text(json.dumps(r, indent=1))
    mp.write_text(markdown(r))
    diffs_csv(D, out_dir / (('FAKE_' if r['fake'] else '') + 'all_differences.csv'))
    return jp, mp


# ── FAD of the new shares ────────────────────────────────
def new_shares():
    out = []
    for job in ('135', '136', '137'):
        out += sr.job_shares(job)
    return out


def fad_done(sh):
    p = fad_json(sh)
    if not p.is_file():
        return False
    m = json.loads(p.read_text())
    return m['n_present'] == sr.N_PROMPTS and m['metrics'].get('fad', -1) >= 0 and \
        m['gen_dir'] == str((sr.ROOT / sh['name'] / 'audio').resolve())


def run_fad(sh):
    out = sr.ROOT / sh['name'] / 'fad'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cmd = [sr.PY_DAC, str(sr.REPO / 'scripts/eval/eval_metrics.py'), '--gen_dir', str(sr.ROOT / sh['name'] / 'audio'),
           '--tsv', sr.TSV, '--exp_name', f'{sh["name"]}_raw_fad', '--out_dir', str(out), '--skip_clap', '--skip_aes',
           '--skip_level', '--fad', '--ref_dir', sr.FAD_REF, '--fad_num_samples', '2048', '--seed', '42']
    p115.log(f'FAD {sh["name"]}')
    with open(out / 'fad.log', 'w') as g:
        subprocess.run(cmd, cwd=sr.REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    if not fad_done(sh):
        raise SystemExit(f'{sh["name"]}: FAD incomplete')


def contract():
    return json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())


def upstream_ready():
    ids = list(sr.read_tsv(sr.TSV))
    cc = sr.c115(sr.ROOT, sr.TSV)
    bad = []
    for sh in new_shares():
        if not sr.generation_complete(sh, sr.ROOT, ids, sr.TSV_SHA):
            bad.append(f'{sh["name"]}: generation')
        elif not sr.scored(cc, sr.sysrec(sh, sr.ROOT)):
            bad.append(f'{sh["name"]}: scoring')
    return bad


def preflight(c):
    b = c['bindings']
    for k in ('action', 'seed_robustness_135', 'ttm_metrics_115', 'eval_metrics'):
        if sr.sha256_file(b[k]) != b[f'{k}_sha256']:
            sys.exit(f'{k} sha mismatch')
    bad = upstream_ready()
    if bad:
        sys.exit('jobs 135-137 are not finished: ' + '; '.join(bad[:6]) + (' ...' if len(bad) > 6 else ''))
    if not Path(sr.FAD_REF).is_dir():
        sys.exit('FAD reference dir missing')
    sr.ROOT.mkdir(parents=True, exist_ok=True)
    print('preflight ok; 30 new shares complete and scored')


def run(c):
    for sh in new_shares():
        if not fad_done(sh):
            run_fad(sh)
    ids = list(sr.read_tsv(sr.TSV))
    D, F = load_real(ids), load_fad()
    r = analyse(ids, D, F)
    r['analysis_sha256'] = sr.sha256_file(__file__)
    jp, mp = write_outputs(r, D, OUT_DIR, c['reports'][0]['path'], c['reports'][1]['path'])
    p115.log(f'analysis written: {jp}, {mp}')


def validate(c):
    for rep in c['reports']:
        if not Path(rep['path']).is_file():
            sys.exit(f'missing {rep["path"]}')
    r = json.loads(Path(c['reports'][0]['path']).read_text())
    if r.get('fake') or r.get('n_prompts') != sr.N_PROMPTS or set(r['hypotheses']) != {'H1', 'H2', 'H3', 'H4'}:
        sys.exit('summary is fake, partial or missing hypotheses')
    if r.get('analysis_sha256') != sr.sha256_file(__file__):
        sys.exit('summary was written by a different analysis program')
    print(json.dumps({h: {'verdict': v['verdict'], 'mean_diff': v['mean_diff'], 'ci95': v['ci95'], 'p_holm': v['p_holm']}
                      for h, v in r['hypotheses'].items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--fake', metavar='OUT_DIR')
    ap.add_argument('--n', type=int, default=256)
    a = ap.parse_args()
    if a.fake:
        t0 = time.time()
        ids, D, F = load_fake(a.n, np.random.default_rng(7))
        r = analyse(ids, D, F, fake=True)
        jp, mp = write_outputs(r, D, a.fake)
        print(f'FAKE analysis on {len(ids)} prompts in {time.time() - t0:.1f}s -> {jp}, {mp}')
        return 0
    c = contract()
    if a.preflight:
        preflight(c)
    elif a.validate_only:
        validate(c)
    else:
        run(c)
    return 0


if __name__ == '__main__':
    sys.exit(main())
