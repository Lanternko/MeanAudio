#!/usr/bin/env python
"""084 NegMF: readout for both arms against the nmv2pair control (prereg
docs/experiments/negprompt_distill_meanflow_084_20260927.md section 5).

Cells (MusicCaps 5521, seed 42, NoMask, --no_q), per arm in {n100, nhi} and training seed:
  arm   <A>_mc_mf25_cfg0 / cfg3_neg, <A>_mc_nfe1_cfg0
  ctrl  <C>_mc_mf25_cfg0 / cfg3_neg (lvl30 may live under d2_075_lvl30/), <C>_mc_nfe1_cfg0
  FAD   <cell>_fad/<cell>_fad/metrics.json for the two MF25 cells

Endpoints (paired by clip; pooled over available seeds by averaging per-seed clip
differences; clip bootstrap 10000, seed 20260927):
  E1 primary  arm cfg0 - ctrl cfg0, PQ lvl30   pass: >= 0.31 and CI low > 0 (stage A: 1 seed;
              stage B: every seed > 0 as well); R = E1 / G_neg(ctrl, lvl30 PQ)
  E2          arm cfg0 - ctrl cfg0, CLAP raw   non-inferiority: CI low > -0.008
  E3          arm cfg0 - ctrl cfg3_neg          (one forward pass vs two)
  E4          arm cfg3_neg - ctrl cfg3_neg      (stacking; watch crest / clipped / LUFS)
  E5          arm nfe1 - ctrl nfe1              (1-step, no inference-time guidance)
Loudness gate: any arm cell with silent_n > 2x the matching control cell -> fail (silence escape).

Stage C (section 6.1; arm rev100 = the 08-31 'reversed' text in the guidance branch), own RNG so
the n100/nhi numbers do not move when rev100 cells appear:
  C1          n100 cfg0 - rev100 cfg0, PQ lvl30   the polarity part of the trained-in gain
  G_rev       ctrl cfg3_revneg - ctrl cfg0          the same reversed text as an inference-time negative
  R_train = E1(rev100) / E1(n100)  vs  R_inf = G_rev / G_neg   (same control checkpoints)
Missing seeds or cells are listed and skipped; nothing here decides launch.
"""
import csv
import json
from pathlib import Path

import numpy as np

EV = Path.home() / 'eval_output_nvme'
LVL_STOCK = EV / 'd2_075_lvl30'
OUT = Path.home() / 'MeanAudio/docs/experiments/results/negmf_084_summary.json'
SEEDS = (14159265, 16180339, 27182818)
ARMS = ('n100', 'nhi', 'rev100')
ARM = 'phase8_qwen_caption2p0_slot0clean_negmf{}_noq_quarter_s{}'
CTRL = 'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{}'
CELLS = {'cfg0': '{}_mc_mf25_cfg0', 'neg': '{}_mc_mf25_cfg3_neg', 'nfe1': '{}_mc_nfe1_cfg0'}
B, RNG_SEED = 10000, 20260927

CONTRASTS = {
    'E1_cfg0_arm_vs_ctrl': [(1, 'arm', 'cfg0'), (-1, 'ctrl', 'cfg0')],
    'E3_arm_cfg0_vs_ctrl_cfg3neg': [(1, 'arm', 'cfg0'), (-1, 'ctrl', 'neg')],
    'E4_cfg3neg_arm_vs_ctrl': [(1, 'arm', 'neg'), (-1, 'ctrl', 'neg')],
    'E5_nfe1_arm_vs_ctrl': [(1, 'arm', 'nfe1'), (-1, 'ctrl', 'nfe1')],
    'ref_ctrl_G_neg': [(1, 'ctrl', 'neg'), (-1, 'ctrl', 'cfg0')],
    'ref_arm_G_neg': [(1, 'arm', 'neg'), (-1, 'arm', 'cfg0')],
}
E1_THRESH, CLAP_NI = 0.31, -0.008
SEED_FLOOR = 0.155
REVNEG = '{}_mc_mf25_cfg3_revneg'
METRICS = (('PQ', True, 'PQ_lvl30'), ('PQ', False, 'PQ_raw'), ('clap', False, 'CLAP_raw'),
           ('clap', True, 'CLAP_lvl30'), ('lufs', False, 'LUFS_raw'), ('crest', False, 'crest_raw'))


def read(path):
    with open(path, encoding='utf-8', newline='') as f:
        return {r['id']: r for r in csv.DictReader(f, delimiter='\t')}


def cell(prefix, key, lvl):
    label = CELLS[key].format(prefix)
    if lvl:
        for root in (EV, LVL_STOCK):
            p = root / f'{label}_lvl30' / f'{label}_lvl30' / 'per_clip.tsv'
            if p.exists():
                return read(p)
        return None
    p = EV / label / label / 'per_clip.tsv'
    return read(p) if p.exists() else None


def fad(prefix, key):
    label = CELLS[key].format(prefix)
    p = EV / f'{label}_fad' / f'{label}_fad' / 'metrics.json'
    return json.loads(p.read_text())['metrics']['fad'] if p.exists() else None


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def load(arm, seed):
    out = {}
    for side, prefix in (('arm', ARM.format(arm, seed)), ('ctrl', CTRL.format(seed))):
        for key in CELLS:
            for lvl in (True, False):
                out[(side, key, lvl)] = cell(prefix, key, lvl)
    return out


def diff(data, terms, metric, lvl):
    cells = [data.get((side, key, lvl)) for _, side, key in terms]
    if any(c is None for c in cells):
        return None
    ids = sorted(set.intersection(*(set(c) for c in cells)))
    return ids, np.array([sum(k * num(c[i][metric]) for (k, _, _), c in zip(terms, cells)) for i in ids])


def ci(x, rng):
    x = x[np.isfinite(x)]
    boots = rng.choice(x, (B, len(x))).mean(1)
    return float(x.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], int(len(x))


def summarize_arm(arm, rng):
    data = {s: load(arm, s) for s in SEEDS}
    seeds = [s for s in SEEDS if data[s][('arm', 'cfg0', False)] is not None]
    res = {'seeds_with_arm_cfg0': seeds, 'cells_missing': {}, 'contrasts': {}, 'level': {}, 'fad': {}}
    for s in SEEDS:
        res['cells_missing'][s] = sorted(f'{side}:{key}:{"lvl30" if lvl else "raw"}'
                                           for (side, key, lvl), v in data[s].items() if v is None)
    for name, terms in CONTRASTS.items():
        rec = {}
        for metric, lvl, tag in METRICS:
            per_seed, stack = {}, []
            for s in seeds:
                d = diff(data[s], terms, metric, lvl)
                if d is None:
                    continue
                ids, x = d
                per_seed[s] = float(np.nanmean(x))
                stack.append(dict(zip(ids, x)))
            if not stack:
                continue
            common = sorted(set.intersection(*(set(m) for m in stack)))
            pooled = np.array([np.mean([m[i] for m in stack]) for i in common])
            mean, bounds, n = ci(pooled, rng)
            rec[tag] = {'mean': mean, 'ci95': bounds, 'n_clips': n, 'n_seeds': len(stack), 'per_seed': per_seed}
        if 'PQ_lvl30' in rec and 'PQ_raw' in rec:
            rec['raw_vs_lvl30_sign_disagree'] = bool(np.sign(rec['PQ_lvl30']['mean']) != np.sign(rec['PQ_raw']['mean']))
        res['contrasts'][name] = rec
    e1 = res['contrasts']['E1_cfg0_arm_vs_ctrl']
    g = res['contrasts']['ref_ctrl_G_neg'].get('PQ_lvl30')
    if 'PQ_lvl30' in e1:
        r = e1['PQ_lvl30']
        e1['threshold'] = E1_THRESH
        e1['pass_stageA'] = bool(r['mean'] >= E1_THRESH and r['ci95'][0] > 0)
        e1['pass_stageB'] = bool(r['n_seeds'] == 3 and e1['pass_stageA'] and all(v > 0 for v in r['per_seed'].values()))
        if g:
            e1['R_over_G_neg'] = r['mean'] / g['mean']
    if 'CLAP_raw' in e1:
        e1['E2_clap_noninferior'] = bool(e1['CLAP_raw']['ci95'][0] > CLAP_NI)
        e1['E2_clap_independent_support'] = bool(e1['CLAP_raw']['mean'] >= -CLAP_NI and e1['CLAP_raw']['ci95'][0] > 0)
        e1['E2_margin'] = CLAP_NI
    escape = []
    for s in SEEDS:
        for key in CELLS:
            for side, prefix in (('arm', ARM.format(arm, s)), ('ctrl', CTRL.format(s))):
                c = data[s].get((side, key, False))
                if c is None:
                    continue
                res['level'][f'{side}_{key}_s{s}'] = {
                    'silent_n': int(sum(int(r['silent']) for r in c.values())),
                    'lufs_mean': float(np.nanmean([num(r['lufs']) for r in c.values()])),
                    'crest_mean': float(np.nanmean([num(r['crest']) for r in c.values()])),
                    'peak_ge_0999_n': int(sum(num(r['peak']) >= 0.999 for r in c.values())),
                }
                if key != 'nfe1':
                    f = fad(prefix, key)
                    if f is not None:
                        res['fad'][f'{side}_{key}_s{s}'] = f
            a, c = res['level'].get(f'arm_{key}_s{s}'), res['level'].get(f'ctrl_{key}_s{s}')
            if a and c and a['silent_n'] > 2 * max(c['silent_n'], 1):
                escape.append(f'{key}_s{s}: {a["silent_n"]} vs ctrl {c["silent_n"]}')
    res['silence_escape'] = escape
    res['loudness_gate_pass'] = not escape
    return res


def lvl30(label):
    for root in (EV, LVL_STOCK):
        p = root / f'{label}_lvl30' / f'{label}_lvl30' / 'per_clip.tsv'
        if p.exists():
            return read(p)
    return None


def pooled_diff(pairs, metric, rng):
    """pairs: {seed: (plus_cells, minus_cells)} -> pooled clip-paired mean diff with CI."""
    per_seed, stack = {}, []
    for s, (a, b) in pairs.items():
        if a is None or b is None:
            continue
        ids = sorted(set(a) & set(b))
        x = dict(zip(ids, [num(a[i][metric]) - num(b[i][metric]) for i in ids]))
        per_seed[s] = float(np.nanmean(list(x.values())))
        stack.append(x)
    if not stack:
        return None
    common = sorted(set.intersection(*(set(m) for m in stack)))
    mean, bounds, n = ci(np.array([np.mean([m[i] for m in stack]) for i in common]), rng)
    return {'mean': mean, 'ci95': bounds, 'n_clips': n, 'n_seeds': len(stack), 'per_seed': per_seed}


def stage_c(res):
    """Section 6.1 readout; returns None until a rev100 cfg0 cell exists."""
    rng = np.random.default_rng(RNG_SEED + 3)
    cfg0 = lambda arm, s: lvl30(CELLS['cfg0'].format(ARM.format(arm, s)))
    if not any(cfg0('rev100', s) for s in SEEDS):
        return None
    out = {}
    for metric, tag in (('PQ', 'PQ_lvl30'), ('clap', 'CLAP_lvl30')):
        out[f'C1_n100_minus_rev100_{tag}'] = pooled_diff(
            {s: (cfg0('n100', s), cfg0('rev100', s)) for s in SEEDS}, metric, rng)
        out[f'G_rev_ctrl_{tag}'] = pooled_diff(
            {s: (lvl30(REVNEG.format(CTRL.format(s))), lvl30(CELLS['cfg0'].format(CTRL.format(s))))
             for s in SEEDS}, metric, rng)
    e1 = {a: res[a]['contrasts']['E1_cfg0_arm_vs_ctrl'].get('PQ_lvl30') for a in ('n100', 'rev100')}
    g_neg = res['n100']['contrasts']['ref_ctrl_G_neg'].get('PQ_lvl30')
    g_rev = out['G_rev_ctrl_PQ_lvl30']
    if e1['n100'] and e1['rev100']:
        out['R_train'] = e1['rev100']['mean'] / e1['n100']['mean']
    if g_neg and g_rev:
        out['R_inf'] = g_rev['mean'] / g_neg['mean']
    r, c1 = e1['rev100'], out['C1_n100_minus_rev100_PQ_lvl30']
    if r and c1 and r['n_seeds'] == 3:
        if r['ci95'][1] < 0:
            v = 'reversed_hurts: polarity acts in training (opposite sign)'
        elif r['mean'] < SEED_FLOOR or r['ci95'][0] <= 0:
            v = 'negative_only: trained-in gain needs negative text'
        elif r['mean'] >= E1_THRESH and c1['ci95'][0] <= 0:
            v = 'any_fidelity_text: no polarity part'
        else:
            v = 'partial: both a domain-vocabulary and a polarity part; read R_train vs R_inf'
        out['verdict'] = v
    return out


def main():
    rng = np.random.default_rng(RNG_SEED)
    res = {a: summarize_arm(a, rng) for a in ARMS}
    sc = stage_c(res)
    if sc is not None:
        res['stage_c'] = sc
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    for arm, r in res.items():
        if arm == 'stage_c':
            print('== stage C  ' + '  '.join(
                f'{k}={v["mean"]:+.3f} [{v["ci95"][0]:+.3f},{v["ci95"][1]:+.3f}]' if isinstance(v, dict)
                else f'{k}={v:+.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items() if v is not None))
            continue
        print(f'== {arm}  seeds={r["seeds_with_arm_cfg0"]}  loudness_gate_pass={r["loudness_gate_pass"]} {r["silence_escape"]}')
        for name, rec in r['contrasts'].items():
            p, c = rec.get('PQ_lvl30'), rec.get('CLAP_raw')
            if not p:
                continue
            extra = ''
            if 'pass_stageA' in rec:
                extra = (f'  A={rec["pass_stageA"]} B={rec["pass_stageB"]} R={rec.get("R_over_G_neg", float("nan")):+.2f}'
                         f' clapNI={rec.get("E2_clap_noninferior")}')
            print(f'  {name:30s} PQ30 {p["mean"]:+.3f} [{p["ci95"][0]:+.3f},{p["ci95"][1]:+.3f}] n_seeds={p["n_seeds"]}'
                  + (f'  CLAP {c["mean"]:+.4f} [{c["ci95"][0]:+.4f},{c["ci95"][1]:+.4f}]' if c else '') + extra)
        if r['fad']:
            print('  FAD ' + '  '.join(f'{k}={v:.3f}' for k, v in sorted(r['fad'].items())))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
