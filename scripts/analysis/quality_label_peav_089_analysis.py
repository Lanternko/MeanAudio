#!/usr/bin/env python
"""089 PE-AV quality-label prefix: pooled 3-seed readout (prereg docs/experiments/quality_label_peav_089_20260929.md).

Same cells, endpoints, thresholds and bootstrap as 081 (quality_label_081_analysis.py), with
  arm  = phase8_qwen_caption2p0_slot0clean_peavlabel_noq_quarter_s{seed}   (PE-AV tiers)
  ctrl = the same nmv2pair control (its cells come from 081)
  a81  = the 081 PQ-tier arm, for side-by-side contrasts (X_* rows: 089 minus 081).

Pre-registered interpretation aid: if the 081 effect were pure "PQ label -> PQ endpoint",
the 089 E1/E3 should scale with how far apart the 089 tiers are in PQ:
  pred = E_081 * (PQ_high - PQ_low | 089 tiers) / (PQ_high - PQ_low | 081 tiers)
from arm_inputs/manifest.json stats. Observed >> pred means a non-PQ label also carries
the effect (081 is not only self-leak); observed ~ pred is consistent with the PQ content
of the tiers driving it. CLAP rows are the non-PE-AV, non-PQ reading.
"""
import csv
import json
from pathlib import Path

import numpy as np

EV = Path.home() / 'eval_output_nvme'
LVL_STOCK = EV / 'd2_075_lvl30'
OUT = Path.home() / 'MeanAudio/docs/experiments/results/quality_label_peav_089_summary.json'
MAN = Path.home() / 'exps_nvme/quality_label_089/arm_inputs/manifest.json'
SUM81 = Path.home() / 'MeanAudio/docs/experiments/results/quality_label_prefix_081_summary.json'
SEEDS = (14159265, 16180339, 27182818)
ARM = 'phase8_qwen_caption2p0_slot0clean_peavlabel_noq_quarter_s{}'
A81 = 'phase8_qwen_caption2p0_slot0clean_qlabel_noq_quarter_s{}'
CTRL = 'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{}'
CELLS = {'cfg0': '{}_mc_mf25_cfg0', 'neg': '{}_mc_mf25_cfg3_neg', 'lq': '{}_mc_mf25_cfg3_lqneg',
         'hq0': '{}_hqpos_mc_mf25_cfg0', 'hqlq': '{}_hqpos_mc_mf25_cfg3_lqneg'}
B, RNG_SEED = 10000, 20260926


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


def load_seed(seed):
    out = {}
    for side, tmpl in (('arm', ARM), ('ctrl', CTRL), ('a81', A81)):
        for key in CELLS:
            out[(side, key, True)] = cell(tmpl.format(seed), key, True)
            out[(side, key, False)] = cell(tmpl.format(seed), key, False)
    return out


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


# contrast = sum of coef * cell; terms: list of (coef, side, key)
CONTRASTS = {
    'E1_lqneg_gain_did': [(1, 'arm', 'lq'), (-1, 'arm', 'cfg0'), (-1, 'ctrl', 'lq'), (1, 'ctrl', 'cfg0')],
    'E1a_arm_lqneg_gain': [(1, 'arm', 'lq'), (-1, 'arm', 'cfg0')],
    'E1b_ctrl_lqneg_gain': [(1, 'ctrl', 'lq'), (-1, 'ctrl', 'cfg0')],
    'E2_arm_lqneg_vs_fidelity8': [(1, 'arm', 'lq'), (-1, 'arm', 'neg')],
    'E2c_ctrl_lqneg_vs_fidelity8': [(1, 'ctrl', 'lq'), (-1, 'ctrl', 'neg')],
    'E3_hqpos_cfg0_did': [(1, 'arm', 'hq0'), (-1, 'arm', 'cfg0'), (-1, 'ctrl', 'hq0'), (1, 'ctrl', 'cfg0')],
    'E3a_arm_hqpos_cfg0_gain': [(1, 'arm', 'hq0'), (-1, 'arm', 'cfg0')],
    'E4_hqpos_lqneg_did': [(1, 'arm', 'hqlq'), (-1, 'arm', 'cfg0'), (-1, 'ctrl', 'hqlq'), (1, 'ctrl', 'cfg0')],
    'E4b_arm_hqpos_lqneg_vs_fidelity8': [(1, 'arm', 'hqlq'), (-1, 'arm', 'neg')],
    'E5_stock_cfg0_arm_vs_ctrl': [(1, 'arm', 'cfg0'), (-1, 'ctrl', 'cfg0')],
    'E5_stock_cfg3neg_arm_vs_ctrl': [(1, 'arm', 'neg'), (-1, 'ctrl', 'neg')],
    'ref_ctrl_fidelity8_gain': [(1, 'ctrl', 'neg'), (-1, 'ctrl', 'cfg0')],
    'ref_arm_fidelity8_gain': [(1, 'arm', 'neg'), (-1, 'arm', 'cfg0')],
    'X_E1_089_minus_081': [(1, 'arm', 'lq'), (-1, 'arm', 'cfg0'), (-1, 'a81', 'lq'), (1, 'a81', 'cfg0')],
    'X_E3_089_minus_081': [(1, 'arm', 'hq0'), (-1, 'arm', 'cfg0'), (-1, 'a81', 'hq0'), (1, 'a81', 'cfg0')],
    'X_hqlq_089_minus_081': [(1, 'arm', 'hqlq'), (-1, 'a81', 'hqlq')],
    'X_cfg0_089_minus_081': [(1, 'arm', 'cfg0'), (-1, 'a81', 'cfg0')],
}
THRESH = {'E1_lqneg_gain_did': 0.19, 'E3_hqpos_cfg0_did': 0.155}
CLAP_NI = {'E5_stock_cfg0_arm_vs_ctrl': -0.004, 'E5_stock_cfg3neg_arm_vs_ctrl': -0.0158}


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


def main():
    data = {s: load_seed(s) for s in SEEDS}
    rng = np.random.default_rng(RNG_SEED)
    res = {'cells_missing': {}, 'contrasts': {}, 'silent': {}, 'lufs_mean': {}}
    for s in SEEDS:
        res['cells_missing'][s] = sorted(f'{side}:{key}:{"lvl30" if lvl else "raw"}'
                                           for (side, key, lvl), v in data[s].items() if v is None)
    for name, terms in CONTRASTS.items():
        rec = {}
        for metric, lvl, tag in (('PQ', True, 'PQ_lvl30'), ('PQ', False, 'PQ_raw'),
                                 ('clap', False, 'CLAP_raw'), ('lufs', False, 'LUFS_raw')):
            per_seed, pooled_ids, stack = {}, None, []
            for s in SEEDS:
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
        if name in THRESH and 'PQ_lvl30' in rec:
            r = rec['PQ_lvl30']
            rec['pass'] = bool(r['n_seeds'] == 3 and r['mean'] >= THRESH[name] and r['ci95'][0] > 0
                               and all(v > 0 for v in r['per_seed'].values()))
            rec['threshold'] = THRESH[name]
        if name in CLAP_NI and 'CLAP_raw' in rec:
            rec['clap_noninferior'] = bool(rec['CLAP_raw']['ci95'][0] > CLAP_NI[name])
            rec['clap_margin'] = CLAP_NI[name]
        if 'PQ_lvl30' in rec and 'PQ_raw' in rec:
            rec['raw_vs_lvl30_sign_disagree'] = bool(np.sign(rec['PQ_lvl30']['mean']) != np.sign(rec['PQ_raw']['mean']))
        res['contrasts'][name] = rec
    for s in SEEDS:
        for side in ('arm', 'ctrl', 'a81'):
            for key in CELLS:
                c = data[s].get((side, key, False))
                if c is not None:
                    res['silent'][f'{side}_{key}_s{s}'] = int(sum(int(r['silent']) for r in c.values()))
                    res['lufs_mean'][f'{side}_{key}_s{s}'] = float(np.nanmean([num(r['lufs']) for r in c.values()]))
    if MAN.exists() and SUM81.exists():
        st = json.load(open(MAN))['stats']
        c = st['pq_contrast_high_minus_low']
        ratio = c['089'] / c['081']
        s81 = json.load(open(SUM81))['contrasts']
        pred = {}
        for name in ('E1_lqneg_gain_did', 'E3_hqpos_cfg0_did'):
            e81 = s81[name]['PQ_lvl30']['mean']
            obs = (res['contrasts'].get(name, {}).get('PQ_lvl30') or {}).get('mean')
            pred[name] = {'e081': e81, 'pq_contrast_ratio': ratio, 'pred_pure_pq': e81 * ratio, 'observed': obs}
        res['pq_scaling_prediction'] = pred
        res['tier_stats'] = {k: st[k] for k in ('spearman_peav_vs_pq', 'pq_contrast_high_minus_low',
                                                  'crosstab_089_by_081', 'peav_by_089_tier', 'pq_by_089_tier')}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    for name, rec in res['contrasts'].items():
        p = rec.get('PQ_lvl30')
        c = rec.get('CLAP_raw')
        if p:
            print(f'{name:36s} PQ30 {p["mean"]:+.3f} [{p["ci95"][0]:+.3f},{p["ci95"][1]:+.3f}] '
                  f'seeds={p["n_seeds"]} ' + ' '.join(f'{v:+.3f}' for v in p['per_seed'].values())
                  + (f'  CLAP {c["mean"]:+.4f} [{c["ci95"][0]:+.4f},{c["ci95"][1]:+.4f}]' if c else '')
                  + (f'  pass={rec["pass"]}' if 'pass' in rec else '')
                  + (f'  clapNI={rec["clap_noninferior"]}' if 'clap_noninferior' in rec else ''))
    for name, p in res.get('pq_scaling_prediction', {}).items():
        print(f"{name}: 081 {p['e081']:+.3f} x ratio {p['pq_contrast_ratio']:.3f} -> pure-PQ pred {p['pred_pure_pq']:+.3f}; "
              f"observed {p['observed']}")
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
