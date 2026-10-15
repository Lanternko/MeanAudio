#!/usr/bin/env python
"""p2 120-122 NegMF N100 at the full recipe: readout against the matched full control
(prereg docs/experiments/negmf_full_120_20261012.md, endpoints carried over from 084).

Cells (MusicCaps 5521, MeanFlow 25 steps, seed 42, NoMask, --no_q), per training seed:
  arm / ctrl  <P>_mc_mf25_cfg0, <P>_mc_mf25_cfg3_neg, <P>_mc_nfe1_cfg0   (+ _lvl30, + _fad for mf25)

Endpoints (paired by clip; pooled over available seeds by averaging per-seed clip differences;
clip bootstrap 10000, seed 20261012):
  E1 primary  arm cfg0 - ctrl cfg0, PQ lvl30   pass: 3 seeds, >= 0.31, CI low > 0, every seed > 0
  E2          arm cfg0 - ctrl cfg0, CLAP raw   non-inferiority: CI low > -0.008
  silence     arm cfg0 silent_n > 2x ctrl and > 1% of clips -> fail regardless of E1
  FAD         always reported, both cells, both sides
Descriptive: arm cfg0 vs ctrl cfg3_neg, cfg3_neg stacking, 1-NFE, the four MIR readouts,
arm cell means next to MusicGen-medium (lvl30 PQ 7.430 / CLAP 0.2241, p2 113/118).
Missing seeds or cells are listed and skipped.
"""
import csv
import json
from pathlib import Path

import numpy as np

EV = Path.home() / 'eval_output_nvme'
OUT = Path.home() / 'MeanAudio/docs/experiments/results/negmf_full_120_summary.json'
SEEDS = (14159265, 16180339, 27182818)
ARM = 'phase8_qwen_caption2p0_slot0clean_negmfn100_noq_full_s{}'
CTRL = 'phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full_s{}'
CELLS = {'cfg0': '{}_mc_mf25_cfg0', 'neg': '{}_mc_mf25_cfg3_neg', 'nfe1': '{}_mc_nfe1_cfg0'}
B, RNG_SEED = 10000, 20261012
E1_THRESH, CLAP_NI, SILENT_FRAC = 0.31, -0.008, 0.01
MUSICGEN = {'PQ_lvl30': 7.430, 'CLAP_lvl30': 0.2241}

CONTRASTS = {
    'E1_cfg0_arm_vs_ctrl': [(1, 'arm', 'cfg0'), (-1, 'ctrl', 'cfg0')],
    'arm_cfg0_vs_ctrl_cfg3neg': [(1, 'arm', 'cfg0'), (-1, 'ctrl', 'neg')],
    'cfg3neg_arm_vs_ctrl': [(1, 'arm', 'neg'), (-1, 'ctrl', 'neg')],
    'nfe1_arm_vs_ctrl': [(1, 'arm', 'nfe1'), (-1, 'ctrl', 'nfe1')],
    'ref_ctrl_G_neg': [(1, 'ctrl', 'neg'), (-1, 'ctrl', 'cfg0')],
    'ref_arm_G_neg': [(1, 'arm', 'neg'), (-1, 'arm', 'cfg0')],
}
# (column, source, tag); source in {'lvl30', 'raw', 'mir'}
METRICS = (('PQ', 'lvl30', 'PQ_lvl30'), ('CE', 'lvl30', 'CE_lvl30'), ('CU', 'lvl30', 'CU_lvl30'),
           ('PC', 'lvl30', 'PC_lvl30'), ('clap', 'lvl30', 'CLAP_lvl30'),
           ('PQ', 'raw', 'PQ_raw'), ('clap', 'raw', 'CLAP_raw'), ('lufs', 'raw', 'LUFS_raw'),
           ('crest', 'raw', 'crest_raw'),
           ('pulse_clarity', 'mir', 'mir_pulse_clarity'), ('ibi_cv', 'mir', 'mir_ibi_cv'),
           ('key_cnn_conf', 'mir', 'mir_key_cnn_conf'), ('chroma_entropy', 'mir', 'mir_chroma_entropy'))


def read(path):
    if not path.exists():
        return None
    with open(path, encoding='utf-8', newline='') as f:
        return {r['id']: r for r in csv.DictReader(f, delimiter='\t')}


def cell(prefix, key, src):
    label = CELLS[key].format(prefix)
    if src == 'lvl30':
        return read(EV / f'{label}_lvl30' / f'{label}_lvl30' / 'per_clip.tsv')
    return read(EV / label / label / ('mir_per_clip.tsv' if src == 'mir' else 'per_clip.tsv'))


def fad(prefix, key):
    label = CELLS[key].format(prefix)
    p = EV / f'{label}_fad' / f'{label}_fad' / 'metrics.json'
    return json.loads(p.read_text())['metrics']['fad'] if p.exists() else None


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def load(seed):
    out = {}
    for side, prefix in (('arm', ARM.format(seed)), ('ctrl', CTRL.format(seed))):
        for key in CELLS:
            for src in ('lvl30', 'raw', 'mir'):
                if src == 'mir' and key == 'nfe1':
                    continue
                out[(side, key, src)] = cell(prefix, key, src)
    return out


def diff(data, terms, metric, src):
    cells = [data.get((side, key, src)) for _, side, key in terms]
    if any(c is None for c in cells):
        return None
    ids = sorted(set.intersection(*(set(c) for c in cells)))
    return dict(zip(ids, (sum(k * num(c[i][metric]) for (k, _, _), c in zip(terms, cells)) for i in ids)))


def ci(x, rng):
    x = x[np.isfinite(x)]
    boots = rng.choice(x, (B, len(x))).mean(1)
    return float(x.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], int(len(x))


def main():
    rng = np.random.default_rng(RNG_SEED)
    data = {s: load(s) for s in SEEDS}
    seeds = [s for s in SEEDS if data[s][('arm', 'cfg0', 'raw')] is not None]
    res = {'seeds_with_arm_cfg0': seeds, 'contrasts': {}, 'level': {}, 'fad': {}, 'cell_means': {},
           'cells_missing': {str(s): sorted(f'{a}:{k}:{src}' for (a, k, src), v in data[s].items() if v is None)
                             for s in SEEDS}}
    for name, terms in CONTRASTS.items():
        rec = {}
        for metric, src, tag in METRICS:
            per_seed, stack = {}, []
            for s in seeds:
                d = diff(data[s], terms, metric, src)
                if d is None:
                    continue
                per_seed[str(s)] = float(np.nanmean(list(d.values())))
                stack.append(d)
            if not stack:
                continue
            common = sorted(set.intersection(*(set(m) for m in stack)))
            mean, bounds, n = ci(np.array([np.mean([m[i] for m in stack]) for i in common]), rng)
            rec[tag] = {'mean': mean, 'ci95': bounds, 'n_clips': n, 'n_seeds': len(stack), 'per_seed': per_seed}
        res['contrasts'][name] = rec

    e1 = res['contrasts']['E1_cfg0_arm_vs_ctrl']
    if 'PQ_lvl30' in e1:
        r = e1['PQ_lvl30']
        e1['threshold'] = E1_THRESH
        e1['pass'] = bool(r['n_seeds'] == 3 and r['mean'] >= E1_THRESH and r['ci95'][0] > 0
                          and all(v > 0 for v in r['per_seed'].values()))
        g = res['contrasts']['ref_ctrl_G_neg'].get('PQ_lvl30')
        if g:
            e1['R_over_G_neg'] = r['mean'] / g['mean']
    if 'CLAP_raw' in e1:
        e1['E2_margin'] = CLAP_NI
        e1['E2_clap_noninferior'] = bool(e1['CLAP_raw']['ci95'][0] > CLAP_NI)

    escape = []
    for s in SEEDS:
        for key in CELLS:
            for side, prefix in (('arm', ARM.format(s)), ('ctrl', CTRL.format(s))):
                c = data[s].get((side, key, 'raw'))
                if c is None:
                    continue
                res['level'][f'{side}_{key}_s{s}'] = {
                    'n': len(c),
                    'silent_n': int(sum(int(r['silent']) for r in c.values())),
                    'lufs_mean': float(np.nanmean([num(r['lufs']) for r in c.values()])),
                    'crest_mean': float(np.nanmean([num(r['crest']) for r in c.values()])),
                    'peak_ge_0999_n': int(sum(num(r['peak']) >= 0.999 for r in c.values())),
                }
                l30 = data[s].get((side, key, 'lvl30'))
                if l30 is not None:
                    res['cell_means'][f'{side}_{key}_s{s}'] = {
                        f'{m}_lvl30': float(np.nanmean([num(r[col]) for r in l30.values()]))
                        for col, m in (('PQ', 'PQ'), ('CE', 'CE'), ('CU', 'CU'), ('PC', 'PC'), ('clap', 'CLAP'))}
                if key != 'nfe1':
                    f = fad(prefix, key)
                    if f is not None:
                        res['fad'][f'{side}_{key}_s{s}'] = f
        a, c = res['level'].get(f'arm_cfg0_s{s}'), res['level'].get(f'ctrl_cfg0_s{s}')
        if a and c and a['silent_n'] > 2 * c['silent_n'] and a['silent_n'] > SILENT_FRAC * a['n']:
            escape.append(f's{s}: {a["silent_n"]} vs ctrl {c["silent_n"]}')
    res['silence_escape'] = escape
    res['silence_gate_pass'] = not escape

    # descriptive: arm cell means (average over seeds) next to MusicGen-medium
    mg = {}
    for key in ('cfg0', 'neg'):
        cm = [res['cell_means'][f'arm_{key}_s{s}'] for s in SEEDS if f'arm_{key}_s{s}' in res['cell_means']]
        if cm:
            mg[f'arm_{key}'] = {t: float(np.mean([c[t] for c in cm])) - v for t, v in MUSICGEN.items()}
    res['arm_minus_musicgen_cell_means'] = mg
    for key in ('cfg0', 'neg'):
        fa = [res['fad'].get(f'arm_{key}_s{s}') for s in SEEDS]
        fc = [res['fad'].get(f'ctrl_{key}_s{s}') for s in SEEDS]
        if all(v is not None for v in fa + fc):
            res['fad'][f'arm_minus_ctrl_{key}_mean'] = float(np.mean(fa) - np.mean(fc))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print(f'seeds={seeds}  silence_gate_pass={res["silence_gate_pass"]} {escape}')
    for name, rec in res['contrasts'].items():
        print(f'== {name}' + (f'  pass={rec.get("pass")} R={rec.get("R_over_G_neg", float("nan")):+.2f}'
                              f' clapNI={rec.get("E2_clap_noninferior")}' if 'threshold' in rec else ''))
        for tag, r in rec.items():
            if isinstance(r, dict):
                ps = ' '.join(f'{v:+.4f}' for v in r['per_seed'].values())
                print(f'  {tag:20s} {r["mean"]:+.4f} [{r["ci95"][0]:+.4f},{r["ci95"][1]:+.4f}] n={r["n_clips"]} seeds: {ps}')
    print('FAD ' + '  '.join(f'{k}={v:.3f}' for k, v in sorted(res['fad'].items())))
    print('silent ' + '  '.join(f'{k}={v["silent_n"]}' for k, v in sorted(res['level'].items())))
    print('vs MusicGen ' + json.dumps(mg))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
