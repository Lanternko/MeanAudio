"""Per-clip score distributions (lvl30) for TTM systems vs MusicCaps real, from p2 115/118 outputs.

Writes a JSON with KDE curves, quantiles, bimodality coefficient, Hartigan dip test and
1-Wasserstein / KS distance to the real-recording distribution, per (system, metric).
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy import stats

R115 = Path.home() / 'eval_output_nvme/ttm_metrics_115_20261009'
R118 = Path.home() / 'eval_output_nvme/ttm_metrics_seeds_118_20261011'
SYSTEMS = [  # key, label, cell dir
    ('real', 'MusicCaps 真實', R115 / 'musiccaps_real'),
    ('musicgen', 'MusicGen-medium', R115 / 'musicgen_medium'),
    ('musicldm', 'MusicLDM', R115 / 'musicldm'),
    ('musicldm_fid8', 'MusicLDM＋fid8', R115 / 'musicldm_fid8'),
    ('ours_cfg0', 'ours CFG0 (s14159265)', R115 / 'meanaudio_cfg0'),
    ('ours_cfg3neg', 'ours CFG3＋neg (s14159265)', R115 / 'meanaudio_cfg3_neg'),
    ('ours_cfg0_s2', 'ours CFG0 (s27182818)', R118 / 'meanaudio_s27182818_cfg0'),
    ('ours_cfg3neg_s2', 'ours CFG3＋neg (s27182818)', R118 / 'meanaudio_s27182818_cfg3_neg'),
    ('ours_cfg0_s3', 'ours CFG0 (s16180339)', R118 / 'meanaudio_s16180339_cfg0'),
    ('ours_cfg3neg_s3', 'ours CFG3＋neg (s16180339)', R118 / 'meanaudio_s16180339_cfg3_neg'),
]
METRICS = ['PQ', 'CE', 'CU', 'PC', 'MEva', 'clap']


def read_tsv(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def load(cell):
    d = cell / 'lvl30'
    pc = list((d / 'scores').glob('*/per_clip.tsv'))
    assert len(pc) == 1, (d, pc)
    rows = {r['id']: r for r in read_tsv(pc[0])}
    meva = {r['id']: float(r['meva_raw']) for r in read_tsv(d / 'meva/meva_per_clip.tsv')}
    out = {}
    for i, r in rows.items():
        if i not in meva:
            continue
        out[i] = {'PQ': float(r['PQ']), 'CE': float(r['CE']), 'CU': float(r['CU']), 'PC': float(r['PC']),
                  'clap': float(r['clap']), 'MEva': meva[i], 'silent': int(float(r['silent']))}
    return out


def bimodality_coef(x):
    n = len(x); g = stats.skew(x); k = stats.kurtosis(x)  # excess
    return (g * g + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def dip(x):
    try:
        import diptest
        d, p = diptest.diptest(np.asarray(x, dtype=float))
        return float(d), float(p)
    except ImportError:
        return None, None


def main(out_path):
    data = {k: load(c) for k, _, c in SYSTEMS}
    common = set.intersection(*[set(v) for v in data.values()])
    res = {'n_common': len(common), 'systems': [{'key': k, 'label': l} for k, l, _ in SYSTEMS], 'metrics': {}}
    for m in METRICS:
        allv = np.concatenate([[data[k][i][m] for i in common] for k, _, _ in SYSTEMS])
        lo, hi = np.percentile(allv, 0.2), np.percentile(allv, 99.8)
        pad = 0.05 * (hi - lo); grid = np.linspace(lo - pad, hi + pad, 200)
        real = np.array([data['real'][i][m] for i in sorted(common)])
        ent = {'grid': grid.round(4).tolist(), 'by_system': {}}
        for k, _, _ in SYSTEMS:
            x = np.array([data[k][i][m] for i in sorted(common)])
            kde = stats.gaussian_kde(x)
            dd, dp = dip(x)
            q = np.percentile(x, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            ent['by_system'][k] = {
                'n': int(len(x)), 'mean': float(x.mean()), 'sd': float(x.std(ddof=1)),
                'quantiles': dict(zip(['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'], q.round(4).tolist())),
                'skew': float(stats.skew(x)), 'excess_kurtosis': float(stats.kurtosis(x)),
                'bimodality_coef': float(bimodality_coef(x)), 'dip': dd, 'dip_p': dp,
                'w1_to_real': float(stats.wasserstein_distance(x, real)),
                'ks_to_real': float(stats.ks_2samp(x, real).statistic),
                'spearman_with_real_same_prompt': float(stats.spearmanr(x, real).statistic),
                'density': kde(grid).round(5).tolist(),
            }
        res['metrics'][m] = ent
    # cross-metric: per-clip scatter sample for PQ vs MEva / PQ vs CLAP (2,000 prompts)
    rng = np.random.default_rng(20261016)
    samp = rng.choice(sorted(common), size=min(2000, len(common)), replace=False)
    res['scatter_ids'] = len(samp)
    res['scatter'] = {k: {m: [round(data[k][i][m], 4) for i in samp] for m in METRICS} for k, _, _ in SYSTEMS}
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False))
    for m in METRICS:
        print(m)
        for k, l, _ in SYSTEMS:
            e = res['metrics'][m]['by_system'][k]
            print(f"  {k:18s} mean {e['mean']:.3f} sd {e['sd']:.3f} p5 {e['quantiles']['p5']:.3f} p50 {e['quantiles']['p50']:.3f} "
                  f"p95 {e['quantiles']['p95']:.3f} BC {e['bimodality_coef']:.3f} dip_p {e['dip_p']} W1 {e['w1_to_real']:.3f} KS {e['ks_to_real']:.3f} rho {e['spearman_with_real_same_prompt']:.3f}")


if __name__ == '__main__':
    main(sys.argv[1])
