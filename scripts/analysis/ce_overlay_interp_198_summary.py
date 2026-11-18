#!/usr/bin/env python3
"""198 summary tables (no GPU): reads ~/eval_output_nvme/ce_overlay_interp_198/summary.json written by the driver and
writes docs/experiments/results/ce_overlay_interp_198_table.md (pooled p, middle / energy-weighted counts, monotonicity,
Spearman block, gate). Design: docs/experiments/ce_overlay_interp_198_20261119.md

  python scripts/analysis/ce_overlay_interp_198_summary.py [--root DIR] [--out FILE]
"""
import argparse
import json
from pathlib import Path

HOME = Path.home()


def f(x, fmt='+.2f'):
    return 'n/a' if x is None else format(x, fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(HOME / 'eval_output_nvme/ce_overlay_interp_198'))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'docs/experiments/results/ce_overlay_interp_198_table.md'))
    a = ap.parse_args()
    s = json.loads((Path(a.root) / 'summary.json').read_text())
    rr = s['r_rel']
    tag = lambda r: f'm{-r}' if r < 0 else f'p{r}'  # noqa: E731
    L = [f'# 198 CE 疊加內插階梯（表；{"smoke" if s["smoke"] else "full"}，n ids = {s["n_ids"]}，bootstrap {s["boot"]["NB"]}）', '',
         f'重現閘門 ok = {s["gate"]["ok"]}（中位 {s["gate"]["median_of_medians"]:.5f}，最大 {s["gate"]["max_abs_dCE"]:.5f}）', '']
    for col in ('CE', 'PQ'):
        L += [f'## {col}：pooled p（9 底平均）依疊入聲 × r_rel', '',
              '| 疊入聲 | ' + ' | '.join(tag(r) for r in rr) + ' | 正中間格數 | 能量加權格數 | 單調底數 | 逐段單調 |',
              '|---|' + '---|' * (len(rr) + 4)]
        for o in s['overlays']:
            cs = s['cells'][col][o]
            n_mid = sum(v['middle'] for v in cs.values())
            n_en = sum(v['energy_weighted'] for v in cs.values())
            cv = s['curves'][col][o]
            L.append(f'| {o} | ' + ' | '.join(f(cs[tag(r)]['pooled_p']) for r in rr) +
                     f' | {n_mid}/9 | {n_en}/9 | {cv["n_bases_mono_expected"]}/{len(s["bases"])} | {cv["mean_per_clip_mono_expected"]:.2f} |')
        L.append('')
        L += [f'跑出區間（{col}，疊入聲 × r_rel 的 frac_above_max / frac_below_min，9 底平均）', '',
              '| 疊入聲 | ' + ' | '.join(tag(r) for r in rr) + ' |', '|---|' + '---|' * len(rr)]
        for o in s['overlays']:
            cs = s['cells'][col][o]
            L.append(f'| {o} | ' + ' | '.join(f'{cs[tag(r)]["frac_above_max_mean"]:.2f}/{cs[tag(r)]["frac_below_min_mean"]:.2f}' for r in rr) + ' |')
        L.append('')
    L += ['## Spearman（r_rel = 0，p 對各因素）', '']
    for col, sp in s['spearman_r0'].items():
        L.append(f'- {col}（n = {sp["n_cells"]}，|gap| ≥ m 的子集 n = {sp["n_cells_gap_ge_m"]}）：' + '；'.join(
            f'{k} rho {f(v["rho"])} (p {v["p"]:.3g})' for k, v in sp.items() if isinstance(v, dict) and 'rho' in v and v) +
            ' ｜ 子集：' + '；'.join(f'{k} {f(v["rho"]) if v else "n/a"}' for k, v in sp['restricted_gap_ge_m'].items()))
    Path(a.out).write_text('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
