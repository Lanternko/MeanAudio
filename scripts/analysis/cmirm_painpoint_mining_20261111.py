#!/usr/bin/env python3
"""192 Part 0: CMI-RM pain-point mining over existing scores (2026-11-11, post hoc, descriptive).

Reads the CMI-RM tables that jobs 161-177 wrote for 126a / 126b / 126c / 134
(~/eval_output_nvme/metric_coverage_20261027/cmirm).  Those runs used prompt_mode='empty' only,
so there is no caption variant to split out; 'alignment' under an empty prompt is not a text
alignment measurement and is reported only as a second head.
Each edited clip is paired with its own base clip (the `base` column).  Scores nothing; CPU.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
C = Path.home() / 'eval_output_nvme/metric_coverage_20261027'
OUT = REPO / 'docs/experiments/results/cmirm_painpoint_mining_20261111_summary.json'
CELLS = REPO / 'docs/experiments/results/cmirm_painpoint_mining_20261111_cells.tsv'
S = json.loads((C / 's_values/s_values_cmirm.json').read_text())['S']
TIE = {h: 0.10 * S[h] for h in ('quality', 'alignment')}
RNG = np.random.default_rng(20261111)
NB = 2000


def ci(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) < 5:
        return float(d.mean()) if len(d) else np.nan, np.nan, np.nan
    b = d[RNG.integers(0, len(d), (NB, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


rows = []
levels = {}
for ds in ('126a', '126b', '126c', '134'):
    t = pd.concat([pd.read_csv(f, sep='\t', dtype={'base': str, 'domain': str}) for f in sorted((C / 'cmirm' / ds).glob('*.tsv'))])
    t = t.drop_duplicates('key').set_index('key')
    ref = t[t.kind == 'ref']
    levels[ds] = {k: round(float(v), 3) for k, v in ref.groupby('source').quality.mean().items()}
    if ds == '126b':
        levels['126b_floor'] = t[t.kind == 'floor'][['case', 'quality']].groupby('case').quality.mean().round(3).to_dict()
    e = t[t.base.notna() & (t.base != '')]
    e = e[e.base.isin(t.index)]
    for h in ('quality', 'alignment'):
        e = e.assign(**{f'd_{h}': e[h].values - t.loc[e.base, h].values})
    for (src, kind, case), g in e.groupby(['source', 'kind', 'case']):
        r = {'ds': ds, 'source': src, 'kind': kind, 'case': case, 'n': len(g),
             'domain': g.domain.dropna().iloc[0] if g.domain.notna().any() else ''}
        for h in ('quality', 'alignment'):
            m, lo, hi = ci(g[f'd_{h}'])
            r.update({f'{h}': m, f'{h}_lo': lo, f'{h}_hi': hi})
        r['frac_up'] = float((g.d_quality > 0).mean())
        rows.append(r)

w = pd.DataFrame(rows)
w.round(4).to_csv(CELLS, sep='\t', index=False)
q = TIE['quality']
res = {'S': S, 'tie_band': TIE, 'prompt_mode': 'empty only (no caption variant on disk)', 'clean_levels': levels, 'by_set': {}}
w['set'] = np.where(w.ds != '126b', w.ds, np.where(w.source.str.startswith('midi'), '126b_midi', '126b_rec'))
for (st, kind), x in w.groupby(['set', 'kind']):
    res['by_set'][f'{st}|{kind}'] = {
        'n': int(len(x)), 'down_CI': int((x.quality_hi < 0).sum()), 'down_beyond_tie': int((x.quality <= -q).sum()),
        'up_CI': int((x.quality_lo > 0).sum()), 'up_beyond_tie': int((x.quality >= q).sum()),
        'within_tie_and_CI_cross0': int(((x.quality.abs() < q) & (x.quality_lo <= 0) & (x.quality_hi >= 0)).sum())}
cols = ['ds', 'source', 'kind', 'case', 'n', 'quality', 'quality_lo', 'quality_hi', 'frac_up']
lose = w[w.kind == 'lose']
res['lose_up_cells'] = lose[lose.quality_lo > 0].sort_values('quality', ascending=False)[cols].round(3).to_dict('records')
res['lose_not_detected'] = lose[lose.quality_hi >= 0].sort_values(['ds', 'case'])[cols].round(3).to_dict('records')
tie = w[w.kind == 'tie']
res['tie_beyond_band'] = tie[tie.quality.abs() >= q].sort_values('quality')[cols].round(3).to_dict('records')
# per-case summary across sources (126a: 5 systems)
a = w[w.ds == '126a'].pivot_table(index=['kind', 'case'], columns='source', values='quality').round(3)
res['126a_case_by_system'] = {f'{k[0]}|{k[1]}': v for k, v in a.to_dict('index').items()}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print('S', S, 'tie', TIE); print(json.dumps(levels, indent=0)); print(json.dumps(res['by_set'], indent=0))
print(a.to_string())
for ds in ('126b', '126c', '134'):
    print('==', ds)
    print(w[w.ds == ds].pivot_table(index=['kind', 'case'], columns='source', values='quality').round(3).to_string())
