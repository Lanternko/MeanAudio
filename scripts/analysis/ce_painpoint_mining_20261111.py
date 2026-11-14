#!/usr/bin/env python3
"""CE pain-point mining over existing scores (2026-11-11, post hoc, descriptive).

Question: AES CE is now half of the MuQ-attack success rule (A1 up AND CE down).
Where does CE itself misbehave?  Reads existing tables only; scores nothing.
  - research/eval/metric_std_20261020/evidence_rows.tsv   (126a/b/c interventions)
  - ~/eval_output_nvme/muq_attractor_probe_188/{multiscorer/all_scores.tsv,per_clip/T_*.tsv}
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
E188 = Path.home() / 'eval_output_nvme/muq_attractor_probe_188'
OUT = REPO / 'docs/experiments/results/ce_painpoint_mining_20261111_summary.json'
TIE_CE = 0.119
RNG = np.random.default_rng(20261111)


def ci(d, nb=5000):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    b = d[RNG.integers(0, len(d), (nb, len(d)))].mean(1)
    return [round(float(d.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


res = {}

# ---- 1. 126 interventions -------------------------------------------------
d = pd.read_csv(REPO / 'research/eval/metric_std_20261020/evidence_rows.tsv', sep='\t')
key = ['ds', 'step', 'system', 'kind']
w = d.pivot_table(index=key, columns='metric', values='mean')
w['CE_lo'] = d[d.metric == 'CE'].set_index(key)['lo']
w['CE_hi'] = d[d.metric == 'CE'].set_index(key)['hi']
w = w.reset_index()
w['midi'] = w.system.str.startswith('midi')
bad = w[w.kind != 'tie']
grp = bad.assign(set=np.where(bad.ds != '126b', bad.ds, np.where(bad.midi, '126b_midi', '126b_rec')))
res['should_be_worse'] = {
    k: {'n': int(len(x)), 'CE_down_CI': int((x.CE_hi < 0).sum()), 'CE_down_beyond_tie': int((x.CE <= -TIE_CE).sum()),
        'CE_up_CI': int((x.CE_lo > 0).sum()), 'CE_up_beyond_tie': int((x.CE >= TIE_CE).sum())}
    for k, x in grp.groupby(['set', 'kind']).__iter__() for k in ['|'.join(k)]}
cols = ['ds', 'step', 'system', 'kind', 'CE', 'CE_lo', 'CE_hi', 'PQ', 'meva', 'key_cnn_conf', 'pulse_clarity']
res['ce_up_cells'] = bad[bad.CE_lo > 0].sort_values('CE', ascending=False)[cols].round(3).to_dict('records')
tie = w[w.kind == 'tie']
res['tie_kind_beyond_band'] = tie[tie.CE.abs() > TIE_CE].sort_values('CE')[cols[:7] + ['PQ']].round(3).to_dict('records')
res['tie_kind_n'] = int(len(tie))

# ---- 2. simple synthetic sounds vs real music (188 multiscorer) -----------
ms = pd.read_csv(E188 / 'multiscorer/all_scores.tsv', sep='\t')
real, syn = ms[ms.group == 'real'], ms[ms.group == 'synth']
res['synth_vs_real'] = {'n_synth': int(len(syn)), 'n_real': int(len(real))}
for m in ['CE', 'PQ', 'CU', 'A1_MI', 'A3a_MI', 'pam', 'se_mus', 'meva_raw']:
    res['synth_vs_real'][m] = {
        'real_median': round(float(real[m].median()), 3), 'synth_max': round(float(syn[m].max()), 3),
        'synth_max_pct_of_real': round(float((real[m] < syn[m].max()).mean()), 3),
        'n_synth_above_real_median': int((syn[m] > real[m].median()).sum())}
top = syn.sort_values('CE', ascending=False).head(5)
res['synth_top_ce'] = top[['id', 'CE', 'PQ', 'A1_MI', 'pam']].round(2).to_dict('records')
res['noise_vs_tone_ce'] = syn[syn.id.str.contains('white|synth__pink$|saw_const|sine_const')][['id', 'CE', 'PQ']].round(2).to_dict('records')
res['ce_rank_corr'] = {g: ms[ms.group == g][['CE', 'PQ', 'A1_MI', 'pam']].corr(method='spearman')['CE'].round(2).to_dict()
                       for g in ['real', 'gen', 'synth']}

# ---- 3. overlays: is the CE drop specific to the loop sound? --------------
res['overlay'] = {}
for base in ('real', 'gen'):
    b = pd.read_csv(E188 / f'per_clip/T_{base}_base.tsv', sep='\t').set_index('id')
    T = {f.stem[len(f'T_{base}_'):]: pd.read_csv(f, sep='\t').set_index('id')
         for f in sorted((E188 / 'per_clip').glob(f'T_{base}_*_r*.tsv'))}
    for k, a in T.items():
        res['overlay'][f'{base}_{k}'] = {m: ci(a[m] - b.loc[a.index, m]) for m in ('A1_MI', 'PQ', 'CE')}
    for r in ('r18', 'r0'):
        for ctl in ('sine1k', 'music', 'click', 'pink'):
            for trt in ('osynth', 'oR'):
                a, c = T[f'{trt}_{r}'], T[f'{ctl}_{r}']
                res['overlay'][f'{base}_{trt}_{r}-minus-{ctl}'] = {m: ci(a[m] - c.loc[a.index, m]) for m in ('A1_MI', 'CE')}

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res['should_be_worse'], indent=1))
print(json.dumps(res['synth_vs_real'], indent=1))
for k, v in res['overlay'].items():
    if 'r18' in k:
        print(k, v)
