#!/usr/bin/env python3
"""Post-hoc re-adjudication of 183 / 187 / 188 attack cells under the 2026-11-11 criterion.

Criterion (Lanternko 2026-11-11, post hoc, NOT pre-registered):
  success  = dA1 CI lower >= m_A1 (0.055)  AND  dPQ CI upper < 0
  guard    = AES CE (and PAM where scored): "caught" when the CI upper of the delta is < 0;
             "beyond tie band" when delta <= -0.10 * S_CE.
Reads per-clip score tables only; scores nothing. CPU, a few seconds.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

E = Path.home() / 'eval_output_nvme'
OUT = Path(__file__).resolve().parents[2] / 'docs/experiments/results/muq_rejudge_new_criterion_20261111_summary.json'
M_A1, S_CE, S_PQ = 0.055, 1.1904778344500349, 0.914
SEEDS = ['14159265', '16180339', '27182818']
RNG = np.random.default_rng(20261111)
NB = 5000


def ci(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    idx = RNG.integers(0, len(d), (NB, len(d)))
    b = d[idx].mean(1)
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(d))]


def judge(a1, pq, ce, pam=None):
    up = a1[1] >= M_A1
    pq_down = pq[2] < 0
    ce_caught = ce[2] < 0
    o = {'dA1': a1, 'dPQ': pq, 'dCE': ce,
         'A1_up': bool(up), 'PQ_down': bool(pq_down),
         'CE_caught': bool(ce_caught), 'CE_beyond_tie': bool(ce[0] <= -0.10 * S_CE)}
    if pam is not None:
        o['dPAM'] = pam; o['PAM_caught'] = bool(pam[2] < 0)
    if not up:
        o['verdict'] = 'no_effect'
    elif not pq_down:
        o['verdict'] = 'not_success_PQ_not_down'
    else:
        o['verdict'] = 'success_caught_by_CE' if ce_caught else 'success_not_caught'
    return o


def seedavg(root, arm, ctrl, cols):
    """per-prompt delta averaged over the three training seeds"""
    ds = []
    for s in SEEDS:
        a = pd.read_csv(root / 'per_cell' / f'{arm}_s{s}.tsv', sep='\t').set_index('id')
        c = pd.read_csv(root / 'per_cell' / f'{ctrl}_s{s}.tsv', sep='\t').set_index('id')
        ds.append((a[cols] - c.loc[a.index, cols]))
    return sum(ds) / len(ds)


res = {'criterion': {'m_A1': M_A1, 'S_CE': S_CE, 'S_PQ': S_PQ, 'post_hoc': True,
                     'rule': 'success = dA1 CI lower >= m_A1 and dPQ CI upper < 0; guard = dCE (dPAM) CI upper < 0'},
       'j183': {}, 'j187': {}, 'j188_partD': {}, 'j188_multiscorer': {}}
COLS = ['A1_MI_lvl30', 'PQ_lvl30', 'CE_lvl30']

r183 = E / 'muq_train_exploit_183'
for arm in ('L', 'R'):
    for samp in ('cfg0', 'cfg3neg', 'nfe1'):
        if not (r183 / 'per_cell' / f'{arm}_{samp}_s{SEEDS[0]}.tsv').exists():
            continue
        d = seedavg(r183, f'{arm}_{samp}', f'C_{samp}', COLS)
        res['j183'][f'{arm}_{samp}'] = judge(*(ci(d[c]) for c in COLS))
# post-hoc 6 kHz lowpass of control: only MuQ was scored on the lowpassed audio
d = []
for s in SEEDS:
    c = pd.read_csv(r183 / 'per_cell' / f'C_cfg0_s{s}.tsv', sep='\t').set_index('id')
    d.append(c['A1_MI_lp6k'] - c['A1_MI_lvl30'])
res['j183']['posthoc_lp6k_control_cfg0'] = {'dA1': ci(sum(d) / 3), 'verdict': 'undetermined_PQ_CE_not_scored'}

r187 = E / 'muq_reward_dose_187'
for arm in ('D010', 'D020', 'D040', 'D060', 'D080', 'D160', 'Z160'):
    d = seedavg(r187, f'{arm}_cfg0', 'C_cfg0', COLS)
    res['j187'][arm] = judge(*(ci(d[c]) for c in COLS))

r188 = E / 'muq_attractor_probe_188' / 'per_clip'
for base in ('real', 'gen'):
    b = pd.read_csv(r188 / f'T_{base}_base.tsv', sep='\t').set_index('id')
    for f in sorted(r188.glob(f'T_{base}_*_r*.tsv')):
        a = pd.read_csv(f, sep='\t').set_index('id')
        d = a[['A1_MI', 'PQ', 'CE']] - b.loc[a.index, ['A1_MI', 'PQ', 'CE']]
        res['j188_partD'][f.stem[2:]] = judge(ci(d['A1_MI']), ci(d['PQ']), ci(d['CE']))

ms = pd.read_csv(E / 'muq_attractor_probe_188/multiscorer/all_scores.tsv', sep='\t')
ms['key'] = ms['id'].str.split('__').str[1]
G = {g: x.set_index('key') for g, x in ms.groupby('group')}
for trt, base in (('realXsynth18', 'real'), ('realXR18', 'real'), ('genXR18', 'gen'),
                  ('R_loop', 'control'), ('R_loop_s2', 'control')):
    a, b = G[trt], G[base].loc[G[trt].index]
    f = lambda c: ci(a[c] - b[c])
    o = judge(f('A1_MI'), f('PQ'), f('CE'), f('pam'))
    o['dCMI_cap_quality'] = f('cmi_cap_quality'); o['dA3a'] = f('A3a_MI')
    res['j188_multiscorer'][f'{trt}-{base}'] = o

OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
fmt = lambda x: f'{x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]'
for job, cells in res.items():
    if job == 'criterion':
        continue
    print('==', job)
    for k, v in cells.items():
        line = f"{k:28s} A1 {fmt(v['dA1'])}"
        if 'dPQ' in v:
            line += f" PQ {fmt(v['dPQ'])} CE {fmt(v['dCE'])}"
        if 'dPAM' in v:
            line += f" PAM {fmt(v['dPAM'])} CMIcap {fmt(v['dCMI_cap_quality'])}"
        print(line, '->', v['verdict'])
