#!/usr/bin/env python3
"""Assemble every ATTM-protocol artifact into one table."""
import json
from pathlib import Path

ATTM = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/attm')
ORDER = ['ours_c2p0_slot0_cfg3_neg', 'ours_c2p0_slot0_cfg0',
         'meanaudio_s_full_topline', 'meanaudio_l_full_topline']
SHORT = {'ours_c2p0_slot0_cfg3_neg': 'ours cfg3+neg',
         'ours_c2p0_slot0_cfg0': 'ours cfg0',
         'meanaudio_s_full_topline': 'MeanAudio-S-Full',
         'meanaudio_l_full_topline': 'MeanAudio-L-Full'}

main = {}
for label in ORDER:
    p = ATTM / f'{label}.json'
    if p.exists():
        d = json.loads(p.read_text())
        d.pop('per_clip_attm_clap', None)
        main[label] = d

ccs = {}
for label in ORDER:
    p = ATTM / f'ccs_{label}.json'
    if p.exists():
        ccs[label] = json.loads(p.read_text())

fadd = {}
p = ATTM / 'fad_disjoint.json'
if p.exists():
    fadd = json.loads(p.read_text())['results']

spec = {}
p = ATTM / 'ccs_specificity.json'
if p.exists():
    spec = json.loads(p.read_text())

hdr = (f"{'arm':18s} {'CLAP-ATTM':>9s} {'CLAP-ours':>9s} {'ratio':>6s} "
       f"{'FAD-all':>8s} {'FAD-disj':>8s} {'CCSmic':>7s} {'CCSmac':>7s} {'PQ':>6s}")
print(hdr)
print('-' * len(hdr))
for label in ORDER:
    d = main.get(label)
    if not d:
        continue
    c = ccs.get(label, {})
    f = fadd.get(label, {})
    print(f"{SHORT[label]:18s} {d['attm_clap_90p14']:9.4f} {d['ours_clap_89p98']:9.4f} "
          f"{d['attm_clap_90p14']/d['ours_clap_89p98']:6.3f} "
          f"{d['attm_fad_90p14']:8.4f} "
          f"{f.get('fad_disjoint', float('nan')):8.4f} "
          f"{c.get('CCS_micro', float('nan')):7.4f} "
          f"{c.get('CCS_macro', float('nan')):7.4f} {d['PQ']:6.3f}")

if spec:
    u = spec.get('usable_tags', [])
    per = spec.get('per_tag', {})
    print(f"\nspecificity: {len(u)}/{len(per)} tags usable (recall>=0.85 and J>=0.3)")
    print('  usable:', ', '.join(u) if u else '(none)')
    dead = [t for t, v in per.items() if v['youden_j'] < 0.3]
    print('  uninformative:', ', '.join(dead) if dead else '(none)')
else:
    print('\nspecificity: not yet computed -- CCS numbers above are NOT interpretable')

if ccs:
    print('\nCCS by category:')
    for label in ORDER:
        c = ccs.get(label)
        if not c:
            continue
        cats = c.get('CCS_by_category', {})
        s = '  '.join(f"{k} {v['rate']:.3f} (n={v['n']})" for k, v in cats.items())
        print(f'  {SHORT[label]:18s} {s}')
