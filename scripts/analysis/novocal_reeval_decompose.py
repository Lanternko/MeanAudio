#!/usr/bin/env python
"""Aggregate and decompose the per-clip re-evaluation produced by novocal_reeval_full_arms.py.

Three questions:
  1. Does the no-vocal subset change the RANKING of arms? (the only reason to switch
     benchmarks; absolute level moving is expected and not informative on its own)
  2. Is the fulltrack advantage uniform across clips, or carried by a subpopulation?
  3. How far is the best non-fulltrack arm from PQ 6.9, and what would have to change?
"""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/novocal_reeval')
FULLTRACK = {'fulltrack_q3_full_q9', 'fulltrack_noq_full'}
KEYS = ('clap', 'CE', 'CU', 'PC', 'PQ')


def load():
    arms = {}
    for path in sorted(OUT.glob('*.json')):
        d = json.loads(path.read_text())
        arms[d['label']] = d
    return arms


def table(arms, subset):
    print(f"\n{'':34s}" + ''.join(f'{k:>9s}' for k in KEYS) + f"{'n':>7s}")
    rows = sorted(arms.items(), key=lambda kv: -kv[1]['aggregates'][subset]['PQ'])
    for label, d in rows:
        a = d['aggregates'][subset]
        mark = '*' if label in FULLTRACK else ' '
        print(f"{mark}{label:33s}" + ''.join(f"{a[k]:9.4f}" for k in KEYS) + f"{a['n']:7d}")
    return [label for label, _ in rows]


def rank_shift(arms):
    print('\n=== Does the no-vocal subset change the ranking? ===')
    for metric in ('PQ', 'CE', 'clap'):
        full = [l for l, _ in sorted(arms.items(), key=lambda kv: -kv[1]['aggregates']['full'][metric])]
        nov = [l for l, _ in sorted(arms.items(), key=lambda kv: -kv[1]['aggregates']['novocal'][metric])]
        moved = [l for i, l in enumerate(full) if nov.index(l) != i]
        # Spearman on ranks
        a = np.array([full.index(l) for l in arms])
        b = np.array([nov.index(l) for l in arms])
        rho = np.corrcoef(a, b)[0, 1] if len(a) > 2 else float('nan')
        print(f"  {metric:5s} spearman(full, novocal) = {rho:+.4f}   "
              f"{'IDENTICAL ORDER' if not moved else f'{len(moved)} arm(s) moved'}")
        if moved:
            for l in moved:
                print(f"       {l}: rank {full.index(l) + 1} -> {nov.index(l) + 1}")


def gap_shape(arms):
    ft = arms.get('fulltrack_q3_full_q9') or arms.get('fulltrack_noq_full')
    best_c2p0 = max((d for l, d in arms.items() if l not in FULLTRACK),
                    key=lambda d: d['aggregates']['full']['PQ'], default=None)
    if ft is None or best_c2p0 is None:
        print('\n(need one fulltrack and one non-fulltrack arm for the gap decomposition)')
        return
    print(f"\n=== Per-clip gap: {ft['label']}  minus  {best_c2p0['label']} ===")
    ids = sorted(set(ft['per_clip']) & set(best_c2p0['per_clip']))
    for metric in ('PQ', 'CE'):
        a = np.array([ft['per_clip'][i][metric] for i in ids])
        b = np.array([best_c2p0['per_clip'][i][metric] for i in ids])
        d = a - b
        print(f"  {metric}: mean {d.mean():+.4f}  median {np.median(d):+.4f}  sd {d.std():.4f}")
        print(f"      fulltrack wins on {100 * (d > 0).mean():.1f}% of clips; "
              f"per-clip corr(ft, c2p0) = {np.corrcoef(a, b)[0, 1]:.3f}")
        q = np.percentile(d, [10, 25, 50, 75, 90])
        print(f"      gap deciles p10 {q[0]:+.3f} p25 {q[1]:+.3f} p50 {q[2]:+.3f} "
              f"p75 {q[3]:+.3f} p90 {q[4]:+.3f}")
        top = np.argsort(-d)[:len(d) // 10]
        print(f"      top decile of clips carries {100 * d[top].sum() / d.sum():.1f}% of the total gap "
              f"(uniform would be 10%)")


def distance_to_target(arms, target=6.9):
    print(f'\n=== Distance to PQ {target} (non-fulltrack arms) ===')
    for label, d in sorted(arms.items(), key=lambda kv: -kv[1]['aggregates']['full']['PQ']):
        if label in FULLTRACK:
            continue
        a = d['aggregates']['full']
        pq = np.array([v['PQ'] for v in d['per_clip'].values()])
        need = target - a['PQ']
        frac_above = 100 * (pq >= target).mean()
        print(f"  {label:33s} PQ {a['PQ']:.4f}  short by {need:+.4f}  "
              f"({frac_above:.1f}% of clips already >= {target})")


def main():
    arms = load()
    if not arms:
        raise SystemExit('no results yet')
    print(f'loaded {len(arms)} arm(s)   (* = fulltrack corpus)')
    for subset in ('full', 'novocal', 'vocal'):
        print(f'\n########## {subset.upper()} ##########')
        table(arms, subset)
    if len(arms) > 2:
        rank_shift(arms)
    gap_shape(arms)
    distance_to_target(arms)


if __name__ == '__main__':
    sys.exit(main())
