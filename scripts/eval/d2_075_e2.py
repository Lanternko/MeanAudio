#!/usr/bin/env python
"""075 E2/E3: negprompt gain and CFG0 non-inferiority, paired by MusicCaps prompt.

G(clip) = metric(CFG3+neg) - metric(CFG0) within one checkpoint; arms are compared
clip by clip (same prompt, same inference seed). Reads per_clip.tsv of the raw cells
or, with --lvl, of the -30 LUFS level-matched rescore (scripts/eval/level_match_rescore.py).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

EV = Path.home() / 'eval_output_nvme'
ARMS = {'lab': 'defectlab', 'unlab': 'defectunlab', 'control': 'nmv2pair'}
METRICS = ['PQ', 'CE', 'CU', 'PC', 'clap']


def cell(arm, c, lvl):
    name = f'phase8_qwen_caption2p0_slot0clean_{ARMS[arm]}_noq_quarter_s14159265_mc_mf25_{c}'
    d = EV / 'd2_075_lvl30' / f'{name}_lvl30' if lvl else EV / name
    rows = {}
    with open(next(d.glob('*/per_clip.tsv')), encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows[r['id']] = {m: float(r[m]) for m in METRICS}
    return rows


def ci(x, rng, n=4000):
    b = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lvl', action='store_true')
    a = ap.parse_args()
    rng = np.random.default_rng(20260923)
    D = {arm: {c: cell(arm, c, a.lvl) for c in ['cfg0', 'cfg3_neg']} for arm in ARMS}
    ids = sorted(set.intersection(*[set(D[arm][c]) for arm in ARMS for c in D[arm]]))
    print(f'# 075 E2/E3 ({"level-matched -30 LUFS" if a.lvl else "raw"}), n={len(ids)} paired clips\n')
    arr = lambda arm, c, m: np.array([D[arm][c][i][m] for i in ids])
    print('| metric | G_lab | G_unlab | G_control | G_lab−G_unlab [95% CI] | G_lab−G_control [95% CI] | G_unlab−G_control [95% CI] |')
    print('|---|---|---|---|---|---|---|')
    for m in METRICS:
        G = {arm: arr(arm, 'cfg3_neg', m) - arr(arm, 'cfg0', m) for arm in ARMS}
        cells = [f'{G[x].mean():+.4f}' for x in ARMS]
        for x, y in [('lab', 'unlab'), ('lab', 'control'), ('unlab', 'control')]:
            mu, lo, hi = ci(G[x] - G[y], rng)
            cells.append(f'{mu:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        print(f'| {m} | ' + ' | '.join(cells) + ' |')
    print('\n## E3: CFG0 arm − control (and CFG3+neg for reference)\n')
    print('| cell | metric | lab−control [95% CI] | unlab−control [95% CI] | lab−unlab [95% CI] |')
    print('|---|---|---|---|---|')
    for c in ['cfg0', 'cfg3_neg']:
        for m in METRICS:
            out = []
            for x, y in [('lab', 'control'), ('unlab', 'control'), ('lab', 'unlab')]:
                mu, lo, hi = ci(arr(x, c, m) - arr(y, c, m), rng)
                out.append(f'{mu:+.4f} [{lo:+.4f}, {hi:+.4f}]')
            print(f'| {c} | {m} | ' + ' | '.join(out) + ' |')


if __name__ == '__main__':
    main()
