#!/usr/bin/env python
"""D1 probe analysis: did the defect prompts move the output toward the
programmatically-degraded reference region?

Reads per_clip.tsv files written by eval_metrics.py and prints, per prompt
group, the mean of every metric with a 95% CI, then the deltas that answer the
probe: defect-prompted minus its matched clean stem, and degraded-reference
minus the same clean stem. If the prompt deltas are near zero while the
reference deltas are large, the model has no reachable defect direction.
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

METRICS = ['clap', 'PQ', 'CU', 'CE', 'PC', 'lufs', 'crest', 'peak', 'silent']


def read_per_clip(path):
    out = {}
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            vals = {}
            for m in METRICS:
                v = row.get(m, '')
                vals[m] = float(v) if v not in ('', None) else float('nan')
            out[row['id']] = vals
    return out


def stats(values):
    v = [x for x in values if not math.isnan(x)]
    n = len(v)
    if n == 0:
        return float('nan'), float('nan'), 0
    mean = sum(v) / n
    if n < 2:
        return mean, float('nan'), n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))
    return mean, 1.96 * sd / math.sqrt(n), n


def delta_ci(a, b):
    """mean(a) - mean(b) with a 95% CI (unpaired, Welch)."""
    ma, _, na = stats(a)
    mb, _, nb = stats(b)
    va = [x for x in a if not math.isnan(x)]
    vb = [x for x in b if not math.isnan(x)]
    if na < 2 or nb < 2:
        return ma - mb, float('nan')
    sa = sum((x - ma) ** 2 for x in va) / (na - 1)
    sb = sum((x - mb) ** 2 for x in vb) / (nb - 1)
    return ma - mb, 1.96 * math.sqrt(sa / na + sb / nb)


def group_gen(per_clip):
    """d1_<group>_<key>_<idx> -> '<group>/<key>'."""
    g = defaultdict(list)
    for cid, vals in per_clip.items():
        if '__' in cid:
            continue
        parts = cid.split('_')
        g[f'{parts[1]}/{parts[2]}'].append(vals)
    return g


def group_ref(per_clip):
    """d1_clean_<stem>_<idx>__<degradation> -> 'ref/<degradation>'."""
    g = defaultdict(list)
    for cid, vals in per_clip.items():
        if '__' not in cid:
            continue
        g[f"ref/{cid.split('__')[1]}"].append(vals)
    return g


def table(title, groups, order=None):
    print(f'\n### {title}')
    print('| group | n | ' + ' | '.join(METRICS) + ' |')
    print('|' + '---|' * (len(METRICS) + 2))
    for k in (order or sorted(groups)):
        vals = groups[k]
        cells = []
        for m in METRICS:
            mean, ci, n = stats([v[m] for v in vals])
            cells.append(f'{mean:.4f} ±{ci:.4f}' if not math.isnan(ci) else f'{mean:.4f}')
        print(f'| {k} | {len(vals)} | ' + ' | '.join(cells) + ' |')


def deltas(title, groups, baseline_key, keys):
    print(f'\n### {title} (vs {baseline_key})')
    print('| group | ' + ' | '.join(METRICS) + ' |')
    print('|' + '---|' * (len(METRICS) + 1))
    base = groups[baseline_key]
    for k in keys:
        if k not in groups:
            continue
        cells = []
        for m in METRICS:
            d, ci = delta_ci([v[m] for v in groups[k]], [v[m] for v in base])
            star = '' if math.isnan(ci) else ('' if abs(d) > ci else ' ns')
            cells.append(f'{d:+.4f} ±{ci:.4f}{star}')
        print(f'| {k} | ' + ' | '.join(cells) + ' |')


def paired(title, ref_per_clip, base_per_clip):
    """Degraded minus its own source clip, so the stem mix cancels out."""
    by_deg = defaultdict(list)
    for cid, vals in ref_per_clip.items():
        if '__' not in cid:
            continue
        src, deg = cid.split('__')
        if src not in base_per_clip:
            continue
        by_deg[deg].append({m: vals[m] - base_per_clip[src][m] for m in METRICS})
    print(f'\n### {title} (paired, vs each clip\'s own clean source)')
    print('| degradation | n | ' + ' | '.join(METRICS) + ' |')
    print('|' + '---|' * (len(METRICS) + 2))
    for deg in sorted(by_deg):
        cells = []
        for m in METRICS:
            mean, ci, n = stats([v[m] for v in by_deg[deg]])
            ns = '' if math.isnan(ci) or abs(mean) > ci else ' ns'
            cells.append(f'{mean:+.4f} ±{ci:.4f}{ns}')
        print(f'| {deg} | {len(by_deg[deg])} | ' + ' | '.join(cells) + ' |')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', action='append', required=True,
                    metavar='NAME=PER_CLIP_TSV', help='generation cell to report')
    ap.add_argument('--ref', action='append', default=[],
                    metavar='NAME=PER_CLIP_TSV', help='degraded-reference scoring')
    ap.add_argument('--ref_baseline', metavar='PER_CLIP_TSV',
                    help='per_clip.tsv of the clips the reference was degraded from; '
                         'enables a paired delta (degraded minus its own source)')
    args = ap.parse_args()

    for spec in args.cell:
        name, path = spec.split('=', 1)
        g = group_gen(read_per_clip(path))
        order = [k for k in ('clean/rock', 'clean/piano', 'clean/edm') if k in g]
        order += sorted(k for k in g if k not in order)
        table(f'cell {name}', g, order)
        deltas(f'cell {name}: defect prompt effect', g, 'clean/rock',
               [k for k in order if k.startswith(('full/rock', 'axis/', 'pure/'))])
        for stem in ('piano', 'edm'):
            if f'full/{stem}' in g:
                deltas(f'cell {name}: defect stack on {stem}', g, f'clean/{stem}',
                       [f'full/{stem}'])

    base = read_per_clip(args.ref_baseline) if args.ref_baseline else None
    for spec in args.ref:
        name, path = spec.split('=', 1)
        per = read_per_clip(path)
        table(f'reference {name}', group_ref(per))
        if base:
            paired(f'reference {name}: degradation effect', per, base)


if __name__ == '__main__':
    main()
