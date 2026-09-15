#!/usr/bin/env python3
"""Synthesise (genre, instrument, mood) tag-triplet prompts, ATTM-style.

ATTM (arXiv 2605.21538) evaluates on 100 prompts synthesised from tag triplets
rather than on natural captions. Those 100 were never released, so this builds
our own draw from the same *kind* of construction, over a tag vocabulary taken
from the instrumental MusicCaps aspect_list (the same pool ccs.py calibrates).

Deviations that must travel with any number produced from this file:
  * their exact 100 prompts and their tag vocabulary are unknown; ours are
    sampled from MusicCaps aspects occurring >= MIN_COUNT times
  * their surface template is unknown; ours is fixed as TEMPLATE below
  * we draw N=1000 rather than 100, so the arm mean is not dominated by
    sampling noise -- resample 100 from the per-clip values to recover the
    spread at their scale
"""
import argparse
import ast
import csv
import json
import random
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ccs import CATEGORIES, CANON, MUSICCAPS_CSV, VOCAL_RE

TEMPLATE = '{art} {mood} {genre} track featuring {instrument}.'


def article(word):
    return 'An' if word[0].lower() in 'aeiou' else 'A'
OUT_TSV = Path('/home/kojiek/eval_tsvs_p100/attm_tag_triplet_1000.tsv')
OUT_JSON = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/attm/tag_triplets.json')


def observed_pools(min_count):
    rows = [r for r in csv.DictReader(MUSICCAPS_CSV.open())
            if not VOCAL_RE.search(r['caption'])]
    counts = Counter()
    for r in rows:
        for a in ast.literal_eval(r['aspect_list']):
            a = a.strip().lower()
            if a in CANON:
                counts[a] += 1
    pools = {cat: sorted(t for t in tags if counts[t] >= min_count)
             for cat, tags in CATEGORIES.items()}
    return pools, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-count', type=int, default=30)
    args = ap.parse_args()

    pools, counts = observed_pools(args.min_count)
    for cat, pool in pools.items():
        print(f'{cat:11s} {len(pool):3d} tags: {", ".join(pool)}')
    rng = random.Random(args.seed)

    seen, triplets = set(), []
    while len(triplets) < args.n:
        t = (rng.choice(pools['genre']), rng.choice(pools['instrument']),
             rng.choice(pools['mood']))
        if t in seen:           # uniform over distinct triplets, like a synth set
            continue
        seen.add(t)
        triplets.append(t)

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['id', 'caption'])
        rec = {}
        for i, (g, ins, m) in enumerate(triplets):
            cid = f'tri{i:04d}'
            w.writerow([cid, TEMPLATE.format(art=article(m), genre=g,
                                             instrument=ins, mood=m)])
            rec[cid] = {'genre': g, 'instrument': ins, 'mood': m}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        'n': len(triplets), 'seed': args.seed, 'min_count': args.min_count,
        'template': TEMPLATE,
        'pools': pools,
        'sampling': 'uniform over each category pool, distinct triplets',
        'triplets': rec}, indent=1))
    print(f'\nwrote {len(triplets)} prompts -> {OUT_TSV}\n       triplets -> {OUT_JSON}')
    for cid in list(rec)[:5]:
        print(f'  {cid}: ' + TEMPLATE.format(art=article(rec[cid]['mood']), **rec[cid]))


if __name__ == '__main__':
    main()
