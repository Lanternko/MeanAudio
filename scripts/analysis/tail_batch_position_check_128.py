#!/usr/bin/env python3
"""128 incident: do the 126a clean vs clean_shuf AES gaps sit only in tail batches?

126a scores `clean` in sorted-ID order and `clean_shuf` after random.Random(126).shuffle()
(metric_win_tie_126a.py), both through score_aes at batch 32. 5,131 clips leave an 11-clip
tail batch in each order. Standard library only; reads the committed snapshot.
"""
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TSV = REPO / 'docs/experiments/harn/metric_invariance_128/smoke_evidence/historical_clean_vs_shuffled_5131.tsv'
AXES, BATCH, TOL = ('PQ', 'CE', 'CU', 'PC'), 32, 1e-4

rows = list(csv.DictReader(open(TSV, encoding='utf-8'), delimiter='\t'))
ids = sorted(r['id'] for r in rows)
shuffled = list(ids)
random.Random(126).shuffle(shuffled)
tail0 = len(ids) // BATCH * BATCH
tails = set(ids[tail0:]) | set(shuffled[tail0:])
gap = {r['id']: max(abs(float(r[f'clean_{m}']) - float(r[f'shuffled_{m}'])) for m in AXES) for r in rows}
over = {i for i, d in gap.items() if d > TOL}
out = {'n': len(ids), 'tail_batch_size': len(ids) - tail0, 'tail_clips_either_order': len(tails),
       'clips_over_1e-4': len(over), 'over_inside_tails': len(over & tails), 'over_outside_tails': len(over - tails),
       'max_gap_inside_tails': max(gap[i] for i in tails), 'max_gap_outside_tails': max(gap[i] for i in gap if i not in tails)}
print(json.dumps(out, indent=1))
