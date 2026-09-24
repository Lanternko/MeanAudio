"""
EXP-B prep: generate qwen_slot0_train.tsv — force ALL ids use Qwen slot 0 caption.

Test H11 (user's hypothesis): Qwen 5-task framing variance is the cause of collapse.
If we force all 251K audio to use the SAME Qwen task slot (slot 0), training caption
distribution becomes structurally uniform; the inductive anchor is preserved within
slot 0's framing.

Output: /home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv
"""

import json
import csv
from pathlib import Path

QWEN_JSONL = '/mnt/HDD/kojiek/phase4_jamendo_data/phase9_omni_captions.jsonl'
SRC_TSV    = '/mnt/HDD/kojiek/phase4_jamendo_data/qwen_singlecap_random_train.tsv'
DST_TSV    = '/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv'

# Read Qwen JSONL → id → 5 caps
qwen = {}
with open(QWEN_JSONL) as f:
    for line in f:
        j = json.loads(line)
        qwen[j['id']] = j['captions']
print(f'Qwen ids: {len(qwen):,}')

# Read source TSV (for id order + q_level if any)
rows = []
with open(SRC_TSV) as f:
    reader = csv.DictReader(f, delimiter='\t')
    fieldnames = reader.fieldnames
    for r in reader:
        rows.append(r)
print(f'TSV rows: {len(rows):,}')

# Replace caption with slot 0
n_replaced = 0
n_missing = 0
for r in rows:
    if r['id'] in qwen:
        r['caption'] = qwen[r['id']][0]   # slot 0
        n_replaced += 1
    else:
        n_missing += 1

print(f'Replaced: {n_replaced:,}  Missing: {n_missing:,}')

Path(DST_TSV).parent.mkdir(parents=True, exist_ok=True)
with open(DST_TSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)
print(f'→ {DST_TSV}')

# Sanity: top 10 prefixes
from collections import Counter
prefs = Counter()
for r in rows[:25000]:
    prefs[r['caption'][:50].lower()] += 1
print('\nTop 10 prefixes (25K sample):')
for p, n in prefs.most_common(10):
    print(f'  {n:5d}  {p!r}')
