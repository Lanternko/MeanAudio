"""
EXP-C: Generate Qwen-with-LP-MC-boilerplate training TSV.

Inverse direction of EXP-A: take EXP-B's qwen_slot0_train.tsv (slot 0 Qwen captions,
251K rows) and prepend the LP-MC writing-task boilerplate "The low quality
recording features a " to every caption. Lowercase first letter of original Qwen
caption to read naturally after the prefix.

Tests H10 cleanly: if anchor template alone is sufficient for healthy training,
MC CLAP should recover from ~0.06 (EXP-B) toward ~0.18 (P8 baseline). If still
collapsed, anchor template is necessary but not sufficient — caption content matters.

Input:  /home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv
Output: /home/kojiek/eval_tsvs_p100/qwen_slot0_boilerplate_train.tsv
"""
import csv
from pathlib import Path

SRC = '/home/kojiek/eval_tsvs_p100/qwen_slot0_train.tsv'
DST = '/home/kojiek/eval_tsvs_p100/qwen_slot0_boilerplate_train.tsv'
PREFIX = 'The low quality recording features a '

def prepend(c: str) -> str:
    c = c.strip()
    if not c:
        return PREFIX.strip()
    # Lowercase first letter so the prefix reads naturally
    first = c[0].lower()
    return PREFIX + first + c[1:]

rows = []
with open(SRC) as f:
    reader = csv.DictReader(f, delimiter='\t')
    fields = reader.fieldnames
    for r in reader:
        r['caption'] = prepend(r['caption'])
        rows.append(r)
print(f'Rows: {len(rows):,}')
print(f'\nSample (first 3):')
for r in rows[:3]:
    print(f'  {r["id"]}: {r["caption"][:120]!r}')

Path(DST).parent.mkdir(parents=True, exist_ok=True)
with open(DST, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
    w.writeheader()
    w.writerows(rows)
print(f'\n-> {DST}')

# Prefix coverage check
from collections import Counter
prefs = Counter()
for r in rows[:25000]:
    prefs[r['caption'][:50].lower()] += 1
print('\nTop 5 prefixes (25K sample):')
for p, n in prefs.most_common(5):
    print(f'  {n:5d}  {p!r}')
