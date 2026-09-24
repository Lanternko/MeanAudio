"""
EXP-A v2: cleaner template-strip on phase4_train.tsv

Fix v1 issues:
  - v1 deleted phrases → orphan punctuation, broken grammar
  - v1 left "the final segment" 3.6K times (regex didn't catch all forms)

v2 strategy: REPLACE template markers with neutral connectors instead of deleting.
This preserves grammar while flattening the temporal-narrative structure.

Replacement table:
  "this <genres> track begins with"   → "<genres> features"
  "this <genres> track is/has"        → "<genres> with"
  "begins with a/an"                  → "with"
  "transitions into a/an"             → "and"
  "then transitions into"             → "and"
  "in the final segment/part"         → "and"
  "in the final"                      → "and"
  "the piece then transitions"        → "and"
  "as the piece progresses"           → "and"
  "shifts to a/an"                    → "and"
  "the (piece|track|music) (then|finally|later) (transitions|shifts|moves)"  → "and"
  "setting the tone for the piece"    → ""
  "creating a/an"                     → "with"
  "evoking a/an"                      → "with"
  "accompanied by"                    → "and"
  "characterized by"                  → "with"

This still removes the temporal structure (begin/middle/end) but keeps grammatical
sentence flow.

Validation: report mean tokens before/after, top trigrams after, sample diff.
"""

import csv
import re
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

# Order matters: longer / more specific patterns first
REPLACEMENTS = [
    (r'\bthis ([a-z, &-]+) (track|piece|composition|soundtrack|song) begins with (a |an )?', r'\1 with '),
    (r'\bthis ([a-z, &-]+) (track|piece|composition|soundtrack|song) (is|has|features|features a|features an)\s*', r'\1 with '),
    (r'\bthis ([a-z, &-]+) (track|piece|composition|soundtrack|song)\s*(is|has)?\s*characterized by\s*', r'\1 with '),
    (r'\bthe (track|piece|music) (then |finally |later |subsequently )?(transitions into|shifts to|moves to|progresses to|gives way to) (a |an )?', r'and '),
    (r'\b(then |finally |later |subsequently )?(transitions into|shifts to|moves to|progresses to) (a |an )?', r'and '),
    (r'\b(track|piece|composition) begins with (a |an )?', r'with '),
    (r'\bbegins with (a |an )?', r'with '),
    (r'\bin the final (segment|part|section)\b\s*,?\s*', r'and '),
    (r'\bin the final\b\s*,?\s*', r'and '),
    (r'\bas the (piece|track|music) progresses\b,?\s*', r'and '),
    (r'\bsetting the tone (for|of) (the rest of )?the (piece|track|music)\b,?\s*', r''),
    (r'\bevoking (a |an )?', r'with '),
    (r'\baccompanied by (a |an )?', r'and '),
    (r'\bcreating (a |an )?', r'with '),
    (r'\bcharacterized by (a |an )?', r'with '),
]
COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in REPLACEMENTS]


def strip_caption(c: str) -> str:
    out = c
    for cp, repl in COMPILED:
        out = cp.sub(repl, out)
    # Cleanup: collapse spaces
    out = re.sub(r'\s+', ' ', out)
    # Fix punctuation gaps
    out = re.sub(r'\s+([.,;:])', r'\1', out)
    out = re.sub(r'([.,;:])([a-zA-Z])', r'\1 \2', out)
    # Collapse repeated "and "
    out = re.sub(r'\b(and )(?:and )+', 'and ', out, flags=re.IGNORECASE)
    # Capitalize first letter
    out = out.strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/mnt/HDD/kojiek/phase4_jamendo_data/phase4_train.tsv')
    ap.add_argument('--dst', default='/home/kojiek/eval_tsvs_p100/phase4_train_destructured.tsv')
    ap.add_argument('--preview', type=int, default=8)
    args = ap.parse_args()

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)

    src_rows = []
    with open(args.src) as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        for r in reader:
            src_rows.append(r)
    print(f'讀 {len(src_rows):,} rows')

    new_rows = []
    o_lens = []
    n_lens = []
    for r in src_rows:
        orig = r['caption']
        new = strip_caption(orig)
        if not new or len(new) < 10:
            new = orig
        nr = dict(r); nr['caption'] = new
        new_rows.append(nr)
        o_lens.append(len(orig.split()))
        n_lens.append(len(new.split()))

    o = np.array(o_lens); n = np.array(n_lens)
    print(f'\nOriginal:    mean={o.mean():.1f}  median={np.median(o):.0f}  p95={np.percentile(o,95):.0f}')
    print(f'Destructured: mean={n.mean():.1f}  median={np.median(n):.0f}  p95={np.percentile(n,95):.0f}')
    print(f'Reduction: {(o.mean() - n.mean()) / o.mean():.1%}')

    print(f'\n=== Preview (first {args.preview}) ===')
    for i in range(args.preview):
        print(f'\nrow {i}: id={src_rows[i]["id"]}')
        print(f'  ORIG ({o_lens[i]:3d}w): {src_rows[i]["caption"][:200]}')
        print(f'  STRP ({n_lens[i]:3d}w): {new_rows[i]["caption"][:200]}')

    # Top trigrams of destructured
    tri = Counter()
    for r in new_rows[:25000]:
        toks = r['caption'].lower().replace('.', ' ').replace(',', ' ').split()
        tri.update(' '.join(toks[i:i+3]) for i in range(len(toks)-2))
    print(f'\nDestructured TSV top 15 trigrams (25K sample):')
    for tg, n in tri.most_common(15):
        print(f'  {n:6d}  "{tg}"')

    # Write
    with open(args.dst, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(new_rows)
    print(f'\n→ {args.dst}')


if __name__ == '__main__':
    main()
