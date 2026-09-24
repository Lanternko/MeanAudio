"""
EXP-A prep: strip phase4_train.tsv's heavy mu-llama template structure
and prepare a "destructured" TSV for retraining.

Hypothesis (H10): the temporal-narrative template ("begins with a → transitions
into → in the final segment") in phase4 captions provides an inductive anchor
that helps MeanAudio learn audio↔text mapping. Qwen captions lack this
structure, hence fail to train.

Test: train P-stripped on phase4_train_stripped.tsv (same captions, template
phrases removed). Compare to P8 baseline (CLAP 0.185).
  - If degrades to Qwen-level (~0.06) → H10 confirmed (template was anchor)
  - If stays near 0.185 → template not key, look elsewhere
  - Intermediate → partial contribution

Strip patterns (regex applied case-insensitive):
  - "this [genres] track" / "this [genres] track begins with"
  - "begins with a/an", "track begins with"
  - "transitions into a/an", "then transitions into"
  - "the final segment / in the final segment / in the final part / in the final"
  - "shifts to a/an", "the piece then", "the track then", "as the piece progresses"

Usage:
  python exp_a_strip_template.py --src phase4_train.tsv --dst phase4_train_stripped.tsv
"""

import csv
import re
import argparse
from pathlib import Path
from collections import Counter

# 樣板片語 — 用 word boundary，case-insensitive
PATTERNS = [
    r'\bthis [a-z, &-]+ (track|piece|composition|soundtrack|song)\b',
    r'\b(track|piece) begins with (a |an )?',
    r'\bbegins with (a |an )?',
    r'\bthen transitions into (a |an )?',
    r'\btransitions into (a |an )?',
    r'\bin the final (segment|part|section)\b,?\s*',
    r'\bthe (piece|track|music) (then |finally |later |subsequently )?(transitions|shifts|moves|progresses)\b',
    r'\bas the (piece|track|music) progresses\b,?\s*',
    r'\bshifts to (a |an )?',
    r'\bsetting the tone for (the rest of )?the piece\b,?\s*',
    r'\bcreating (a |an )?',  # very common filler
    r'\bevoking (a |an )?',
    r'\baccompanied by (a |an )?',
    r'\bfeaturing (a |an )?',
    r'\bcharacterized by (a |an )?',
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in PATTERNS]


def strip_caption(c: str) -> str:
    out = c
    for cp in COMPILED:
        out = cp.sub(' ', out)
    # collapse multiple spaces, fix punctuation
    out = re.sub(r'\s+', ' ', out)
    out = re.sub(r'\s*([.,;:])', r'\1', out)
    out = re.sub(r'^\s*[.,;:]+\s*', '', out)
    return out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/mnt/HDD/kojiek/phase4_jamendo_data/phase4_train.tsv')
    ap.add_argument('--dst', default='/home/kojiek/eval_tsvs_p100/phase4_train_stripped.tsv')
    ap.add_argument('--preview', type=int, default=10)
    args = ap.parse_args()

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)

    # 讀
    rows = []
    with open(args.src) as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)
    print(f'讀 {len(rows):,} rows from {args.src}')

    # 處理 + 統計
    orig_lengths = []
    new_lengths = []
    for r in rows:
        orig = r['caption']
        new = strip_caption(orig)
        if not new:
            new = orig  # 防止空字串
        r['caption'] = new
        orig_lengths.append(len(orig.split()))
        new_lengths.append(len(new.split()))

    import numpy as np
    o = np.array(orig_lengths)
    n = np.array(new_lengths)
    print(f'\nOriginal:  mean={o.mean():.1f} words  median={np.median(o):.0f}')
    print(f'Stripped:  mean={n.mean():.1f} words  median={np.median(n):.0f}')
    print(f'Reduction: {(o.mean() - n.mean()) / o.mean():.1%}')

    # Preview
    print(f'\n=== Preview (first {args.preview} rows) ===')
    for i in range(args.preview):
        print(f'\nrow {i}: id={rows[i]["id"]}')
        print(f'  ORIG  ({orig_lengths[i]} w): {rows[i]["caption"][:200]}...' if False else '')
        # We already overwrote caption — need to print before. Fix:

    # Re-read source to compare
    src_rows = []
    with open(args.src) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            src_rows.append(row)
    print(f'\n=== Preview comparison ===')
    for i in range(args.preview):
        print(f'\nrow {i}: id={src_rows[i]["id"]}')
        print(f'  ORIG ({orig_lengths[i]:3d}w): {src_rows[i]["caption"][:160]}')
        print(f'  STRP ({new_lengths[i]:3d}w): {rows[i]["caption"][:160]}')

    # 寫
    with open(args.dst, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n→ {args.dst}')

    # 統計新 trigram top
    from collections import Counter
    tri = Counter()
    for r in rows[:10000]:
        toks = r['caption'].lower().replace('.', ' ').replace(',', ' ').split()
        tri.update(' '.join(toks[i:i+3]) for i in range(len(toks)-2))
    print(f'\nStripped TSV top 10 trigrams (10K sample):')
    for tg, n in tri.most_common(10):
        print(f'  {n:5d}  "{tg}"')


if __name__ == '__main__':
    main()
