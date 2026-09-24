"""
EXP-A v3: strip phase7_v1_train.tsv's LP-MC writing-task boilerplate

CORRECTION: P8 baseline (LP-MC NoQ, MC CLAP 0.185) was trained on phase7_v1_train.tsv,
NOT phase4_train.tsv. v1/v2 targeted the wrong TSV. This v3 strips the actual P8 boilerplate.

Target boilerplate phrases in phase7_v1_train.tsv (50k sample analysis):
  - "The low quality recording features a/an"  (>50% of captions, dominant)
  - "The song is a/an" / "the song is instrumental"
  - "This music is instrumental/electronic/..."
  - "This is a [genre] music piece"
  - "This [genre] song features"
  - "There is a [instrument] playing"
  - Common structural fragments:
      "the recording is noisy and in mono"  (LP-MC writing-task fingerprint)
      "the song is medium tempo / fast / slow"
      "the [music|song] is [adj] and [adj]"
      "the rhythm is being played by"
      "the [instrument] is playing in the background"

Strategy: replace boilerplate prefixes with content-bearing fragments to keep grammar.
  - "The low quality recording features [a/an] [genre] song that [verb]"
    → "[genre] song with [verb]"
  - "the recording is noisy and in mono" → ""  (drop entirely; production fingerprint)
  - "this is a [genre] music piece. there is" → "[genre] piece with"
  - "this music is instrumental." → ""  (the rest carries the info)
  - "this song features" → "with"
  - "the rhythm is being played by" → "with"
  - "the X is playing in the background" → "with X"

Output: ~/eval_tsvs_p100/phase7_v1_train_destructured.tsv

Test: regen NPZ from this TSV with text_features (T5) + text_features_c (CLAP) only,
keeping the existing audio mean/std cache. Retrain P8 from scratch with this NPZ.
Compare MC CLAP vs P8 baseline 0.185:
  - drops to Qwen-level → H10 confirmed (LP-MC writing-task boilerplate was the anchor)
  - stays near 0.185 → H10 falsified, look elsewhere
"""

import csv
import re
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

# Order: longer/more specific first
REPLACEMENTS = [
    # LP-MC writing-task signature
    (r'\bthe low quality recording features (a |an )?(live performance of (a |an )?)?', r''),
    (r'\bthe recording is noisy and in mono\b\s*[.,]?\s*', r''),
    (r'\b(?:as|because|since) it was probably recorded with (a |an )?(?:phone|mobile|webcam)\b\.?', r''),
    # "this is a/an X music piece"
    (r'\bthis is (a |an )?([a-z- ]+?) (music|song|track) piece\b\s*\.?\s*', r'\2 piece. '),
    (r'\bthis is (a |an )?([a-z- ]+?) (music|song|track)\b\s*\.?\s*', r'\2 \3. '),
    # "this music is instrumental. " etc.
    (r'\bthis music is (an? )?instrumental\b\s*\.?\s*', r''),
    (r'\bthis song is (an? )?instrumental\b\s*\.?\s*', r''),
    (r'\bthe song is (an? )?instrumental\b\s*\.?\s*', r''),
    (r'\bthis music is (an? )?', r''),
    (r'\bthis song is (an? )?', r''),
    # "the song is medium tempo / fast / slow"
    (r'\bthe song is (medium|fast|slow|moderate)(?: tempo)?\b\s*[.,]?\s*', r''),
    (r'\bthe tempo is (medium|fast|slow|moderate)(?: tempo)?\b\s*[.,]?\s*', r''),
    # "this song features" / "this song contains" / "this audio contains"
    (r'\bthis (song|music|track|audio) (features|contains|consists of) (a |an )?', r'with '),
    (r'\bthe (song|music|track) (features|contains|consists of) (a |an )?', r'with '),
    # "there is a/an X playing" / "X is playing"
    (r'\bthere is (a |an )?', r'with '),
    (r'\bthe rhythm is (being )?played by (a |an )?', r'with '),
    (r'\bthe ([a-z]+(?:\s+[a-z]+)?) is playing (in the background|the (lead|main|tune|melody))\b\s*[.,]?\s*', r'with \1 '),
    (r'\bthe ([a-z]+(?:\s+[a-z]+)?) is playing\s*', r'with \1 '),
    # Common closers
    (r'\bthe atmosphere is\b\s*', r'with '),
]
COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in REPLACEMENTS]


def strip_caption(c: str) -> str:
    out = c
    for cp, repl in COMPILED:
        out = cp.sub(repl, out)
    out = re.sub(r'\s+', ' ', out)
    out = re.sub(r'\s+([.,;:])', r'\1', out)
    out = re.sub(r'([.,;:])([a-zA-Z])', r'\1 \2', out)
    out = re.sub(r'\b(with )(?:with )+', 'with ', out, flags=re.IGNORECASE)
    out = re.sub(r'\b(and )(?:and )+', 'and ', out, flags=re.IGNORECASE)
    out = re.sub(r'^\s*[.,;:]+\s*', '', out)
    out = out.strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/mnt/HDD/kojiek/phase4_jamendo_data/phase7_v1_train.tsv')
    ap.add_argument('--dst', default='/home/kojiek/eval_tsvs_p100/phase7_v1_train_destructured.tsv')
    ap.add_argument('--preview', type=int, default=10)
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
    o_lens, n_lens = [], []
    for r in src_rows:
        orig = r['caption']
        new = strip_caption(orig)
        if not new or len(new) < 8:
            new = orig  # 防 empty
        nr = dict(r); nr['caption'] = new
        new_rows.append(nr)
        o_lens.append(len(orig.split()))
        n_lens.append(len(new.split()))

    o = np.array(o_lens); n = np.array(n_lens)
    print(f'\nOriginal:    mean={o.mean():.1f}  median={np.median(o):.0f}')
    print(f'Destructured: mean={n.mean():.1f}  median={np.median(n):.0f}')
    print(f'Reduction: {(o.mean() - n.mean()) / o.mean():.1%}')

    print(f'\n=== Preview (first {args.preview}) ===')
    for i in range(args.preview):
        print(f'\nrow {i}: id={src_rows[i]["id"]}')
        print(f'  ORIG ({o_lens[i]:3d}w): {src_rows[i]["caption"][:200]}')
        print(f'  STRP ({n_lens[i]:3d}w): {new_rows[i]["caption"][:200]}')

    tri = Counter()
    for r in new_rows[:25000]:
        toks = r['caption'].lower().replace('.', ' ').replace(',', ' ').split()
        tri.update(' '.join(toks[i:i+3]) for i in range(len(toks)-2))
    print(f'\nDestructured top 15 trigrams (25K sample):')
    for tg, cnt in tri.most_common(15):
        print(f'  {cnt:6d}  "{tg}"')

    # Coverage of original boilerplate prefix in destructured output
    orig_match = sum(1 for r in src_rows[:25000] if 'low quality recording' in r['caption'].lower())
    new_match = sum(1 for r in new_rows[:25000] if 'low quality recording' in r['caption'].lower())
    print(f'\n"low quality recording" coverage:  ORIG {orig_match}/25k ({orig_match/250:.1f}%)  STRP {new_match}/25k ({new_match/250:.1f}%)')

    with open(args.dst, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(new_rows)
    print(f'\n→ {args.dst}')


if __name__ == '__main__':
    main()
