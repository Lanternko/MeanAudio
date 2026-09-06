#!/usr/bin/env python
"""Strip fidelity/production-quality language out of the Caption 2.0 training captions.

Mechanism this targets (docs/experiments/results/phase8/fulltrack_pq_gap_mechanism_2026_08_28.md):

  Caption 2.0 mentions "quality" in 82.8% of its captions, almost always positively.
  MusicCaps mentions low-fidelity language in 37% of its prompts. Per-clip decomposition
  showed c2p0's worst decile is 47.1% low-fidelity prompts (best decile: 27.2%) -- the
  model learned a sharp fidelity axis and faithfully renders degraded audio when asked.
  The fulltrack corpus, which wins AES under canonical CFG 0, mentions quality in only
  7.3% of its captions and therefore largely ignores those words.

  Removing fidelity language from TRAINING captions should stop the model learning that
  axis at all, without touching anything else about the caption. Unlike the modular
  template arm, this changes one clearly-identified property rather than the whole
  surface form.

Everything else in each caption -- genre, instruments, mood, arrangement -- is preserved
verbatim. Only fidelity clauses and adjectives are removed.

Usage:
  python scripts/preprocess/build_fidelity_stripped_captions.py --out <tsv> [--limit N]
"""
import argparse
import csv
import re
import sys
from pathlib import Path

SRC = Path('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv')

# A whole sentence goes only if fidelity is the ONLY thing it talks about. A sentence that
# also names an instrument, a mood or an arrangement detail is kept and edited in place,
# so musical content is never collateral damage.
FIDELITY = re.compile(
    r'\b(production quality|recording quality|sound quality|audio quality|recording fidelity|'
    r'fidelity|mastering|mix(?:ed|ing)?|production|recorded|recording)\b', re.I)
MUSICAL = re.compile(
    r'\b(guitar|piano|drum\w*|bass|synth\w*|string\w*|percussion|melod\w*|harmon\w*|'
    r'rhythm\w*|tempo|chord\w*|vocal\w*|horn|brass|flute|violin|cello|organ|beat|groove|'
    r'genre|mood|atmosphere|energetic|melancholic|upbeat|calm|dark|bright|arrangement)\b', re.I)

# In-place removals: fidelity adjectives and the phrases that carry them.
PHRASES = [
    (r',?\s*with (?:a )?(?:high[- ]quality|excellent|professional|clear|crisp|pristine|'
     r'poor|low[- ]quality|amateur|muddy|noisy|rough)\s+(?:recording|production|mix|sound)'
     r'(?:\s+quality)?(?:\s+that[^.,]*)?', ''),
    (r'\b(?:the\s+)?(?:overall\s+)?(?:production|recording|sound|audio|mix)\s+quality\s+'
     r'(?:is|appears to be|seems)\s+[^.,;]*', ''),
    (r'\b(?:high|low)[- ]quality\b', ''),
    (r'\b(?:high|low)[- ]fidelity\b', ''),
    (r'\bwell[- ](?:balanced|mixed|produced|recorded)\b', ''),
    (r'\b(?:professionally|poorly|cleanly)\s+(?:recorded|mixed|produced|mastered)\b', ''),
    (r'\b(?:pristine|polished|immaculate|impeccable)\b', ''),
    (r'\b(?:amateur|amateurish|lo-?fi|muffled)\b', ''),
    (r'\bclear (?:mix|mixing|separation|sound|recording)\b', ''),
    (r'\bbalanced (?:mix|mixing|dynamics|sound)\b', ''),
    (r'\bgood (?:recording|sound|audio)(?:\s+fidelity)?\b', ''),
    (r'\bwith (?:no|minimal|noticeable) (?:distortion|noise|hiss|clipping)\b', ''),
]
CLEANUP = [
    (r'\s*,\s*,', ','), (r'\(\s*\)', ''), (r'\s+([,.;:])', r'\1'),
    (r'([,;:])\s*\.', '.'), (r'\.\s*\.', '.'), (r'\s{2,}', ' '),
    (r'^\s*[,;:]\s*', ''), (r'\b(is|are|has|have|with)\s*\.', '.'),
]


def strip(caption):
    sentences = re.split(r'(?<=[.!?])\s+', caption)
    kept = []
    for sent in sentences:
        if FIDELITY.search(sent) and not MUSICAL.search(sent):
            continue                      # pure fidelity verdict -- drop the sentence
        for pattern, repl in PHRASES:
            sent = re.sub(pattern, repl, sent, flags=re.I)
        for pattern, repl in CLEANUP:
            sent = re.sub(pattern, repl, sent)
        sent = sent.strip()
        if len(re.findall(r'[A-Za-z]+', sent)) >= 3:
            kept.append(sent)
    out = ' '.join(kept).strip()
    for pattern, repl in CLEANUP:
        out = re.sub(pattern, repl, out)
    out = out.strip()
    return out if len(re.findall(r'[A-Za-z]+', out)) >= 5 else caption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=SRC)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    with args.src.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if args.limit:
        rows = rows[:args.limit]

    changed = fallback = 0
    words_before = words_after = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter='\t')
        writer.writeheader()
        for row in rows:
            new = strip(row['caption'])
            changed += new != row['caption']
            fallback += new == row['caption'] and bool(FIDELITY.search(row['caption']))
            words_before += len(row['caption'].split())
            words_after += len(new.split())
            writer.writerow({**row, 'caption': new})

    n = len(rows)
    print(f'wrote {n} rows -> {args.out}')
    print(f'  changed          {changed:6d} ({changed / n * 100:5.1f}%)')
    print(f'  fell back intact {fallback:6d} (stripping would have emptied the caption)')
    print(f'  mean words       {words_before / n:.1f} -> {words_after / n:.1f}')


if __name__ == '__main__':
    sys.exit(main())
