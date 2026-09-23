#!/usr/bin/env python
"""L1 probe: can a text prompt move output LOUDNESS?

Easier control target than D1's defects: in the training corpus (peak-normalised
windows) captions with "loud" sit at -14.1 LUFS / crest 4.4 and "quiet|subdued|whisper"
at -19.5 LUFS / crest 6.2 (n=400 each; random -16.8), i.e. a 5.5 LU supervised
signal carried by compression, not peak level. If the model cannot reproduce even a
fraction of that from a loudness word, text control of level is absent too.

Rows: 4 stems x {base, loud/quiet as adjective (training style), loud/quiet as an
explicit volume phrase} x N samples, one row per sample.
"""
import argparse
import csv

STEMS = {
    'rock': ('rock song', 'with electric guitar, bass and drums'),
    'piano': ('solo piano piece', ''),
    'edm': ('electronic dance track', 'with synth and a steady beat'),
    'acoustic': ('acoustic guitar ballad', ''),
}


def cap(noun, tail, adj=None, vol=None):
    s = f"a {adj + ' ' if adj else ''}{noun}{' ' + tail if tail else ''}"
    return f'{s}, {vol}' if vol else s


def prompts():
    out = []
    for k, (noun, tail) in STEMS.items():
        out += [
            ('base', k, cap(noun, tail)),
            ('loudadj', k, cap(noun, tail, adj='loud')),
            ('quietadj', k, cap(noun, tail, adj='quiet')),
            ('loudvol', k, cap(noun, tail, vol='played very loudly at high volume')),
            ('quietvol', k, cap(noun, tail, vol='played very quietly at low volume')),
        ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = [(f'l1_{g}_{k}_{i:03d}', c) for g, k, c in prompts() for i in range(a.n)]
    assert len({r[0] for r in rows}) == len(rows)
    with open(a.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['id', 'caption'])
        w.writerows(rows)
    print(f'{a.out}: {len(rows)} rows')


if __name__ == '__main__':
    main()
