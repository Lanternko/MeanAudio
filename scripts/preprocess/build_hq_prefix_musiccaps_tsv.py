#!/usr/bin/env python
"""081: MusicCaps TSV with every caption prefixed "High quality recording. ".

Used as --gen_tsv for the HQ-prompt cells of the quality-label arm; CLAP is still
scored against the plain captions. Records are read and written with the csv
module (five MusicCaps captions span two physical lines), no q_level column.
The output is checked to parse to the same ids, in order, with both csv and
pandas.

Usage: python build_hq_prefix_musiccaps_tsv.py OUT_TSV [--src MC_TSV]
"""
import argparse
import csv
import os
from pathlib import Path

import pandas as pd

PREFIX = 'High quality recording. '
MC_TSV = '/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv'


def read(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--src', default=MC_TSV)
    a = ap.parse_args()
    src = read(a.src)
    assert src and list(src[0]) == ['id', 'caption'], list(src[0])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'caption'], delimiter='\t', lineterminator='\n')
        w.writeheader()
        for r in src:
            w.writerow({'id': r['id'], 'caption': PREFIX + r['caption']})
    back = read(tmp)
    pdf = pd.read_csv(tmp, sep='\t', dtype=str, keep_default_na=False)
    assert [r['id'] for r in back] == [r['id'] for r in src]
    assert [r['caption'] for r in back] == [PREFIX + r['caption'] for r in src]
    assert list(pdf['id']) == [r['id'] for r in src] and list(pdf['caption']) == [r['caption'] for r in back]
    os.replace(tmp, out)
    print(f'wrote {out}: {len(back)} rows, ids identical to {a.src}')


if __name__ == '__main__':
    main()
