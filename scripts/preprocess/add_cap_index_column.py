#!/usr/bin/env python3
"""Bind a single-caption train TSV to a slot of an existing stacked text overlay.

Adds a `cap_index` column giving, per row, which slot of the stacked overlay holds
that row's caption. Lets a single-caption arm reuse a multi-caption overlay instead
of re-encoding an identical ~76 GiB copy of the same T5 features.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--extraction-tsv', type=Path, required=True,
                        help='id / caption / source_slot rows that the overlay was built from')
    parser.add_argument('--train-tsv', type=Path, required=True)
    parser.add_argument('--out-tsv', type=Path, required=True)
    parser.add_argument('--slot-order', default='slot0,slot1,slot3',
                        help='registered stacking order of the overlay')
    args = parser.parse_args()

    order = args.slot_order.split(',')
    slots: dict[str, dict[str, str]] = defaultdict(dict)
    with args.extraction_tsv.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            slots[row['id']][row['source_slot']] = row['caption'].strip()

    with args.train_tsv.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if 'cap_index' in fields:
        raise ValueError('train TSV already has a cap_index column')

    histogram: dict[int, int] = defaultdict(int)
    for index, row in enumerate(rows):
        stacked = slots.get(row['id'])
        if stacked is None:
            raise ValueError(f'row {index} id {row["id"]!r} is absent from the extraction TSV')
        caption = row['caption'].strip()
        matches = [pos for pos, slot in enumerate(order) if stacked.get(slot) == caption]
        if len(matches) != 1:
            raise ValueError(
                f'row {index} id {row["id"]!r}: caption matches {len(matches)} of the {len(order)} stacked slots'
            )
        row['cap_index'] = matches[0]
        histogram[matches[0]] += 1

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out_tsv.with_suffix(args.out_tsv.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ['cap_index'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(args.out_tsv)
    print(f'rows={len(rows)} slot_order={order} histogram=' +
          ' '.join(f'{order[k]}(idx{k})={histogram[k]}' for k in sorted(histogram)))


if __name__ == '__main__':
    main()
