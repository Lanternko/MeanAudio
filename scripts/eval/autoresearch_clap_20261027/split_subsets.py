#!/usr/bin/env python
"""Pre-registered MusicCaps split for the autoresearch inference-search line.

The split is by csv record (the canonical TSV has five captions whose quoted text
spans a physical line break; slicing by physical line is off by one, see memory
reference_musiccaps_tsv_multiline_captions).

  dev       records [0, 1024)      search set (all selection happens here)
  heldout   records [1024, 2048)   confirmation set, never used for selection
  rest      records [2048, 5521)   reported only inside full-5521 runs

Writes the three subset TSVs and a manifest with record counts and sha256, so the
split cannot drift after the pre-registration is committed.

    python scripts/eval/autoresearch_clap_20261027/split_subsets.py --out <dir>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

MC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
SPLITS = {'dev256': (0, 256), 'dev1024': (0, 1024),
          'heldout1024': (1024, 2048), 'rest3473': (2048, None)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tsv', default=str(MC_TSV))
    p.add_argument('--out', required=True, help='directory for the subset TSVs + manifest.json')
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.tsv, encoding='utf-8', newline=''), delimiter='\t'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'source_tsv': str(a.tsv), 'source_sha256': sha256(a.tsv),
                'n_records': len(rows), 'rule': 'csv-record slices; no shuffling',
                'split_by_file_order': True, 'splits': {}}
    for name, (lo, hi) in SPLITS.items():
        part = rows[lo:hi]
        dst = out / f'{name}.tsv'
        with open(dst, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
            w.writeheader()
            w.writerows(part)
        manifest['splits'][name] = {'first_record': lo, 'n': len(part),
                                    'sha256': sha256(dst), 'ids_sha256': hashlib.sha256(
                                        '\n'.join(r['id'] for r in part).encode()).hexdigest()}
    if len({r['id'] for r in rows}) != len(rows):
        raise SystemExit('duplicate ids in source TSV')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
