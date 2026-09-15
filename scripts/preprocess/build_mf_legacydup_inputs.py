#!/usr/bin/env python3
"""Exact legacy duplicate-caption backfill; CPU-only, atomic, fail-closed audit.

Never repairs text. A failed gate is an artifact for operator review, not a
training authorization. Re-running verifies source hashes and reproduces bytes.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/caption10s_pipeline'))
from repair_multisent_first_entity_line import classify as inherited_classify

VERSION = 'mf-legacydup-structural-v1'
TAXONOMY = ['null', 'duplicate_id', 'missing_extra_id', 'row_order', 'metadata_drift',
            'cjk', 'turn_marker', 'degenerate_leadin', 'json_wrapper', 'character_run',
            'markdown_wrapper', 'url', 'latex', 'multiline', 'repeated_leadin',
            'bracket_wrapper', 'meta_disclaimer', 'question_terminal',
            'no_terminal_punctuation', 'missing_terminal_punctuation', 'code']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_json(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def classify(cap):
    if not isinstance(cap, str) or not cap.strip():
        return ['null']
    # Benign short captions must not be filtered by the corpus policy.
    tags = [t for t in inherited_classify(cap) if t != 'too_short']
    if not re.search(r'[.!?][\"\u201d\u2019\x27)\]}]*$', cap.rstrip()):
        tags.append('missing_terminal_punctuation')
    if re.search(r'\b(?:def \w+\(|import (?:os|sys)|function\s*\(|console\.log\()', cap):
        tags.append('code')
    return sorted(set(tags))


def backfill(base, old, limit):
    for rows in (base, old):
        if any(not r.get('id') or not isinstance(r.get('caption'), str) for r in rows):
            raise ValueError('null id/caption')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('duplicate_id')
    counts = collections.Counter(r['caption'] for r in old)
    selected = {r['id']: r['caption'] for r in old if counts[r['caption']] > 1}
    result, changes = [], []
    for row in base:
        cid, slot = row['id'].rsplit('_', 1)
        if not slot.isdigit():
            raise ValueError('base id lacks numeric slot suffix')
        replaced = cid in selected and len(changes) < limit
        cap = selected[cid] if replaced else row['caption']
        result.append({'id': row['id'], 'caption': cap})
        if replaced:
            changes.append({'id': row['id'], 'legacy_id': cid,
                            'old_group_size': counts[cap],
                            'caption_sha256': hashlib.sha256(cap.encode()).hexdigest()})
    return result, changes, len(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--science', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.science.read_text())
    for source in spec['sources'].values():
        if sha(source['path']) != source['sha256']:
            raise ValueError('source hash drift: ' + source['path'])
    read = lambda key: list(csv.DictReader(open(spec['sources'][key]['path'], newline=''), delimiter='\t'))
    base, old = read('base_tsv'), read('legacy_tsv')
    assert len(base) == spec['selection']['base_rows']
    assert len(old) == spec['selection']['legacy_rows']
    rows, changes, eligible = backfill(base, old, spec['selection']['max_replacements'])
    names = Path(spec['sources']['cache_list']['path']).read_text().splitlines()
    assert len(names) == len(rows) == len(set(names))
    assert [r['id'] for r in rows] == [r['id'] for r in base]
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['id', 'caption'], delimiter='\t', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    tsv = args.out / 'mf_legacydup_train.tsv'
    atomic(tsv, stream.getvalue())
    changed = {r['id'] for r in changes}
    defects = [{'id': r['id'], 'categories': classify(r['caption']),
                'replaced': r['id'] in changed, 'observed': r['caption'],
                'expected': 'plain English single-line caption with terminal punctuation'}
               for r in rows if classify(r['caption'])]
    write_json(args.out / 'defects.json', defects)
    write_json(args.out / 'replacements.json', changes)
    caps = collections.Counter(r['caption'] for r in rows)
    report = {'experiment_id': spec['experiment_id'], 'run_id': spec['run_id'],
              'status': 'failed' if defects else 'passed',
              'science_sha256': sha(args.science), 'corpus_sha256': sha(tsv),
              'classifier_version': VERSION, 'classifier_sha256': sha(__file__),
              'inherited_classifier_sha256': sha(ROOT / 'scripts/caption10s_pipeline/repair_multisent_first_entity_line.py'),
              'taxonomy': TAXONOMY, 'rows': len(rows), 'legacy_duplicate_rows': eligible,
              'replaced_rows': len(changes), 'unique_rate': len(caps) / len(rows),
              'shared_caption_rows': sum(n for n in caps.values() if n > 1),
              'max_duplicate_group': max(caps.values()), 'defect_rows': len(defects),
              'defect_counts': dict(collections.Counter(t for d in defects for t in d['categories'])),
              'defects_sha256': sha(args.out / 'defects.json'),
              'generation_stop_evidence': 'unavailable; historical captions reused without generation',
              'gpu_launch_allowed': False}
    write_json(args.out / 'gate.json', report)
    write_json(args.out / 'tsv_manifest.json', {'corpus_sha256': sha(tsv),
               'tsv_sha256': sha(tsv), 'sources': spec['sources'],
               'row_mapping': 'base order, exact ID; replacement map separate',
               'replacements_sha256': sha(args.out / 'replacements.json')})
    print(json.dumps({k: report[k] for k in ('status', 'rows', 'replaced_rows', 'unique_rate', 'shared_caption_rows', 'defect_rows', 'defect_counts')}, indent=2))
    return 2 if defects else 0


if __name__ == '__main__':
    sys.exit(main())
