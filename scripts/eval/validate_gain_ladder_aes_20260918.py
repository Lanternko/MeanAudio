#!/usr/bin/env python3
"""063 preflight: immutable identity, provenance and capacity checks before any GPU work."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from gain_ladder_aes_20260918 import arms, binding, config, digest, ladder_arms  # noqa: E402


def main() -> int:
    c = config()
    problems = []

    for path in c['resource_budget']['writable_filesystems']:
        Path(path).mkdir(parents=True, exist_ok=True)
        fs = os.statvfs(path)
        if fs.f_bavail * fs.f_frsize < c['storage']['hard_stop_free_bytes']:
            print(f'capacity wait: {path}', flush=True)
            return 75

    for key in ('source_manifest', 'source_audio_root', 'tsv', 'aes_snapshot', 'clap_checkpoint'):
        if not Path(c[key]).exists():
            problems.append(f'missing input {key}: {c[key]}')

    if not problems:
        manifest = json.loads(Path(c['source_manifest']).read_text())
        from score_musiccaps_per_item import read_musiccaps_tsv
        ids = {r.id for r in read_musiccaps_tsv(Path(c['tsv']), expected_count=c['protocol']['rows'])}
        if set(manifest['audio_sha256']) != ids:
            problems.append('051 baseline manifest does not cover the registered rows')
        else:
            # Provenance: the retained 051 audio is the experimental unit, so it is
            # verified here rather than trusted, on a fixed deterministic sample.
            sample = sorted(ids)[::max(1, len(ids) // 40)]
            for i in sample:
                p = Path(c['source_audio_root']) / (i + '.flac')
                if not p.is_file() or digest(p) != manifest['audio_sha256'][i]:
                    problems.append(f'baseline audio drift: {i}')

    a = c['analysis']
    names = list(arms(c))
    ladder = ladder_arms(c)
    if a['reference_arm'] not in names or a['above_baseline_reference_arm'] not in names:
        problems.append('analysis arms not declared in ladder.arms')
    if a['replication_reference_arm'] not in names:
        problems.append('replication reference arm not declared')
    for arm in a['replication_targets']:
        if arm not in ladder:
            problems.append(f'replication target {arm} is not a ladder arm')
    if ladder and a['reference_arm'] != ladder[0]:
        problems.append('reference arm must be the quietest rung (the registered zero of the deltas)')
    if len(set(arms(c)[n]['gain_db'] for n in names)) != len(names):
        problems.append('duplicate gain_db among arms')
    if any(arms(c)[n]['gain_db'] > 0 for n in ladder):
        problems.append('ladder must not contain an above-baseline arm')
    if c['protocol']['scored_conditions'] != c['protocol']['rows'] * len(names):
        problems.append('declared condition count does not match rows x arms')

    for name, want in (('action', c['commands']['run'][-1]), ('preflight', c['commands']['preflight'][-1])):
        if not Path(want).is_file():
            problems.append(f'{name} script missing: {want}')

    if problems:
        for p in problems:
            print('INVALID: ' + p, flush=True)
        return 2
    print(f'PASS: contract {binding()[:12]} / {c["protocol"]["rows"]} rows x {len(names)} arms '
          f'({", ".join(sorted(names, key=lambda n: arms(c)[n]["gain_db"]))}) / '
          f'baseline provenance sampled', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
