#!/usr/bin/env python3
"""072 preflight: identity, provenance and capacity checks before any GPU work.

Self-contained on purpose: the 063 preflight imports config()/binding() from the 063
action module, where CONTRACT is a hardcoded path, so reusing it here would validate
the wrong contract.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from gain_ladder_aes_20260918 import digest  # noqa: E402
from shallow_limit_ladder_aes_20260922 import (CROSS_CHECK_RUNG_DB, FLOOR065,  # noqa: E402
                                               LADDER_DB, LIMIT_DB, configure)


def main() -> int:
    contract_path = Path(os.environ['GPU_QUEUE_CONTRACT'])
    c = json.loads(contract_path.read_text())
    cfg = configure()
    problems = []

    for path in c['resource_budget']['writable_filesystems']:
        Path(path).mkdir(parents=True, exist_ok=True)
        fs = os.statvfs(path)
        if fs.f_bavail * fs.f_frsize < c['storage']['hard_stop_free_bytes']:
            print(f'capacity wait: {path}', flush=True)
            return 75

    for key in ('source_manifest', 'source_audio_root', 'tsv', 'aes_snapshot', 'clap_checkpoint'):
        if not Path(cfg[key]).exists():
            problems.append(f'missing input {key}: {cfg[key]}')

    # The two prior runs this one is read against must still be on disk, item by item:
    # 063 supplies the replication gates inside analyze(), 065 the D18 cross-check.
    for name, root in (('063 gain ladder items', Path(cfg['gain_ladder_063_items'])),
                       ('065 floor ladder items', Path(FLOOR065) / 'items')):
        if not root.is_dir():
            problems.append(f'missing {name}: {root}')

    if not problems:
        manifest = json.loads(Path(cfg['source_manifest']).read_text())
        from score_musiccaps_per_item import read_musiccaps_tsv
        ids = {r.id for r in read_musiccaps_tsv(Path(cfg['tsv']), expected_count=cfg['rows'])}
        if set(manifest['audio_sha256']) != ids:
            problems.append('051 baseline manifest does not cover the registered rows')
        else:
            # Provenance: the retained 051 audio is the experimental unit, so it is
            # verified rather than trusted, on a fixed deterministic sample.
            sample = sorted(ids)[::max(1, len(ids) // 40)]
            for i in sample:
                p = Path(cfg['source_audio_root']) / (i + '.flac')
                if not p.is_file() or digest(p) != manifest['audio_sha256'][i]:
                    problems.append(f'baseline audio drift: {i}')
        # 065 must actually hold the rung this run is gated against.
        missing = [i for i in sorted(ids)[:1]
                   if not (Path(FLOOR065) / 'items' / (i + '.json')).is_file()]
        if missing:
            problems.append(f'065 items incomplete: {missing[0]}')
        elif CROSS_CHECK_RUNG_DB not in json.loads(
                (Path(FLOOR065) / 'summary.json').read_text())['config']['limit_db']:
            problems.append(f'065 did not measure D{CROSS_CHECK_RUNG_DB}; no cross-check possible')

    l = c['ladder']
    if l['scalar_rungs_db'] != LADDER_DB or l['limited_rungs_db'] != LIMIT_DB:
        problems.append('contract rungs do not match the action script')
    if 0 not in LADDER_DB or 18 not in LADDER_DB:
        problems.append('z0 and m18 rungs are required by the 063 replication gates')
    if not set(LIMIT_DB) <= set(LADDER_DB):
        problems.append('every limited rung needs its scalar rung as the matched control')
    if CROSS_CHECK_RUNG_DB not in LIMIT_DB:
        problems.append('the cross-check rung must be measured in this run')
    if c['protocol']['scored_conditions'] != cfg['rows'] * (
            2 * len(LADDER_DB) + 2 * len(LIMIT_DB) + len(LIMIT_DB) + 1):
        problems.append('declared condition count does not match rows x conditions')

    for name, want in (('action', c['commands']['run'][-1]),
                       ('preflight', c['commands']['preflight'][-1])):
        if not Path(want).is_file():
            problems.append(f'{name} script missing: {want}')

    if problems:
        for p in problems:
            print('INVALID: ' + p, flush=True)
        return 2
    print(f'PASS: contract {digest(contract_path)[:12]} / {cfg["rows"]} rows / '
          f'scalar {LADDER_DB} / limited {LIMIT_DB} / baseline provenance sampled / '
          f'065 D{CROSS_CHECK_RUNG_DB} available', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
