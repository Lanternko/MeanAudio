#!/usr/bin/env python3
"""073 preflight: identity, provenance and capacity checks before any GPU work.

Self-contained on purpose (same reason as the 072 preflight): the action module
hardcodes its own paths, so a shared validator would check the wrong contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import guidance_geometry_adg_apg_20260922 as gg  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    contract_path = Path(os.environ['GPU_QUEUE_CONTRACT'])
    c = json.loads(contract_path.read_text())
    problems = []

    for path in c['resource_budget']['writable_filesystems']:
        Path(path).mkdir(parents=True, exist_ok=True)
        fs = os.statvfs(path)
        if fs.f_bavail * fs.f_frsize < c['storage']['hard_stop_free_bytes']:
            print(f'capacity wait: {path}', flush=True)
            return 75

    for key, path in (('checkpoint', gg.CKPT), ('musiccaps_tsv', gg.MC_TSV),
                      ('pilot_subset_tsv', gg.SUBSET_TSV),
                      ('baseline_manifest', gg.BASELINE_MANIFEST),
                      ('clap_checkpoint', Path(c['clap_checkpoint'])),
                      ('aes_snapshot', Path(c['aes_snapshot']))):
        if not Path(path).exists():
            problems.append(f'missing input {key}: {path}')

    # The replication gate is only meaningful if the reference manifest is the run
    # this experiment claims (c2p0_slot0, CFG3 + fidelity8, 5521 rows).
    if gg.BASELINE_MANIFEST.exists():
        manifest = json.loads(gg.BASELINE_MANIFEST.read_text())
        if len(manifest['audio_sha256']) != c['protocol']['rows']:
            problems.append('051 baseline manifest does not cover the registered rows')
        if gg.MC_TSV.exists():
            import csv
            with open(gg.MC_TSV, encoding='utf-8', newline='') as f:
                ids = {r['id'] for r in csv.DictReader(f, delimiter='\t')}
            if set(manifest['audio_sha256']) != ids:
                problems.append('051 manifest ids do not equal the MusicCaps TSV ids')

    if gg.SUBSET_TSV.exists():
        import csv
        with open(gg.SUBSET_TSV, encoding='utf-8', newline='') as f:
            n = sum(1 for _ in csv.DictReader(f, delimiter='\t'))
        if n != c['protocol']['pilot_rows']:
            problems.append(f'pilot subset has {n} rows, contract says '
                            f'{c["protocol"]["pilot_rows"]}')

    # Identity: contract and action script must describe the same intervention.
    if c['geometry']['definitions'] != gg.GEOMETRIES:
        problems.append('contract geometry definitions do not match the action script')
    if c['analysis']['early_kill'] != {'pq': gg.GATE_PQ, 'crest': gg.GATE_CREST,
                                       'clap_floor': gg.GATE_CLAP_FLOOR}:
        problems.append('contract early-kill thresholds do not match the action script')
    if c['analysis']['loudness_gate_lu'] != gg.LOUDNESS_GATE_LU:
        problems.append('contract loudness gate does not match the action script')
    if c['protocol']['pilot_cells'] != len(gg.pilot_cells()) + len(gg.gate_cells()):
        problems.append('declared pilot cell count does not match the action script')
    if c['protocol']['clap_batch_size'] != 1 or c['protocol']['aes_batch_size'] != gg.AES_BATCH:
        problems.append('contract scoring batch sizes do not match the action script')

    # The two files this run must never edit.
    for name, want in (('networks.py', c['immutable']['networks_sha256']),
                       ('eval.py', c['immutable']['eval_sha256'])):
        got = digest(ROOT / ('meanaudio/model/networks.py' if name == 'networks.py' else 'eval.py'))
        if got != want:
            problems.append(f'{name} changed since the contract was pinned '
                            f'({got[:12]} != {want[:12]})')

    # The geometry algebra itself, before any GPU time is spent on it.
    eq = gg.equivalence_check(n_clips=8)
    if not eq['passed']:
        problems.append(f'geometry equivalence check failed: {eq["worst_relative_error"]}')

    for name, want in (('action', c['commands']['run'][1]),
                       ('preflight', c['commands']['preflight'][1])):
        if not Path(want).is_file():
            problems.append(f'{name} script missing: {want}')

    if problems:
        for p in problems:
            print('INVALID: ' + p, flush=True)
        return 2
    print(f'PASS: contract {digest(contract_path)[:12]} / '
          f'{c["protocol"]["pilot_cells"]} pilot+gate cells / '
          f'equivalence worst rel err '
          f'{max(eq["worst_relative_error"].values()):.2e} / '
          f'networks.py and eval.py unchanged', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
