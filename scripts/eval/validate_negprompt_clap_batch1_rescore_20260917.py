#!/usr/bin/env python3
"""062 preflight: inputs, checkpoints, source results and capacity before any GPU work."""
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from negprompt_clap_batch1_rescore_20260917 import (  # noqa: E402
    EXPECTED, OUT, TSV, checkpoint, jobs, result_path, source_result)
from rescore_clap_batch1 import CLAP_CKPT  # noqa: E402

HARD_STOP_FREE_BYTES = 15_000_000_000


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fs = os.statvfs(OUT)
    if fs.f_bavail * fs.f_frsize < HARD_STOP_FREE_BYTES:
        print(f'capacity wait: {OUT}', flush=True)
        return 75

    problems = []
    for path in (TSV, CLAP_CKPT, ROOT / 'eval.py'):
        if not path.is_file():
            problems.append(f'missing input: {path}')
    for family, label, exp_id, qflags in jobs():
        if result_path(family, label).exists():
            continue
        if not checkpoint(exp_id).is_file():
            problems.append(f'{family}/{label}: missing checkpoint {checkpoint(exp_id)}')
        src = source_result(family, label)
        if not src.is_file():
            problems.append(f'{family}/{label}: missing source result {src}')
            continue
        r = json.loads(src.read_text())
        if r.get('exp_id') != exp_id or list(r.get('q_flags') or []) != qflags:
            problems.append(f'{family}/{label}: source result identity mismatch')
        if len(r.get('per_clip') or {}) != EXPECTED:
            problems.append(f'{family}/{label}: source per_clip is not {EXPECTED} rows')

    if problems:
        for p in problems:
            print('INVALID: ' + p, flush=True)
        return 2
    print(f'PASS: {len(jobs())} arms, inputs and source results present', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
