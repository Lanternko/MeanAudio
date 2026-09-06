#!/usr/bin/env python
"""Bind a freshly trained checkpoint's sha256 into its preregistered contract cell.

`scripts/eval/validate_caption2p0_cfg0_report.py` compares `digest(checkpoint)`
against `cells[i].checkpoint_sha256`, but that value cannot exist at preregistration
time -- the checkpoint has not been trained yet, which is why contracts ship it as
`pending_runtime_output`. This fills it in exactly once, and refuses to repoint an
already-bound cell at a different checkpoint.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PLACEHOLDER = 'pending_runtime_output'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', type=Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f'FAIL missing checkpoint {args.checkpoint}')
    contract = json.loads(args.contract.read_text())
    cells = [c for c in contract['cells'] if c['cell_id'] == args.arm]
    if len(cells) != 1:
        raise SystemExit(f'FAIL unknown or duplicate arm {args.arm!r}')
    cell = cells[0]

    if str(args.checkpoint) != cell['checkpoint']:
        raise SystemExit(f"FAIL checkpoint path drift: {args.checkpoint} != {cell['checkpoint']}")
    digest = sha256(args.checkpoint)
    current = cell.get('checkpoint_sha256')
    if current not in (PLACEHOLDER, digest):
        raise SystemExit('FAIL cell is already bound to a different checkpoint; refusing to repoint')
    if current == digest:
        print(f'[OK] already bound: {digest}')
        return 0

    cell['checkpoint_sha256'] = digest
    args.contract.write_text(json.dumps(contract, indent=2) + '\n')
    print(f'[OK] bound {args.arm} checkpoint_sha256={digest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
