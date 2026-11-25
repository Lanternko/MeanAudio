#!/usr/bin/env python3
"""MIR four readings on the perturbed rec clips of 126b (the 126b mir.json only had the 619 refs).

CPU only; same scorer (mir_metrics.score_mir via adapters.Mir) and shard runner as job 176.
Output: OUT_ROOT/mir/126b shards, read by judge_coverage for the rec column.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import common as C  # noqa: E402
import adapters  # noqa: E402
import score_coverage as SC  # noqa: E402


def main():
    items = [it for it in C.load_items_file('126b') if it.get('block') == 'rec']
    print(f'rec items: {len(items)}', flush=True)
    if '--dry-run' in sys.argv:
        return 0
    t0 = time.time()
    out = C.OUT_ROOT / 'mir' / '126b'
    out.mkdir(parents=True, exist_ok=True)
    r = SC.run_protocol('mir', '126b', items, out, adapters.build('mir'), '0/1')
    print(r, f'{time.time() - t0:.0f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
