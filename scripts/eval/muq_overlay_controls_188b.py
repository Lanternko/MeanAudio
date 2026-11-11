#!/usr/bin/env python3
"""188b: fill in the missing control overlays of 188 Part D (post hoc, 2026-11-11).

188 scored the control sounds (1 kHz sine, another music clip) only at r = 0 and 18 LU, so the
tightened attack rule (treatment must beat the controls on A1 and on CE) left the 12 treatment
cells at r = -6 / 6 / 12 undetermined.  This scores sine1k and music at those three levels on the
same 500 test ids, real and generated bases, with the 188 driver's own mix / scorer code, and writes
per_clip/T_{real,gen}_{sine1k,music}_r{-6,6,12}.tsv next to the 188 tables.

Replication gate first: T_real_sine1k_r18 is re-scored under a scratch key and compared with the
table on disk (AES drifts ~1e-3 with batch composition; anything larger stops the run).
Direct run (about 4 min of GPU), not a queue job; the launcher holds the notify trap.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import muq_attractor_probe_188 as P  # noqa: E402

LEVELS = (12, 6, -6)
CTLS = ('sine1k', 'music')
TOL = {'A1_MI': 5e-3, 'PQ': 1e-2, 'CE': 1e-2}


def main():
    P.POOL = ThreadPoolExecutor(P.NTHREAD)
    P.H.load_scales()
    P.H.P = P.H.Paths(P.OUT / 'h183')
    P.ATT.update(json.loads((P.OUT / 'attractor.json').read_text()))
    sel = P.load_selection()
    pl = P.pools()
    test, dev, caps = pl['test'], pl['dev'], P.caps_all()
    assert len(test) == 500 and sorted(test) == sorted(P.tsv_read(P.OUT / 'per_clip/T_real_base.tsv'))

    def builder(getter, ov, r):
        if ov == 'music':
            return lambda i: P.mix(getter(i), P.real30(dev[test.index(i) % len(dev)]), r)
        o = P.overlay(ov, sel)
        return lambda i: P.mix(getter(i), o, r)

    old = P.tsv_read(P.OUT / 'per_clip/T_real_sine1k_r18.tsv')
    new = P.run_set('X188b_gate_real_sine1k_r18', test, builder(P.real30, 'sine1k', 18), caps, stage='188b-gate')
    gate = {c: float(np.max(np.abs(P.arr(new, test, c) - P.arr(old, test, c)))) for c in TOL}
    P.log(f'replication gate (max abs diff vs disk): {gate}')
    P.jdump(P.OUT / 'controls_188b_gate.json', {'max_abs_diff': gate, 'tol': TOL})
    if any(gate[c] > TOL[c] for c in TOL):
        P.log('gate failed: stop')
        return 1
    for base, getter in (('real', P.real30), ('gen', P.gen30)):
        for ov in CTLS:
            for r in LEVELS:
                P.run_set(P.cell_key(base, ov, r), test, builder(getter, ov, r), caps, stage='188b')
    P.progress('188b-done', '', 0, 0)
    return 0


if __name__ == '__main__':
    sys.exit(main())
