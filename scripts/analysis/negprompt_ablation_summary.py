"""Summarise the CFG x negative-prompt ablation matrix.

Three things this pulls out that the raw cell files do not show directly:

  1. CFG-only vs CFG+negative at each cfg. The 'none' cells are textbook
     classifier-free guidance against the stored null, so (fidelity - none) is
     the part attributable to the negative wording and 'none - cfg0' is the part
     attributable to guidance itself. The single-point negprompt_reeval sweep
     confounded these two completely.

  2. A conditional-negative protocol, synthesised with no extra GPU: take the
     clean-prompt clips from a negative cell and the low-fidelity-prompt clips
     from the same arm's cfg 0 cell. That is exactly "apply the negative only
     where it does not contradict the prompt", and it answers the metric-gaming
     objection -- our fidelity negative pushes away from what 35.7% of MusicCaps
     prompts explicitly ask for, while PQ rewards exactly that push.

  3. Whether a PQ gain is accompanied by a level or brightness shift. A cell
     that gains PQ while also gaining several dB RMS has not been shown to gain
     quality, only loudness.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation')
KEYS = ('PQ', 'CE', 'CU', 'PC', 'clap')


def load():
    cells = {}
    for path in sorted(OUT.glob('*.json')):
        d = json.loads(path.read_text())
        cells[d['label']] = d
    return cells


def lofi_ids(cell):
    """Recover the split from the stored aggregates' membership counts."""
    return set(cell.get('lofi_ids', []))


def conditional(cell, base):
    """clean clips from `cell`, low-fidelity clips from `base` (same arm, cfg 0)."""
    per_c, per_b = cell['per_clip'], base['per_clip']
    lofi = cell.get('lofi_ids')
    if lofi is None:
        return None
    lofi = set(lofi)
    merged = {}
    for cid in per_b:
        src = per_b if cid in lofi else per_c
        if cid in src:
            merged[cid] = src[cid]
    return {k: float(np.mean([v[k] for v in merged.values()])) for k in KEYS} | {'n': len(merged)}


def main():
    cells = load()
    if not cells:
        print('no cells yet')
        return
    arms = sorted({c['arm'] for c in cells.values()})

    for arm in arms:
        base = cells.get(f'{arm}__cfg0__none')
        print(f'\n=== {arm} ===')
        hdr = f"{'cell':34s} {'PQ':>7s} {'dPQ':>7s} {'CLAP':>7s} {'dCLAP':>7s} {'RMSdB':>7s} {'cntrHz':>7s} {'crest':>6s}"
        print(hdr)
        rows = [c for c in cells.values() if c['arm'] == arm]
        rows.sort(key=lambda c: (c['cfg_strength'], c['negative_key']))
        for c in rows:
            a, s = c['aggregates']['full'], c['signal_stats']
            dpq = a['PQ'] - base['aggregates']['full']['PQ'] if base else float('nan')
            dcl = a['clap'] - base['aggregates']['full']['clap'] if base else float('nan')
            name = f"cfg{c['cfg_strength']}__{c['negative_key']}"
            print(f"{name:34s} {a['PQ']:7.4f} {dpq:+7.4f} {a['clap']:7.4f} {dcl:+7.4f} "
                  f"{s['rms_db_mean']:7.2f} {s['spectral_centroid_hz_mean']:7.0f} {s['crest_min']:6.2f}")

        # lo-fi vs clean split: where does the gain actually come from?
        print(f"\n{'cell':34s} {'PQ(lofi)':>9s} {'PQ(clean)':>10s} {'gap':>7s}")
        for c in rows:
            ag = c['aggregates']
            if ag['lofi_prompt']['n'] and ag['clean_prompt']['n']:
                lo, cl = ag['lofi_prompt']['PQ'], ag['clean_prompt']['PQ']
                name = f"cfg{c['cfg_strength']}__{c['negative_key']}"
                print(f'{name:34s} {lo:9.4f} {cl:10.4f} {cl - lo:+7.4f}')
        # conditional protocol: negative only where it does not fight the prompt
        if base:
            print(f"\n{'conditional (neg on clean only)':34s} {'PQ':>7s} {'dPQ':>7s} {'CLAP':>7s}")
            for c in rows:
                if c['negative_key'] == 'none' and c['cfg_strength'] == 0.0:
                    continue
                m = conditional(c, base)
                if not m:
                    continue
                name = f"cfg{c['cfg_strength']}__{c['negative_key']}"
                print(f"{name:34s} {m['PQ']:7.4f} "
                      f"{m['PQ'] - base['aggregates']['full']['PQ']:+7.4f} {m['clap']:7.4f}")


if __name__ == '__main__':
    main()
