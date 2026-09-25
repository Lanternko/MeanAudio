#!/usr/bin/env python
"""075 follow-up: which part of the ODE trajectory carries the fidelity8 PQ gain,
and where does lab lose it?

Branch geometry and defect-direction projections ruled out "push size" and
"push away from programmatic defects". This measures the first-order effect
directly: fidelity8 CFG3 is applied only inside one third of the trajectory and
the model runs CFG0 (conditional branch only) elsewhere.

  in segment : u = 3*A + (1-3)*B_fidelity8   (the exact expression of the stock CFG3+neg cell)
  elsewhere  : u = A                          (the exact expression of the stock CFG0 cell)
  segments   : hi t>2/3 (9 of 25 steps), mid 1/3<t<=2/3 (8), lo t<=1/3 (8)

Generation uses the first N rows of the standard MusicCaps TSV with the stock
flags, so every clip is paired with the stock cfg0 / cfg3_neg cells of the same
checkpoint (same prompt, same noise). Replication gate before anything else:
segment 'all' and 'none' on control s14159265 must reproduce the stock
cfg3_neg / cfg0 audio sample-for-sample.

Scoring: -30 LUFS level match (level_match_rescore.py), then paired
G_seg(clip) = PQ(seg) - PQ(stock cfg0), both level matched.

Usage: python d2_075_segment_cfg.py [rows=256]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/kojiek/MeanAudio/scripts/eval')
import guidance_geometry_adg_apg_20260922 as G  # noqa: E402
from d2_075_branch_geometry import ARMS, SEEDS, ckpt  # noqa: E402

OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/d2_075_segment_cfg')
EV = Path.home() / 'eval_output_nvme'
SEGMENTS = {'hi': (2 / 3, 1.01), 'mid': (1 / 3, 2 / 3), 'lo': (-0.01, 1 / 3),
            'all': (-0.01, 1.01), 'none': (2.0, 3.0)}
GATE = ('control', 14159265)
STATE = {'seg': None}


def stock(arm, seed, c):
    return f'phase8_qwen_caption2p0_{ARMS[arm]}_noq_quarter_s{seed}_mc_mf25_{c}'


def recorder():
    def ode_wrapper(self, t, r, latent, conditions, empty_conditions, cfg, q=None):
        import torch
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if q is None:
            q = torch.full((len(latent),), 10, dtype=torch.long, device=latent.device)
        lo, hi = SEGMENTS[STATE['seg']]
        if lo < float(t[0]) <= hi:
            return (cfg * self.predict_flow(latent, t, r, conditions, q) +
                    (1 - cfg) * self.predict_flow(latent, t, r, empty_conditions, q))
        return self.predict_flow(latent, t, r, conditions, q)
    return ode_wrapper


def cell_dir(arm, seed, seg):
    return OUT / 'cells' / f'{arm}_s{seed}_{seg}'


def gen(arm, seed, seg, rows):
    from meanaudio.model.networks import MeanAudio
    d = cell_dir(arm, seed, seg)
    if (d / 'audio').exists() and len(list((d / 'audio').glob('*.flac'))) == rows:
        return d
    G.CKPT = ckpt(arm, seed)
    assert G.CKPT.exists(), G.CKPT
    G.AUDIO_ROOT = d
    STATE['seg'] = seg
    original = MeanAudio.ode_wrapper
    MeanAudio.ode_wrapper = recorder()
    try:
        G.generate({'name': 'audio', 'family': 'N8', 'cfg': 3.0, 'geometry': 'vanilla',
                    'tsv': str(G.MC_TSV), 'stage': 'diag'}, limit=rows)
    finally:
        MeanAudio.ode_wrapper = original
    n = len(list((d / 'audio').glob('*.flac')))
    assert n == rows, (d, n)
    return d


def replication_gate(rows):
    import soundfile as sf
    arm, seed = GATE
    res = {}
    for seg, c in (('all', 'cfg3_neg'), ('none', 'cfg0')):
        d = gen(arm, seed, seg, rows)
        ref = EV / stock(arm, seed, c) / 'audio'
        bad = 0
        files = sorted((d / 'audio').glob('*.flac'))
        for f in files:
            a, _ = sf.read(f, dtype='int16')
            b, _ = sf.read(ref / f.name, dtype='int16')
            bad += int(a.shape != b.shape or not np.array_equal(a, b))
        res[seg] = {'vs': c, 'n': len(files), 'mismatch': bad}
        print(f'[gate] {seg} vs stock {c}: {bad}/{len(files)} clips differ', flush=True)
    if any(v['mismatch'] for v in res.values()):
        raise SystemExit(f'[FAIL] replication gate: {res}')
    return res


def rescore(d, tsv):
    lvl = d.parent / f'{d.name}_lvl30'
    if not list(lvl.glob('*/per_clip.tsv')):
        subprocess.run([sys.executable, str(G.ROOT / 'scripts/eval/level_match_rescore.py'),
                        '--cell_dir', str(d), '--tsv', str(tsv)], check=True)
    return next(lvl.glob('*/per_clip.tsv'))


def per_clip(path, metric='PQ'):
    import csv
    with open(path, encoding='utf-8', newline='') as f:
        return {r['id']: float(r[metric]) for r in csv.DictReader(f, delimiter='\t')}


def stock_lvl(arm, seed, c):
    name = stock(arm, seed, c) + '_lvl30'
    d = EV / 'd2_075_lvl30' / name
    if not d.exists():
        d = EV / name
    return next(d.glob('*/per_clip.tsv'))


def main():
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    OUT.mkdir(parents=True, exist_ok=True)
    G.OUT = OUT
    tsv = OUT / f'_smoke_{rows}.tsv'
    gate = replication_gate(rows)
    for seed in SEEDS:
        for arm in ARMS:
            for seg in ('hi', 'mid', 'lo'):
                gen(arm, seed, seg, rows)
                print(f'[gen] {arm} s{seed} {seg} done', flush=True)
    rng = np.random.default_rng(20260926)
    out = {'rows': rows, 'gate': gate, 'cells': {}}
    for seed in SEEDS:
        for arm in ARMS:
            base = per_clip(stock_lvl(arm, seed, 'cfg0'))
            full = per_clip(stock_lvl(arm, seed, 'cfg3_neg'))
            ids = sorted(per_clip(rescore(cell_dir(arm, seed, 'hi'), tsv)))
            rec = {'all_stock': float(np.mean([full[i] - base[i] for i in ids]))}
            for seg in ('hi', 'mid', 'lo'):
                pq = per_clip(rescore(cell_dir(arm, seed, seg), tsv))
                g = np.array([pq[i] - base[i] for i in ids])
                boots = rng.choice(g, (4000, len(g))).mean(1)
                rec[seg] = {'G': float(g.mean()), 'ci95': [float(np.percentile(boots, 2.5)),
                                                          float(np.percentile(boots, 97.5))]}
            rec['sum_segments'] = sum(rec[s]['G'] for s in ('hi', 'mid', 'lo'))
            out['cells'][f'{arm}_s{seed}'] = rec
            print(f'[{arm} s{seed}] G_all={rec["all_stock"]:+.3f} hi={rec["hi"]["G"]:+.3f} '
                  f'mid={rec["mid"]["G"]:+.3f} lo={rec["lo"]["G"]:+.3f} sum={rec["sum_segments"]:+.3f}',
                  flush=True)
    path = OUT / f'segment_cfg_n{rows}.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
