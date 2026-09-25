#!/usr/bin/env python
"""075 follow-up P0: does the fidelity8 negative branch sit closer to the
conditional branch in the defect-labelled arm?

Hypothesis from the 075 results doc: in control/unlab, fidelity8 has no trained
meaning, so B (negative branch) is a far-away "domain average"; in lab it is
bound to programmatic defects, B becomes "the same music, degraded", and the
extrapolation direction A-B shrinks. Records per ODE step |A|, |B|, cos(A,B)
and |A-B|/|A| for lab / unlab / control x 3 training seeds under the exact
CFG3+neg generation flags (runpy harness of the 073 driver, CKPT swapped).

Usage: python d2_075_branch_geometry.py [rows=32]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/kojiek/MeanAudio/scripts/eval')
import guidance_geometry_adg_apg_20260922 as G  # noqa: E402

OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/d2_075_branch_geometry')
ARMS = {'lab': 'slot0clean_defectlab', 'unlab': 'slot0clean_defectunlab',
        'control': 'slot0clean_nmv2pair'}
SEEDS = (14159265, 27182818, 16180339)
REC = defaultdict(list)


def ckpt(arm, seed):
    exp = f'phase8_qwen_caption2p0_{ARMS[arm]}_noq_quarter_s{seed}_stage2_50000'
    return G.ROOT / 'exps' / exp / f'{exp}_ema_final.pth'


def recorder():
    import torch

    def ode_wrapper(self, t, r, latent, conditions, empty_conditions, cfg, q=None):
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if q is None:
            q = torch.full((len(latent),), 10, dtype=torch.long, device=latent.device)
        a = self.predict_flow(latent, t, r, conditions, q)
        b = self.predict_flow(latent, t, r, empty_conditions, q)
        na, nb = G._norm(a, 'global'), G._norm(b, 'global')
        dims = tuple(range(1, a.ndim))
        cos = (a * b).sum(dim=dims, keepdim=True) / (na * nb)
        REC['t'] += t.flatten().tolist()
        REC['norm_a'] += na.flatten().tolist()
        REC['norm_b'] += nb.flatten().tolist()
        REC['cos'] += cos.flatten().tolist()
        REC['delta_rel'] += (G._norm(a - b, 'global') / na).flatten().tolist()
        return a + (cfg - 1.0) * (a - b)

    return ode_wrapper


def summarise(rec):
    t = np.asarray(rec['t'])
    out = {}
    for k in ('norm_a', 'norm_b', 'cos', 'delta_rel'):
        v = np.asarray(rec[k])
        out[k] = {'mean': float(v.mean()), 'p05': float(np.percentile(v, 5)),
                  'p95': float(np.percentile(v, 95)), 'n': int(v.size)}
        # early / mid / late thirds of the trajectory (t runs 1 -> 0)
        for name, lo, hi in (('t_hi', 2 / 3, 1.01), ('t_mid', 1 / 3, 2 / 3), ('t_lo', -0.01, 1 / 3)):
            m = (t >= lo) & (t < hi)
            out[k][name] = float(v[m].mean()) if m.any() else None
    return out


def main():
    from meanaudio.model.networks import MeanAudio
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    OUT.mkdir(parents=True, exist_ok=True)
    G.OUT, G.AUDIO_ROOT = OUT, OUT / '_audio'
    path = OUT / f'branch_geometry_n{rows}.json'
    out = json.loads(path.read_text()) if path.exists() else {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f'{arm}_s{seed}'
            if key in out:
                continue
            G.CKPT = ckpt(arm, seed)
            assert G.CKPT.exists(), G.CKPT
            REC.clear()
            original = MeanAudio.ode_wrapper
            MeanAudio.ode_wrapper = recorder()
            try:
                G.generate({'name': f'{key}_N8_cfg3', 'family': 'N8', 'cfg': 3.0,
                            'geometry': 'vanilla', 'tsv': str(G.SUBSET_TSV),
                            'stage': 'diag'}, limit=rows)
            finally:
                MeanAudio.ode_wrapper = original
            out[key] = {'checkpoint': str(G.CKPT), 'rows': rows, **summarise(REC)}
            path.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
            s = out[key]
            print(f'[{key}] cos={s["cos"]["mean"]:.4f} |A-B|/|A|={s["delta_rel"]["mean"]:.4f} '
                  f'|B|/|A|={s["norm_b"]["mean"] / s["norm_a"]["mean"]:.4f}', flush=True)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
