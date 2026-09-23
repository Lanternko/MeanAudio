#!/usr/bin/env python
"""073 extension, step 0: how far apart are the two CFG branches' norms?

"Normalise first, then subtract" only differs from vanilla CFG to the extent
that |A| != |B|. If the conditional and negative branches already carry the
same norm, the cell is a near-no-op and must not be spent GPU time. This
records |A|, |B| and the angle between them at every ODE step on real clips,
under the same runpy harness as the 073 driver.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/kojiek/MeanAudio/scripts/eval')
import guidance_geometry_adg_apg_20260922 as G  # noqa: E402

REC = defaultdict(list)


def recorder(cfg_strength):
    import torch

    def ode_wrapper(self, t, r, latent, conditions, empty_conditions, cfg, q=None):
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if q is None:
            q = torch.full((len(latent),), 10, dtype=torch.long, device=latent.device)
        if cfg < 1.0:
            return self.predict_flow(latent, t, r, conditions, q)
        a = self.predict_flow(latent, t, r, conditions, q)
        b = self.predict_flow(latent, t, r, empty_conditions, q)
        na, nb = G._norm(a, 'global'), G._norm(b, 'global')
        cos = ((a * b).sum(dim=tuple(range(1, a.ndim)), keepdim=True) / (na * nb))
        REC['norm_a'] += na.flatten().tolist()
        REC['norm_b'] += nb.flatten().tolist()
        REC['ratio'] += (nb / na).flatten().tolist()
        REC['cos'] += cos.flatten().tolist()
        # the part of (A-B) that pre-normalisation would remove
        delta = a - b
        delta_pre = na * (a / na - b / nb)
        REC['rel_change_of_delta'] += (
            G._norm(delta_pre - delta, 'global') / G._norm(delta, 'global')
        ).flatten().tolist()
        return a + (cfg - 1.0) * delta

    return ode_wrapper


def main():
    from meanaudio.model.networks import MeanAudio
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = {}
    for family in ('N0', 'N8'):
        for cfg in (3.0, 4.5):
            REC.clear()
            original = MeanAudio.ode_wrapper
            MeanAudio.ode_wrapper = recorder(cfg)
            try:
                G.generate({'name': f'_prenorm_diag_{family}_{cfg}', 'family': family,
                            'cfg': cfg, 'geometry': 'vanilla',
                            'tsv': str(G.SUBSET_TSV), 'stage': 'diag'}, limit=rows)
            finally:
                MeanAudio.ode_wrapper = original
            out[f'{family}_cfg{cfg}'] = {
                k: {'mean': float(np.mean(v)), 'p05': float(np.percentile(v, 5)),
                    'p95': float(np.percentile(v, 95)), 'n': len(v)}
                for k, v in REC.items()}
    path = G.OUT / 'prenorm_branch_norm_diag.json'
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out, indent=2, sort_keys=True))
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
