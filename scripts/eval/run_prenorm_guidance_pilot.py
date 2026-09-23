#!/usr/bin/env python
"""073 extension — "normalise first, then subtract" (pre-normalised CFG).

073 covers subtract-then-rescale (ADG-style) and orthogonal projection (APG).
The third ordering in the 2026-09-22 direction memo is not covered: equalise the
two branches' norms BEFORE taking the difference.

    A_hat = A/|A| , B_hat = B/|B|
    delta_pre = |A| * (A_hat - B_hat)
    delta_mix = (1-beta)*delta + beta*delta_pre
    v = A + (c-1)*delta_mix                       beta=0 -> vanilla

Mechanism it isolates: when |A| != |B|, part of (A-B) is a pure MAGNITUDE
difference between the branches rather than a direction difference. ADG cannot
remove it (it rescales the composed vector, after the difference is taken); APG
cannot either (it splits the difference along A, not along B). Pre-normalising
removes it at the source.

Not a no-op: prenorm_branch_norm_diag.json measures |B|/|A| = 1.0006 (N0) /
1.0353 (N8) on average, but the branches are nearly parallel (cos 0.993/0.975),
so |A-B| is small and those norm differences move the update by 12-14% on
average (p95 up to 48% for N8).

Reuses the 073 driver verbatim for generation, scoring, pairing, the loudness
gate and the early-kill thresholds; vanilla partners are loaded from the 073
cell cache, not regenerated. Writes its own summary and never touches
073's summary.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/kojiek/MeanAudio/scripts/eval')
import guidance_geometry_adg_apg_20260922 as G  # noqa: E402

NEW_GEOMETRIES = {
    'prenorm_b1.0': {'kind': 'prenorm', 'beta': 1.0, 'axis': 'global'},
    'prenorm_b0.5': {'kind': 'prenorm', 'beta': 0.5, 'axis': 'global'},
}
SUMMARY = G.OUT / 'prenorm_extension_summary.json'


def patched_make_ode_wrapper(spec):
    """G.make_ode_wrapper plus the prenorm kind."""
    if spec['kind'] != 'prenorm':
        return _orig_make(spec)
    import torch

    def ode_wrapper(self, t, r, latent, conditions, empty_conditions, cfg_strength, q=None):
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if q is None:
            q = torch.full((len(latent),), 10, dtype=torch.long, device=latent.device)
        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, r, conditions, q)
        a = self.predict_flow(latent, t, r, conditions, q)
        b = self.predict_flow(latent, t, r, empty_conditions, q)
        axis = spec['axis']
        na, nb = G._norm(a, axis), G._norm(b, axis)
        delta = a - b
        delta_pre = na * (a / na - b / nb)
        beta = spec['beta']
        return a + (cfg_strength - 1.0) * ((1.0 - beta) * delta + beta * delta_pre)

    return ode_wrapper


_orig_make = G.make_ode_wrapper
G.make_ode_wrapper = patched_make_ode_wrapper
G.GEOMETRIES.update(NEW_GEOMETRIES)


def equivalence_check(n_clips=32):
    """beta=0 must reproduce vanilla to <=1e-5 relative, as 073 requires of
    gamma=0 / eta=1. Float ordering differs, so this is numeric, not bit-exact."""
    import numpy as np
    import torch
    torch.manual_seed(20260923)
    rng = np.random.default_rng(20260923)
    patched = patched_make_ode_wrapper({'kind': 'prenorm', 'beta': 0.0, 'axis': 'global'})
    worst = 0.0
    for _ in range(n_clips):
        a = torch.from_numpy(rng.standard_normal((1, 312, 20), dtype=np.float32))
        b = torch.from_numpy(rng.standard_normal((1, 312, 20), dtype=np.float32))
        c = float(rng.uniform(1.5, 5.0))

        class Stub:
            def predict_flow(self, latent, t, r, conditions, q):
                return a if conditions == 'cond' else b
        v_ref = c * a + (1 - c) * b
        v_new = patched(Stub(), torch.tensor(0.5), torch.tensor(0.0),
                        torch.zeros(1, 312, 20), 'cond', 'uncond', c,
                        q=torch.zeros(1, dtype=torch.long))
        worst = max(worst, (v_new - v_ref).abs().max().item()
                    / max(v_ref.abs().max().item(), 1e-12))
    return {'tolerance': 1e-5, 'worst_relative_error': {'prenorm_beta0': worst},
            'passed': worst <= 1e-5}


def main():
    import eval_metrics as em
    eq = equivalence_check()
    if not eq['passed']:
        raise SystemExit(f'[FAIL] equivalence check: {eq}')
    print(f"[prenorm] equivalence beta=0 ok (worst rel err "
          f"{eq['worst_relative_error']['prenorm_beta0']:.2e})")

    clap_model = em.load_clap()
    rows, t0 = [], time.time()
    for fam in G.FAMILIES:
        van = G.score_cell(G.cell(f'A__{fam}__cfg4.5__vanilla', family=fam, cfg=4.5,
                                  geometry='vanilla', tsv=G.SUBSET_TSV, stage='pilot'),
                           clap_model)
        for geom in NEW_GEOMETRIES:
            name = f'A__{fam}__cfg4.5__{geom}'
            print(f'[prenorm] {name}', flush=True)
            rec = G.score_cell(G.cell(name, family=fam, cfg=4.5, geometry=geom,
                                      tsv=G.SUBSET_TSV, stage='pilot'), clap_model)
            rows.append(G.contrast(rec, van, clap_model))

    payload = {'document_kind': 'guidance_geometry_prenorm_extension_v1',
               'experiment_id': 'guidance-geometry-prenorm-20260923',
               'extends': str(G.SUMMARY), 'checkpoint': str(G.CKPT),
               'geometries': NEW_GEOMETRIES, 'equivalence': eq,
               'branch_norm_diag': str(G.OUT / 'prenorm_branch_norm_diag.json'),
               'thresholds': {'pq': G.GATE_PQ, 'crest': G.GATE_CREST,
                              'clap_floor': G.GATE_CLAP_FLOOR},
               'early_kill': dict(zip(('passed', 'triggering_cell'),
                                      G.passes_early_kill(rows))),
               'pilot': rows, 'wall_seconds': round(time.time() - t0, 1)}
    SUMMARY.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print(f'\n[prenorm] wrote {SUMMARY}')
    for r in rows:
        p = r['paired']
        print(f"  {r['cell']:34s} dPQ {p['PQ']['mean_delta']:+.4f} "
              f"dCLAP {p['clap']['mean_delta']:+.4f} "
              f"dcrest {p['crest']['mean_delta']:+.4f} "
              f"dLUFS {p['lufs']['mean_delta']:+.3f}"
              f"{'  [loudness gate]' if r['loudness_gate']['matched_read_required'] else ''}")
    print(f"[prenorm] early-kill gate: {payload['early_kill']}")


if __name__ == '__main__':
    main()
