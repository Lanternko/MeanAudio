#!/usr/bin/env python
"""075 follow-up P0 #2: which way does the fidelity8 push point?

The branch-geometry probe showed lab's |A-B| is the LARGEST of the three arms
while its negprompt gain is the smallest, so push size does not explain the
gain. Hypothesis: in lab, A-B points along the programmatic-defect directions
the arm was trained on (noise/clip/lowpass/bitcrush/crackle), which MusicCaps
generations do not contain, so pushing away from them buys no PQ.

Two definitions of "defect direction", per ODE step and clip:
  data  D_k   = mean over training pairs of (z_degraded - z_clean), class k,
                in the network's normalised latent space (5 classes).
  model M_k   = A - A_k, where A_k is the conditional branch with the lab
                defect sentence of class k prefixed to the same caption.
Recorded: cos(A-B, D_k), the share of |A-B|^2 inside span{D_1..D_5},
cos(A-B, M_k), and the manipulation check cos(M_k, D_k).
Sign convention: data estimate = x - t*u, so data_B - data_A = t*(A-B);
cos(A-B, D_k) > 0 means the negative branch sits on the defect side.

Usage: python d2_075_defect_direction.py [rows=32]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/kojiek/MeanAudio/scripts/eval')
import guidance_geometry_adg_apg_20260922 as G  # noqa: E402
from d2_075_branch_geometry import ARMS, SEEDS, ckpt  # noqa: E402

OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/d2_075_defect_direction')
DEFECT_DIR = Path('/home/kojiek/exps_nvme/defect_negsample')
SRC_CACHE = Path('/home/kojiek/exps_nvme/slot0clean_nmv2matched/arm_inputs/cache_train.txt')
NPZ_DIR = Path('/mnt/HDD/kojiek/phase8_qwen_official_matched_npz')
KINDS = ('noise', 'clip', 'lowpass', 'bitcrush', 'crackle')
# first template of each class in build_defect_negsample_arm_inputs.py (lab prefix)
SENTENCES = {
    'noise': 'A noisy recording buried in hiss and static.',
    'clip': 'A badly clipped recording with harsh distortion.',
    'lowpass': 'A muffled recording that lacks high frequencies.',
    'bitcrush': 'A lo-fi, bit-crushed recording with grainy digital artifacts.',
    'crackle': 'A recording full of clicks and crackles.',
}
PAIRS_PER_KIND = 1000
REC = defaultdict(list)
STATE = {'fu': None, 'texts': [], 'key': None, 'conds': None, 'D': None}


def data_directions():
    """Raw-latent mean(deg - clean) per class, cached; normalised later by the net's std."""
    path = OUT / f'defect_directions_raw_n{PAIRS_PER_KIND}.npz'
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in KINDS}
    import csv
    names = [l.strip() for l in open(SRC_CACHE) if l.strip()]
    rows = list(csv.DictReader(open(DEFECT_DIR / 'degradations.tsv'), delimiter='\t'))
    out = {}
    for k in KINDS:
        sel = [r for r in rows if r['kind'] == k][:PAIRS_PER_KIND]
        acc = np.zeros((312, 20), np.float64)
        for r in sel:
            deg = np.load(DEFECT_DIR / 'latents' / r['file'])
            assert str(deg['degradation']) == k, (r['file'], deg['degradation'])
            clean = np.load(NPZ_DIR / names[int(r['src_index'])])['mean']
            acc += deg['mean'].astype(np.float64) - clean
        out[k] = (acc / len(sel)).astype(np.float32)
        print(f'[dir] {k}: n={len(sel)} |d|={np.linalg.norm(out[k]):.3f}', flush=True)
    np.savez(path, **out)
    return out


def patch_encode_text():
    from meanaudio.model.utils.features_utils import FeaturesUtils
    orig = FeaturesUtils.encode_text

    def encode_text(self, text, **kw):
        STATE['fu'] = self
        STATE['texts'].append(list(text))
        return orig(self, text, **kw)

    FeaturesUtils.encode_text = encode_text
    STATE['orig_encode'] = orig
    return FeaturesUtils, orig


def recorder(raw_dirs):
    import torch

    def flat(x):
        return x.reshape(x.shape[0], -1)

    def cos(x, y):
        return torch.nn.functional.cosine_similarity(flat(x), flat(y), dim=1)

    def ode_wrapper(self, t, r, latent, conditions, empty_conditions, cfg, q=None):
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if q is None:
            q = torch.full((len(latent),), 10, dtype=torch.long, device=latent.device)
        bs = len(latent)
        if STATE['D'] is None:
            std = self.latent_std.detach().float()  # (1, 1, 20)
            D = torch.stack([torch.from_numpy(raw_dirs[k]).to(latent.device) / std[0] for k in KINDS])
            STATE['D'] = D.to(latent.dtype)                          # (5, 312, 20)
            Q, _ = torch.linalg.qr(flat(STATE['D']).T)               # (6240, 5) orthonormal basis
            STATE['Q'] = Q
        # the positive texts of this batch: last encode_text call that is not the negative
        pos = [x for x in STATE['texts'] if len(x) == bs and x[0] != G.FIDELITY8][-1]
        key = tuple(pos)
        if STATE['key'] != key:
            conds = {}
            for k in KINDS:
                # call the unpatched encoder so the prefixed texts are not logged as positives
                tf, tfc = STATE['orig_encode'](STATE['fu'], [f'{SENTENCES[k]} {c}' for c in pos])[:2]
                conds[k] = self.preprocess_conditions(tf, tfc, None)   # NoMask, as the cell itself
            STATE['key'], STATE['conds'] = key, conds
        a = self.predict_flow(latent, t, r, conditions, q)
        b = self.predict_flow(latent, t, r, empty_conditions, q)
        delta = a - b
        REC['t'] += t.flatten().tolist()
        dflat = flat(delta)
        proj = dflat @ STATE['Q']
        REC['span_share'] += ((proj ** 2).sum(1) / (dflat ** 2).sum(1)).tolist()
        for i, k in enumerate(KINDS):
            Dk = STATE['D'][i].expand_as(delta)
            REC[f'cos_data_{k}'] += cos(delta, Dk).tolist()
            m = a - self.predict_flow(latent, t, r, STATE['conds'][k], q)
            REC[f'cos_model_{k}'] += cos(delta, m).tolist()
            REC[f'check_{k}'] += cos(m, Dk).tolist()
        return a + (cfg - 1.0) * delta

    return ode_wrapper


def summarise(rec):
    t = np.asarray(rec['t'])
    out = {}
    for k, v in rec.items():
        if k == 't':
            continue
        v = np.asarray(v)
        out[k] = {'mean': float(v.mean()), 'n': int(v.size)}
        for name, lo, hi in (('t_hi', 2 / 3, 1.01), ('t_mid', 1 / 3, 2 / 3), ('t_lo', -0.01, 1 / 3)):
            m = (t >= lo) & (t < hi)
            out[k][name] = float(v[m].mean())
    return out


def main():
    from meanaudio.model.networks import MeanAudio
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    OUT.mkdir(parents=True, exist_ok=True)
    raw_dirs = data_directions()
    G.OUT, G.AUDIO_ROOT = OUT, OUT / '_audio'
    path = OUT / f'defect_direction_n{rows}.json'
    out = json.loads(path.read_text()) if path.exists() else {}
    FU, orig_encode = patch_encode_text()
    try:
        for seed in SEEDS:
            for arm in ARMS:
                key = f'{arm}_s{seed}'
                if key in out:
                    continue
                G.CKPT = ckpt(arm, seed)
                assert G.CKPT.exists(), G.CKPT
                REC.clear()
                STATE.update(texts=[], key=None, conds=None, D=None)
                original = MeanAudio.ode_wrapper
                MeanAudio.ode_wrapper = recorder(raw_dirs)
                try:
                    G.generate({'name': f'{key}_N8_cfg3', 'family': 'N8', 'cfg': 3.0,
                                'geometry': 'vanilla', 'tsv': str(G.SUBSET_TSV),
                                'stage': 'diag'}, limit=rows)
                finally:
                    MeanAudio.ode_wrapper = original
                out[key] = {'checkpoint': str(G.CKPT), 'rows': rows, **summarise(REC)}
                path.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
                s = out[key]
                print(f'[{key}] span_share={s["span_share"]["mean"]:.4f} '
                      + ' '.join(f'{k}:data={s[f"cos_data_{k}"]["mean"]:+.3f}/model={s[f"cos_model_{k}"]["mean"]:+.3f}'
                                 f'/chk={s[f"check_{k}"]["mean"]:+.3f}' for k in KINDS), flush=True)
    finally:
        FU.encode_text = orig_encode
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
