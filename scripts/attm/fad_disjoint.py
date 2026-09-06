#!/usr/bin/env python3
"""Recompute FAD with a reference set disjoint from the prompts.

attm_protocol_eval.py scores FAD against all 2,382 instrumental MusicCaps
reference clips -- including the very clips whose captions produced the
generated audio. That is structurally easier than ATTM's setup, where prompts
are synthesised tag triplets and the reference is an unrelated held-out sample,
and it inflates our FAD advantage.

Fix: split the instrumental subset in half by a hash of the clip id. Score the
generations made from half A's captions against half B's real audio only. No
generated clip's own source recording is ever in its reference set.

Costs one CLAP embedding pass per arm; no regeneration.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import linalg

ROOT = Path('/home/kojiek/MeanAudio')
ATTM = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/attm')
REF_DIR = ATTM / 'musiccaps_instrumental_ref'
CLAP_CKPT = ROOT / 'weights/music_audioset_epoch_15_esc_90.14.pt'


def half(cid):
    return int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 2


def embed(model, paths, batch=32):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            e = model.get_audio_embedding_from_filelist(
                [str(p) for p in paths[i:i + batch]], use_tensor=True)
            out.append(e.cpu().numpy())
    return np.concatenate(out, 0)


def frechet(a, b):
    d = a.mean(0) - b.mean(0)
    sa, sb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cm, _ = linalg.sqrtm(sa.dot(sb), disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(d.dot(d) + np.trace(sa) + np.trace(sb) - 2 * np.trace(cm))


def main():
    import torch
    import laion_clap

    ref_all = sorted(REF_DIR.glob('*.wav'))
    ref_b = [p for p in ref_all if half(p.stem) == 1]
    print(f'reference half B: {len(ref_b)}/{len(ref_all)}', flush=True)

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(CLAP_CKPT))
    model = model.eval().cuda()
    ref_emb = embed(model, ref_b)

    results = {}
    for d in sorted((ATTM / '_audio').iterdir()):
        gen_a = [p for p in sorted(d.glob('*.flac')) if half(p.stem) == 0]
        if not gen_a:
            continue
        fad = frechet(ref_emb, embed(model, gen_a))
        results[d.name] = {'n_gen': len(gen_a), 'n_ref': len(ref_b), 'fad_disjoint': fad}
        print(f'{d.name:32s} n_gen={len(gen_a):5d}  FAD_disjoint={fad:.4f}', flush=True)

    del model
    torch.cuda.empty_cache()
    (ATTM / 'fad_disjoint.json').write_text(json.dumps({
        'note': ('prompts from hash-half A, reference audio from hash-half B; '
                 'no generated clip shares its source recording with the reference'),
        'clap_ckpt': str(CLAP_CKPT), 'results': results}, indent=1))


if __name__ == '__main__':
    main()
