#!/usr/bin/env python3
"""ATTM-protocol evaluation on the instrumental MusicCaps subset.

Reproduces the objective half of the ICME 2026 ATTM scorecard (arXiv 2605.21538)
as closely as our data allows:

  * CLAP score with ATTM's embedder, LAION-CLAP music_audioset_epoch_15_esc_90.14
  * CLAP score with our historical embedder, music_speech_audioset_..._89.98,
    on the SAME audio, so every number already on record can be rescaled
  * FAD computed in the ATTM embedding space (they use the same music_audioset
    checkpoint as the FAD feature extractor, not VGGish)
  * Audiobox CE/CU/PC/PQ, which ATTM does not have -- kept because it is our
    own primary axis

Deviations from ATTM that must be reported alongside any number this produces:
  * prompts are instrumental-filtered MusicCaps, not their 100 synthesised
    tag-triplet prompts (those were never released)
  * the FAD reference is instrumental MusicCaps, not their hidden 1,000-track
    MTG-Jamendo subset
So these are ATTM-protocol numbers, not ATTM leaderboard numbers.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import linalg

ROOT = Path('/home/kojiek/MeanAudio')
PYTHON = '/home/kojiek/venvs/dac/bin/python'
ATTM_CLAP = ROOT / 'weights/music_audioset_epoch_15_esc_90.14.pt'
OURS_CLAP = ROOT / 'weights/music_speech_audioset_epoch_15_esc_89.98.pt'
TSV = Path('/home/kojiek/eval_tsvs_p100/musiccaps_instrumental_test.tsv')
REF_DIR = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/attm/musiccaps_instrumental_ref')
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/attm')

NEG = ('low quality recording, noisy, amateur, distorted, muffled, '
       'poor fidelity, hiss, lo-fi')

# (label, checkpoint, extra eval.py flags)
ARMS = {
    # ATTM topline. "Official checkpoint" means upstream MeanAudio defaults:
    # single-step MeanFlow at cfg 4.5, no negative prompt.
    'meanaudio_s_full_topline': (
        ROOT / 'weights/meanaudio_s_full.pth',
        ['--num_steps', '1', '--cfg_strength', '4.5', '--no_q']),
    'meanaudio_l_full_topline': (
        ROOT / 'weights/meanaudio_l_full.pth',
        ['--num_steps', '1', '--cfg_strength', '4.5', '--no_q']),
    # our best known arm, at the cfg 3.0 + negative-prompt setting
    'ours_c2p0_slot0_cfg3_neg': (
        ROOT / 'exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/'
               'phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth',
        ['--num_steps', '25', '--cfg_strength', '3.0', '--negative_prompt', NEG,
         '--no_text_attention_mask', '--no_q']),
    # same arm at our canonical CFG 0, to keep the negprompt delta visible here too
    'ours_c2p0_slot0_cfg0': (
        ROOT / 'exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000/'
               'phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth',
        ['--num_steps', '25', '--cfg_strength', '0.0',
         '--no_text_attention_mask', '--no_q']),
}


def load_rows():
    with TSV.open(encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def generate(label, ckpt, flags, audio_dir, expected):
    audio_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(audio_dir.glob('*.flac')))
    if have >= expected * 0.99:
        print(f'  [skip gen] {have} clips present', flush=True)
        return
    if not Path(ckpt).is_file():
        raise SystemExit(f'[FAIL] missing checkpoint {ckpt}')
    variant = 'meanaudio_l' if 'meanaudio_l' in label else 'meanaudio_s'
    cmd = [PYTHON, 'eval.py', '--variant', variant, '--model_path', str(ckpt),
           '--output', str(audio_dir), '--tsv', str(TSV), '--use_meanflow',
           '--encoder_name', 't5_clap', '--text_c_dim', '512', '--seed', '42',
           '--full_precision'] + flags
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f'[FAIL] generation failed for {label}')
    got = len(list(audio_dir.glob('*.flac')))
    print(f'  [gen] {got} clips in {(time.time()-t0)/60:.1f} min', flush=True)
    if got < expected * 0.99:
        raise SystemExit(f'[FAIL] only {got}/{expected} clips for {label}')


def clap_embeddings(model, paths, captions=None, batch=32):
    """batch 32 throughout -- CLAP is batch-size sensitive at roughly the same
    magnitude as the between-arm gaps we are trying to read."""
    import torch
    audio, text = [], []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            ae = model.get_audio_embedding_from_filelist(
                [str(p) for p in chunk], use_tensor=True)
            audio.append(ae.cpu().numpy())
            if captions is not None:
                te = model.get_text_embedding(captions[i:i + batch], use_tensor=True)
                text.append(te.cpu().numpy())
    audio = np.concatenate(audio, 0)
    text = np.concatenate(text, 0) if captions is not None else None
    return audio, text


def frechet(a, b):
    mu_a, mu_b = a.mean(0), b.mean(0)
    sig_a = np.cov(a, rowvar=False)
    sig_b = np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(sig_a.dot(sig_b), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig_a) + np.trace(sig_b) - 2 * np.trace(covmean))


def score_clap(ckpt, present, ref_paths, cache):
    import torch
    import laion_clap
    key = Path(ckpt).stem
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(ckpt))
    model = model.eval().cuda()

    gen_paths = [p for _, _, p in present]
    caps = [c for _, c, _ in present]
    gen_a, gen_t = clap_embeddings(model, gen_paths, caps)
    # cosine similarity, L2-normalised, matching laion_clap's own convention
    gn = gen_a / np.linalg.norm(gen_a, axis=1, keepdims=True)
    tn = gen_t / np.linalg.norm(gen_t, axis=1, keepdims=True)
    per_clip = (gn * tn).sum(1)

    if key not in cache:
        cache[key], _ = clap_embeddings(model, ref_paths)
    fad = frechet(cache[key], gen_a)

    del model
    torch.cuda.empty_cache()
    return float(per_clip.mean()), fad, {i: float(s) for (i, _, _), s in zip(present, per_clip)}


def score_aes(present):
    import torch
    import audiobox_aesthetics.infer as aes_infer

    def read_wav(meta):
        wav, sr = sf.read(meta['path'], dtype='float32', always_2d=True)
        wav = torch.from_numpy(wav.T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav, sr

    aes_infer.read_wav = read_wav
    from audiobox_aesthetics.infer import AesPredictor
    predictor = AesPredictor(checkpoint_pth=None, batch_size=32)
    per = {}
    for i in range(0, len(present), 32):
        chunk = present[i:i + 32]
        res = predictor.forward([{'path': str(p)} for _, _, p in chunk])
        for (cid, _, _), r in zip(chunk, res):
            per[cid] = {k: float(r[k]) for k in ('CE', 'CU', 'PC', 'PQ')}
    del predictor
    torch.cuda.empty_cache()
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arms', nargs='*', default=None)
    ap.add_argument('--gen-only', action='store_true')
    args = ap.parse_args()

    rows = load_rows()
    expected = len(rows)
    ref_paths = sorted(REF_DIR.glob('*.wav'))
    print(f'prompts={expected}  fad_reference={len(ref_paths)}', flush=True)

    selected = args.arms or list(ARMS)
    ref_cache = {}
    for label in selected:
        ckpt, flags = ARMS[label]
        print(f'\n=== {label} ===', flush=True)
        audio_dir = OUT / '_audio' / label
        generate(label, ckpt, flags, audio_dir, expected)
        if args.gen_only:
            continue

        present = [(r['id'], r['caption'], audio_dir / f"{r['id']}.flac") for r in rows]
        present = [(i, c, p) for i, c, p in present if p.exists()]

        attm_clap, attm_fad, attm_per = score_clap(ATTM_CLAP, present, ref_paths, ref_cache)
        ours_clap, ours_fad, _ = score_clap(OURS_CLAP, present, ref_paths, ref_cache)
        aes = score_aes(present)
        agg = {k: float(np.mean([v[k] for v in aes.values()]))
               for k in ('CE', 'CU', 'PC', 'PQ')}

        result = {
            'label': label,
            'checkpoint': str(ckpt),
            'flags': flags,
            'n': len(present),
            'protocol': ('instrumental MusicCaps 2535; seed 42; full precision; '
                         'CLAP batch 32; FAD in LAION-CLAP music_audioset space '
                         f'vs {len(ref_paths)} instrumental MusicCaps refs'),
            'attm_clap_90p14': attm_clap,
            'attm_fad_90p14': attm_fad,
            'ours_clap_89p98': ours_clap,
            'ours_fad_89p98': ours_fad,
            **agg,
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f'{label}.json').write_text(json.dumps(
            {**result, 'per_clip_attm_clap': attm_per}, indent=1))
        print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
