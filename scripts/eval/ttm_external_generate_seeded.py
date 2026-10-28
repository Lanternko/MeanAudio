#!/usr/bin/env python3
"""Seed-parameterised copy of ttm_external_generate.py (p2 135-139, docs/experiments/seed_robustness_table_b_135_20261021.md).

The only change against the contract-pinned original (p2 113) is --base_seed (default 42):
at 42 every code path, argument and written byte is the original's, and the manifest
fields are identical (the reproduction gate in seed_robustness_135.py checks this against the
113 audio). Nothing else is tuned. Seed scheme: MusicGen torch.manual_seed(base_seed + batch
index); MusicLDM per-prompt generator base_seed + row index. Note that with base seeds 101/202
the per-prompt MusicLDM generator seeds overlap numerically with seed-42 seeds of *other*
rows (101 + i == 42 + (i + 59)); they are attached to different prompts, so no clip repeats.

Original docstring follows.

Generate MusicCaps audio with external TTM baselines at their official defaults.

Runs in ~/venvs/ttm (torch 2.11 cu128, transformers 4.57.3, diffusers 0.39.0).
Design: docs/experiments/ttm_quality_comparison_plan_20261007.md.

Systems (revisions pinned; settings are the upstream defaults, not tuned here):
  musicgen_medium  facebook/musicgen-medium, AudioCraft defaults: sampling, top_k 250,
                   temperature 1.0, cfg 3.0, 500 tokens = 10 s, fp16 autocast, 32 kHz.
                   transformers' own generation_config uses top_k 50, so every sampling
                   argument is passed explicitly. AudioCraft's top_p=0 means "off"; in
                   transformers top_p=0.0 keeps only the argmax token (greedy, near-silent
                   output in the 2026-10-08 smoke), so "off" is top_p=1.0 here.
  musicldm         ucsd-reach/musicldm, diffusers defaults: 200 steps, guidance 2.0,
                   one waveform per prompt (no CLAP reranking), fp32, 16 kHz, 10 s.
  musicldm_fid8    musicldm + negative_prompt = fidelity8 (same string as our CFG3+neg cell).

Output: <out_dir>/audio/{id}.flac written like eval.py (sf.write default PCM_16, native sr),
<out_dir>/manifest.json (revision, settings, seed scheme, versions, timings, peak VRAM).

Seeds: musicldm* use one generator per prompt, seed 42 + row index, so a clip does not
depend on batch composition. MusicGen samples the whole batch from one global RNG, so the
seed is 42 + batch index and the batch size is part of the protocol (recorded in the
manifest; resume refuses a different batch size). A partially written batch is regenerated
whole.
"""
import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

FIDELITY8 = ('low quality recording, noisy, amateur, distorted, muffled, poor fidelity, '
             'hiss, lo-fi')
SYSTEMS = {
    'musicgen_medium': {
        'repo': 'facebook/musicgen-medium',
        'revision': 'd3bd7b00761b78ad7a8a05145ee31e7832e9916c',
        'settings': {'do_sample': True, 'top_k': 250, 'top_p': 1.0, 'temperature': 1.0,
                     'guidance_scale': 3.0, 'max_new_tokens': 500, 'autocast': 'fp16'},
    },
    'musicldm': {
        'repo': 'ucsd-reach/musicldm',
        'revision': 'b8135e32e6b85e752c513d2e2ad44269e46024f7',
        'settings': {'num_inference_steps': 200, 'guidance_scale': 2.0,
                     'audio_length_in_s': 10.0, 'num_waveforms_per_prompt': 1,
                     'dtype': 'fp32', 'negative_prompt': None},
    },
    'musicldm_fid8': {
        'repo': 'ucsd-reach/musicldm',
        'revision': 'b8135e32e6b85e752c513d2e2ad44269e46024f7',
        'settings': {'num_inference_steps': 200, 'guidance_scale': 2.0,
                     'audio_length_in_s': 10.0, 'num_waveforms_per_prompt': 1,
                     'dtype': 'fp32', 'negative_prompt': FIDELITY8},
    },
}
DEFAULT_BASE_SEED = 42


def read_rows(tsv, limit):
    with open(tsv, newline='') as f:
        rows = [(r['id'], r['caption']) for r in csv.DictReader(f, delimiter='\t')]
    if len(rows) != len({i for i, _ in rows}):
        raise SystemExit('duplicate ids in TSV')
    return rows[:limit] if limit else rows


def load(system, device):
    spec = SYSTEMS[system]
    if system == 'musicgen_medium':
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        proc = AutoProcessor.from_pretrained(spec['repo'], revision=spec['revision'])
        model = MusicgenForConditionalGeneration.from_pretrained(
            spec['repo'], revision=spec['revision'], torch_dtype=torch.float32).to(device).eval()
        return (proc, model), model.config.audio_encoder.sampling_rate
    from diffusers import MusicLDMPipeline
    pipe = MusicLDMPipeline.from_pretrained(spec['repo'], revision=spec['revision'],
                                            torch_dtype=torch.float32).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, pipe.vocoder.config.sampling_rate


@torch.inference_mode()
def generate(system, handle, captions, first_idx, batch_idx, device, base_seed):
    s = SYSTEMS[system]['settings']
    if system == 'musicgen_medium':
        proc, model = handle
        torch.manual_seed(base_seed + batch_idx)
        inputs = proc(text=captions, padding=True, return_tensors='pt').to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            out = model.generate(**inputs, do_sample=True, top_k=s['top_k'], top_p=s['top_p'],
                                 temperature=s['temperature'], guidance_scale=s['guidance_scale'],
                                 max_new_tokens=s['max_new_tokens'])
        return [w[0].float().cpu().numpy() for w in out]
    gens = [torch.Generator(device).manual_seed(base_seed + first_idx + k) for k in range(len(captions))]
    neg = s['negative_prompt']
    out = handle(captions, num_inference_steps=s['num_inference_steps'],
                 guidance_scale=s['guidance_scale'], audio_length_in_s=s['audio_length_in_s'],
                 num_waveforms_per_prompt=1, generator=gens,
                 negative_prompt=[neg] * len(captions) if neg else None).audios
    return [np.asarray(a, dtype=np.float32) for a in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True, choices=sorted(SYSTEMS))
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--base_seed', type=int, default=DEFAULT_BASE_SEED)
    args = ap.parse_args()

    device = 'cuda'
    out = Path(args.out_dir)
    audio_dir = out / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.tsv, args.limit)
    spec = SYSTEMS[args.system]
    man_path = out / 'manifest.json'
    manifest = json.loads(man_path.read_text()) if man_path.is_file() else None
    fixed = {'system': args.system, 'repo': spec['repo'], 'revision': spec['revision'],
             'settings': spec['settings'], 'tsv': str(Path(args.tsv).resolve()),
             'n_rows': len(rows), 'batch_size': args.batch_size, 'base_seed': args.base_seed,
             'seed_scheme': (f'{args.base_seed} + batch index (global RNG)' if args.system == 'musicgen_medium'
                             else f'{args.base_seed} + row index (per-prompt generator)')}
    if manifest and any(manifest.get(k) != v for k, v in fixed.items()):
        raise SystemExit(f'manifest in {out} disagrees with this invocation; refusing to mix')
    if not manifest:
        manifest = {**fixed, 'batches': [], 'versions': {
            'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': __import__('transformers').__version__,
            'diffusers': __import__('diffusers').__version__,
            'gpu': torch.cuda.get_device_name(0)}}

    handle, sr = load(args.system, device)
    manifest['sampling_rate'] = sr
    torch.cuda.reset_peak_memory_stats()
    n_batches = math.ceil(len(rows) / args.batch_size)
    t_start = time.time()
    for b in range(n_batches):
        chunk = rows[b * args.batch_size:(b + 1) * args.batch_size]
        paths = [audio_dir / f'{i}.flac' for i, _ in chunk]
        if all(p.is_file() for p in paths):
            continue
        t0 = time.time()
        wavs = generate(args.system, handle, [c for _, c in chunk], b * args.batch_size, b, device, args.base_seed)
        for p, w in zip(paths, wavs):
            if not np.isfinite(w).all():
                raise SystemExit(f'non-finite audio for {p.name}')
            tmp = p.with_suffix('.tmp.flac')
            sf.write(str(tmp), w, sr)
            tmp.replace(p)
        dt = time.time() - t0
        manifest['batches'].append({'batch': b, 'n': len(chunk), 'seconds': round(dt, 3),
                                    'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 2**30, 3)})
        done = (b + 1) * args.batch_size
        print(f'[{args.system}] batch {b + 1}/{n_batches} {dt:.1f}s '
              f'({dt / len(chunk):.2f} s/clip, eta {(time.time() - t_start) / done * (len(rows) - done) / 60:.1f} min)',
              flush=True)
        if b % 10 == 0 or b == n_batches - 1:
            man_path.write_text(json.dumps(manifest, indent=1))
    n = sum((audio_dir / f'{i}.flac').is_file() for i, _ in rows)
    manifest['n_written'] = n
    manifest['complete'] = n == len(rows)
    man_path.write_text(json.dumps(manifest, indent=1))
    print(f'[{args.system}] {n}/{len(rows)} clips in {audio_dir}')
    return 0 if n == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
