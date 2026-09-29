#!/usr/bin/env python
"""AES PQ audit: why does generated audio score above the MusicCaps reference?

Stage 1 (this script, scoring only):
  sets
    mc_ref        MusicCaps reference clips (16 kHz, 10 s, as downloaded)
    mc_ref_pk     same, peak-normalised to 0.95 (training preprocessing)
    mc_ref_vae    mc_ref_pk -> mel -> VAE encode (mean) -> decode -> BigVGAN
    jam           Jamendo training segments: whole 30 s file peak-normed to 0.95,
                  first 10 s, resampled to 16 kHz (= extract_audio_latents.py)
    jam_vae       jam through the same VAE+vocoder round trip
  per clip: AES CE/CU/PC/PQ + level + a small set of spectral / temporal features.
  Generated arms are scored elsewhere; their per_clip.tsv is read by the analysis.

Output: <out>/<set>/audio/*.flac, <out>/scores.tsv (one row per set x clip)
"""
import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'eval'))
from eval_metrics import score_aes  # noqa: E402

SR = 16_000
N = SR * 10


def peak_norm(x):
    m = np.abs(x).max()
    return x if m == 0 else x / m * 0.95


def load_mc(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    assert sr == SR, (path, sr)
    x = x[:N]
    return np.pad(x, (0, N - len(x)))


def load_jam(path):
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = torch.from_numpy(x.T.copy())
    x = torchaudio.functional.resample(x, sr, SR).mean(0).numpy()
    x = peak_norm(x)                       # whole 30 s file, then crop
    x = x[:N]
    return np.pad(x, (0, N - len(x)))


class RoundTrip:
    def __init__(self):
        from meanaudio.ext.autoencoder import AutoEncoderModule
        from meanaudio.ext.mel_converter import get_mel_converter
        self.tod = AutoEncoderModule(vae_ckpt_path=str(ROOT / 'weights/v1-16.pth'),
                                     vocoder_ckpt_path=str(ROOT / 'weights/best_netG.pt'),
                                     mode='16k').eval().cuda()
        self.mel = get_mel_converter('16k').eval().cuda()

    @torch.inference_mode()
    def __call__(self, batch):
        x = torch.from_numpy(np.stack(batch)).cuda()
        z = self.tod.encode(self.mel(x)).mode()
        y = self.tod.vocode(self.tod.decode(z)).squeeze(1)
        return [a[:N].float().cpu().numpy() for a in y]


def features(x):
    import librosa
    import pyloudnorm
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    peak = float(np.abs(x).max())
    lufs = pyloudnorm.Meter(SR).integrated_loudness(x.astype(np.float64))
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    tot = S.sum() + 1e-12
    frame_db = 10 * np.log10(S.sum(0) + 1e-12)
    onset_env = librosa.onset.onset_strength(y=x, sr=SR)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=SR)[0]
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR)
    H, P = librosa.decompose.hpss(S)
    return {
        'lufs': lufs if math.isfinite(lufs) else float('nan'),
        'rms_db': 20 * math.log10(rms + 1e-9),
        'crest': peak / rms if rms > 0 else float('nan'),
        'centroid': float(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR).mean()),
        'rolloff95': float(librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=SR, roll_percent=0.95).mean()),
        'flatness': float(librosa.feature.spectral_flatness(S=np.sqrt(S)).mean()),
        'hf_4k': float(S[freqs >= 4000].sum() / tot),
        'hf_6k': float(S[freqs >= 6000].sum() / tot),
        'lf_150': float(S[freqs < 150].sum() / tot),
        'frame_db_std': float(frame_db.std()),
        'noise_floor_db': float(np.percentile(frame_db, 10) - np.median(frame_db)),
        'onset_rate': len(onsets) / 10.0,
        'tempo': float(tempo),
        'harm_ratio': float(H.sum() / (H.sum() + P.sum() + 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/seagate/aes_audit/stage1')
    ap.add_argument('--mc_dir', default='/mnt/HDD/kojiek/musiccaps_reference')
    ap.add_argument('--jam_dir', default='/mnt/HDD/hsiehyian/segments_no_vocals')
    ap.add_argument('--clips_tsv', default='/mnt/HDD/kojiek/phase4_jamendo_data/clips.tsv')
    ap.add_argument('--n_jam', type=int, default=2000)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--seed', type=int, default=20260929)
    a = ap.parse_args()
    out = Path(a.out)
    rng = random.Random(a.seed)

    mc = sorted(Path(a.mc_dir).glob('*.wav'))[:a.limit]
    with open(a.clips_tsv) as fh:
        names = [r['name'] for r in csv.DictReader(fh, delimiter='\t')]
    # name = <bucket>_<track>_segment_<k>  ->  <jam_dir>/<bucket>/<track>/segment_<k>.mp3
    jam_all = [Path(a.jam_dir) / n.split('_', 2)[0] / n.split('_', 2)[1] / (n.split('_', 2)[2] + '.mp3')
               for n in names]
    jam = rng.sample(jam_all, a.n_jam if a.limit is None else a.limit)
    print(f'mc {len(mc)}  jam {len(jam)} / {len(jam_all)}', flush=True)

    rt = RoundTrip()
    sets = {k: {} for k in ('mc_ref', 'mc_ref_pk', 'mc_ref_vae', 'jam', 'jam_vae')}

    def run(src, loader, base, pk_name, vae_name, keep_raw):
        B = 32
        for i in range(0, len(src), B):
            chunk = src[i:i + B]
            raws = [loader(p) for p in chunk]
            pks = [peak_norm(r) for r in raws]
            vaes = rt(pks)
            for p, r, k, v in zip(chunk, raws, pks, vaes):
                cid = p.stem if base == 'mc' else f'{p.parent.name}_{p.stem}'
                if keep_raw:
                    sets['mc_ref'][cid] = r
                sets[pk_name][cid] = k
                sets[vae_name][cid] = v
            print(f'{base} {i + len(chunk)}/{len(src)}', flush=True)

    run(mc, load_mc, 'mc', 'mc_ref_pk', 'mc_ref_vae', True)
    run(jam, load_jam, 'jam', 'jam', 'jam_vae', False)
    del rt
    torch.cuda.empty_cache()

    paths = {}
    for s, clips in sets.items():
        d = out / s / 'audio'
        d.mkdir(parents=True, exist_ok=True)
        for cid, x in clips.items():
            p = d / f'{cid}.flac'
            sf.write(p, x, SR, subtype='PCM_24')
            paths[(s, cid)] = str(p)

    aes, failed = score_aes(list(paths.values()))
    assert not failed, failed

    from multiprocessing import Pool
    keys = list(paths)
    with Pool(24) as pool:
        feats = pool.map(features, [sets[s][c] for s, c in keys], chunksize=16)

    rows = []
    for (s, c), f in zip(keys, feats):
        rows.append({'set': s, 'id': c, **aes[paths[(s, c)]], **f})
    with open(out / 'scores.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    for s in sets:
        pq = [r['PQ'] for r in rows if r['set'] == s]
        print(f'{s:12s} n={len(pq):5d} PQ={np.mean(pq):.4f}')


if __name__ == '__main__':
    main()
