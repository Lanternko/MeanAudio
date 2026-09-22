#!/usr/bin/env python
"""D1 probe, reference arm: where IS the defect region in metric space?

Takes the clean-prompt clips the model generated in the D1 probe and applies
programmatic degradations that match the defect vocabulary used in the prompts
(noise / clipping / muffling / bitcrush). Scoring these with the canonical
scorer gives the coordinates a defect prompt would have to reach. Without them,
"the defect prompt did not move the output" is unfalsifiable.

Every degraded clip is renormalised to its source clip's integrated LUFS, so
the reference region is the effect of the degradation and not of level (063/065:
PQ/CU rise as level falls, CLAP rises as level rises). The pre-match loudness of
each clip is recorded in degradations.tsv so the level story is not lost.

Writes <out>/audio/<source_id>__<degradation>.flac plus two TSVs for the
canonical scorer: one pairing each clip with its source's CLEAN caption (does
CLAP fall?), one with the matching DEFECT caption (does CLAP rise?).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt

# degradation -> the defect prompt a listener would write for it
DEFECT_CAPTION = {
    'noise_snr20': 'noisy recording with loud hiss and background static',
    'noise_snr10': 'noisy recording with loud hiss and background static',
    'noise_snr0': 'static noise and tape hiss, no music',
    'clip_6db': 'harsh digital distortion and clipping',
    'clip_18db': 'harsh digital distortion and clipping, no music',
    'lowpass_2k': 'muffled low-bitrate recording with no high frequencies',
    'lowpass_1k': 'a muffled low-bitrate recording with no high frequencies, no music',
    'bitcrush_8': 'lo-fi amateur recording, poor fidelity',
    'bitcrush_4': 'low quality recording, noisy, distorted, clipping, muffled, hiss, lo-fi, poor fidelity',
}


def add_noise(x, snr_db, rng):
    n = rng.standard_normal(x.shape)
    sp, np_ = np.mean(x ** 2), np.mean(n ** 2)
    if sp <= 0 or np_ <= 0:
        return x
    return x + n * np.sqrt(sp / (np_ * 10 ** (snr_db / 10)))


def hard_clip(x, drive_db):
    peak = np.max(np.abs(x))
    if peak <= 0:
        return x
    return np.clip(x * 10 ** (drive_db / 20), -peak, peak)


def lowpass(x, cutoff, sr):
    sos = butter(8, cutoff / (sr / 2), btype='low', output='sos')
    return sosfilt(sos, x)


def bitcrush(x, bits):
    peak = np.max(np.abs(x))
    if peak <= 0:
        return x
    step = 2 * peak / (2 ** bits)
    return np.round(x / step) * step


def degrade(name, x, sr, rng):
    if name.startswith('noise_snr'):
        return add_noise(x, float(name[len('noise_snr'):]), rng)
    if name.startswith('clip_'):
        return hard_clip(x, float(name[len('clip_'):-2]))
    if name.startswith('lowpass_'):
        khz = name[len('lowpass_'):]
        return lowpass(x, float(khz[:-1]) * 1000, sr)
    if name.startswith('bitcrush_'):
        return bitcrush(x, int(name[len('bitcrush_'):]))
    raise ValueError(name)


def match_lufs(y, target, meter):
    """Scale y to `target` LUFS; returns (scaled, measured_lufs_before)."""
    before = meter.integrated_loudness(y)
    if not np.isfinite(before):
        return y, before
    return y * 10 ** ((target - before) / 20), before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_dir', required=True, help='D1 cfg0 audio dir')
    ap.add_argument('--src_tsv', required=True, help='the D1 probe TSV (for captions)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--group', default='clean', help='id prefix group to degrade')
    ap.add_argument('--per_prompt', type=int, default=64,
                    help='source clips per clean prompt')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    captions = {}
    with open(args.src_tsv, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            captions[row['id']] = row['caption']

    # d1_clean_<stem>_<idx>: keep the first per_prompt of each stem
    srcs, seen = [], {}
    for cid in sorted(captions):
        if not cid.startswith(f'd1_{args.group}_'):
            continue
        stem = cid.rsplit('_', 1)[0]
        seen[stem] = seen.get(stem, 0) + 1
        if seen[stem] <= args.per_prompt:
            srcs.append(cid)
    if not srcs:
        raise SystemExit(f'[FAIL] no ids with prefix d1_{args.group}_ in {args.src_tsv}')

    out = Path(args.out)
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows_clean, rows_defect, manifest = [], [], []

    for cid in srcs:
        path = Path(args.src_dir) / f'{cid}.flac'
        x, sr = sf.read(path, dtype='float64', always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        meter = pyln.Meter(sr)
        src_lufs = meter.integrated_loudness(x)
        for name in DEFECT_CAPTION:
            y = degrade(name, x, sr, rng)
            y, raw_lufs = match_lufs(y, src_lufs, meter)
            peak = np.max(np.abs(y))
            if peak > 0.999:  # LUFS matching can overshoot full scale
                y = y * (0.999 / peak)
            did = f'{cid}__{name}'
            sf.write(out / 'audio' / f'{did}.flac', y.astype(np.float32), sr)
            rows_clean.append((did, captions[cid]))
            rows_defect.append((did, DEFECT_CAPTION[name]))
            manifest.append((did, cid, name, f'{src_lufs:.3f}', f'{raw_lufs:.3f}',
                             f'{float(np.max(np.abs(y))):.4f}'))

    def write_tsv(path, header, rows):
        with open(path, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            w.writerow(header)
            w.writerows(rows)

    write_tsv(out / 'score_clean_caption.tsv', ['id', 'caption'], rows_clean)
    write_tsv(out / 'score_defect_caption.tsv', ['id', 'caption'], rows_defect)
    write_tsv(out / 'degradations.tsv',
              ['id', 'source_id', 'degradation', 'source_lufs',
               'degraded_lufs_before_match', 'peak_after'], manifest)
    print(f'{len(manifest)} clips from {len(srcs)} sources '
          f'x {len(DEFECT_CAPTION)} degradations -> {out}')


if __name__ == '__main__':
    main()
