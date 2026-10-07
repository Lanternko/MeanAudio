#!/usr/bin/env python
"""Rescore the Codex 2026-10-01 clean MIDI / karaoke originals at -30 LUFS.

Codex's aes-extended-20261001 reported AES at -23 LUFS only; our reference lines
(Jamendo 7.73, Jamendo VAE ceiling 7.33, 081 8.20) are at -30 LUFS (lvl30).
Same gain rule as aes_audit_stage2.py (scalar gain, peak cap 0.999, PCM_24).

Variants per clip:
  full_lvl23   whole clip at -23 LUFS   -> sanity vs Codex's own lufs23 scores
  full_lvl30   whole clip at -30 LUFS   (Codex window: AES averages 10 s windows)
  head10_lvl30 first 10 s at -30 LUFS   (matches our 10 s reference sets;
               identical to full_lvl30 for 10 s karaoke contexts, so skipped there)

Output: <out>/scores.tsv (variant, domain, id, bank, family, CE, CU, PC, PQ, lufs_src, gain_db, status)
"""
import argparse
import csv
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'eval'))
from eval_metrics import score_aes  # noqa: E402

SRC = Path.home() / 'Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs/aes-extended-20261001'
PEAK_CAP = 0.999


def gain_one(args):
    src, dst, target, head_s = args
    import pyloudnorm
    wav, sr = sf.read(src, dtype='float64', always_2d=True)
    wav = wav.mean(1, keepdims=True)
    if head_s:
        wav = wav[:int(head_s * sr)]
    lufs = pyloudnorm.Meter(sr).integrated_loudness(wav)
    peak = float(np.abs(wav).max())
    if not math.isfinite(lufs) or peak == 0:
        status, g = 'no_finite_loudness', 0.0
    else:
        status, g = 'ok', target - lufs
        if peak * 10 ** (g / 20) > PEAK_CAP:
            g, status = 20 * math.log10(PEAK_CAP / peak), 'peak_capped'
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, np.clip(wav * 10 ** (g / 20), -1, 1), sr, subtype='PCM_24')
    return (lufs if math.isfinite(lufs) else float('nan')), g, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/seagate/aes_audit/codex_lvl30_20261007')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)

    midi = [x for x in json.load(open(SRC / 'midi_items.json')) if x['id'].endswith('_clean')]
    kara = [x for x in json.load(open(SRC / 'karaoke_items.json')) if x['case'] == 'clean']
    assert len(midi) == 2304 and len(kara) == 27, (len(midi), len(kara))
    clips = midi + kara
    if a.limit:
        clips = midi[:a.limit] + kara[:2]

    jobs = []   # (variant, item, dst, gain args)
    for it in clips:
        dur = sf.info(it['path']).duration
        for variant, target, head in (('full_lvl23', -23.0, 0), ('full_lvl30', -30.0, 0), ('head10_lvl30', -30.0, 10)):
            if head and dur <= head + 1e-3:
                continue
            dst = out / 'audio' / variant / f"{it['id']}.wav"
            jobs.append((variant, it, str(dst), (it['path'], str(dst), target, head)))

    with Pool(16) as pool:
        levels = pool.map(gain_one, [j[3] for j in jobs], chunksize=16)
    aes, failed = score_aes([j[2] for j in jobs])

    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'scores.tsv', 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['variant', 'domain', 'id', 'bank', 'family', 'CE', 'CU', 'PC', 'PQ', 'lufs_src', 'gain_db', 'status'])
        for (variant, it, dst, _), (lufs, g, status) in zip(jobs, levels):
            s = aes.get(dst)
            if s is None:
                status = 'aes_failed'
                s = dict.fromkeys(['CE', 'CU', 'PC', 'PQ'], float('nan'))
            w.writerow([variant, it['domain'], it['id'], it['bank'], it['family'],
                        s['CE'], s['CU'], s['PC'], s['PQ'], lufs, g, status])
    print(f'wrote {len(jobs)} rows, {len(failed)} AES failures -> {out / "scores.tsv"}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
