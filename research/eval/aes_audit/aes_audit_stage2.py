#!/usr/bin/env python
"""AES audit stage 2 (scoring): level-matched rescoring + human-rated sets.

  1. Every stage-1 set rescored at -30 LUFS (scalar gain, peak cap 0.999, same
     rule as scripts/eval/level_match_rescore.py) -> <set>_lvl30.
  2. PAM human_eval music (audioldm2 / musicgen_large / musicgen_melody /
     musicldm / real, 100 each, 10 human PQ ratings per clip) scored with AES,
     native level and -30 LUFS.

Output: <out>/scores_stage2.tsv  (set, id, CE, CU, PC, PQ, lufs_src, gain_db, status)
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

TARGET = -30.0
PEAK_CAP = 0.999


def gain_one(args):
    src, dst = args
    import pyloudnorm
    wav, sr = sf.read(src, dtype='float64', always_2d=True)
    wav = wav.mean(1, keepdims=True)
    lufs = pyloudnorm.Meter(sr).integrated_loudness(wav)
    peak = float(np.abs(wav).max())
    if not math.isfinite(lufs) or peak == 0:
        status, g = 'no_finite_loudness', 0.0
    else:
        status, g = 'ok', TARGET - lufs
        if peak * 10 ** (g / 20) > PEAK_CAP:
            g, status = 20 * math.log10(PEAK_CAP / peak), 'peak_capped'
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, np.clip(wav * 10 ** (g / 20), -1, 1), sr, subtype='PCM_24')
    return (lufs if math.isfinite(lufs) else float('nan')), g, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default='/mnt/seagate/aes_audit/stage1')
    ap.add_argument('--pam', default='/mnt/seagate/aes_audit/pam/human_eval/music')
    ap.add_argument('--out', default='/mnt/seagate/aes_audit/stage2')
    a = ap.parse_args()
    s1, out = Path(a.stage1), Path(a.out)

    items = []   # (set, id, path_to_score, lvl_job or None)
    for sd in sorted(p for p in s1.iterdir() if (p / 'audio').is_dir()):
        for f in sorted((sd / 'audio').glob('*.flac')):
            items.append((sd.name + '_lvl30', f.stem, str(out / (sd.name + '_lvl30') / f.name), (str(f), str(out / (sd.name + '_lvl30') / f.name))))
    for sysd in sorted(p for p in Path(a.pam).iterdir() if p.is_dir()):
        for f in sorted(sysd.glob('*.wav')):
            cid = f'{sysd.name}/{f.stem}'
            items.append(('pam_' + sysd.name, cid, str(f), None))
            dst = out / ('pam_' + sysd.name + '_lvl30') / f'{f.stem}.wav'
            items.append(('pam_' + sysd.name + '_lvl30', cid, str(dst), (str(f), str(dst))))
    print(f'{len(items)} items', flush=True)

    jobs = [it[3] for it in items if it[3]]
    with Pool(24) as pool:
        gres = dict(zip([j[1] for j in jobs], pool.map(gain_one, jobs, chunksize=32)))

    aes, failed = score_aes([it[2] for it in items])
    assert not failed, list(failed.items())[:3]

    with open(out / 'scores_stage2.tsv', 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['set', 'id', 'CE', 'CU', 'PC', 'PQ', 'lufs_src', 'gain_db', 'status'])
        for s, cid, p, job in items:
            lufs, g, st = gres.get(p, (float('nan'), 0.0, 'native'))
            r = aes[p]
            w.writerow([s, cid, r['CE'], r['CU'], r['PC'], r['PQ'], lufs, g, st])
    by = {}
    for s, cid, p, _ in items:
        by.setdefault(s, []).append(aes[p]['PQ'])
    print(json.dumps({k: round(float(np.mean(v)), 4) for k, v in by.items()}, indent=1))


if __name__ == '__main__':
    main()
