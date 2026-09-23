#!/usr/bin/env python
"""Scalar-gain every clip of an eval cell to a fixed integrated LUFS, then rescore.

Loudness gate for arm comparisons (063/065: PQ/CU rise as level drops, CLAP rises
with level). Each clip gets one scalar gain to --target LUFS; if that gain would push
peak above --peak_cap the gain is capped (status peak_capped) and the clip is kept but
flagged. Clips with no finite loudness are copied unchanged and flagged.

Output: <out_root>/<cell>_lvl<target>/audio/*.flac (same sr / PCM_16 as input),
gain_status.tsv, then scripts/eval/eval_metrics.py into the same directory.

    python scripts/eval/level_match_rescore.py --cell_dir ~/eval_output_nvme/<EXP>_mc_mf25_cfg0 \
        --tsv /mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _gain_one(args):
    src, dst, target, peak_cap = args
    import pyloudnorm
    import soundfile as sf
    info = sf.info(src)
    wav, sr = sf.read(src, dtype='float64', always_2d=True)
    lufs = pyloudnorm.Meter(sr).integrated_loudness(wav)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if not math.isfinite(lufs) or peak == 0.0:
        status, gain_db = 'no_finite_loudness', 0.0
    else:
        gain_db = target - lufs
        status = 'ok'
        if peak * 10 ** (gain_db / 20) > peak_cap:
            gain_db = 20 * math.log10(peak_cap / peak)
            status = 'peak_capped'
    out = np.clip(wav * 10 ** (gain_db / 20), -1.0, 1.0)
    sf.write(dst, out, sr, subtype=info.subtype)
    return Path(src).stem, status, lufs if math.isfinite(lufs) else None, gain_db


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cell_dir', required=True, help='eval cell dir containing audio/')
    p.add_argument('--tsv', required=True)
    p.add_argument('--target', type=float, default=-30.0)
    p.add_argument('--peak_cap', type=float, default=0.999)
    p.add_argument('--out_root', default=None, help='default: parent of cell_dir')
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()

    cell = Path(a.cell_dir).expanduser().resolve()
    src_audio = cell / 'audio'
    tag = f'lvl{int(round(-a.target))}'
    out = Path(a.out_root).expanduser() if a.out_root else cell.parent
    out = out / f'{cell.name}_{tag}'
    dst_audio = out / 'audio'
    dst_audio.mkdir(parents=True, exist_ok=True)

    srcs = sorted(src_audio.glob('*.flac'))
    if not srcs:
        sys.exit(f'no flac in {src_audio}')
    jobs = [(str(s), str(dst_audio / s.name), a.target, a.peak_cap) for s in srcs]
    with Pool(a.workers) as pool:
        res = pool.map(_gain_one, jobs, chunksize=64)

    counts = {}
    with open(out / 'gain_status.tsv', 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['id', 'status', 'src_lufs', 'gain_db'])
        for r in res:
            w.writerow(r)
            counts[r[1]] = counts.get(r[1], 0) + 1
    summary = {'n': len(res), 'target': a.target, 'peak_cap': a.peak_cap, 'status': counts,
               'source': str(cell)}
    (out / 'gain_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)

    subprocess.run([sys.executable, str(HERE / 'eval_metrics.py'), '--gen_dir', str(dst_audio),
                    '--tsv', a.tsv, '--exp_name', f'{cell.name}_{tag}', '--out_dir', str(out)],
                   check=True)


if __name__ == '__main__':
    main()
