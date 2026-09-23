#!/usr/bin/env python
"""075 E1: waveform-signature read of the D1 probe on the three quarter checkpoints.

CLAP is not used (D1: pure digital silence outscores real clipping / lowpass on
defect captions). Per clip: spectral flatness and centroid over non-silent frames,
crest and silence from eval_metrics' per_clip.tsv. Per prompt group, the effect is
group mean minus the same checkpoint's d1_clean_rock mean (unpaired: eval.py draws
noise sequentially, so clip k of two groups shares no noise). The prereg question is
whether defectlab's effect exceeds defectunlab's and control066's; those
differences-of-differences get a bootstrap CI.

Usage: d2_075_signature.py [--root ~/eval_output_nvme/d2_075_d1probe] [--out md]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

TSV = Path('/home/kojiek/eval_tsvs_p100/d1_noise_probe_n128.tsv')
TAGS = ['defectlab', 'defectunlab', 'control066']
CELLS = ['cfg0', 'cfg3']
# prompt group -> (signature, expected sign of the effect vs clean rock)
AXES = {
    'd1_axis_noisy': ('flatness', +1),
    'd1_axis_distorted': ('crest', -1),
    'd1_axis_muffled': ('centroid', -1),
    'd1_axis_lofi': ('flatness', +1),
    'd1_full_rock': ('flatness', +1),
    'd1_pure_white': ('flatness', +1),
    'd1_pure_static': ('flatness', +1),
    'd1_pure_clip': ('crest', -1),
    'd1_pure_muffled': ('centroid', -1),
}
FEATS = ['flatness', 'centroid', 'crest', 'rms_dbfs', 'silent']
N_FFT, HOP = 1024, 512


def spectral(path):
    x, sr = sf.read(path, dtype='float32', always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = 1 + (len(x) - N_FFT) // HOP
    fr = np.lib.stride_tricks.as_strided(x, (n, N_FFT), (x.strides[0] * HOP, x.strides[0]))
    rms = np.sqrt((fr ** 2).mean(1))
    keep = 20 * np.log10(rms + 1e-12) > -60
    if keep.sum() < 4:
        return float('nan'), float('nan')
    S = np.abs(np.fft.rfft(fr[keep] * np.hanning(N_FFT), axis=1)) ** 2 + 1e-12
    flat = np.exp(np.log(S).mean(1)) / S.mean(1)
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    cen = (S * f).sum(1) / S.sum(1)
    return float(flat.mean()), float(cen.mean())


def load_cell(d):
    per = {}
    with open(next(d.glob('*/per_clip.tsv')), encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            per[r['id']] = r
    out = defaultdict(lambda: defaultdict(list))
    for cid, r in per.items():
        grp = cid.rsplit('_', 1)[0]
        fl, ce = spectral(d / 'audio' / f'{cid}.flac')
        vals = {'flatness': fl, 'centroid': ce, 'crest': float(r['crest']),
                'rms_dbfs': float(r['rms_dbfs']), 'silent': float(r['silent'])}
        for k, v in vals.items():
            out[grp][k].append(v)
    return {g: {k: np.array(v) for k, v in fs.items()} for g, fs in out.items()}


def effect(cell, grp, feat):
    a, b = cell[grp][feat], cell['d1_clean_rock'][feat]
    return np.nanmean(a) - np.nanmean(b), a[~np.isnan(a)], b[~np.isnan(b)]


def boot_dd(c1, c2, grp, feat, rng, n=4000):
    _, a1, b1 = effect(c1, grp, feat)
    _, a2, b2 = effect(c2, grp, feat)
    s = lambda v: rng.choice(v, (n, len(v))).mean(1)
    dd = (s(a1) - s(b1)) - (s(a2) - s(b2))
    return np.percentile(dd, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path.home() / 'eval_output_nvme/d2_075_d1probe')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(20260923)
    data = {t: {c: load_cell(args.root / t / f'd1_noise_probe_{c}') for c in CELLS} for t in TAGS}
    L = ['# 075 E1 signature read (D1 probe, held-out prompts)', '']
    for c in CELLS:
        L += [f'## {c}', '', '| group | signature | expected | ' + ' | '.join(TAGS)
              + ' | lab−unlab [95% CI] | lab−control [95% CI] |', '|' + '---|' * (5 + len(TAGS))]
        for grp, (feat, sign) in AXES.items():
            eff = [effect(data[t][c], grp, feat)[0] for t in TAGS]
            ci_u = boot_dd(data['defectlab'][c], data['defectunlab'][c], grp, feat, rng)
            ci_c = boot_dd(data['defectlab'][c], data['control066'][c], grp, feat, rng)
            L.append(f'| {grp} | Δ{feat} | {"+" if sign > 0 else "−"} | '
                     + ' | '.join(f'{e:+.4g}' for e in eff)
                     + f' | {eff[0]-eff[1]:+.4g} [{ci_u[0]:+.3g}, {ci_u[1]:+.3g}]'
                     + f' | {eff[0]-eff[2]:+.4g} [{ci_c[0]:+.3g}, {ci_c[1]:+.3g}] |')
        L += ['', f'### {c}: silence rate / RMS on pure-defect prompts', '',
              '| group | ' + ' | '.join(f'{t} silent / rms dBFS' for t in TAGS) + ' |', '|' + '---|' * (1 + len(TAGS))]
        for grp in ['d1_pure_white', 'd1_pure_static', 'd1_pure_clip', 'd1_pure_muffled', 'd1_clean_rock']:
            L.append(f'| {grp} | ' + ' | '.join(
                f'{np.mean(data[t][c][grp]["silent"]):.3f} / {np.mean(data[t][c][grp]["rms_dbfs"]):.1f}'
                for t in TAGS) + ' |')
        L.append('')
    txt = '\n'.join(L)
    print(txt)
    if args.out:
        args.out.write_text(txt + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
