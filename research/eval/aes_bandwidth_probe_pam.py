"""Is AES's overrating of 16 kHz systems (PAM) a bandwidth effect?

AES resamples every input to 16 kHz, so it cannot see content above 8 kHz.
Hypothesis: humans reward >8 kHz content that AES cannot see, so 16 kHz-output
systems (audioldm2, musicldm) look relatively better to AES than to humans.

Low-passing musicgen and re-scoring with AES cannot test the human side (no
human ratings of low-passed audio exist); it only checks the premise that AES
is blind above 8 kHz. So two parts:

  A. premise   brick-wall low-pass at 8 kHz on the >16 kHz clips (musicgen_large,
               musicgen_melody, real); AES delta vs. original should be ~0.
  B. human     per clip, HF share = energy fraction above 8 kHz. If humans reward
               HF that AES cannot see, residual = human - AES should rise with HF
               share *within* each wide-band system (system-demeaned pooled r
               plus per-system r). Human score vs HF share is reported too, but
               HF share also tracks general production quality, so the residual
               is the test.

Needs output/aes_human_corr_pam/per_clip.tsv (aes_human_corr_pam.py).

Usage:
    python research/eval/aes_bandwidth_probe_pam.py --lp_dir <scratch> \
        --out_dir research/eval/output/aes_bandwidth_probe_pam
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import stats

from aes_human_corr import AXES, boot, score_aes

HERE = Path(__file__).resolve().parent
WIDE = ('musicgen_large', 'musicgen_melody', 'real')
CUT = 8000.0


def hf_share_and_lowpass(path, lp_path):
    wav, sr = sf.read(path, dtype='float64', always_2d=True)
    wav = wav.mean(1)
    spec = np.fft.rfft(wav)
    f = np.fft.rfftfreq(len(wav), 1 / sr)
    p = np.abs(spec) ** 2
    share = float(p[f > CUT].sum() / max(p.sum(), 1e-20))
    spec[f > CUT] = 0
    sf.write(lp_path, np.fft.irfft(spec, n=len(wav)).astype(np.float32), sr, subtype='FLOAT')
    return share


def r_ci(x, y):
    r = stats.pearsonr(x, y)[0]
    ci = boot(lambda i: np.array([stats.pearsonr(x[i], y[i])[0], 0.0]), len(x))[:, 0]
    return float(r), ci.tolist(), float(stats.spearmanr(x, y)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_clip', default=str(HERE / 'output' / 'aes_human_corr_pam' / 'per_clip.tsv'))
    ap.add_argument('--audio_root', default=str(HERE / 'pam_human_eval' / 'human_eval' / 'music'))
    ap.add_argument('--lp_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    out, lp_dir = Path(args.out_dir), Path(args.lp_dir)
    out.mkdir(parents=True, exist_ok=True)
    lp_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.per_clip), delimiter='\t'))
    for r in rows:
        sysname, ytid = r['key'].split('__', 1)
        r['path'] = Path(args.audio_root) / sysname / f'{ytid}.wav'
        r['lp'] = lp_dir / f"{r['key']}.wav"
        r['hf'] = hf_share_and_lowpass(r['path'], r['lp'])

    wide = [r for r in rows if r['system'] in WIDE]
    orig = [str(r['path']) for r in wide]
    lp = [str(r['lp']) for r in wide]
    # score both in one scorer pass layout (same batch composition) for fairness
    per_o, fo = score_aes(orig)
    per_l, fl = score_aes(lp)
    if fo or fl:
        raise SystemExit(f'[FAIL] AES failed on {len(fo)} orig / {len(fl)} lp clips')

    res = {'cut_hz': CUT, 'premise': {}, 'human': {}, 'hf_share_by_system': {}}
    lines = ['# A. premise: AES(lowpass 8k) - AES(orig)',
             'axis\tsystem\tmean_delta\tmean_|delta|\tmax_|delta|\tbatch_rescore_noise_ref']
    for a in AXES:
        for s in WIDE:
            d = np.array([per_l[str(r['lp'])][a] - per_o[str(r['path'])][a] for r in wide if r['system'] == s])
            res['premise'].setdefault(a, {})[s] = {'mean': float(d.mean()), 'mean_abs': float(np.abs(d).mean()),
                                                   'max_abs': float(np.abs(d).max())}
            lines.append(f'{a}\t{s}\t{d.mean():+.4f}\t{np.abs(d).mean():.4f}\t{np.abs(d).max():.4f}\t~1e-3')

    systems = sorted({r['system'] for r in rows})
    lines += ['', '# HF share (>8 kHz energy fraction) by system', 'system\tmedian\tmean\tp90']
    for s in systems:
        h = np.array([r['hf'] for r in rows if r['system'] == s])
        res['hf_share_by_system'][s] = {'median': float(np.median(h)), 'mean': float(h.mean()),
                                        'p90': float(np.percentile(h, 90))}
        lines.append(f'{s}\t{np.median(h):.4f}\t{h.mean():.4f}\t{np.percentile(h, 90):.4f}')

    lines += ['', '# B. human side: residual (human - AES raw) vs log10 HF share, wide-band systems',
              'axis\tscope\tn\tr(resid,HF) [CI]\trho\tr(human,HF)\tr(AES,HF)']
    for a in AXES:
        res['human'][a] = {}
        scopes = [('pooled_demeaned', WIDE)] + [(s, (s,)) for s in WIDE]
        for name, ss in scopes:
            sub = [r for r in rows if r['system'] in ss]
            x = np.log10(np.array([r['hf'] for r in sub]) + 1e-8)
            hum = np.array([float(r[f'human_{a}']) for r in sub])
            aes = np.array([float(r[f'raw_{a}']) for r in sub])
            resid = hum - aes
            if name == 'pooled_demeaned':
                sy = np.array([r['system'] for r in sub])
                for s in ss:
                    m = sy == s
                    x[m] -= x[m].mean(); hum[m] -= hum[m].mean(); aes[m] -= aes[m].mean(); resid[m] -= resid[m].mean()
            r_, ci, rho = r_ci(x, resid)
            d = {'n': len(sub), 'r_resid_hf': r_, 'ci': ci, 'rho': rho,
                 'r_human_hf': float(stats.pearsonr(x, hum)[0]), 'r_aes_hf': float(stats.pearsonr(x, aes)[0])}
            res['human'][a][name] = d
            lines.append(f"{a}\t{name}\t{d['n']}\t{r_:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]\t{rho:+.3f}\t"
                         f"{d['r_human_hf']:+.3f}\t{d['r_aes_hf']:+.3f}")

    table = '\n'.join(lines)
    print(table)
    (out / 'summary.txt').write_text(table + '\n')
    (out / 'summary.json').write_text(json.dumps(res, indent=2))
    with open(out / 'per_clip.tsv', 'w') as f:
        f.write('key\tsystem\thf_share\t' + '\t'.join(f'orig_{a}\tlp_{a}' for a in AXES) + '\n')
        for r in wide:
            f.write(f"{r['key']}\t{r['system']}\t{r['hf']:.6f}\t" + '\t'.join(
                f"{per_o[str(r['path'])][a]:.4f}\t{per_l[str(r['lp'])][a]:.4f}" for a in AXES) + '\n')


if __name__ == '__main__':
    main()
