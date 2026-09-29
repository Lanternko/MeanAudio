"""AES predictor vs. released human ratings on MusicCaps (AES-natural music subset).

Meta released 10-rater scores for 549 MusicCaps clips (all in our eval TSV);
522 of them are in /mnt/HDD/kojiek/musiccaps_reference. We score those clips
with the same AES scorer as eval_metrics.py under two input conditions:

  raw    file as-is (what our standard eval does: AES has no loudness norm)
  lufs23 gain-only normalized to -23 LUFS (ffmpeg-normalize's default EBU R128
         target; the paper loudness-normalized audio before human annotation)

and report Pearson/Spearman vs. the 10-rater mean with bootstrap CIs, the paired
raw-vs-lufs23 difference, and a human noise ceiling (split-half + leave-one-out).

Our reference audio is our own YouTube re-download (16 kHz mono PCM_16), not
Meta's copy, so encode/window differences are part of the error.

Usage:
    python research/eval/aes_human_corr.py --out_dir research/eval/output/aes_human_corr
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts' / 'eval'))
from eval_metrics import score_aes  # noqa: E402

AXES = {'PQ': 'Production_Quality', 'PC': 'Production_Complexity',
        'CE': 'Content_Enjoyment', 'CU': 'Content_Usefulness'}
REF_DIR = Path('/mnt/HDD/kojiek/musiccaps_reference')
TARGET_LUFS = -23.0


def load_human(jsonl):
    out = {}
    for line in open(jsonl):
        r = json.loads(line)
        if '/musiccaps/' not in r['data_path']:
            continue
        ytid = os.path.basename(r['data_path'])[:-4]
        out[ytid] = {k: np.array(r[v], dtype=float) for k, v in AXES.items()}
    return out


def match_ref(ytids):
    by_ytid = {}
    for f in REF_DIR.glob('*.wav'):
        by_ytid[f.stem.rsplit('_', 1)[0]] = f
    return {y: by_ytid[y] for y in ytids if y in by_ytid}


def normalize_lufs(paths, out_dir):
    import pyloudnorm as pyln
    out_dir.mkdir(parents=True, exist_ok=True)
    new, lufs, clipped = {}, {}, 0
    for y, p in paths.items():
        wav, sr = sf.read(p, dtype='float32')
        loud = pyln.Meter(sr).integrated_loudness(wav)
        lufs[y] = loud
        g = 10 ** ((TARGET_LUFS - loud) / 20) if np.isfinite(loud) else 1.0
        w = wav * g
        if np.abs(w).max() > 1.0:
            clipped += 1  # kept as float (no clipping applied); AES reads float32
        q = out_dir / f'{y}.wav'
        sf.write(q, w, sr, subtype='FLOAT')
        new[y] = q
    return new, lufs, clipped


def corr(x, y):
    return stats.pearsonr(x, y)[0], stats.spearmanr(x, y)[0]


def boot(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    return np.percentile(vals, [2.5, 97.5], axis=0)


def noise_ceiling(mat, seed=0):
    """mat: clips x 10 raters (rater slots are not identities; ratings per clip
    come from different people). Returns split-half r (Spearman-Brown to 10)
    and mean leave-one-out single-rating vs. mean-of-other-9 r."""
    rng = np.random.default_rng(seed)
    sh = []
    for _ in range(200):
        perm = np.argsort(rng.random(mat.shape), axis=1)
        m = np.take_along_axis(mat, perm, axis=1)
        r = stats.pearsonr(m[:, :5].mean(1), m[:, 5:].mean(1))[0]
        sh.append(2 * r / (1 + r))
    loo = [stats.pearsonr(mat[:, j], np.delete(mat, j, 1).mean(1))[0] for j in range(mat.shape[1])]
    return float(np.mean(sh)), float(np.mean(loo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human', default=str(HERE / 'aes_human_ratings' / 'AES_natural_music.jsonl'))
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--norm_dir', required=True, help='where -23 LUFS copies are written')
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    human = load_human(args.human)
    paths = match_ref(sorted(human))
    ids = sorted(paths)
    print(f'human MusicCaps clips {len(human)}, matched in ref dir {len(ids)}')

    norm_paths, lufs, clipped = normalize_lufs(paths, Path(args.norm_dir))
    print(f'lufs raw: mean {np.mean([lufs[y] for y in ids]):.2f}  sd {np.std([lufs[y] for y in ids]):.2f}  '
          f'peaks>1 after norm: {clipped}')

    pred = {}
    for cond, pmap in (('raw', paths), ('lufs23', norm_paths)):
        per, failed = score_aes([str(pmap[y]) for y in ids], batch_size=args.batch_size)
        if failed:
            raise SystemExit(f'[FAIL] {cond}: {len(failed)} clips failed AES')
        pred[cond] = {k: np.array([per[str(pmap[y])][k] for y in ids]) for k in AXES}

    H = {k: np.stack([human[y][k] for y in ids]) for k in AXES}
    L = np.array([lufs[y] for y in ids])
    n = len(ids)
    res = {'n': n, 'n_human_musiccaps': len(human), 'target_lufs': TARGET_LUFS,
           'raw_lufs_mean': float(L.mean()), 'raw_lufs_sd': float(L.std()),
           'peaks_over_1_after_norm': clipped, 'axes': {}}
    rows = []
    for k in AXES:
        h = H[k].mean(1)
        sh, loo = noise_ceiling(H[k])
        a = {'human_mean': float(h.mean()), 'human_sd': float(h.std()),
             'noise_ceiling_splithalf_sb10': sh, 'single_rater_vs_rest_r': loo,
             'human_vs_lufs_pearson': float(stats.pearsonr(h, L)[0]),
             'human_vs_lufs_spearman': float(stats.spearmanr(h, L)[0])}
        for cond in pred:
            p = pred[cond][k]
            pr, sr_ = corr(p, h)
            ci = boot(lambda i: corr(p[i], h[i]), n)
            a[cond] = {'pred_mean': float(p.mean()), 'pearson': pr, 'spearman': sr_,
                       'pearson_ci': ci[:, 0].tolist(), 'spearman_ci': ci[:, 1].tolist(),
                       'pred_vs_lufs_pearson': float(stats.pearsonr(p, L)[0])}
        pr_d = lambda i: (corr(pred['lufs23'][k][i], h[i])[0] - corr(pred['raw'][k][i], h[i])[0])
        d = pr_d(np.arange(n))
        a['pearson_diff_lufs23_minus_raw'] = d
        a['pearson_diff_ci'] = boot(pr_d, n).tolist()
        # partial r(pred_raw, human | LUFS)
        def resid(v):
            b = np.polyfit(L, v, 1)
            return v - np.polyval(b, L)
        a['raw_partial_pearson_given_lufs'] = float(stats.pearsonr(resid(pred['raw'][k]), resid(h))[0])
        res['axes'][k] = a
        rows.append(f"{k}\t{a['human_mean']:.2f}\t{sh:.3f}\t{loo:.3f}\t"
                    f"{a['raw']['pearson']:.3f} [{a['raw']['pearson_ci'][0]:.3f},{a['raw']['pearson_ci'][1]:.3f}]\t"
                    f"{a['lufs23']['pearson']:.3f} [{a['lufs23']['pearson_ci'][0]:.3f},{a['lufs23']['pearson_ci'][1]:.3f}]\t"
                    f"{d:+.3f} [{a['pearson_diff_ci'][0]:+.3f},{a['pearson_diff_ci'][1]:+.3f}]\t"
                    f"{a['raw']['spearman']:.3f}\t{a['lufs23']['spearman']:.3f}\t"
                    f"{a['human_vs_lufs_pearson']:+.3f}\t{a['raw']['pred_vs_lufs_pearson']:+.3f}\t"
                    f"{a['lufs23']['pred_vs_lufs_pearson']:+.3f}")
    hdr = ('axis\thuman_mean\tceil_sb10\t1rater_r\tr_raw [CI]\tr_lufs23 [CI]\tdiff [CI]\t'
           'rho_raw\trho_lufs23\thuman~LUFS\traw~LUFS\tlufs23~LUFS')
    table = hdr + '\n' + '\n'.join(rows)
    print(table)
    (out / 'summary.tsv').write_text(table + '\n')
    (out / 'summary.json').write_text(json.dumps(res, indent=2))
    with open(out / 'per_clip.tsv', 'w') as f:
        f.write('ytid\tlufs\t' + '\t'.join(f'human_{k}\traw_{k}\tlufs23_{k}' for k in AXES) + '\n')
        for i, y in enumerate(ids):
            f.write(f'{y}\t{L[i]:.3f}\t' + '\t'.join(
                f"{H[k][i].mean():.2f}\t{pred['raw'][k][i]:.4f}\t{pred['lufs23'][k][i]:.4f}" for k in AXES) + '\n')


if __name__ == '__main__':
    main()
