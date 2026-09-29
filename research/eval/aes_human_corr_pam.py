"""AES predictor vs. released human ratings on PAM generated music.

`AES_PAM.jsonl` has 10-rater AES-axis scores for PAM's human_eval set
(zenodo 10737388): music = audioldm2 / musicgen_large / musicgen_melody /
musicldm / real, 100 clips each. Audio is downloaded to
research/eval/pam_human_eval/.

Reports, for raw and -23 LUFS inputs:
  pooled    all 500 music clips (includes real -> easy between-system spread)
  gen       400 generated clips only
  within    per-system correlation (n=100 each) -- closest to comparing clips
            from one model, which is what arm-vs-arm comparison needs
  system    5-point system-level mean ranking (Spearman; tiny n, descriptive)
  paired    all 5 systems share the same 100 MusicCaps prompts, so for each
            prompt and each pair of *generated* systems (6 pairs) we ask whether
            AES picks the same winner as the human mean -- this is the
            arm-vs-arm question (same prompt, different model). Reported vs.
            the AES-axis human mean and vs. PAM's own OVL panel (scores.csv),
            binned by the size of the human difference. Human-panel ceiling =
            agreement between two random halves of the 10 raters.

Usage:
    python research/eval/aes_human_corr_pam.py \
        --out_dir research/eval/output/aes_human_corr_pam --norm_dir <scratch>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from aes_human_corr import AXES, boot, corr, noise_ceiling, normalize_lufs, score_aes

HERE = Path(__file__).resolve().parent


def load(jsonl, audio_root, kind):
    items = {}
    for line in open(jsonl):
        r = json.loads(line)
        rel = r['data_path'].split('/your_path/PAM/', 1)[1]  # human_eval/<kind>/<system>/<file>
        parts = rel.split('/')
        if parts[1] != kind:
            continue
        p = audio_root / rel
        key = f'{parts[2]}__{Path(parts[-1]).stem}'
        items[key] = {'path': p, 'system': parts[2],
                      'human': {k: np.array(r[v], dtype=float) for k, v in AXES.items()}}
    return items


def block(pred, human, idx):
    p, h = pred[idx], human[idx]
    r, rho = corr(p, h)
    ci = boot(lambda i: corr(p[i], h[i]), len(idx))
    return {'n': int(len(idx)), 'pearson': float(r), 'spearman': float(rho),
            'pearson_ci': ci[:, 0].tolist(), 'spearman_ci': ci[:, 1].tolist()}


def load_pam_scores(csv_path):
    import csv
    out = {}
    for r in csv.DictReader(open(csv_path)):
        out[f"{r['Model']}__{r['File Name']}"] = {'OVL': float(r['OVL']), 'REL': float(r['REL'])}
    return out


def pairs_of(keys, systems):
    """(i, j) index pairs: same prompt, two different generated systems."""
    by_prompt = {}
    for i, k in enumerate(keys):
        if systems[i] != 'real':
            by_prompt.setdefault(k.split('__', 1)[1], []).append(i)
    prompts = sorted(by_prompt)
    pp = [[(a, b) for x, a in enumerate(by_prompt[q]) for b in by_prompt[q][x + 1:]] for q in prompts]
    return prompts, pp


def agreement(pred, ref, pp, idx_prompts, thr_lo=None, thr_hi=None):
    hit = tot = 0
    for q in idx_prompts:
        for a, b in pp[q]:
            d = ref[a] - ref[b]
            if d == 0 or (thr_lo is not None and abs(d) < thr_lo) or (thr_hi is not None and abs(d) >= thr_hi):
                continue
            tot += 1
            hit += np.sign(pred[a] - pred[b]) == np.sign(d)
    return (hit / tot if tot else float('nan')), tot


def paired_block(pred, ref, pp, bins=((None, None), (None, 0.5), (0.5, 1.0), (1.0, None))):
    n = len(pp)
    out = {}
    for lo, hi in bins:
        name = f"diff[{lo or 0},{hi or 'inf'})"
        acc, tot = agreement(pred, ref, pp, range(n), lo, hi)
        rng = np.random.default_rng(0)
        bs = [agreement(pred, ref, pp, rng.integers(0, n, n), lo, hi)[0] for _ in range(1000)]
        out[name] = {'acc': float(acc), 'n_pairs': tot, 'ci': np.nanpercentile(bs, [2.5, 97.5]).tolist()}
    return out


def split_half_agreement(mat, pp, reps=200, seed=0):
    """Human-panel ceiling: rater half A predicts the winner under half B."""
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(reps):
        perm = np.argsort(rng.random(mat.shape), axis=1)
        m = np.take_along_axis(mat, perm, axis=1)
        accs.append(agreement(m[:, :5].mean(1), m[:, 5:].mean(1), pp, range(len(pp)))[0])
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--human', default=str(HERE / 'aes_human_ratings' / 'AES_PAM.jsonl'))
    ap.add_argument('--audio_root', default=str(HERE / 'pam_human_eval'))
    ap.add_argument('--kind', default='music')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--norm_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = load(args.human, Path(args.audio_root), args.kind)
    missing = [k for k, v in items.items() if not v['path'].exists()]
    if missing:
        raise SystemExit(f'[FAIL] {len(missing)} audio files missing (first: {missing[:3]})')
    keys = sorted(items)
    systems = np.array([items[k]['system'] for k in keys])
    print(f'{args.kind}: {len(keys)} clips, systems {sorted(set(systems))}')

    raw_paths = {k: items[k]['path'] for k in keys}
    norm_paths, lufs, clipped = normalize_lufs(raw_paths, Path(args.norm_dir))
    L = np.array([lufs[k] for k in keys])
    print(f'raw LUFS mean {np.nanmean(L):.2f} sd {np.nanstd(L):.2f}; peaks>1 after norm {clipped}')

    pam = load_pam_scores(Path(args.audio_root) / 'human_eval' / args.kind / 'scores.csv')
    OVL = np.array([pam[k]['OVL'] for k in keys])
    REL = np.array([pam[k]['REL'] for k in keys])
    prompts, pp = pairs_of(keys, systems)

    pred = {}
    for cond, pmap in (('raw', raw_paths), ('lufs23', norm_paths)):
        per, failed = score_aes([str(pmap[k]) for k in keys], batch_size=args.batch_size)
        if failed:
            raise SystemExit(f'[FAIL] {cond}: {len(failed)} clips failed AES')
        pred[cond] = {a: np.array([per[str(pmap[k])][a] for k in keys]) for a in AXES}

    H = {a: np.stack([items[k]['human'][a] for k in keys]) for a in AXES}
    gen = np.where(systems != 'real')[0]
    allidx = np.arange(len(keys))
    sysnames = sorted(set(systems))
    res = {'n': len(keys), 'systems': sysnames, 'raw_lufs_mean': float(np.nanmean(L)),
           'peaks_over_1_after_norm': clipped, 'axes': {}}
    lines = ['axis\tcond\tpooled r [CI]\tgen r [CI]\t' + '\t'.join(f'within:{s}' for s in sysnames)
             + '\tsys_rho\tceil_gen_sb10\t1rater_gen']
    paired_lines = ['axis\tcond\thuman_splithalf\tvsAxis:all\tvsAxis:<0.5\tvsAxis:0.5-1\tvsAxis:>=1\t'
                    'vsOVL:all\tvsOVL:<0.5\tvsOVL:0.5-1\tvsOVL:>=1\tr(AES,OVL)gen\tr(humanAxis,OVL)gen']
    sys_lines = ['axis\tsystem\thuman_mean\traw_mean\tlufs23_mean\tlufs_mean']
    for a in AXES:
        h = H[a].mean(1)
        ceil, one = noise_ceiling(H[a][gen])
        ra = {'noise_ceiling_gen_sb10': ceil, 'single_rater_gen': one,
              'human_vs_lufs_gen_pearson': float(stats.pearsonr(h[gen], L[gen])[0])}
        for cond in pred:
            p = pred[cond][a]
            d = {'pooled': block(p, h, allidx), 'gen': block(p, h, gen),
                 'within': {s: block(p, h, np.where(systems == s)[0]) for s in sysnames}}
            hm = [h[systems == s].mean() for s in sysnames]
            pm = [p[systems == s].mean() for s in sysnames]
            d['system_means_human'] = dict(zip(sysnames, map(float, hm)))
            d['system_means_pred'] = dict(zip(sysnames, map(float, pm)))
            d['system_spearman'] = float(stats.spearmanr(hm, pm)[0])
            d['pred_vs_lufs_gen_pearson'] = float(stats.pearsonr(p[gen], L[gen])[0])
            ra[cond] = d
            fmt = lambda b: f"{b['pearson']:.3f} [{b['pearson_ci'][0]:.3f},{b['pearson_ci'][1]:.3f}]"
            lines.append(f"{a}\t{cond}\t{fmt(d['pooled'])}\t{fmt(d['gen'])}\t"
                         + '\t'.join(f"{d['within'][s]['pearson']:.3f}" for s in sysnames)
                         + f"\t{d['system_spearman']:+.2f}\t{ceil:.3f}\t{one:.3f}")
        for s in sysnames:
            m = systems == s
            sys_lines.append(f"{a}\t{s}\t{h[m].mean():.2f}\t{pred['raw'][a][m].mean():.2f}\t"
                             f"{pred['lufs23'][a][m].mean():.2f}\t{np.nanmean(L[m]):.1f}")
        ra['paired_ceiling_splithalf'] = split_half_agreement(H[a], pp)
        ra['human_axis_vs_pam_OVL_gen_pearson'] = float(stats.pearsonr(h[gen], OVL[gen])[0])
        ra['human_axis_vs_pam_REL_gen_pearson'] = float(stats.pearsonr(h[gen], REL[gen])[0])
        ra['paired_human_axis_vs_OVL'] = paired_block(h, OVL, pp)
        for cond in pred:
            p = pred[cond][a]
            ra[cond]['paired_vs_human_axis'] = paired_block(p, h, pp)
            ra[cond]['paired_vs_pam_OVL'] = paired_block(p, OVL, pp)
            ra[cond]['vs_pam_OVL_gen'] = block(p, OVL, gen)
            ra[cond]['vs_pam_REL_gen'] = block(p, REL, gen)
            pa = ra[cond]['paired_vs_human_axis']
            po = ra[cond]['paired_vs_pam_OVL']
            paired_lines.append(
                f"{a}\t{cond}\t{ra['paired_ceiling_splithalf']:.3f}\t"
                + '\t'.join(f"{v['acc']:.3f} (n={v['n_pairs']})" for v in pa.values()) + '\t'
                + '\t'.join(f"{v['acc']:.3f} (n={v['n_pairs']})" for v in po.values())
                + f"\t{ra[cond]['vs_pam_OVL_gen']['pearson']:.3f}\t{ra['human_axis_vs_pam_OVL_gen_pearson']:.3f}")
        res['axes'][a] = ra
    table = '\n'.join(lines) + '\n\n' + '\n'.join(paired_lines) + '\n\n' + '\n'.join(sys_lines)
    print(table)
    (out / 'summary.tsv').write_text(table + '\n')
    (out / 'summary.json').write_text(json.dumps(res, indent=2))
    with open(out / 'per_clip.tsv', 'w') as f:
        f.write('key\tsystem\tlufs\tpam_OVL\tpam_REL\t' + '\t'.join(f'human_{a}\traw_{a}\tlufs23_{a}' for a in AXES) + '\n')
        for i, k in enumerate(keys):
            f.write(f'{k}\t{systems[i]}\t{L[i]:.3f}\t{OVL[i]}\t{REL[i]}\t' + '\t'.join(
                f"{H[a][i].mean():.2f}\t{pred['raw'][a][i]:.4f}\t{pred['lufs23'][a][i]:.4f}" for a in AXES) + '\n')


if __name__ == '__main__':
    main()
