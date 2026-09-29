#!/usr/bin/env python
"""AES audit analysis: reference vs training corpus vs generated, and AES vs humans.

Reads stage-1/2 score tables, existing generated per_clip.tsv files (no audio needed)
and the official AES human ratings (AES_natural_music.jsonl, AES_PAM.jsonl).
Prints a report; writes <out>/analysis.json.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

EVN = Path.home() / 'eval_output_nvme'
P8 = 'phase8_qwen_caption2p0_slot0clean_'
GEN = {
    'gen_cfg0':            P8 + 'defectunlab_noq_quarter_s14159265_mc_mf25_cfg0',
    'gen_cfg3neg':         P8 + 'defectunlab_noq_quarter_s14159265_mc_mf25_cfg3_neg',
    'gen_cfg0_lvl30':      P8 + 'defectunlab_noq_quarter_s14159265_mc_mf25_cfg0_lvl30',
    'gen_cfg3neg_lvl30':   P8 + 'defectunlab_noq_quarter_s14159265_mc_mf25_cfg3_neg_lvl30',
    'gen081_hqlq':         P8 + 'qlabel_noq_quarter_s27182818_hqpos_mc_mf25_cfg3_lqneg',
    'gen081_hqlq_lvl30':   P8 + 'qlabel_noq_quarter_s27182818_hqpos_mc_mf25_cfg3_lqneg_lvl30',
    'gen084_n100_lvl30':   P8 + 'negmfn100_noq_quarter_s27182818_mc_mf25_cfg3_neg_lvl30',
}
LQ_RE = re.compile(r'low[- ]quality|poor (audio )?quality|bad (audio )?quality|noisy|amateur|'
                   r'muffled|lo-?fi|distorted|mono(phonic)? recording|recorded (on|with) a (phone|mobile)|'
                   r'live performance|crowd', re.I)
FEATS = ['lufs', 'crest', 'centroid', 'rolloff95', 'flatness', 'hf_4k', 'hf_6k', 'lf_150',
         'frame_db_std', 'noise_floor_db', 'onset_rate', 'tempo', 'harm_ratio']


def read_tsv(p):
    with open(p, newline='') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(t):
    return f'{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]' if t[0] < 0 or abs(t[0]) < 3 else f'{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/seagate/aes_audit')
    ap.add_argument('--tsv', default='/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
    a = ap.parse_args()
    root = Path(a.root)
    res = {}

    # ---------- load ----------
    s1 = read_tsv(root / 'stage1' / 'scores.tsv')
    s2 = read_tsv(root / 'stage2' / 'scores_stage2.tsv')
    pq = defaultdict(dict)       # set -> id -> PQ
    feat = defaultdict(dict)     # set -> id -> {feat}
    for r in s1:
        pq[r['set']][r['id']] = float(r['PQ'])
        feat[r['set']][r['id']] = {k: float(r[k]) for k in FEATS}
    for r in s2:
        pq[r['set']][r['id']] = float(r['PQ'])
    for name, d in GEN.items():
        f = EVN / d / d / 'per_clip.tsv'
        for r in read_tsv(f):
            pq[name][r['id']] = float(r['PQ'])
            if 'lvl30' not in name:
                feat[name][r['id']] = {'lufs': float(r['lufs'] or 'nan'), 'crest': float(r['crest'] or 'nan')}
    caps = {r['id']: r['caption'] for r in read_tsv(a.tsv)}

    # ---------- 1. set means ----------
    print('\n== 1. PQ by set (mean [95% CI]) ==')
    res['set_means'] = {}
    for s in sorted(pq):
        v = list(pq[s].values())
        t = boot_ci(v)
        res['set_means'][s] = {'n': len(v), 'PQ': t}
        print(f'{s:28s} n={len(v):5d}  PQ {fmt(t)}')

    # ---------- 2. paired gen - ref on the same MusicCaps id ----------
    print('\n== 2. paired gen - mc_ref (same id), overall and by caption "low-quality" flag ==')
    res['paired'] = {}
    lq_ids = {i for i, c in caps.items() if LQ_RE.search(c)}
    print(f'caption flagged low-quality/live: {len(lq_ids & set(pq["mc_ref"]))} / {len(pq["mc_ref"])} ref clips')
    for g in GEN:
        refset = 'mc_ref_lvl30' if 'lvl30' in g else 'mc_ref'
        ids = sorted(set(pq[g]) & set(pq[refset]))
        d_all = [pq[g][i] - pq[refset][i] for i in ids]
        d_lq = [pq[g][i] - pq[refset][i] for i in ids if i in lq_ids]
        d_hq = [pq[g][i] - pq[refset][i] for i in ids if i not in lq_ids]
        ref_lq = np.mean([pq[refset][i] for i in ids if i in lq_ids])
        ref_hq = np.mean([pq[refset][i] for i in ids if i not in lq_ids])
        res['paired'][g] = {'vs': refset, 'n': len(ids), 'all': boot_ci(d_all), 'lq': boot_ci(d_lq),
                            'clean': boot_ci(d_hq), 'ref_lq_mean': ref_lq, 'ref_clean_mean': ref_hq,
                            'frac_gen_gt_ref': float(np.mean(np.array(d_all) > 0))}
        print(f'{g:22s} vs {refset:13s} n={len(ids)}  all {fmt(boot_ci(d_all))}  '
              f'LQ-caption {fmt(boot_ci(d_lq))}  clean-caption {fmt(boot_ci(d_hq))}  '
              f'(ref PQ LQ {ref_lq:.3f} / clean {ref_hq:.3f}; gen>ref {np.mean(np.array(d_all) > 0):.1%})')

    # ---------- 3. humans: MusicCaps AES-natural (522 in our ref) ----------
    print('\n== 3. AES vs human PQ ==')
    hum_mc = {}
    for l in open(root / 'AES_natural_music.jsonl'):
        r = json.loads(l)
        if 'musiccaps' in r['data_path']:
            hum_mc[Path(r['data_path']).stem] = float(np.mean(r['Production_Quality']))
    ref_by_yt = {i[:11]: i for i in pq['mc_ref']}
    res['human_mc'] = {}
    for s in ('mc_ref', 'mc_ref_lvl30', 'mc_ref_vae'):
        pairs = [(hum_mc[y], pq[s][ref_by_yt[y]]) for y in hum_mc if y in ref_by_yt]
        h, m = map(np.array, zip(*pairs))
        sp = stats.spearmanr(h, m).statistic
        pr = stats.pearsonr(h, m).statistic
        slope, icpt = np.polyfit(h, m, 1)
        res['human_mc'][s] = {'n': len(h), 'human_mean': h.mean(), 'aes_mean': m.mean(), 'pearson': pr,
                              'spearman': sp, 'aes_sd': m.std(), 'human_sd': h.std(), 'fit_slope': slope,
                              'fit_icpt': icpt}
        print(f'MusicCaps {s:13s} n={len(h)}  human {h.mean():.3f}±{h.std():.2f}  AES {m.mean():.3f}±{m.std():.2f}  '
              f'pearson {pr:.3f} spearman {sp:.3f}  AES≈{slope:.3f}*human+{icpt:.2f}')

    hum_pam = defaultdict(dict)
    for l in open(root / 'AES_PAM.jsonl'):
        r = json.loads(l)
        p = Path(r['data_path'])
        if p.parts[-3] != 'music':
            continue
        hum_pam[p.parts[-2]][f'{p.parts[-2]}/{p.stem}'] = float(np.mean(r['Production_Quality']))
    print('\nPAM music systems (human 10 raters; AES native / lvl30):')
    res['pam'] = {}
    allh, alla = [], []
    for sysn in sorted(hum_pam):
        for suf in ('', '_lvl30'):
            s = f'pam_{sysn}{suf}'
            ids = [i for i in hum_pam[sysn] if i in pq.get(s, {})]
            h = np.array([hum_pam[sysn][i] for i in ids])
            m = np.array([pq[s][i] for i in ids])
            if suf == '':
                allh += list(h); alla += list(m)
            res['pam'][s] = {'n': len(ids), 'human': h.mean(), 'aes': m.mean(),
                             'spearman_within': stats.spearmanr(h, m).statistic}
            print(f'  {s:28s} n={len(ids)} human {h.mean():.3f}  AES {m.mean():.3f}  within-sys spearman {stats.spearmanr(h, m).statistic:.3f}')
    print(f'  PAM music pooled clip-level spearman (native): {stats.spearmanr(allh, alla).statistic:.3f}')
    res['pam_pooled_spearman'] = stats.spearmanr(allh, alla).statistic

    # ---------- 4. what does PQ track? features ----------
    print('\n== 4. feature means by set, and within-set spearman(feature, PQ) ==')
    res['features'] = {}
    sets_f = [s for s in ('mc_ref', 'mc_ref_pk', 'mc_ref_vae', 'jam', 'jam_vae') if s in feat]
    print('feature        ' + ''.join(f'{s:>12s}' for s in sets_f) + '   rho(mc_ref) rho(jam)')
    for k in FEATS:
        row = []
        for s in sets_f:
            v = np.array([feat[s][i][k] for i in feat[s]])
            row.append(np.nanmean(v))
        rhos = []
        for s in ('mc_ref', 'jam'):
            ids = list(feat[s])
            x = np.array([feat[s][i][k] for i in ids]); y = np.array([pq[s][i] for i in ids])
            ok = np.isfinite(x)
            rhos.append(stats.spearmanr(x[ok], y[ok]).statistic)
        res['features'][k] = {'means': dict(zip(sets_f, row)), 'rho_mc_ref': rhos[0], 'rho_jam': rhos[1]}
        print(f'{k:14s} ' + ''.join(f'{v:12.4g}' for v in row) + f'   {rhos[0]:+.3f}      {rhos[1]:+.3f}')

    # AES-minus-human residual on the 522 human-rated MusicCaps clips vs features
    print('\n== 5. where does AES disagree with humans? spearman(feature, AES_z - human_z) on MusicCaps n≈522 ==')
    ids = [ref_by_yt[y] for y in hum_mc if y in ref_by_yt]
    h = np.array([hum_mc[i[:11]] for i in ids]); m = np.array([pq['mc_ref'][i] for i in ids])
    resid = (m - m.mean()) / m.std() - (h - h.mean()) / h.std()
    res['resid_feature_rho'] = {}
    for k in FEATS:
        x = np.array([feat['mc_ref'][i][k] for i in ids]); ok = np.isfinite(x)
        r = stats.spearmanr(x[ok], resid[ok])
        res['resid_feature_rho'][k] = (r.statistic, r.pvalue)
        print(f'  {k:14s} rho {r.statistic:+.3f}  p {r.pvalue:.2g}')

    (root / 'analysis.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
