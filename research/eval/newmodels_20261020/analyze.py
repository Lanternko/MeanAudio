"""Screening analysis per docs/experiments/newmodels_screening_20261020.md (pre-registered). Cluster bootstrap B=2000 seed 20261020."""
import csv, json, os, sys, itertools
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
B = 2000
SCORES = {'muqeval_A1': ['muqeval_A1_MI', 'muqeval_A1_TA'], 'muqeval_A3a': ['muqeval_A3a_MI', 'muqeval_A3a_TA'],
          'cmirm': ['cmi_empty_quality', 'cmi_empty_alignment', 'cmi_caption_quality', 'cmi_caption_alignment'], 'pam': ['pam']}
BENCH = {'mc522': 'bench_musiccaps522.tsv', 'pam500': 'bench_pam500.tsv', 'me2748': 'bench_musiceval2748.tsv'}

def load(b):
    rows = list(csv.DictReader(open(BENCH[b]), delimiter='\t'))
    d = {r['key']: r for r in rows}
    if b == 'mc522':
        for r in csv.DictReader(open('scores_meva_mc522.tsv'), delimiter='\t'): d[r['key']]['meva'] = r['meva']
    for m in SCORES:
        f = f'scores_{m}_{b}.tsv'
        for r in csv.DictReader(open(f), delimiter='\t'):
            d[r['key']].update({k: v for k, v in r.items() if k != 'key'})
    return rows

def col(rows, c): return np.array([float(r[c]) if r.get(c) not in (None, '') else np.nan for r in rows])

def wcorr(kind, x, y):
    if kind == 'spearman': return spearmanr(x, y)[0]
    if kind == 'pearson': return pearsonr(x, y)[0]
    return kendalltau(x, y)[0]

def demean(v, sysid):
    out = v.copy()
    for s in np.unique(sysid): out[sysid == s] -= v[sysid == s].mean()
    return out

def analyse(bench, rows, metrics, targets, cluster, sysid, pair_mask, tag):
    """returns list of cell dicts. rows already filtered."""
    n = len(rows); cl = np.array([r[cluster] for r in rows]); ucl, cinv = np.unique(cl, return_inverse=True)
    rng = np.random.default_rng(20261020)
    cnt = rng.multinomial(len(ucl), np.ones(len(ucl)) / len(ucl), size=B)  # prompt counts per bootstrap
    idx_by_cl = [np.where(cinv == i)[0] for i in range(len(ucl))]
    boots = [np.concatenate([np.repeat(idx_by_cl[i], c) for i, c in enumerate(cnt[b]) if c]) for b in range(B)]
    # pair table
    pairs = []
    if pair_mask is not None:
        for i in range(len(ucl)):
            ix = [j for j in idx_by_cl[i] if pair_mask(rows[j])]
            for a, b_ in itertools.combinations(ix, 2): pairs.append((i, a, b_))
    P = np.array(pairs) if pairs else np.zeros((0, 3), int)
    out = []
    H = {t: col(rows, t) for t in targets}; M = {m: col(rows, m) for m in metrics}
    stat_cache = {}
    for t in targets:
        h = H[t]
        for m in metrics:
            x = M[m]; ok = np.isfinite(x) & np.isfinite(h)
            if ok.sum() < 30: continue
            cell = dict(bench=tag, target=t, metric=m, n=int(ok.sum()))
            res = {}
            for kind in ('spearman', 'pearson', 'kendall'):
                pt = wcorr(kind, x[ok], h[ok])
                nb = B if kind != 'kendall' or n < 1500 else 500
                bs = []
                for bi in boots[:nb]:
                    bi = bi[ok[bi]]
                    bs.append(wcorr(kind, x[bi], h[bi]))
                res[kind] = (pt, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), np.array(bs))
                cell[kind] = [float(pt), float(res[kind][1]), float(res[kind][2])]
            if sysid is not None:
                xs, hs = demean(x, sysid), demean(h, sysid)
                pt = spearmanr(xs[ok], hs[ok])[0]; bs = []
                for bi in boots:
                    bi = bi[ok[bi]]
                    bs.append(spearmanr(demean(x[bi], sysid[bi]), demean(h[bi], sysid[bi]))[0])
                cell['within'] = [float(pt), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
                res['within'] = np.array(bs)
            if len(P) >= 200:
                a, b_ = P[:, 1], P[:, 2]
                hd = h[a] - h[b_]; md = x[a] - x[b_]
                valid = np.isfinite(hd) & np.isfinite(md) & (hd != 0)
                credit = np.where(md == 0, 0.5, (np.sign(md) == np.sign(hd)).astype(float))
                bins = {'all': valid, 'lt05': valid & (np.abs(hd) < 0.5), '05_1': valid & (np.abs(hd) >= 0.5) & (np.abs(hd) < 1), 'ge1': valid & (np.abs(hd) >= 1)}
                for k, mk in bins.items():
                    num = np.bincount(P[:, 0], weights=credit * mk, minlength=len(ucl)); den = np.bincount(P[:, 0], weights=mk.astype(float), minlength=len(ucl))
                    pt = num.sum() / den.sum() if den.sum() else np.nan
                    bs = (cnt @ num) / np.maximum(cnt @ den, 1e-9)
                    cell['pair_' + k] = [float(pt), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)), int(den.sum())]
                    if k == 'all': res['pair'] = bs
            stat_cache[(t, m)] = res
            out.append(cell)
    # paired deltas vs AES PQ / CE (same bootstrap draws)
    for c in out:
        for ref in ('aes_PQ', 'aes_CE'):
            r0, r1 = stat_cache.get((c['target'], ref)), stat_cache[(c['target'], c['metric'])]
            if r0 is None or c['metric'] == ref: continue
            for kind in ('spearman', 'within', 'pair'):
                if kind in r0 and kind in r1:
                    d = (r1[kind][3] if isinstance(r1[kind], tuple) else r1[kind]) - (r0[kind][3] if isinstance(r0[kind], tuple) else r0[kind])
                    pt = c[kind][0] if kind != 'pair' else c['pair_all'][0]
                    pt0 = [x for x in out if x['target'] == c['target'] and x['metric'] == ref][0]
                    pt0 = pt0[kind][0] if kind != 'pair' else pt0['pair_all'][0]
                    c[f'd_{ref}_{kind}'] = [float(pt - pt0), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    return out

AES = ['aes_PQ', 'aes_PC', 'aes_CE', 'aes_CU']
NEW = [m for v in SCORES.values() for m in v]
allcells = []
r = load('mc522')
allcells += analyse('mc522', r, AES + ['meva'] + NEW, ['human_PQ', 'human_CE', 'human_CU', 'human_PC'], 'key', None, None, 'MC522')
r = [x for x in load('pam500') if x['system'] != 'real']
sysid = np.array([x['system'] for x in r])
allcells += analyse('pam400', r, AES + ['meva'] + NEW, ['pam_OVL', 'pam_REL', 'human_PQ', 'human_CE', 'human_CU', 'human_PC'], 'prompt', sysid, lambda _: True, 'PAM400')
r = load('me2748')
sysid = np.array([x['system'] for x in r])
allcells += analyse('me', r, AES + ['meva'] + NEW, ['human_OVL', 'human_REL'], 'prompt', sysid, lambda x: x['shared'] == 'True', 'ME')
rt = [x for x in r if x['split'] == 'test']
allcells += analyse('me_test', rt, AES + ['meva'] + NEW, ['human_OVL', 'human_REL'], 'prompt', np.array([x['system'] for x in rt]), lambda x: x['shared'] == 'True', 'ME-test')
json.dump(allcells, open('analysis_cells.json', 'w'), indent=1)
print('cells', len(allcells))
