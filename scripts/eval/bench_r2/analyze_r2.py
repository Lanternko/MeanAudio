"""round-2 benchmark screening 分析（benchmark_screening_r2_20261025.md §6–§8）。

讀 /mnt/seagate/benchmarks_r2/scores/<dataset>/<model>__<variant>_<loud>.tsv 與各資料集的人類標籤，
輸出 analysis/<dataset>_*.tsv 與 analysis/summary.json、analysis/REPORT.md。
bootstrap 2000 次、seed 20261025、percentile 95%；同一資料集同一比較內所有指標共用同一組重抽索引。

用法：
  python -I analyze_r2.py [--datasets odaq,mc522,...] [--nboot 2000] [--out DIR]
"""
import argparse, base64, csv, importlib.util, json, sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bench_datasets', HERE / 'bench_datasets.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

SEED = 20261025
ROOT = D.ROOT

# 主要格（各資料集的 G1–G3 用哪一格）
PRIMARY = {'odaq': ('native', 'raw'), 'aime': ('native', 'raw'), 'singmos': ('native', 'raw'),
           'mc522': ('V10', 'raw'), 'songeval': ('VFULL', 'raw'), 'arena': ('native', 'raw')}

# 進閘門的候選欄位（§4）；其餘欄位只描述
CAND_COL = {'aes': 'aes_PQ', 'meva': 'meva', 'muqeval_A1': 'muqeval_A1_MI', 'muqeval_A3a': 'muqeval_A3a_MI',
            'cmirm': 'cmi_empty_quality', 'pam': 'pam', 'songeval': 'se_mus', 'singmos': 'singmos'}

# §5：各指標在哪些 dataset 上「獨立」（可進閘門）
INDEP = {'aes': {'odaq', 'aime', 'singmos', 'arena'},
         'meva': {'odaq', 'singmos', 'mc522'},
         'muqeval_A1': {'odaq', 'singmos', 'mc522'},
         'muqeval_A3a': {'odaq', 'singmos', 'mc522'},
         'cmirm': {'odaq', 'singmos', 'mc522'},
         'pam': {'odaq', 'aime', 'singmos', 'mc522', 'arena'},
         'singmos': {'singmos'},
         'songeval': set(), 'clap': set()}
GENERATIVE = {'aime', 'singmos', 'arena', 'songeval'}

# §7.2：各指標輸入 Nyquist 類別（LP 可見性）
PATH = {'aes': 16, 'singmos': 16, 'meva': 16, 'muqeval_A1': 24, 'muqeval_A3a': 24, 'cmirm': 24, 'embed': 24,
        'pam': 44, 'clap': 48, 'songeval': 24}
LP_VISIBLE = {16: {'Q1'}, 24: {'Q1', 'Q2', 'Q3'}, 44: {'Q1', 'Q2', 'Q3', 'Q4', 'Q5'}, 48: {'Q1', 'Q2', 'Q3', 'Q4', 'Q5'}}


# ───────────────────────── loading ─────────────────────────
def load_scores(ds, root=None):
    """{(model, variant, loud): {item_id: {col: float}}}"""
    out = {}
    for p in sorted((Path(root or ROOT) / 'scores' / ds).glob('*.tsv')):
        if p.name.endswith('.err.tsv'):
            continue
        model, rest = p.stem.split('__', 1)
        variant, loud = rest.rsplit('_', 1)
        rows = {}
        with open(p, newline='') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                v = {}
                for k, x in r.items():
                    if k in ('unit_id', 'item_id', 'n_win') or x is None:
                        continue
                    if model == 'embed':
                        v[k] = x
                        continue
                    try:
                        fx = float(x)
                    except ValueError:
                        continue
                    if np.isfinite(fx):
                        v[k] = fx
                if v:
                    rows[r['item_id']] = v
        out[(model, variant, loud)] = rows
    return out


def metric_table(sc, variant, loud):
    """{metric_name: {item: value}} for one (variant, loud) cell. 欄位名即指標名（_med/_first 只描述）。"""
    t = {}
    for (model, v, l), rows in sc.items():
        if (v, l) != (variant, loud) or model == 'embed':
            continue
        for item, d in rows.items():
            for col, x in d.items():
                t.setdefault((model, col), {})[item] = x
    return t


def unpack_emb(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


# ───────────────────────── stats helpers ─────────────────────────
def rank_rows(a):
    return stats.rankdata(a, axis=-1)


def _pearson_rows(X, Y):
    X = X - X.mean(-1, keepdims=True)
    Y = Y - Y.mean(-1, keepdims=True)
    den = np.sqrt((X * X).sum(-1) * (Y * Y).sum(-1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (X * Y).sum(-1) / den


def boot_idx(n, nboot, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(nboot, n))


def boot_ci(vals):
    v = vals[np.isfinite(vals)]
    if len(v) < 20:
        return (float('nan'), float('nan'))
    return tuple(np.percentile(v, [2.5, 97.5]))


def corr_stats(x, y, idx, kendall=True):
    """Pearson/Spearman/Kendall 點估計；Pearson/Spearman 附 percentile CI（idx 為共用重抽索引）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return dict(n=len(x))
    out = dict(n=len(x))
    out['pearson'] = float(stats.pearsonr(x, y)[0])
    out['spearman'] = float(stats.spearmanr(x, y)[0])
    if kendall:
        out['kendall'] = float(stats.kendalltau(x, y)[0])
    xs, ys = x[idx], y[idx]
    out['pearson_ci'] = boot_ci(_pearson_rows(xs, ys))
    out['spearman_ci'] = boot_ci(_pearson_rows(rank_rows(xs), rank_rows(ys)))
    return out


def center_by(vals, groups):
    vals = np.asarray(vals, float)
    g = np.asarray(groups)
    out = vals.copy()
    for k in np.unique(g):
        m = g == k
        out[m] = vals[m] - vals[m].mean()
    return out


def fmt(v, nd=3):
    if isinstance(v, tuple):
        return f'[{v[0]:.{nd}f}, {v[1]:.{nd}f}]'
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'nan'
    return f'{v:.{nd}f}' if isinstance(v, float) else str(v)


def gate_flags(st, thr, ci_key='spearman_ci', key='spearman'):
    if key not in st:
        return None
    return bool(st[key] >= thr and st[ci_key][0] > 0.30)


# ───────────────────────── per-item datasets (G1/G2) ─────────────────────────
def human_targets(ds):
    """{item_id: {target_name: value}}, {item_id: system} — 逐 item 的人類分數。"""
    items = D.read_manifest(ds)
    H, S = {}, {}
    for it in items:
        ex = json.loads(it['extra'])
        if ds == 'singmos':
            if 'test' not in it['subset'].split(';'):
                continue
            H[it['item_id']] = {'mos': ex['mos']}
            S[it['item_id']] = ex['system']
        elif ds == 'mc522':
            H[it['item_id']] = {'human_PQ': ex['human_PQ']}
        elif ds == 'songeval':
            H[it['item_id']] = dict(ex['mean'])
            S[it['item_id']] = ex['gender']
    return H, S


def analyze_item_level(ds, nboot, out_dir, log):
    sc = load_scores(ds)
    H, S = human_targets(ds)
    primary_target = {'singmos': 'mos', 'mc522': 'human_PQ', 'songeval': 'Musicality'}[ds]
    rows = []
    cells = sorted({(v, l) for (_, v, l) in sc})
    cache_idx = {}
    for variant, loud in cells:
        tab = metric_table(sc, variant, loud)
        for (model, col), vals in sorted(tab.items()):
            items = [i for i in vals if i in H]
            if len(items) < 20:
                continue
            n = len(items)
            if n not in cache_idx:
                cache_idx[n] = boot_idx(n, nboot)
            idx = cache_idx[n]
            x = np.array([vals[i] for i in items])
            for target in H[items[0]]:
                y = np.array([H[i][target] for i in items])
                st = corr_stats(x, y, idx, kendall=(target == primary_target))
                if 'spearman' not in st:
                    continue
                r = dict(dataset=ds, model=model, col=col, variant=variant, loud=loud, target=target,
                         primary=int(target == primary_target and (variant, loud) == PRIMARY[ds]), **{
                             'n': st['n'], 'pearson': st['pearson'], 'pearson_ci': st['pearson_ci'],
                             'spearman': st['spearman'], 'spearman_ci': st['spearman_ci'],
                             'kendall': st.get('kendall', float('nan'))})
                if ds in S and target == primary_target and ds != 'songeval':
                    sysn = [S[i] for i in items]
                    xw, yw = center_by(x, sysn), center_by(y, sysn)
                    sw = corr_stats(xw, yw, idx, kendall=False)
                    r['within_sys'] = sw.get('spearman', float('nan'))
                    r['within_sys_ci'] = sw.get('spearman_ci', (float('nan'),) * 2)
                rows.append(r)
    write_rows(out_dir / f'{ds}_corr.tsv', rows)
    log(f'{ds}: {len(rows)} correlation rows')
    return rows, sc, H


def paired_delta(ds, nboot, sc, H, target, out_dir, kind, log):
    """段長 Δρ（V* − V10）與響度 Δρ（norm − raw），paired bootstrap（共用重抽索引）。"""
    pairs = []
    if kind == 'length':
        for v in ('V5', 'V30', 'VFULL'):
            pairs.append(((v, 'raw'), ('V10', 'raw')))
    else:
        for v in ('native', 'V10', 'VFULL'):
            pairs.append(((v, 'norm'), (v, 'raw')))
    rows = []
    for (c1, c0) in pairs:
        t1, t0 = metric_table(sc, *c1), metric_table(sc, *c0)
        for key in sorted(set(t1) & set(t0)):
            common = [i for i in t1[key] if i in t0[key] and i in H]
            if len(common) < 20:
                continue
            idx = boot_idx(len(common), nboot)
            y = np.array([H[i][target] for i in common])
            x1 = np.array([t1[key][i] for i in common])
            x0 = np.array([t0[key][i] for i in common])
            r1, r0 = stats.spearmanr(x1, y)[0], stats.spearmanr(x0, y)[0]
            ys = y[idx]
            b1 = _pearson_rows(rank_rows(x1[idx]), rank_rows(ys))
            b0 = _pearson_rows(rank_rows(x0[idx]), rank_rows(ys))
            d = b1 - b0
            ci = boot_ci(d)
            dr = r1 - r0
            if kind == 'length':
                lab = ('stable' if abs(dr) < 0.05 and ci[0] <= 0 <= ci[1] else
                       'sensitive' if abs(dr) >= 0.05 and not (ci[0] <= 0 <= ci[1]) else 'inconclusive')
            else:
                lab = ('loudness-robust' if abs(dr) < 0.05 and ci[0] <= 0 <= ci[1] else
                       'loudness-dependent' if abs(dr) >= 0.10 and not (ci[0] <= 0 <= ci[1]) else 'inconclusive')
            rows.append(dict(dataset=ds, model=key[0], col=key[1], cmp=f'{c1[0]}_{c1[1]}-{c0[0]}_{c0[1]}',
                             rho_a=float(r1), rho_b=float(r0), delta=float(dr), delta_ci=ci, n=len(common), label=lab))
    write_rows(out_dir / f'{ds}_{kind}_delta.tsv', rows)
    log(f'{ds}: {len(rows)} {kind} delta rows')
    return rows


# ───────────────────────── ODAQ (G6) ─────────────────────────
def odaq_human():
    import pandas as pd
    d = pd.read_csv(D.DL / 'ODAQ/ODAQ/ODAQ_listening_test/ODAQ_results.csv')
    m = d.groupby(['item', 'process', 'method', 'condition']).score.mean().reset_index()
    sub = d.pivot_table(index=['item', 'process'], columns='subject', values='score')
    return m, sub


def lp_visible(model, method, cond):
    if method != 'LP' or cond not in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
        return True
    return cond in LP_VISIBLE[PATH.get(model, 44)]


def analyze_odaq(nboot, out_dir, log):
    m, sub = odaq_human()
    hum = {f'{r.item}/{r.process}': r.score for r in m.itertuples()}
    meta = {f'{r.item}/{r.process}': (r.item, r.method, r.condition) for r in m.itertuples()}
    sc = load_scores('odaq')
    rows, pair_rows = [], []
    rng = np.random.default_rng(SEED)
    # embed 參考距離（MERT-ref／MuQ-ref）：−cosim(x, 該 trial 的 hidden ref)
    emb = {}
    for (model, v, l), r in sc.items():
        if model != 'embed':
            continue
        for k in ('mert_emb', 'muq_emb'):
            vals = {i: unpack_emb(d[k]) for i, d in r.items() if k in d}
            ref = {meta[i][0]: e for i, e in vals.items() if i in meta and i.endswith('/reference')}
            sc[(k.replace('_emb', '_ref'), v, l)] = {
                i: {k.replace('_emb', '_ref'): -float(np.dot(e, ref[meta[i][0]]) / (np.linalg.norm(e) * np.linalg.norm(ref[meta[i][0]])))}
                for i, e in vals.items() if i in meta and meta[i][0] in ref}
        del sc[(model, v, l)]
    for variant, loud in sorted({(v, l) for (_, v, l) in sc}):
        tab = metric_table(sc, variant, loud)
        for (model, col), vals in sorted(tab.items()):
            fam = model if model in PATH else ('embed' if model.endswith('_ref') else model)
            trials = {}
            for i, x in vals.items():
                if i in hum:
                    trials.setdefault(meta[i][0], []).append(i)
            rho_all, rho_vis, q_ok, q_n, ref_top, ref_n = [], [], [], [], 0, 0
            by_method = {}
            for t, ids in trials.items():
                qs = [i for i in ids if meta[i][2].startswith('Q')]
                method = meta[qs[0]][1] if qs else ''
                vis = [i for i in qs if lp_visible(fam, method, meta[i][2])]
                # G6a：可見的 Q1–Q5 條件上的 within-trial Spearman（≥3 條件）
                if len(vis) >= 3:
                    r = stats.spearmanr([vals[i] for i in vis], [hum[i] for i in vis])[0]
                    if np.isfinite(r):
                        rho_vis.append((t, r))
                        by_method.setdefault(method, []).append(r)
                if len(qs) >= 3:
                    r = stats.spearmanr([vals[i] for i in qs], [hum[i] for i in qs])[0]
                    if np.isfinite(r):
                        rho_all.append((t, r))
                # G6b：可見條件中人類最高 vs 最低；LP 在 24 kHz 用 Q1 vs Q3、16 kHz 整組排除
                cand = vis
                if method == 'LP':
                    cand = [i for i in qs if meta[i][2] in (('Q1', 'Q3') if PATH.get(fam, 44) == 24 else ())] if PATH.get(fam, 44) < 44 else qs
                if len(cand) >= 2:
                    hi = max(cand, key=lambda i: hum[i]); lo = min(cand, key=lambda i: hum[i])
                    if hum[hi] > hum[lo]:
                        q_n.append(t); q_ok.append(float(vals[hi] > vals[lo]))
                # hidden ref 是否 trial 內最高／前 2
                refs = [i for i in ids if i.endswith('/reference')]
                if refs:
                    ref_n += 1
                    rk = sorted(ids, key=lambda i: -vals[i])
                    ref_top += int(refs[0] in rk[:2])
            r = dict(dataset='odaq', model=model, col=col, variant=variant, loud=loud, fam=fam,
                     n_trials_g6a=len(rho_vis), ref_top2=(ref_top / ref_n if ref_n else float('nan')))
            if rho_vis:
                a = np.array([x[1] for x in rho_vis])
                bs = a[rng.integers(0, len(a), size=(nboot, len(a)))].mean(1)
                r.update(g6a=float(a.mean()), g6a_ci=boot_ci(bs))
            if rho_all:
                r['within_trial_all'] = float(np.mean([x[1] for x in rho_all]))
            if q_ok:
                a = np.array(q_ok)
                bs = a[rng.integers(0, len(a), size=(nboot, len(a)))].mean(1)
                r.update(g6b=float(a.mean()), g6b_ci=boot_ci(bs), n_trials_g6b=len(a))
            r['by_method'] = {k: float(np.mean(v)) for k, v in by_method.items()}
            r['g6_pass'] = bool('g6a' in r and 'g6b' in r and r['g6a'] >= 0.5 and r['g6a_ci'][0] > 0.30
                                and r['g6b'] >= 0.80 and r['g6b_ci'][0] > 0.60)
            rows.append(r)
            # 成對（描述）：trial 內 C(5,2) 對，人類 ΔMUSHRA 的 listener-bootstrap CI 不含 0
            if (variant, loud) == PRIMARY['odaq']:
                agree, tot = 0, 0
                for t, ids in trials.items():
                    qs = [i for i in ids if meta[i][2].startswith('Q')]
                    for a_i in range(len(qs)):
                        for b_i in range(a_i + 1, len(qs)):
                            i, j = qs[a_i], qs[b_i]
                            ci_, cj_ = (meta[i][0], i.split('/')[1]), (meta[j][0], j.split('/')[1])
                            d_ = (sub.loc[ci_] - sub.loc[cj_]).dropna().values
                            if len(d_) < 10:
                                continue
                            bsd = d_[rng.integers(0, len(d_), size=(500, len(d_)))].mean(1)
                            lo_, hi_ = np.percentile(bsd, [2.5, 97.5])
                            if lo_ <= 0 <= hi_:
                                continue
                            tot += 1
                            agree += int((vals[i] - vals[j]) * d_.mean() > 0)
                if tot:
                    pair_rows.append(dict(dataset='odaq', model=model, col=col, n_pairs=tot, agree=agree / tot))
    write_rows(out_dir / 'odaq_g6.tsv', rows)
    write_rows(out_dir / 'odaq_pairs.tsv', pair_rows)
    # 方法別 Q1 vs Q5 方向（§8 規則 2）
    log(f'odaq: {len(rows)} G6 rows, {len(pair_rows)} pair rows')
    return rows


# ───────────────────────── pairwise datasets (AIME, Arena) ─────────────────────────
def aime_pairs():
    import pandas as pd
    s = pd.read_parquet(D.META / 'aime_survey/train.parquet')
    out = []
    s = s.reset_index(drop=True)
    for k in range(len(s)):
        r = s.iloc[k]
        a = f"{r['model-1']}|{int(r['track-1-id']):05d}|{r['track-1-begin']}-{r['track-1-end']}"
        b = f"{r['model-2']}|{int(r['track-2-id']):05d}|{r['track-2-begin']}-{r['track-2-end']}"
        out.append(dict(q=r['question-type'], a=a, b=b, sys_a=r['model-1'], sys_b=r['model-2'],
                        win=0 if int(r['answer']) == 1 else 1))
    return out


def arena_pairs():
    pick = json.loads(D.ARENA_PICK.read_text())
    out = []
    for b in pick['battles']:
        out.append(dict(q='pref', a=b['audio_a'], b=b['audio_b'], sys_a=b['system_a'], sys_b=b['system_b'],
                        pref=b['preference'], win={'A': 0, 'B': 1}.get(b['preference'], -1), uuid=b['battle_uuid']))
    return out


def bradley_terry(sys_list, pairs, iters=200):
    """MM 迭代；pairs: (winner_sys, loser_sys)。"""
    ix = {s: k for k, s in enumerate(sorted(sys_list))}
    n = len(ix)
    W = np.zeros((n, n))
    for w, l in pairs:
        if w in ix and l in ix and w != l:
            W[ix[w], ix[l]] += 1
    p = np.ones(n)
    N = W + W.T
    for _ in range(iters):
        for i in range(n):
            den = sum(N[i, j] / (p[i] + p[j]) for j in range(n) if N[i, j] > 0)
            if den > 0:
                p[i] = max(W[i].sum(), 0.1) / den
        p /= np.exp(np.log(p).mean())
    return {s: float(np.log(p[k])) for s, k in ix.items()}


def analyze_pairwise(ds, nboot, out_dir, log):
    sc = load_scores(ds)
    pairs = [p for p in (aime_pairs() if ds == 'aime' else arena_pairs())
             if ds == 'arena' or p['q'] == 'Music Quality']
    dec = [p for p in pairs if p['win'] in (0, 1)]
    cell = PRIMARY[ds]
    tab = metric_table(sc, *cell)
    rows, sysrows = [], []
    # 人類 per-item 目標：AIME 為該 snippet 的勝率；Arena 為勝負 0/1（單票，描述）
    wins = {}
    for p in dec:
        for it, w in ((p['a'], 1 - p['win']), (p['b'], p['win'])):
            wins.setdefault(it, []).append(w)
    hw = {i: float(np.mean(v)) for i, v in wins.items()}
    isys = {it.get('item_id'): it['group'] for it in D.read_manifest(ds)}
    rng = np.random.default_rng(SEED)
    for (model, col), vals in sorted(tab.items()):
        # 成對一致率
        S = max(vals.values()) - min(vals.values())
        m = 0.10 * S
        use = [p for p in dec if p['a'] in vals and p['b'] in vals]
        if len(use) < 50:
            continue
        d = np.array([vals[p['a']] - vals[p['b']] for p in use])
        h = np.array([1.0 if p['win'] == 0 else -1.0 for p in use])  # +1 = A 被偏好
        score_m = np.where(np.abs(d) < m, 0.5, (np.sign(d) == h).astype(float))
        score_0 = np.where(d == 0, 0.5, (np.sign(d) == h).astype(float))
        clus = {}
        for k, p in enumerate(use):
            clus.setdefault(p.get('uuid') or p['a'], []).append(k)  # AIME：以 A 側 snippet 為 cluster
        keys = list(clus)
        bs = []
        for _ in range(nboot):
            pick = rng.integers(0, len(keys), len(keys))
            ids = np.concatenate([clus[keys[q]] for q in pick])
            bs.append(score_m[ids].mean())
        r = dict(dataset=ds, model=model, col=col, n_pairs=len(use), m=float(m),
                 agree_m=float(score_m.mean()), agree_m_ci=boot_ci(np.array(bs)),
                 agree_forced=float(score_0.mean()), tie_frac=float((np.abs(d) < m).mean()))
        # G1/G2：snippet 勝率（AIME）／勝負 0/1（Arena）
        items = [i for i in vals if i in hw]
        if len(items) >= 20:
            x = np.array([vals[i] for i in items]); y = np.array([hw[i] for i in items])
            idx = boot_idx(len(items), nboot)
            st = corr_stats(x, y, idx, kendall=True)
            if 'spearman' in st:
                r.update(pearson=st['pearson'], spearman=st['spearman'], spearman_ci=st['spearman_ci'],
                         kendall=st['kendall'], n_items=len(items))
                sysn = [isys.get(i, '') for i in items]
                sw = corr_stats(center_by(x, sysn), center_by(y, sysn), idx, kendall=False)
                if 'spearman' in sw:
                    r.update(within_sys=sw['spearman'], within_sys_ci=sw['spearman_ci'])
        rows.append(r)
        # 系統排名：BT（人類）vs metric 系統平均
        sysm = {}
        for i, v in vals.items():
            sysm.setdefault(isys.get(i, ''), []).append(v)
        sysm = {k: float(np.mean(v)) for k, v in sysm.items() if k}
        bt = bradley_terry(list(sysm), [((p['sys_a'], p['sys_b']) if p['win'] == 0 else (p['sys_b'], p['sys_a']))
                                         for p in dec])
        common = [s for s in sysm if s in bt]
        if len(common) >= 5:
            tau = stats.kendalltau([sysm[s] for s in common], [bt[s] for s in common])[0]
            sysrows.append(dict(dataset=ds, model=model, col=col, n_sys=len(common), kendall_tau=float(tau)))
        if ds == 'arena':
            tie = [p for p in pairs if p['pref'] == 'TIE' and p['a'] in vals and p['b'] in vals]
            bad = [p for p in pairs if p['pref'] == 'BOTH_BAD' and p['a'] in vals and p['b'] in vals]
            r['n_tie'] = len(tie); r['n_both_bad'] = len(bad)
            if tie:
                r['tie_small_diff_frac'] = float(np.mean([abs(vals[p['a']] - vals[p['b']]) < m for p in tie]))
    write_rows(out_dir / f'{ds}_pairwise.tsv', rows)
    write_rows(out_dir / f'{ds}_system_rank.tsv', sysrows)
    # 段長／響度（Arena norm800 子集）
    log(f'{ds}: {len(rows)} pairwise rows')
    return rows


# ───────────────────────── gates / decision ─────────────────────────
def decide(results):
    """§7.3：candidate / excluded / undetermined。results: 各 ds 的 rows。"""
    cands = {}
    for ds, rows in results.items():
        for r in rows:
            fam, col = r['model'], r['col']
            if CAND_COL.get(fam) != col or ds not in INDEP.get(fam, set()):
                continue
            if (r.get('variant'), r.get('loud')) not in (PRIMARY[ds], (None, None)):
                continue
            if r.get('target') not in (None, {'singmos': 'mos', 'mc522': 'human_PQ'}.get(ds)):
                continue
            c = cands.setdefault(fam, {}).setdefault(ds, {})
            if 'spearman' in r:
                c['G1'] = bool(r['spearman'] >= 0.5 and r['spearman_ci'][0] > 0.30)
                c['G1_upper_lt'] = bool(r['spearman_ci'][1] < 0.5)
            if 'within_sys' in r and ds in ('aime', 'singmos', 'arena'):
                c['G2'] = bool(r['within_sys'] >= 0.5 and r['within_sys_ci'][0] > 0.30)
            if 'agree_m' in r and ds in ('aime', 'arena'):
                c['G3'] = bool(r['agree_m_ci'][0] > 0.55)
                c['G3_upper_lt'] = bool(r['agree_m_ci'][1] < 0.55)
            if 'g6_pass' in r:
                c['G6'] = bool(r['g6_pass'])
    out = {}
    for fam, per in cands.items():
        ok_ds = [ds for ds, c in per.items() if all(v for k, v in c.items() if k in ('G1', 'G2', 'G3'))
                 and any(k in c for k in ('G1', 'G2', 'G3'))]
        gen = [d for d in ok_ds if d in GENERATIVE]
        odaq_ok = per.get('odaq', {}).get('G6', None)
        excluded = any(ds in GENERATIVE and (c.get('G1_upper_lt') or c.get('G3_upper_lt')) for ds, c in per.items())
        if excluded:
            lab = 'excluded'
        elif len(ok_ds) >= 2 and gen and odaq_ok is not False:
            lab = 'round-2 candidate'
        else:
            lab = 'undetermined'
        out[fam] = dict(label=lab, passing_datasets=ok_ds, per_dataset=per)
    return out


def write_rows(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(keys)
        for r in rows:
            w.writerow([fmt(r.get(k, '')) if not isinstance(r.get(k), (dict,)) else json.dumps(r[k]) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='odaq,mc522,singmos,songeval,aime,arena')
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--out', default=str(ROOT / 'analysis'))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    results, deltas = {}, {}
    for ds in a.datasets.split(','):
        if not (ROOT / 'scores' / ds).exists():
            log(f'{ds}: no scores, skipped')
            continue
        if ds == 'odaq':
            results[ds] = analyze_odaq(a.nboot, out, log)
        elif ds in ('aime', 'arena'):
            results[ds] = analyze_pairwise(ds, a.nboot, out, log)
        else:
            rows, sc, H = analyze_item_level(ds, a.nboot, out, log)
            results[ds] = rows
            tgt = {'singmos': 'mos', 'mc522': 'human_PQ', 'songeval': 'Musicality'}[ds]
            deltas[ds] = paired_delta(ds, a.nboot, sc, H, tgt, out, 'length', log) + \
                paired_delta(ds, a.nboot, sc, H, tgt, out, 'loudness', log)
    dec = decide({k: v for k, v in results.items() if k != 'songeval'})
    (out / 'summary.json').write_text(json.dumps({'decision': dec, 'seed': SEED, 'nboot': a.nboot}, indent=1, default=str))
    (out / 'REPORT.md').write_text('# round-2 analysis log\n\n' + '\n'.join(f'- {l}' for l in lines) +
                                   '\n\n## 判定（§7.3）\n\n' + '\n'.join(f'- {k}: {v["label"]}（通過：{v["passing_datasets"]}）'
                                                                    for k, v in dec.items()) + '\n')
    log('decision: ' + json.dumps({k: v['label'] for k, v in dec.items()}))


if __name__ == '__main__':
    main()
