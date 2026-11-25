#!/usr/bin/env python
"""178 judgement heads + the preregistered test evaluation (prereg §2-§4).

  python scripts/eval/wrongnote_probe_178_analyze.py --root <dir>

Protocol (frozen by docs/experiments/wrongnote_probe_mert_178_20261103.md):
  * heads are fitted on train (generaluser only), C and the MERT layer are chosen on val,
  * test is evaluated once: T1 per-intensity AUC (cluster bootstrap 2,000), T2 pairwise
    original-vs-wronged, T3 false alarms on changes that are not wrong notes,
  * controls: random-weight MERT, AES 4 axes, chroma out-of-key ratio, madmom key confidence,
    and the 2-d rule pair; the symbolic out-of-scale ratio is reported as an oracle ceiling.

Writes <root>/summary.json, <root>/results.md and copies of both into
docs/experiments/results/.
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrongnote_probe_178_common as C  # noqa: E402

CS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
BOOT = 2000
T3_THRESHOLD = 0.5
PRIMARY = 'wrong_10_in+out'
CONTROLS = ('mert_random', 'aes', 'rule_chroma', 'rule_key', 'rule_2d')
DEVIATIONS = [
    'MAESTRO 官方 test 只有 177 場演奏、validation 只有 137 場，湊不到 250／150 段且每段都來自不同演奏；'
    '先用每個演奏一段，不足的部分（test 73、val 13）用同一批演奏裡第二個互不重疊的 10 s 窗口補，'
    '並把 bootstrap 的抽樣單位改成「演奏」而不是「段」。manifest 的 window 欄可查。',
    'T3 的誤報門檻固定在 p>=0.5（prereg 沒指定門檻）。',
    '「最強的非 oracle 對照組」取 test 上同一組比較（wrong_10 合併）中 AUC 最高的對照組，'
    '用 test 最大值因此偏嚴格（只會讓我們更難過關）。',
    '符號 oracle 的調外音比例用該段 clean 版本估出的調性計算，不隨版本重新估調；'
    '所以 transpose 版本的 oracle 讀數會變成 ~0.6（整段被移到調外），這是字面定義的結果。',
    'madmom 只算 key_cnn_conf（不跑拍點），其餘約定與 mir_metrics.py 相同。',
    'MLP 次要頭用線性頭選出的同一層特徵，early stopping 以 val log-loss 選 epoch。',
]


# ── data ──────────────────────────────────────────────────────────────────────
def load(root):
    rows = C.read_tsv(Path(root) / 'manifest.tsv')
    n = len(rows)
    feat = {}
    fd = Path(root) / 'features'
    for name in ('mert', 'mert_random', 'aes'):
        p = fd / name / f'{name}.npy'
        if not p.is_file():
            raise SystemExit(f'[FAIL] missing feature array {p}')
        feat[name] = np.load(p)
        if feat[name].shape[0] != n:
            raise SystemExit(f'[FAIL] {name}: {feat[name].shape[0]} rows vs manifest {n}')
    rules = {}
    for mode in ('chroma', 'madmom'):
        p = fd / f'rule_{mode}.tsv'
        if not p.is_file():
            raise SystemExit(f'[FAIL] missing rule table {p}')
        vals = {}
        with open(p, encoding='utf-8') as f:
            for ln in f:
                if ln.strip():
                    k, v = ln.rstrip('\n').split('\t')
                    vals[k] = float(v)
        rules[mode] = np.asarray([vals.get(r['path'], np.nan) for r in rows], dtype=np.float64)
    feat['rule_chroma'] = rules['chroma'][:, None]
    feat['rule_key'] = rules['madmom'][:, None]
    feat['rule_2d'] = np.stack([rules['chroma'], rules['madmom']], axis=1)
    feat['oracle'] = np.asarray([float(r['oracle_out_ratio']) for r in rows], dtype=np.float64)[:, None]
    for k, v in feat.items():
        if not np.all(np.isfinite(v)):
            raise SystemExit(f'[FAIL] {k}: non-finite values')
    y = np.asarray([1 if r['label'] == 'wrong' else 0 for r in rows], dtype=np.int64)
    variant = np.asarray([r['variant'] for r in rows])
    split = np.asarray([r['split'] for r in rows])
    font = np.asarray([r['font'] for r in rows])
    perf = np.asarray([r['perf'] for r in rows])
    tr = (split == 'train') & (font == 'generaluser')
    va = (split == 'val') & (font == 'generaluser')
    te = (split == 'test') & (font == 'generaluser')
    tf = (split == 'test') & (font == 'fluidr3')
    return dict(rows=rows, n=n, feat=feat, y=y, variant=variant, split=split, font=font, perf=perf,
                tr=tr, va=va, te=te, tf=tf)


def stack(X, mask):
    return np.asarray(X[mask], dtype=np.float64)


# ── heads ─────────────────────────────────────────────────────────────────────
def fit_logit(Xtr, ytr, Xva, yva, Cval):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=Cval, penalty='l2',
                                                              max_iter=3000, solver='lbfgs'))
    pipe.fit(Xtr, ytr)
    return pipe, float(roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1]))


def train_source(Xtr, ytr, Xva, yva, layers=False):
    """Pick C (and the layer) on val; returns the best estimator and its setting."""
    best = None
    grid = range(Xtr.shape[1]) if layers else (None,)
    for li in grid:
        a, b = (Xtr[:, li, :], Xva[:, li, :]) if layers else (Xtr, Xva)
        for Cval in CS:
            model, auc = fit_logit(a, ytr, b, yva, Cval)
            if best is None or auc > best['val_auc']:
                best = {'model': model, 'layer': li, 'C': Cval, 'val_auc': auc}
    return best


def train_mlp(Xtr, ytr, Xva, yva):
    sc = StandardScaler().fit(Xtr)
    a, b = sc.transform(Xtr), sc.transform(Xva)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-4, batch_size=64, max_iter=1,
                        random_state=0, learning_rate_init=1e-3)
    best = (np.inf, None)
    for _ep in range(120):
        mlp.partial_fit(a, ytr, classes=np.asarray([0, 1]))
        p = mlp.predict_proba(b)[:, 1]
        loss = float(-np.mean(yva * np.log(p + 1e-9) + (1 - yva) * np.log(1 - p + 1e-9)))
        if loss < best[0] - 1e-5:
            best = (loss, copy.deepcopy(mlp))
    model = make_pipeline(sc, best[1])
    return model, best[0]


def proba(model_fit, X):
    return model_fit['model'].predict_proba(X)[:, 1]


# ── bootstrap ─────────────────────────────────────────────────────────────────
def bootstrap_auc(pos_idx, neg_idx, gid, scores, B=BOOT, seed=0):
    """Cluster (performance) bootstrap; returns (B, n_scores) AUCs. Clusters present in
    the comparison are remapped so a test-only contrast never resamples train performances."""
    rng = np.random.default_rng(seed)
    used = np.unique(np.r_[gid[pos_idx], gid[neg_idx]])
    gid = np.searchsorted(used, gid)
    p_gid, n_gid = gid[pos_idx], gid[neg_idx]
    nperf = len(used)
    pos_by = [pos_idx[p_gid == g] for g in range(nperf)]
    neg_by = [neg_idx[n_gid == g] for g in range(nperf)]
    out = np.full((B, len(scores)), np.nan)
    for b in range(B):
        pick = rng.integers(0, nperf, nperf)
        ps = np.concatenate([pos_by[i] for i in pick])
        ns = np.concatenate([neg_by[i] for i in pick])
        for j, s in enumerate(scores):
            v = np.concatenate([s[ps], s[ns]])
            r = rankdata(v)
            m, k = len(ps), len(ns)
            out[b, j] = (r[:m].sum() - m * (m + 1) / 2.0) / (m * k)
    return out


def ci(v, lo=2.5, hi=97.5):
    return [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))]


# ── T1/T2/T3 ──────────────────────────────────────────────────────────────────
def t1_table(scores, data, mask, gid, variants, boot=True):
    """Per-variant AUC (positives = that variant, negatives = the 4 ok variants) + cluster CI."""
    out = {}
    y = data['y']
    ok = mask & (y == 0)
    for v in variants:
        pos = np.where(mask & (data['variant'] == v))[0]
        neg = np.where(ok)[0]
        s = scores[pos], scores[neg]
        auc = float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[s[0], s[1]]))
        c = ci(bootstrap_auc(pos, neg, gid, [scores], seed=C.h('boot', v))[:, 0]) if boot else None
        out[v] = {'auc': auc, 'ci': c, 'n_pos': int(len(pos)), 'n_neg': int(len(neg))}
    return out


def pooled_variant(data, mask, v):
    if v == PRIMARY:
        return np.where(mask & np.isin(data['variant'], ['wrong_10_in', 'wrong_10_out']))[0]
    return np.where(mask & (data['variant'] == v))[0]


def t2_table(scores, data, mask):
    """Pair (clean, wrong) of the same excerpt+font: lower wrong probability = the original."""
    out = {}
    idx = {}
    for r_i in np.where(mask)[0]:
        row = data['rows'][r_i]
        idx[(row['excerpt'], row['font'], row['variant'])] = r_i
    for v in C.WRONG_VARIANTS:
        hits, tot = 0, 0
        for (ex, font, var), i in idx.items():
            if var != v or (ex, font, 'clean') not in idx:
                continue
            j = idx[(ex, font, 'clean')]
            hits += int(scores[j] < scores[i])
            tot += 1
        out[v] = {'acc': hits / tot if tot else float('nan'), 'n': tot}
    return out


def t3_table(scores, data, mask):
    out = {}
    for v in C.OK_VARIANTS:
        rows = np.where(mask & (data['variant'] == v))[0]
        out[v] = {'fpr': float(np.mean(scores[rows] >= T3_THRESHOLD)), 'n': int(len(rows))}
    out['threshold'] = T3_THRESHOLD
    return out


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(C.ROOT))
    ap.add_argument('--boot', type=int, default=BOOT)
    ap.add_argument('--no-mlp', action='store_true')
    a = ap.parse_args()
    root = Path(a.root)
    t0 = time.time()
    d = load(root)
    print(f'[analyze] n={d["n"]} train={d["tr"].sum()} val={d["va"].sum()} '
          f'test={d["te"].sum()} test_fluidr3={d["tf"].sum()}', flush=True)

    perf_ids = np.asarray([C.h('perfid', p) for p in d['perf']])
    _, perf_index = np.unique(perf_ids, return_inverse=True)
    gid = perf_index.astype(np.int64)

    ytr, yva = d['y'][d['tr']], d['y'][d['va']]
    fits = {}
    t_fit = time.time()
    for name, layers in (('mert', True), ('mert_random', True)):
        X = d['feat'][name]
        fits[name] = train_source(stack(X, d['tr']), ytr, stack(X, d['va']), yva, layers=True)
        print(f'[analyze] {name}: layer={fits[name]["layer"]} C={fits[name]["C"]} '
              f'val_auc={fits[name]["val_auc"]:.3f} ({time.time() - t_fit:.0f}s)', flush=True)
    for name in ('aes', 'rule_chroma', 'rule_key', 'rule_2d'):
        X = d['feat'][name]
        fits[name] = train_source(stack(X, d['tr']), ytr, stack(X, d['va']), yva, layers=False)
        print(f'[analyze] {name}: C={fits[name]["C"]} val_auc={fits[name]["val_auc"]:.3f}', flush=True)

    def scores_of(name, mask):
        X = d['feat'][name]
        if fits[name]['layer'] is not None:
            X = X[:, fits[name]['layer'], :]
        return fits[name]['model'].predict_proba(np.asarray(X[mask], dtype=np.float64))[:, 1]

    te, tf_mask = d['te'], d['tf']
    # global score vectors (only filled where needed) so the T2/T3 loops can index any row
    S = {}
    for name in fits:
        for tag, m in (('te', te), ('tf', tf_mask)):
            s = np.full(d['n'], np.nan)
            s[m] = scores_of(name, m)
            S[(name, tag)] = s
    s_oracle = d['feat']['oracle'][:, 0]

    # secondary MLP on the MERT layer chosen by the linear head
    mlp = None
    if not a.no_mlp:
        li = fits['mert']['layer']
        Xm = d['feat']['mert'][:, li, :]
        model, vloss = train_mlp(stack(Xm, d['tr']), ytr, stack(Xm, d['va']), yva)
        mlp = model
        for tag, m in (('te', te), ('tf', tf_mask)):
            s = np.full(d['n'], np.nan)
            s[m] = model.predict_proba(np.asarray(Xm[m], dtype=np.float64))[:, 1]
            S[('mlp', tag)] = s
        print(f'[analyze] mlp: layer={li} val_logloss={vloss:.3f}', flush=True)

    # ── test evaluation (once) ──
    pos10 = pooled_variant(d, te, PRIMARY)
    neg10 = np.where(te & (d['y'] == 0))[0]
    scores_list = [S[('mert', 'te')], S[('mert_random', 'te')], S[('aes', 'te')],
                   S[('rule_chroma', 'te')], S[('rule_key', 'te')], S[('rule_2d', 'te')]]
    if mlp is not None:
        scores_list.append(S[('mlp', 'te')])
    bs = bootstrap_auc(pos10, neg10, gid, scores_list, B=a.boot, seed=C.h('boot', PRIMARY))
    names = ['mert', 'mert_random', 'aes', 'rule_chroma', 'rule_key', 'rule_2d'] + \
            (['mlp'] if mlp is not None else [])
    primary = {}
    for j, name in enumerate(names):
        s = scores_list[j]
        auc = float(roc_auc_score(np.r_[np.ones(len(pos10)), np.zeros(len(neg10))],
                                  np.r_[s[pos10], s[neg10]]))
        primary[name] = {'auc': auc, 'ci': ci(bs[:, j])}
    oracle_auc = float(roc_auc_score(np.r_[np.ones(len(pos10)), np.zeros(len(neg10))],
                                     np.r_[s_oracle[pos10], s_oracle[neg10]]))

    ctl_names = [n for n in CONTROLS if n in primary]
    best_ctl = max(ctl_names, key=lambda n: primary[n]['auc'])
    delta = primary['mert']['auc'] - primary[best_ctl]['auc']
    j_mert, j_ctl = names.index('mert'), names.index(best_ctl)
    dci = ci(bs[:, j_mert] - bs[:, j_ctl])
    crit = {
        'primary_variant': PRIMARY,
        'auc': primary['mert']['auc'], 'auc_ci': primary['mert']['ci'],
        'pass_auc_080': bool(primary['mert']['auc'] >= 0.80),
        'best_control': best_ctl, 'best_control_auc': primary[best_ctl]['auc'],
        'delta': float(delta), 'delta_ci': dci,
        'pass_margin': bool(delta >= 0.05 and dci[0] > 0),
        'pass': bool(primary['mert']['auc'] >= 0.80 and delta >= 0.05 and dci[0] > 0),
    }

    t1 = t1_table(S[('mert', 'te')], d, te, gid, C.WRONG_VARIANTS)
    t1_f = t1_table(S[('mert', 'tf')], d, tf_mask, gid, C.WRONG_VARIANTS)
    t2 = t2_table(S[('mert', 'te')], d, te)
    t3 = t3_table(S[('mert', 'te')], d, te)
    t3_fluid = t3_table(S[('mert', 'tf')], d, tf_mask)
    # controls on the same per-variant table (descriptive)
    ctl_variants = ['wrong_1_in', 'wrong_1_out', 'wrong_5_in', 'wrong_5_out', 'wrong_10_in',
                    'wrong_10_out', 'wrong_30_in', 'wrong_30_out']
    ctl_t1 = {n: t1_table(S[(n, 'te')], d, te, gid, ctl_variants, boot=False) for n in ctl_names}
    oracle_t1 = t1_table(s_oracle, d, te, gid, C.WRONG_VARIANTS)

    verdict = []
    if crit['pass']:
        verdict.append('MERT 在 MIDI 鋼琴分布內判得出錯音（主判準兩條都過）')
    else:
        if not crit['pass_auc_080']:
            verdict.append('MERT 沒有達到 AUC 0.80，分布內判不出錯音')
        else:
            verdict.append('MERT 判得出「有改動」，但沒有明顯超過對照組（判得出改動，判不出錯）')
        t3_ok = (t3['transpose']['fpr'] <= t3['clean']['fpr'] + 0.10 and
                 t3['tempo']['fpr'] <= t3['clean']['fpr'] + 0.10)
        if not t3_ok:
            verdict.append('且 T3 顯示 transpose/tempo 被誤判成錯音 → 判得出改動，判不出錯')
    if t1['wrong_10_out']['auc'] >= 0.75 and t1['wrong_10_in']['auc'] < 0.60:
        verdict.append('只有調外錯音（out）明顯高於 0.5，in 貼近 0.5 → 學到調性，不是錯音')
    if abs(t1['wrong_10_in']['auc'] - t1['wrong_10_out']['auc']) > 0.15 and not crit['pass']:
        verdict.append('in/out 落差大')

    t3_ok = (t3['transpose']['fpr'] <= t3['clean']['fpr'] + 0.10 and
             t3['tempo']['fpr'] <= t3['clean']['fpr'] + 0.10)
    summary = {
        'experiment': 'wrongnote_probe_mert_178',
        'prereg': str(REPO_PREREG),
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'stimuli': C.read_json(root / 'stimuli.json'),
        'feature_speed': {p.stem: C.read_json(p) for p in sorted((root / 'features').glob('rule_*.json'))},
        'selection': {k: {'layer': v['layer'], 'C': v['C'], 'val_auc': v['val_auc']}
                      for k, v in fits.items()},
        'mlp': None if mlp is None else {'layer': fits['mert']['layer']},
        'test_primary': primary,
        'criteria': crit,
        't1_generaluser': t1,
        't1_fluidr3': t1_f,
        't2_generaluser': t2,
        't3_generaluser': t3,
        't3_fluidr3': t3_fluid,
        'oracle_generaluser': oracle_t1,
        'oracle_primary_auc': oracle_auc,
        'controls_t1': ctl_t1,
        't3_condition': {
            'ok': bool(t3_ok),
            'transpose_excess': t3['transpose']['fpr'] - t3['clean']['fpr'],
            'tempo_excess': t3['tempo']['fpr'] - t3['clean']['fpr'],
        },
        'verdict': verdict,
        'deviations': DEVIATIONS,
    }
    C.write_json(root / 'summary.json', summary)
    md = render_md(summary)
    C.atomic_text(root / 'results.md', md)
    if root.resolve() == C.ROOT.resolve():   # smoke roots stay out of the repo docs
        C.RESULTS.mkdir(parents=True, exist_ok=True)
        C.write_json(C.RESULTS / 'wrongnote_probe_mert_178_summary.json', summary)
        C.atomic_text(C.RESULTS / 'wrongnote_probe_mert_178_results.md', md)
    print('[analyze] verdict: ' + ' | '.join(verdict), flush=True)
    print(f'[analyze] done in {time.time() - t0:.1f} s', flush=True)


REPO_PREREG = C.REPO / 'docs/experiments/wrongnote_probe_mert_178_20261103.md'


def _f(x, n=3):
    return 'nan' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{n}f}'


def render_md(s):
    c = s['criteria']
    p = s['test_primary']
    t1, t1f, t2, t3 = s['t1_generaluser'], s['t1_fluidr3'], s['t2_generaluser'], s['t3_generaluser']
    sel = s['selection']
    lines = []
    lines.append('# 總評')
    lines.append('')
    lines.append(f'**判準：{"通過" if c["pass"] else "不通過"}**  '
                 f'test generaluser wrong_10（in＋out 合併）AUC {_f(c["auc"])} '
                 f'（95% CI {_f(c["auc_ci"][0])}–{_f(c["auc_ci"][1])}）；'
                 f'最強非 oracle 對照組 {c["best_control"]} {_f(c["best_control_auc"])}，'
                 f'差值 {_f(c["delta"])}（CI {_f(c["delta_ci"][0])}–{_f(c["delta_ci"][1])}）。')
    lines.append('')
    lines.append('**核心結論**：' + '；'.join(s['verdict']) + '。')
    lines.append('')
    lines.append('**關鍵數據與例子**')
    lines.append('')
    lines.append('| 版本 | MERT AUC（generaluser） | 最強對照組 | 對照組 AUC | oracle |')
    lines.append('|---|---|---|---|---|')
    for v in ('wrong_1_in', 'wrong_1_out', 'wrong_5_in', 'wrong_5_out', 'wrong_10_in',
              'wrong_10_out', 'wrong_30_in', 'wrong_30_out'):
        ctl = max(CONTROLS, key=lambda n: s['controls_t1'][n][v]['auc'])
        lines.append(f'| {v} | {_f(t1[v]["auc"])} | {ctl} | {_f(s["controls_t1"][ctl][v]["auc"])} | '
                     f'{_f(s["oracle_generaluser"][v]["auc"])} |')
    lines.append(f'| {PRIMARY}（主判準） | {_f(c["auc"])} | {c["best_control"]} | '
                 f'{_f(c["best_control_auc"])} | {_f(s["oracle_primary_auc"])} |')
    lines.append('')
    lines.append(f'- 未見音色 fluidr3（wrong_10）：in {_f(t1f["wrong_10_in"]["auc"])}、'
                 f'out {_f(t1f["wrong_10_out"]["auc"])}。')
    lines.append(f'- T2 兩段挑原版：wrong_10 合併 '
                 f'{_f(np.mean([t2["wrong_10_in"]["acc"], t2["wrong_10_out"]["acc"]]))}，'
                 f'wrong_30 {_f(np.mean([t2["wrong_30_in"]["acc"], t2["wrong_30_out"]["acc"]]))}，'
                 f'wrong_1 {_f(np.mean([t2["wrong_1_in"]["acc"], t2["wrong_1_out"]["acc"]]))}。')
    lines.append(f'- T3 誤報（p>=0.5）：clean {_f(t3["clean"]["fpr"])}、transpose '
                 f'{_f(t3["transpose"]["fpr"])}、tempo {_f(t3["tempo"]["fpr"])}、gain −6 dB '
                 f'{_f(t3["gain_m6"]["fpr"])} → 附帶條件{"過" if s["t3_condition"]["ok"] else "不過"}。')
    lines.append(f'- 選模（全部由 val 決定）：MERT 第 {sel["mert"]["layer"]} 層／C={sel["mert"]["C"]}'
                 f'（val AUC {_f(sel["mert"]["val_auc"])}）；'
                 f'隨機權重 MERT 第 {sel["mert_random"]["layer"]} 層／C={sel["mert_random"]["C"]}'
                 f'（{_f(sel["mert_random"]["val_auc"])}）；AES {_f(sel["aes"]["val_auc"])}；'
                 f'chroma 規則 {_f(sel["rule_chroma"]["val_auc"])}；madmom key {_f(sel["rule_key"]["val_auc"])}。')
    lines.append(f'- 其他對照組（wrong_10 合併 AUC）：' +
                 '、'.join(f'{k} {_f(p[k]["auc"])}' for k in CONTROLS if k in p) +
                 (f'、MLP {_f(p["mlp"]["auc"])}' if 'mlp' in p else '') +
                 f'、隨機權重 MERT {_f(p["mert_random"]["auc"])}。')
    lines.append('')
    lines.append('**要你決定的事**')
    lines.append('')
    if c['pass']:
        lines.append('- 主判準通過；建議 **approve**，下一步照 prereg §4 的界線：只能寫「MIDI 鋼琴分布內」，'
                     'OOD（AI 生成音樂）要另立設計（真值難取得，只能用人類標註或可控生成）。')
        lines.append('- 選項 B：先在 fluidr3 未見音色與 T3 上檢查泛化，再決定要不要擴到更多 SoundFont／真實錄音。')
    else:
        lines.append('- 主判準不通過；建議 **不採用 MERT 當錯音守門員**，這個結果本身可寫進指標分析'
                     '（呼應 126d／133：神經指標對音符層錯誤無感）。')
        lines.append('- 選項 B：把頭換成「有原版的差值特徵」（133 的對稱距離）或直接餵 chroma，'
                     '成本低但只能回答「改動」不是「錯」。')
        lines.append('- 選項 C：收到 126 的證據表，不再投入 OOD 錯音偵測。')
    lines.append('')
    lines.append('## 設定與偏離')
    lines.append('')
    st = s.get('stimuli') or {}
    lines.append(f'- 刺激：{st.get("excerpts", "?")} 段 × 版本，共 {st.get("clips", "?")} 段音檔；'
                 f'val／test 額外窗口 {st.get("extra_windows", "?")} 段。')
    for dv in s['deviations']:
        lines.append(f'- 偏離：{dv}')
    lines.append('- 不可寫：過了也只能寫「MIDI 鋼琴分布內」；不可寫成能判斷 AI 生成音樂的錯音。')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
