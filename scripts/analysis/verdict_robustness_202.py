#!/usr/bin/env python3
"""202 判定穩健性（零 GPU）。設計：docs/experiments/paper_closing_evidence_200_202_20261125.md §3。

只重算 197 表 1／表 2 A 區既有逐段分數；讀檔、bootstrap、verdict 全部 import 自
scripts/analysis/attack_metric_matrix_tables.py（不修改）。
  a. m × {0.5,1,2} 重判
  b. 逐 id Δ（6 底等權平均）Wilcoxon signed-rank + Holm（表 1 全部格 × 4 指標一起校正）
  c. 逐底一致數 k/6
  d. 方向錯的格 |Δ| 對 seed 底線（3 個 MeanAudio cfg3neg 乾淨均值兩兩最大差）的倍數
  e. 9 底乾淨均值排名，套用該列處理到單一底後越過幾個底
  python scripts/analysis/verdict_robustness_202.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attack_metric_matrix_tables as T  # noqa: E402

ROOT = T.EV / 'attack_metric_matrix_197'
RES = T.RES
FACT = [0.5, 1.0, 2.0]
MS = {m[0]: m[2] for m in T.METRICS}
SEED_BASES = ['ma_cfg3neg_s14159265', 'ma_cfg3neg_s16180339', 'ma_cfg3neg_s27182818']
CFG3 = SEED_BASES


def coarse(v, exp):
    """把 verdict 歸成粗類：方向錯／抓得到(或變差)／其他。"""
    if v == '**方向錯**':
        return '方向錯'
    if exp == 'W':
        return {'抓得到': '變差', '沒反應': '沒反應'}.get(v, '其他')
    if v in ('過敏', '可接受(變差)'):
        return '變差'
    return '其他'


def wil_class(mean, p_adj, exp):
    """b：不用 m；方向 + Holm 後 p<0.05。"""
    if exp == 'N':
        return '—'
    sig = p_adj < 0.05
    if sig and mean > 0:
        return '方向錯'
    if sig and mean < 0:
        return '變差'
    return '沒反應' if exp == 'W' else '其他'


def holm(ps):
    ps = np.asarray(ps, float)
    n = len(ps)
    o = np.argsort(ps)
    adj = np.empty(n)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (n - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def wil_p(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or np.all(x == 0):
        return 1.0
    return float(wilcoxon(x).pvalue)


def clean_means(root):
    """9 底乾淨（base）均值，每指標；對應 197 的 T1_<base>_base.tsv。"""
    out = {}
    for b in T.BASES:
        r = T.read_tsv(T.per_clip_path('197', b, 'base', root))
        out[b] = {mk: float(np.nanmean([v[col] for v in r.values()])) for mk, col, _ in T.METRICS}
    return out


def rank_cross(clean, mk, deltas):
    """clean: {base: mean}；deltas: {base: Δ mean}。對每個底單獨套 Δ，數有幾個別的底被越過。"""
    res = {}
    for b in T.BASES:
        if b not in deltas:
            continue
        new = clean[b] + deltas[b]
        crossed = 0
        for o in T.BASES:
            if o == b:
                continue
            if (clean[b] - clean[o]) * (new - clean[o]) < 0:
                crossed += 1
        res[b] = crossed
    return res


def main():
    summary = {'seed': T.SEED, 'nboot': T.NB, 'factors': FACT, 'm': MS}
    # ---------------- 表 1：建立每格資料 ----------------
    cells = {}  # (key, mk) -> dict
    selfcheck = json.loads((RES / 'attack_metric_matrix_197_summary.json').read_text())
    maxdiff = 0.0
    maxdiff_ci = 0.0
    ncheck = 0
    for key, label, exp, src, tok, reft in T.T1_ROWS:
        for mk, col, m in T.METRICS:
            per, ds = {}, {}
            miss = False
            for b in T.BASES:
                ids, d = T.t1_cell(ROOT, src, tok, reft, b, col)
                if ids is None:
                    if b in T.LISTEN:
                        miss = True
                    continue
                per[b] = T.boot_mean(d)
                ds[b] = dict(zip(ids, d))
            if miss:
                continue
            common = sorted(set.intersection(*[set(ds[b]) for b in T.LISTEN]))
            Mx = np.array([[ds[b][i] for i in common] for b in T.LISTEN])
            avg = np.nanmean(Mx, axis=0)
            s = T.boot_mean(avg)
            ref = selfcheck['t1'][key][mk]['main']
            maxdiff = max(maxdiff, abs(s['mean'] - ref['mean']))
            maxdiff_ci = max(maxdiff_ci, abs(s['lo'] - ref['lo']), abs(s['hi'] - ref['hi']))
            ncheck += 1
            cells[(key, mk)] = dict(label=label, exp=exp, src=src, m=m, s=s, per=per, avg=avg,
                                    perbase_mean={b: per[b]['mean'] for b in per})
    summary['selfcheck'] = {'cells': ncheck, 'max_abs_mean_diff': maxdiff, 'max_abs_ci_diff': maxdiff_ci,
                            'pass': bool(maxdiff < 1e-6)}
    print('selfcheck T1', summary['selfcheck'])
    if maxdiff >= 1e-6:
        print('SELF-CHECK FAILED')
        (RES / 'verdict_robustness_202_summary.json').write_text(json.dumps(summary, indent=1, default=float))
        sys.exit(2)

    # Wilcoxon + Holm（表 1 全部格 × 4 指標）
    keys = list(cells)
    ps = [wil_p(cells[k]['avg']) for k in keys]
    padj = holm(ps)
    for k, p, pa in zip(keys, ps, padj):
        cells[k]['p'] = p
        cells[k]['p_holm'] = float(pa)
    clean = clean_means(ROOT)
    cl_floor = {}
    for mk, _, _ in T.METRICS:
        v = [clean[b][mk] for b in SEED_BASES]
        cl_floor[mk] = max(v) - min(v)
    summary['seed_floor'] = cl_floor
    summary['clean_means'] = clean

    # 逐格判定
    t1 = {}
    unstable = []
    R = {f: {'R1': {m[0]: 0 for m in T.METRICS}, 'R2': {m[0]: 0 for m in T.METRICS}} for f in FACT}
    Rb = {'R1': {m[0]: 0 for m in T.METRICS}, 'R2': {m[0]: 0 for m in T.METRICS}}
    for (key, mk), c in cells.items():
        exp, m, s = c['exp'], c['m'], c['s']
        row = {'label': c['label'], 'expected': exp, 'mean': s['mean'], 'lo': s['lo'], 'hi': s['hi'],
               'dz': s['mean'] / s['sd'] if s['sd'] else float('nan'), 'up_frac': s['up'],
               'p': c['p'], 'p_holm': c['p_holm']}
        vs = {}
        for f in FACT:
            v = T.verdict(s, m * f, exp)
            vs[f] = v
            if exp in ('U', 'UW') and v == '**方向錯**':
                R[f]['R1'][mk] += 1
            if exp == 'W' and v == '沒反應':
                R[f]['R2'][mk] += 1
        row['verdict_by_m'] = {str(f): vs[f] for f in FACT}
        cs = {f: coarse(vs[f], exp) for f in FACT}
        wb = wil_class(s['mean'], c['p_holm'], exp)
        row['wilcoxon_class'] = wb
        if exp in ('U', 'UW') and wb == '方向錯':
            Rb['R1'][mk] += 1
        if exp == 'W' and wb == '沒反應':
            Rb['R2'][mk] += 1
        # c. 逐底一致
        kk = sum(1 for b in T.LISTEN if np.sign(c['per'][b]['mean']) == np.sign(s['mean'])
                 and (c['per'][b]['lo'] > 0 or c['per'][b]['hi'] < 0))
        kk_big = sum(1 for b in T.LISTEN if np.sign(c['per'][b]['mean']) == np.sign(s['mean'])
                     and (c['per'][b]['lo'] > m or c['per'][b]['hi'] < -m))
        row['k_same_sign_ci'] = kk
        row['k_same_sign_big'] = kk_big
        # d. seed 底線
        row['abs_over_seed_floor'] = abs(s['mean']) / cl_floor[mk] if cl_floor[mk] > 0 else float('nan')
        # e. 排名
        row['rank_crossed'] = rank_cross({b: clean[b][mk] for b in clean}, mk, c['perbase_mean'])
        # 穩健
        base_cls = cs[1.0]
        same_m = len({cs[f] for f in FACT}) == 1
        same_b = (coarse(vs[1.0], exp) == wb) if exp != 'N' else True
        row['robust'] = bool(same_m and same_b) if exp != 'N' else None
        row['coarse_by_m'] = {str(f): cs[f] for f in FACT}
        if exp != 'N' and not row['robust']:
            why = []
            if not same_m:
                why.append('m 換倍率翻類：' + ' / '.join(f'×{f}={cs[f]}' for f in FACT))
            if not same_b:
                why.append(f'Wilcoxon+Holm={wb}（m×1={base_cls}）')
            unstable.append(f'{c["label"]} / {mk}：' + '；'.join(why))
        t1.setdefault(key, {'label': c['label'], 'expected': exp})[mk] = row
    summary['t1'] = t1
    summary['t1_R_by_m'] = {str(f): R[f] for f in FACT}
    summary['t1_R_wilcoxon'] = Rb
    summary['t1_unstable'] = unstable
    nonN = sum(1 for k, c in cells.items() if c['exp'] != 'N')
    summary['t1_n_cells'] = nonN

    # ---------------- 表 2 A 區 ----------------
    exc = json.loads((T.EV / 'midi_realpiece_123/excerpts.json').read_text())
    perf = {x['id']: x['midi'] for x in exc}
    pc = ROOT / 'per_clip'
    base = T.read_tsv(pc / 'T2A_base.tsv')
    t2cells = {}
    for key, label, exp in T.T2A_ROWS:
        t = T.read_tsv(pc / f'T2A_{key}.tsv')
        for mk, col, m in T.METRICS:
            ids = [i for i in base if i in t]
            d = np.array([t[i][col] - base[i][col] for i in ids])
            cl = [perf.get(i.rsplit('_', 1)[0], i) for i in ids]
            s = T.boot_mean(d, cl)
            ref = selfcheck['t2a'][key]['combined'][mk]
            if abs(s['mean'] - ref['mean']) > 1e-6:
                print('T2A SELFCHECK FAIL', key, mk)
                sys.exit(2)
            # 演奏層平均（Wilcoxon 單位 = 演奏）
            un = {}
            for c_, x in zip(cl, d):
                un.setdefault(c_, []).append(x)
            pm = np.array([np.mean(v) for v in un.values()])
            t2cells[(key, mk)] = dict(label=label, exp=exp, m=m, s=s, pm=pm)
    ks = list(t2cells)
    ps2 = [wil_p(t2cells[k]['pm']) for k in ks]
    pa2 = holm(ps2)
    t2 = {}
    unstable2 = []
    R2m = {f: {'R1': {m[0]: 0 for m in T.METRICS}, 'R2': {m[0]: 0 for m in T.METRICS}} for f in FACT}
    R2b = {'R1': {m[0]: 0 for m in T.METRICS}, 'R2': {m[0]: 0 for m in T.METRICS}}
    for k, p, pa in zip(ks, ps2, pa2):
        c = t2cells[k]
        key, mk = k
        exp, m, s = c['exp'], c['m'], c['s']
        vs = {f: T.verdict(s, m * f, exp) for f in FACT}
        for f in FACT:
            if exp in ('U', 'UW') and vs[f] == '**方向錯**':
                R2m[f]['R1'][mk] += 1
            if exp == 'W' and vs[f] == '沒反應':
                R2m[f]['R2'][mk] += 1
        wb = wil_class(s['mean'], float(pa), exp)
        if exp in ('U', 'UW') and wb == '方向錯':
            R2b['R1'][mk] += 1
        if exp == 'W' and wb == '沒反應':
            R2b['R2'][mk] += 1
        cs = {f: coarse(vs[f], exp) for f in FACT}
        robust = None
        if exp != 'N':
            robust = bool(len(set(cs.values())) == 1 and cs[1.0] == wb)
            if not robust:
                unstable2.append(f'{c["label"]} / {mk}：' + ' / '.join(f'×{f}={cs[f]}' for f in FACT) + f'；Wilcoxon+Holm={wb}')
        t2.setdefault(key, {'label': c['label'], 'expected': exp})[mk] = {
            'mean': s['mean'], 'lo': s['lo'], 'hi': s['hi'], 'n_perf': int(len(c['pm'])),
            'dz': float(c['pm'].mean() / c['pm'].std(ddof=1)) if len(c['pm']) > 1 else float('nan'),
            'up_frac_perf': float((c['pm'] > 0).mean()), 'p': p, 'p_holm': float(pa),
            'verdict_by_m': {str(f): vs[f] for f in FACT}, 'wilcoxon_class': wb, 'robust': robust}
    summary['t2a'] = t2
    summary['t2a_R_by_m'] = {str(f): R2m[f] for f in FACT}
    summary['t2a_R_wilcoxon'] = R2b
    summary['t2a_unstable'] = unstable2
    (RES / 'verdict_robustness_202_summary.json').write_text(json.dumps(summary, indent=1, default=float))

    # ---------------- 報告 ----------------
    L = ['# 202 判定穩健性結果（2026-11-25，零 GPU，重算 197 既有逐段分數）', '',
         '設計：`docs/experiments/paper_closing_evidence_200_202_20261125.md` §3。腳本：`scripts/analysis/verdict_robustness_202.py`；'
         f'bootstrap {T.NB} 次、種子 {T.SEED}（沿用 197 tables 腳本）。無人類評分。', '',
         f'自我核對：表 1 共 {ncheck} 格重建的 Δ 平均與 197 summary 最大差 {maxdiff:.2e}（CI 端點最大差 {maxdiff_ci:.2e}），表 2 A 區 combined 同樣一致 → 通過。', '',
         '穩健＝同一格的分類在 m×{0.5,1,2} 與「Wilcoxon + Holm（p<0.05，不看大小）」下相同。'
         '粗類：預期變差的列為 抓得到(變差)／方向錯／沒反應；預期不變或不變或變差的列為 方向錯／變差(過敏)／其他。'
         '「沒反應」在 Wilcoxon 下只等於檢定不顯著，不等於證明沒差。', '']

    def cell_line(key, mk):
        r = t1[key][mk]
        return (f'{r["mean"]:+.3f} [{r["lo"]:+.3f},{r["hi"]:+.3f}] dz {r["dz"]:+.2f} 上升{100 * r["up_frac"]:.0f}% '
                f'k={r["k_same_sign_ci"]}/6(大於m {r["k_same_sign_big"]}) 倍底線{r["abs_over_seed_floor"]:.1f} p_holm={r["p_holm"]:.1e} '
                f'; m×0.5/1/2={r["coarse_by_m"]["0.5"]}/{r["coarse_by_m"]["1.0"]}/{r["coarse_by_m"]["2.0"]}; W={r["wilcoxon_class"]} '
                f'; {"穩健" if r["robust"] else "依門檻而定"}')

    # headline 1
    L += ['## 標題結論 1：A1 的 5 格方向錯', '', '| 格 | A1 統計（Δ [CI]／dz／上升比例／k／倍底線／p_holm；各門檻判定；穩健與否） |', '|---|---|']
    h1 = ['ps_m1', 'ps_m2', 'ts_090', 'lp6k', 'eq250_p10']
    for key in h1:
        r = t1[key]['A1']
        L.append(f'| {r["label"]}（原判 {r["verdict_by_m"]["1.0"]}） | {cell_line(key, "A1")} |')
    r1 = [k for k in h1 if t1[k]['A1']['robust']]
    L += ['', f'穩健者：{", ".join(t1[k]["A1"]["label"] for k in r1) or "無"}；依門檻而定者：'
          f'{", ".join(t1[k]["A1"]["label"] for k in h1 if k not in r1) or "無"}。', '']
    L += ['排名後果（A1，單獨把處理套在一個底上，越過的底數；每個底各自的 Δ）：', '',
          '| 格 | ' + ' | '.join(b.replace('ma_', '') for b in T.BASES) + ' |', '|---|' + '---|' * len(T.BASES)]
    for key in h1:
        rc = t1[key]['A1']['rank_crossed']
        L.append(f'| {t1[key]["A1"]["label"]} | ' + ' | '.join(str(rc.get(b, '—')) for b in T.BASES) + ' |')
    L += ['', f'A1 的乾淨均值：' + '；'.join(f'{b.replace("ma_", "")} {clean[b]["A1"]:.3f}' for b in T.BASES) +
          f'。seed 底線（三個 cfg3neg 兩兩最大差）A1 {cl_floor["A1"]:.3f}、PQ {cl_floor["PQ"]:.3f}、CE {cl_floor["CE"]:.3f}、CMI {cl_floor["CMI"]:.3f}。', '']

    # headline 2
    L += ['## 標題結論 2：`pno0`（同響度調內鋼琴琶音）四指標全方向錯', '',
          '| 指標 | 統計（Δ [CI]／dz／上升比例／k／倍底線／p_holm；各門檻判定；穩健與否） |', '|---|---|']
    for mk, _, _ in T.METRICS:
        L.append(f'| {mk}（原判 {t1["pno0"][mk]["verdict_by_m"]["1.0"]}） | {cell_line("pno0", mk)} |')
    L.append('')

    # headline 3
    L += ['## 標題結論 3：表 2 A 區 PQ／CE 對錯音沒反應', '',
          '單位＝演奏（150 個 MAESTRO 演奏、兩 SoundFont 先平均）；p 值為 Holm（表 2 A 區 12 列 × 4 指標）。', '',
          '| 列 | 指標 | Δ [CI] | dz | 上升演奏比例 | p_holm | m×0.5/1/2 | Wilcoxon | 穩健 |', '|---|---|---|---|---|---|---|---|---|']
    for key in ['wrong_10', 'wrong_30', 'wrong_100', 'extra_rand_30']:
        for mk in ['PQ', 'CE']:
            r = t2[key][mk]
            L.append(f'| {t2[key]["label"]} | {mk} | {r["mean"]:+.3f} [{r["lo"]:+.3f},{r["hi"]:+.3f}] | {r["dz"]:+.2f} | '
                     f'{100 * r["up_frac_perf"]:.0f}% | {r["p_holm"]:.1e} | '
                     f'{r["verdict_by_m"]["0.5"]}/{r["verdict_by_m"]["1.0"]}/{r["verdict_by_m"]["2.0"]} | {r["wilcoxon_class"]} | '
                     f'{"穩健" if r["robust"] else "依門檻而定"} |')
    L.append('')

    # R tables
    L += ['## R1／R2 隨設定變化', '', '表 1（主表，6 底平均）：', '',
          '| 設定 | R1 A1／PQ／CE／CMI | R2 A1／PQ／CE／CMI |', '|---|---|---|']
    for f in FACT:
        L.append(f'| m×{f} | ' + '／'.join(str(R[f]['R1'][m[0]]) for m in T.METRICS) + ' | ' + '／'.join(str(R[f]['R2'][m[0]]) for m in T.METRICS) + ' |')
    L.append('| Wilcoxon+Holm | ' + '／'.join(str(Rb['R1'][m[0]]) for m in T.METRICS) + ' | ' + '／'.join(str(Rb['R2'][m[0]]) for m in T.METRICS) + ' |')
    L += ['', '表 2 A 區（combined）：', '', '| 設定 | R1 | R2 |', '|---|---|---|']
    for f in FACT:
        L.append(f'| m×{f} | ' + '／'.join(str(R2m[f]['R1'][m[0]]) for m in T.METRICS) + ' | ' + '／'.join(str(R2m[f]['R2'][m[0]]) for m in T.METRICS) + ' |')
    L.append('| Wilcoxon+Holm | ' + '／'.join(str(R2b['R1'][m[0]]) for m in T.METRICS) + ' | ' + '／'.join(str(R2b['R2'][m[0]]) for m in T.METRICS) + ' |')
    L.append('')

    n_t1 = sum(1 for k, c in cells.items() if c['exp'] != 'N')
    L += [f'## 所有不穩健的格（表 1：{len(unstable)}／{n_t1}；表 2 A 區：{len(unstable2)}／{sum(1 for k, c in t2cells.items() if c["exp"] != "N")}）', '']
    L += ['### 表 1', ''] + [f'- {u}' for u in unstable] + ['', '### 表 2 A 區', ''] + [f'- {u}' for u in unstable2] + ['']

    # 全部格 方向錯 總表
    L += ['## 表 1 所有「方向錯」格的 (c)(d)(e)', '',
          '| 格 | 指標 | Δ | k/6(CI) | 倍 seed 底線 | 最大越過底數 | 穩健 |', '|---|---|---|---|---|---|---|']
    for (key, mk), c in cells.items():
        r = t1[key][mk]
        if r['verdict_by_m']['1.0'] == '**方向錯**':
            rc = r['rank_crossed']
            L.append(f'| {r["label"]} | {mk} | {r["mean"]:+.3f} | {r["k_same_sign_ci"]}/6 | {r["abs_over_seed_floor"]:.1f}× | {max(rc.values())} | '
                     f'{"穩健" if r["robust"] else "依門檻而定"} |')
    L += ['', '## 不能宣稱的事', '',
          '- 穩健只表示這幾種門檻與檢定下分類一致；Wilcoxon 在 n=1000 的 id 上對微小偏移很敏感，「方向錯」在 b 之下不含大小，需配合 m 與倍底線讀。',
          '- 「沒反應」不等於證明沒差；`seed 底線` 只有 3 個 MeanAudio seed，估計很粗。',
          '- 排名後果只是把某處理套在單一底上的簡單位移，不是新生成；無人類評分，不能寫變好或變差。']
    (RES / 'verdict_robustness_202_results.md').write_text('\n'.join(L) + '\n')
    print('unstable T1', len(unstable), 'T2A', len(unstable2))


if __name__ == '__main__':
    main()
