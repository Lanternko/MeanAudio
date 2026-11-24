#!/usr/bin/env python3
"""201a ODAQ 人類錨點（零 GPU）。設計：docs/experiments/paper_closing_evidence_200_202_20261125.md §1。

對每個條件群（LP3.5／LP7 錨點 20 個音樂 trial；各方法類別的 Q1–Q5 各 4 個 trial；DE 附錄）：
  人類：逐聽者算「條件 − Ref」，對 trial 平均，聽者 bootstrap（5000、seed 20261125）；閘門＝平均 ≤ −10 且 CI 上界 < 0。
  評分器：Δ = score(條件) − score(reference)，trial bootstrap 同 seed；分類沿用 attack_metric_matrix_tables.verdict（預期＝變差）。
  主格 native_norm；native_raw 當敏感度。
  python scripts/analysis/odaq_human_anchor_201.py
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.pop('MKL_NUM_THREADS', None)
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'eval/bench_r2'))
import attack_metric_matrix_tables as T  # noqa: E402
import analyze_r2 as AR  # noqa: E402  (只用 lp_visible／PATH／load_scores，import 無副作用)

SEED = 20261125
NB = 5000
RES = T.RES
MAN = Path('/mnt/seagate/benchmarks_r2/manifests/odaq.tsv')
HUM = Path('/mnt/seagate/benchmarks_r2/downloads/ODAQ/ODAQ/ODAQ_listening_test/ODAQ_results.csv')
MUSIC = ['LP', 'TM', 'UN', 'SH', 'PE']
# (model, 欄位, 顯示名, m 或 None, 進主判定)
SC = [('muqeval_A1', 'muqeval_A1_MI', 'A1 MI', 0.055), ('aes', 'aes_PQ', 'PQ', 0.0914), ('aes', 'aes_CE', 'CE', 0.119),
      ('cmirm', 'cmi_empty_quality', 'CMI q', 0.0657), ('muqeval_A3a', 'muqeval_A3a_MI', 'A3a MI', None),
      ('pam', 'pam', 'PAM', None), ('meva', 'meva', 'MEva', None),
      ('aes', 'aes_CU', 'CU(描述)', None), ('aes', 'aes_PC', 'PC(描述)', None)]


def boot_idx(K, seed_off=0):
    return np.random.default_rng(SEED + seed_off).integers(0, K, size=(NB, K))


def boot_stat(mat_or_vec, idx):
    """vec: 長度 K 的 trial 平均值；回傳 dict(mean, lo, hi, n, sd, up)。"""
    v = np.asarray(mat_or_vec, float)
    bm = v[idx].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return {'mean': float(v.mean()), 'lo': float(lo), 'hi': float(hi), 'n': int(len(v)),
            'sd': float(v.std(ddof=1)) if len(v) > 1 else float('nan'), 'up': float((v > 0).mean())}


def main():
    # ---- manifest ----
    man = {}
    trial_meta = {}
    with open(MAN, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            e = json.loads(r['extra'])
            man[(e['trial'], e['condition'])] = r['item_id']
            trial_meta[e['trial']] = e['method']
    trials_by_method = {}
    for t, m in trial_meta.items():
        trials_by_method.setdefault(m, []).append(t)
    for m in trials_by_method:
        trials_by_method[m].sort()

    # ---- 人類 ----
    d = pd.read_csv(HUM)
    piv = d.pivot_table(index=['item', 'subject'], columns='condition', values='score')
    subs = sorted(d.subject.unique())
    nsub = len(subs)
    H = {}  # (trial, cond) -> array(nsub) 條件 − Ref
    for t in trial_meta:
        sub_t = piv.loc[t].reindex(subs)
        for c in sub_t.columns:
            if c != 'Ref':
                H[(t, c)] = (sub_t[c] - sub_t['Ref']).to_numpy()
    sidx = boot_idx(nsub, 1)

    # ---- 評分器 ----
    S = {}
    for loud in ('norm', 'raw'):
        sc = AR.load_scores('odaq')
        tab = {}
        for (model, v, l), rows in sc.items():
            if v != 'native' or l != loud or model not in ('aes', 'cmirm', 'muqeval_A1', 'muqeval_A3a', 'pam', 'meva'):
                continue
            for item, dd in rows.items():
                for col, x in dd.items():
                    tab.setdefault(col, {})[item] = x
        S[loud] = tab

    def groups():
        g = []
        for cond, name in (('LP3.5', 'LP35 錨點（3.5 kHz）'), ('LP7', 'LP70 錨點（7 kHz）')):
            g.append((name, 'anchor', cond, MUSIC, [t for m in MUSIC for t in trials_by_method[m]]))
            for m in MUSIC:
                g.append((f'{name}｜{m}', 'anchor_m', cond, [m], trials_by_method[m]))
        for m in MUSIC:
            for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
                g.append((f'{m} {q}', 'q', q, [m], trials_by_method[m]))
        for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
            g.append((f'音樂池 {q}（描述）', 'qpool', q, MUSIC, [t for m in MUSIC for t in trials_by_method[m]]))
        for cond in ('LP3.5', 'LP7', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
            if all((t, cond) in H for t in trials_by_method['DE']):
                g.append((f'DE {cond}（附錄）', 'de', cond, ['DE'], trials_by_method['DE']))
        return g

    # 評分器參考分布 sd（20 個音樂 reference）
    mus_tr = [t for m in MUSIC for t in trials_by_method[m]]
    ref_sd = {}
    for loud in ('norm', 'raw'):
        for model, col, nm, m in SC:
            vals = [S[loud][col][man[(t, 'Ref')]] for t in mus_tr if man[(t, 'Ref')] in S[loud][col]]
            ref_sd[(loud, col)] = float(np.std(vals, ddof=1))

    out = {'seed': SEED, 'nboot': NB, 'n_listeners': nsub, 'groups': {}, 'm': {x[2]: x[3] for x in SC}}
    G = groups()
    for gname, kind, cond, methods, trs in G:
        # 方法類別組：每個 trial 的人類差 (nsub,) → trial 平均
        hm = np.stack([H[(t, cond)] for t in trs], 0).mean(0)  # 每聽者
        hb = hm[sidx].mean(1)
        lo, hi = np.percentile(hb, [2.5, 97.5])
        hum = {'mean': float(hm.mean()), 'lo': float(lo), 'hi': float(hi)}
        hum['gate'] = bool(hum['mean'] <= -10 and hum['hi'] < 0)
        rec = {'kind': kind, 'cond': cond, 'n_trials': len(trs), 'human': hum, 'scorers': {}}
        tidx = boot_idx(len(trs), 2)
        for loud in ('norm', 'raw'):
            for model, col, nm, m in SC:
                vis = all(AR.lp_visible(model, trial_meta[t], cond) for t in trs)
                dl = []
                for t in trs:
                    a = S[loud][col].get(man[(t, cond)])
                    b = S[loud][col].get(man[(t, 'Ref')])
                    dl.append(np.nan if a is None or b is None else a - b)
                dl = np.array(dl)
                if np.isnan(dl).any():
                    rec['scorers'].setdefault(nm, {})[loud] = {'missing': True}
                    continue
                s = boot_stat(dl, tidx)
                std = s['mean'] / ref_sd[(loud, col)]
                if not vis:
                    cls = 'invisible'
                else:
                    cls = T.verdict(s, m or 0.0, 'W')
                rec['scorers'].setdefault(nm, {})[loud] = {
                    'delta': s['mean'], 'lo': s['lo'], 'hi': s['hi'], 'std_effect': float(std),
                    'k_up': int((dl > 0).sum()), 'k_down': int((dl < 0).sum()), 'n': len(dl),
                    'class': cls.replace('*', ''), 'm': m, 'visible': bool(vis)}
        out['groups'][gname] = rec
    out['ref_sd_norm'] = {col: ref_sd[('norm', col)] for _, col, _, _ in SC}

    # ---- 敏感度與清單 ----
    changed = []
    flagged = []
    for gname, rec in out['groups'].items():
        for nm, dd in rec['scorers'].items():
            if 'class' in dd.get('norm', {}) and 'class' in dd.get('raw', {}) and dd['norm']['class'] != dd['raw']['class']:
                changed.append((gname, nm, dd['norm']['class'], dd['raw']['class']))
            n = dd.get('norm', {})
            if rec['human']['gate'] and n.get('class') in ('**方向錯**', '沒反應') and rec['kind'] in ('anchor', 'anchor_m', 'q', 'qpool'):
                flagged.append((gname, nm, n['class'], n['delta']))
    out['norm_vs_raw_changed'] = changed
    out['gate_pass_but_not_caught'] = flagged
    (RES / 'odaq_human_anchor_201_summary.json').write_text(json.dumps(out, indent=1, default=float))

    # ---- md ----
    def c_(x):
        return x.replace('**', '')

    L = ['# 201a ODAQ 人類錨點結果（2026-11-25，零 GPU，既有 r2 分數，無新評分）', '',
         '設計：`docs/experiments/paper_closing_evidence_200_202_20261125.md` §1。腳本：`scripts/analysis/odaq_human_anchor_201.py`；'
         f'bootstrap {NB} 次、種子 {SEED}（人類＝聽者 {nsub} 位重抽；評分器＝trial 重抽）。'
         '主格 `native_norm`（−23 LUFS）；`native_raw` 為敏感度。**201b（改變量欄）待 199 模組 commit 後補，這輪未做。**', '',
         '判讀規則：人類閘門＝條件 − Ref 的平均 ≤ −10 且 CI 上界 < 0；評分器預期方向＝變差，沿用 197 `verdict()`：'
         '抓得到＝CI 整段 < −m；**方向錯**＝CI 整段 > m；沒反應＝其餘（無 m 的評分器 m=0，只看 CI 是否跨零）；'
         '`invisible`＝依 `analyze_r2.lp_visible()` 該評分器依構造看不到（只用在 LP 類別的 Q1–Q5），不進判定。'
         '格內數字＝標準化效果（Δ ÷ 20 個音樂 reference 的 SD）｜上升 trial 數/總數。', '']
    # E1/E2
    def e(gname, nm):
        r = out['groups'][gname]
        return r['human'], r['scorers'][nm]['norm']
    e1h, e1 = e('LP70 錨點（7 kHz）', 'A1 MI')
    e2h, e2 = e('LP35 錨點（3.5 kHz）', 'A1 MI')
    L += ['## 主端點', '', '| 端點 | 人類 條件−Ref [CI] | 人類閘門 | A1 MI Δ [CI] | 標準化 | 上升 trial | 分類 | 設計預期 |', '|---|---|---|---|---|---|---|---|']
    for nme, h, s, exp in (('E1：LP70（7 kHz，20 trial）', e1h, e1, '人類明顯變差；A1 沒反應或方向錯（193 的 6 kHz 在真實錄音底 +0.061 未過 m）'),
                           ('E2：LP35（3.5 kHz，20 trial）', e2h, e2, '人類明顯變差；A1 抓得到（193 的 3 kHz 以下 A1 為負）')):
        L.append(f'| {nme} | {h["mean"]:+.1f} [{h["lo"]:+.1f},{h["hi"]:+.1f}] | {"過" if h["gate"] else "未過"} | '
                 f'{s["delta"]:+.3f} [{s["lo"]:+.3f},{s["hi"]:+.3f}] | {s["std_effect"]:+.2f} | {s["k_up"]}/{s["n"]} | **{c_(s["class"])}** | {exp} |')
    L.append('')
    # 主表
    names = [x[2] for x in SC]
    hdr = '| 條件群 | trial | 人類 條件−Ref [CI] | 閘門 | ' + ' | '.join(names) + ' |'
    sep = '|---|---|---|---|' + '---|' * len(names)

    def row(gname, rec, loud='norm'):
        cells = []
        for nm in names:
            x = rec['scorers'][nm].get(loud, {})
            if 'delta' not in x:
                cells.append('—')
            else:
                cells.append(f'{x["std_effect"]:+.2f}｜{x["k_up"]}/{x["n"]}｜{c_(x["class"])}' + ('' if x['visible'] else ''))
        h = rec['human']
        return (f'| {gname} | {rec["n_trials"]} | {h["mean"]:+.1f} [{h["lo"]:+.1f},{h["hi"]:+.1f}] | {"過" if h["gate"] else "未過"} | ' + ' | '.join(cells) + ' |')

    sections = [('## 錨點（20 個音樂 trial 合併，以及各方法類別）', ('anchor', 'anchor_m')),
                ('## 各方法類別 Q1–Q5（每格只有 4 個 trial，只當描述）', ('q',)),
                ('## 音樂池各 Q 合併（描述；不同方法的 Q 不是同一劑量）', ('qpool',)),
                ('## 附錄：DE（對白分離；Q 編號不是劑量順序，10 trial）', ('de',))]
    for title, kinds in sections:
        L += [title, '', hdr, sep]
        for gname, rec in out['groups'].items():
            if rec['kind'] in kinds:
                L.append(row(gname, rec))
        L.append('')
    # 數值表（錨點）
    L += ['## 錨點的評分器原始 Δ（native_norm；20 trial）', '', '| 條件 | 評分器 | Δ [CI] | 標準化 | 上升 | 分類 | raw 分類 |', '|---|---|---|---|---|---|---|']
    for gname in ('LP70 錨點（7 kHz）', 'LP35 錨點（3.5 kHz）'):
        for nm in names:
            n_ = out['groups'][gname]['scorers'][nm]['norm']
            r_ = out['groups'][gname]['scorers'][nm]['raw']
            L.append(f'| {gname} | {nm} | {n_["delta"]:+.3f} [{n_["lo"]:+.3f},{n_["hi"]:+.3f}] | {n_["std_effect"]:+.2f} | {n_["k_up"]}/{n_["n"]} | {c_(n_["class"])} | {c_(r_["class"])} |')
    L += ['', f'評分器在 20 個 reference 上的 SD（norm）：' + '；'.join(f'{nm} {out["ref_sd_norm"][col]:.3f}' for _, col, nm, _ in SC), '']
    L += ['## native_raw 敏感度', '']
    if changed:
        L += [f'norm → raw 分類改變 {len(changed)} 格：', ''] + [f'- {g} / {n}：{a} → {b}' for g, n, a, b in changed]
    else:
        L.append('分類在 norm 與 raw 之間完全一致。')
    L += ['', '### 主端點在 norm／raw 的讀法（A1 MI）', '',
          '分類翻轉來自 CI 一端落在 ±m 附近，Δ 的符號在兩種響度處理下相同：', '']
    for gname in ('LP70 錨點（7 kHz）', 'LP35 錨點（3.5 kHz）'):
        for loud in ('norm', 'raw'):
            x = out['groups'][gname]['scorers']['A1 MI'][loud]
            L.append(f'- {gname}／{loud}：{x["delta"]:+.3f} [{x["lo"]:+.3f},{x["hi"]:+.3f}]，上升 {x["k_up"]}/{x["n"]}，'
                     f'CI {"整段 > 0" if x["lo"] > 0 else ("整段 < 0" if x["hi"] < 0 else "跨零")} → {c_(x["class"])}')
    L += ['', '## 過人類閘門卻被判「方向錯」或「沒反應」（非 DE；norm）', '']
    if flagged:
        L += ['| 條件群 | 評分器 | 分類 | Δ |', '|---|---|---|---|'] + [f'| {g} | {n} | {c_(c)} | {dl:+.3f} |' for g, n, c, dl in flagged]
    else:
        L.append('無。')
    L += ['', '## 不能宣稱的事', '',
          '- ODAQ 是 48 kHz 高品質真實錄音（MUSHRA），不是生成音樂；只支持「低通、量化、時間調變等這幾類處理人會判差」，不支持移調、變速、疊入循環音。',
          '- 每個 Q 類別只有 4 個 trial、分類在 n=4 的 trial bootstrap 下 CI 很寬；方法類別列只當描述。',
          '- 人類分數是條件相對 hidden reference 的差，不是絕對品質；MUSHRA 刻度本身的大小不等於 m。',
          '- 「沒反應」只表示 CI 跨零或落在 ±m 內，不證明沒差；A3a／PAM／MEva 無 m，只看 CI。',
          '- 201b（客觀改變量三欄對照）尚未做，待 199 的改變量模組。']
    (RES / 'odaq_human_anchor_201_results.md').write_text('\n'.join(L) + '\n')
    print('E1', e1h, e1)
    print('E2', e2h, e2)
    print('changed', len(changed), 'flagged', len(flagged))


if __name__ == '__main__':
    main()
