#!/usr/bin/env python3
"""199 attack change-amount tables (no GPU).

Reads 199 per_clip (~/eval_output_nvme/attack_change_amount_199[_smoke]/per_clip) and the 197 summary json (the four scorer
deltas are read, not recomputed), applies the acceptance gates (design section 4) and the classification (section 5), writes
  docs/experiments/results/attack_change_amount_199_summary.json
  docs/experiments/results/attack_change_amount_199_table1.md
  docs/experiments/results/attack_change_amount_199_table2.md
Design: docs/experiments/attack_change_amount_199_20261125.md

  python scripts/analysis/attack_change_amount_tables.py [--root DIR] [--out-prefix P] [--nboot N]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attack_metric_matrix_tables as T  # noqa: E402

EV, RES = T.EV, T.RES
SEED, LISTEN = T.SEED, T.LISTEN
QS = [('rho', 'rho'), ('mel', 'mel'), ('d_clap', 'd_clap'), ('d_mert', 'd_mert')]
EMB = ['d_clap', 'd_mert']
R_LIST = [24, 18, 12, 6, 0, -6, -12, -18]
MIX = [f'mix_r{r}' if r >= 0 else f'mix_rm{-r}' for r in R_LIST]
WN = [f'wn_snr{s}' for s in (40, 30, 20, 10, 0)]
BANKS = ['generaluser', 'fluidr3']
NB = 5000


# ───────────── io ─────────────
def read(root, key):
    p = root / 'per_clip' / f'{key}.tsv'
    if not p.is_file():
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    ids = [r['id'] for r in rows]
    return ids, {q: np.array([float(r[q]) for r in rows]) for q, _ in QS}


def lag_of(root, key):
    r = read(root, key)
    return None if r is None else float(np.median(r[1]['rho']))


def med_boot(M, nb):
    """M: (bases, ids) -> pooled median and cluster (id) bootstrap 95% CI."""
    K = M.shape[1]
    pooled = float(np.nanmedian(M))
    rng = np.random.default_rng(SEED + K)
    lo = []
    for s in range(0, nb, 250):
        idx = rng.integers(0, K, size=(min(250, nb - s), K))
        lo.append(np.nanmedian(M[:, idx], axis=(0, 2)))
    lo = np.concatenate(lo)
    a, b = np.percentile(lo, [2.5, 97.5])
    return pooled, float(a), float(b)


def interp_mix(vals, ladder_meds):
    """equivalent mix amount (LU, r) of a median value; ladder_meds in MIX order (increasing)."""
    if not np.all(np.diff(ladder_meds) > 0):
        return None, 'ladder 非單調'
    if vals < ladder_meds[0]:
        return None, '<r24'
    if vals > ladder_meds[-1]:
        return None, '>m18'
    return float(np.interp(vals, ladder_meds, R_LIST)), ''


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    ev = np.linalg.eigvals(c1 @ c2).real
    tr = np.sqrt(np.clip(ev, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1) + np.trace(c2) - 2 * tr)


# ───────────── gates ─────────────
def gates(per, bases):
    """per[(base, cell)] = {q: median}.  -> (dict q -> {pass, detail})."""
    out = {}
    for q, _ in QS:
        det = {}
        det['identity'] = all(per[(b, 'identity')][q] <= 0.1 * per[(b, 'mix_r24')][q] for b in bases)
        det['mix_mono'] = all(all(per[(b, MIX[k])][q] < per[(b, MIX[k + 1])][q] for k in range(len(MIX) - 1)) for b in bases)
        det['wn_mono'] = all(all(per[(b, WN[k])][q] < per[(b, WN[k + 1])][q] for k in range(len(WN) - 1)) for b in bases)
        det['order'] = all(per[(b, 'other_clip')][q] > per[(b, 'mix_r0')][q] > per[(b, 'mix_r12')][q] for b in bases)
        out[q] = {'pass': all(det.values()), **det}
    return out


def classify(meds, ladders, ok_q):
    """meds: {unit: {q: median}}, ladders: {unit: (m12, m0 dict q)}; unit = base (T1) or bank (T2)."""
    qs = [q for q in EMB if ok_q.get(q)]
    if not qs:
        return '無法分類（嵌入量未過驗收）', {}
    votes = {}
    for u, m in meds.items():
        r12, r0 = ladders[u]
        if any(m[q] >= r0[q] for q in qs):
            votes[u] = '換成另一段音樂'
        elif all(m[q] <= r12[q] for q in qs):
            votes[u] = '小改變'
        else:
            votes[u] = '中等'
    n = len(votes)
    best = max(set(votes.values()), key=list(votes.values()).count)
    need = 4 if n >= 6 else n          # T1: >= 4 of 6 bases; T2: both banks
    cnt = list(votes.values()).count(best)
    return (best if cnt >= need else '依底而定'), votes


# ───────────── tables ─────────────
def fmtv(x):
    return f'{x:.3f}' if abs(x) >= 0.01 or x == 0 else f'{x:.4f}'


def make(root, nb):
    s197 = json.loads((RES / 'attack_metric_matrix_197_summary.json').read_text())
    summ = {'root': str(root), 'nboot': nb, 'seed': SEED}
    bases = [b for b in LISTEN if (root / 'per_clip' / f'T1_{b}_identity.tsv').is_file()]
    smoke = 'smoke' in str(root)
    summ['bases'] = bases

    # ---------- T1 ladders / gates ----------
    cells = {}
    for b in bases:
        for c in set(MIX + WN + ['identity', 'other_clip', 'other_seed']):
            r = read(root, f'T1_{b}_{c}')
            if r is not None:
                cells[(b, c)] = r
    per = {k: {q: float(np.nanmedian(v[1][q])) for q, _ in QS} for k, v in cells.items()}
    g = gates(per, bases)
    summ['acceptance'] = g
    ok_q = {q: g[q]['pass'] for q, _ in QS}
    ladder_pool = {}
    for c in MIX + WN + ['identity', 'other_clip']:
        ids = cells[(bases[0], c)][0]
        ladder_pool[c] = {}
        for q, _ in QS:
            M = np.array([cells[(b, c)][1][q] for b in bases])
            ladder_pool[c][q] = float(np.nanmedian(M))
    summ['ladders'] = {c: ladder_pool[c] for c in ladder_pool}
    summ['other_seed'] = {b: per[(b, 'other_seed')] for b in bases if (b, 'other_seed') in per}

    head = ['# 199 表 1：197 表 1 各處理的客觀改變量', '',
            f'底：{", ".join(bases)}（{len(bases)} 個；{"smoke 子集，不可判讀" if smoke else "6 個能聽的底"}）。'
            f'每格 = 逐段（6 底合併）中位數［95% CI，依 id 為單位重抽 {nb} 次］。'
            'rho＝訊號對不上的比例（0 同、1 無關）；mel＝log-mel 差（dB）；d_clap／d_mert＝1−cos。'
            '嵌入量括號內 C＝該列中位數 ÷ 同底 other_clip 中位數（1＝已換成另一段音樂）。', '',
            '## 驗收（design §4；未過的量整欄不分類）', '',
            '| 量 | identity ≤ 1/10 mix_r24 | 混音階梯嚴格單調 | 白噪階梯嚴格單調 | other_clip > mix_r0 > mix_r12 | 結論 |',
            '|---|---|---|---|---|---|']
    for q, _ in QS:
        x = g[q]
        head.append(f'| {q} | {"過" if x["identity"] else "不過"} | {"過" if x["mix_mono"] else "不過"} | '
                    f'{"過" if x["wn_mono"] else "不過"} | {"過" if x["order"] else "不過"} | {"通過" if x["pass"] else "整欄剔除"} |')
    head += ['', '## 兩條階梯（6 底合併中位數）', '',
             '| 階梯點 | ' + ' | '.join(q for q, _ in QS) + ' |', '|---|' + '---|' * len(QS)]
    for c in ['identity'] + MIX + WN + ['other_clip']:
        head.append(f'| {c} | ' + ' | '.join(fmtv(ladder_pool[c][q]) for q, _ in QS) + ' |')
    if summ['other_seed']:
        head += ['', 'other_seed（同 id 換另一 seed 的 MeanAudio 生成）：' + '；'.join(
            f'{b.replace("ma_cfg3neg_", "")} ' + ' '.join(f'{q} {fmtv(v[q])}' for q, _ in QS) for b, v in summ['other_seed'].items())]

    # ---------- T1 rows ----------
    L = list(head) + ['', '## 逐列', '',
                      '| 處理 | 預期 | 197 A1 | 197 PQ | 197 CE | 197 CMI | rho | mel | d_clap (C) | d_mert (C) | 等效混合量（CLAP／MERT） | fd_clap | 類別 | 各底 |',
                      '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    summ['t1'] = {}
    for key, label, exp, src, tok, reft in T.T1_ROWS:
        rows = {b: read(root, f'T1_{b}_{tok}') for b in bases}
        if any(v is None for v in rows.values()):
            L.append(f'| {label} | {T.EXPECT_TXT[exp]} | ' + ' | '.join(['待 199'] * 12) + ' |')
            continue
        d197 = []
        for mk, _, m in T.METRICS:
            s = s197['t1'][key][mk].get('main')
            d197.append('待 197' if not s else f'{T.fmt(s["mean"])}{T.mark(s, m)}')
        ent = {}
        meds, lad = {}, {}
        for q, _ in QS:
            M = np.array([rows[b][1][q] for b in bases])
            ent[q] = dict(zip(('median', 'lo', 'hi'), med_boot(M, nb)))
        for b in bases:
            meds[b] = {q: float(np.nanmedian(rows[b][1][q])) for q, _ in QS}
            lad[b] = (per[(b, 'mix_r12')], per[(b, 'mix_r0')])
        cls, votes = classify(meds, lad, ok_q)
        cc = {q: ent[q]['median'] / ladder_pool['other_clip'][q] for q in EMB}
        eq = {}
        for q in EMB:
            if ok_q[q]:
                lm = np.array([ladder_pool[c][q] for c in MIX])
                r, note = interp_mix(ent[q]['median'], lm)
                eq[q] = 'r%+.1f' % r if r is not None else note
            else:
                eq[q] = '—'
        # fd_clap
        fds = []
        for b in bases:
            ep = root / 'emb' / f'T1_{b}_{tok}.npy'
            rp = root / 'emb' / f'T1_{b}_ref_{"ext" if reft == "ext" else "base"}.npy'
            if ep.is_file() and rp.is_file():
                fds.append(frechet(np.load(rp), np.load(ep)))
        fd = float(np.mean(fds)) if fds else float('nan')
        cell = lambda q: f'{fmtv(ent[q]["median"])} [{fmtv(ent[q]["lo"])}, {fmtv(ent[q]["hi"])}]'  # noqa: E731
        cq = lambda q: (cell(q) + f' (C {cc[q]:.2f})') if ok_q[q] else cell(q) + ' (未過驗收)'  # noqa: E731
        L.append(f'| {label} | {T.EXPECT_TXT[exp]} | ' + ' | '.join(d197) + f' | {cell("rho")} | {cell("mel")} | {cq("d_clap")} | '
                 f'{cq("d_mert")} | {eq["d_clap"]}／{eq["d_mert"]} | {fd:.3f} | {cls} | '
                 + ' '.join(f'{b.replace("ma_cfg3neg_", "ma").replace("musicgen", "mg").replace("musicldm", "ml")[:6]}:{votes.get(b, "")[:2]}' for b in bases) + ' |')
        summ['t1'][key] = {'expected': exp, 'class': cls, 'votes': votes, 'quantities': ent, 'C': cc, 'equiv_mix': eq,
                           'fd_clap': fd, 'per_base_median': meds}
    t1 = L

    # ---------- T2 ----------
    t2ok = {}
    sec = {}
    for bank in BANKS:
        c2 = {c: read(root, f'T2A_{bank}_{c}') for c in ['identity', 'other_excerpt', 'other_bank'] + MIX}
        if any(v is None for v in c2.values()):
            continue
        sec[bank] = c2
    L2 = ['# 199 表 2：197 表 2 MIDI 處理的客觀改變量（A 區：MAESTRO 150 × 2 SoundFont）', '',
          f'{"smoke 子集，不可判讀。" if smoke else ""}每格 = 逐段中位數［95% CI］（兩個 SoundFont 各自一列）。'
          'rho／mel／d_clap／d_mert 定義同表 1；B 區和弦進行不在 199 範圍。分類沿用表 1 驗收（T2 階梯用各 SoundFont 自己的 mix 與 other_excerpt／other_bank）。', '',
          '## 兩條階梯與參照點（逐 SoundFont 中位數）', '', '| 階梯點 | SoundFont | ' + ' | '.join(q for q, _ in QS) + ' |',
          '|---|---|' + '---|' * len(QS)]
    summ['t2_ladders'] = {}
    for bank, c2 in sec.items():
        summ['t2_ladders'][bank] = {}
        for c in ['identity'] + MIX + ['other_excerpt', 'other_bank']:
            m = {q: float(np.nanmedian(c2[c][1][q])) for q, _ in QS}
            summ['t2_ladders'][bank][c] = m
            L2.append(f'| {c} | {bank} | ' + ' | '.join(fmtv(m[q]) for q, _ in QS) + ' |')
    L2 += ['', '## 逐列', '', '| 處理 | 預期 | SoundFont | 197 A1 | 197 PQ | 197 CE | 197 CMI | rho | mel | d_clap | d_mert | 等效混合量（CLAP／MERT） | 類別 |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    summ['t2'] = {}
    for key, label, exp in T.T2A_ROWS:
        rows = {bank: read(root, f'T2A_{bank}_{key}') for bank in sec}
        if not rows or any(v is None for v in rows.values()):
            L2.append(f'| {label} | {T.EXPECT_TXT[exp]} | — | ' + ' | '.join(['待 199'] * 10) + ' |')
            continue
        meds = {bank: {q: float(np.nanmedian(rows[bank][1][q])) for q, _ in QS} for bank in rows}
        lad = {bank: (summ['t2_ladders'][bank]['mix_r12'], summ['t2_ladders'][bank]['mix_r0']) for bank in rows}
        cls, votes = classify(meds, lad, ok_q)
        summ['t2'][key] = {'expected': exp, 'class': cls, 'votes': votes, 'per_bank_median': meds, 'quantities': {}}
        for bank in rows:
            d197 = []
            for mk, _, m in T.METRICS:
                s = s197['t2a'][key][bank].get(mk)
                d197.append('待 197' if not s else f'{T.fmt(s["mean"])}{T.mark(s, m)}')
            ent = {q: dict(zip(('median', 'lo', 'hi'), med_boot(rows[bank][1][q][None, :], nb))) for q, _ in QS}
            summ['t2'][key]['quantities'][bank] = ent
            eq = {}
            for q in EMB:
                if ok_q[q]:
                    lm = np.array([summ['t2_ladders'][bank][c][q] for c in MIX])
                    r, note = interp_mix(ent[q]['median'], lm)
                    eq[q] = 'r%+.1f' % r if r is not None else note
                else:
                    eq[q] = '—'
            cell = lambda q: f'{fmtv(ent[q]["median"])} [{fmtv(ent[q]["lo"])}, {fmtv(ent[q]["hi"])}]'  # noqa: E731
            L2.append(f'| {label} | {T.EXPECT_TXT[exp]} | {bank} | ' + ' | '.join(d197) + ' | ' + ' | '.join(cell(q) for q, _ in QS)
                      + f' | {eq["d_clap"]}／{eq["d_mert"]} | {cls} |')
    return t1, L2, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(EV / 'attack_change_amount_199'))
    ap.add_argument('--out-prefix', default=str(RES / 'attack_change_amount_199'))
    ap.add_argument('--nboot', type=int, default=NB)
    a = ap.parse_args()
    t1, t2, summ = make(Path(a.root), a.nboot)
    P = a.out_prefix
    Path(P + '_table1.md').write_text('\n'.join(t1) + '\n')
    Path(P + '_table2.md').write_text('\n'.join(t2) + '\n')
    Path(P + '_summary.json').write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=float))
    print('wrote', P + '_{table1.md,table2.md,summary.json}')


if __name__ == '__main__':
    main()
