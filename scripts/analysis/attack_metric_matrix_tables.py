#!/usr/bin/env python3
"""197 attack x metric tables (no GPU).

Reads 197 per_clip (~/eval_output_nvme/attack_metric_matrix_197[_smoke]/per_clip) plus the
existing 189 / 190 / 193 per_clip rows, and writes
  docs/experiments/results/attack_metric_matrix_197_table1.md
  docs/experiments/results/attack_metric_matrix_197_table2.md
  docs/experiments/results/attack_metric_matrix_197_summary.json
Cells without 197 scores are marked "待 197".  Design: docs/experiments/attack_metric_matrix_197_20261117.md

  python scripts/analysis/attack_metric_matrix_tables.py [--root DIR] [--out-prefix P] [--nboot N]
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

HOME = Path.home()
EV = HOME / 'eval_output_nvme'
RES = Path(__file__).resolve().parents[2] / 'docs/experiments/results'
SEED = 20261117
NB = 5000

METRICS = [('A1', 'A1_MI', 0.055), ('PQ', 'PQ', 0.0914), ('CE', 'CE', 0.119), ('CMI', 'CMI_q', 0.0657)]
BASES = ['real', 'ma_cfg0_s14159265', 'ma_cfg0_s16180339', 'ma_cfg0_s27182818',
         'ma_cfg3neg_s14159265', 'ma_cfg3neg_s16180339', 'ma_cfg3neg_s27182818', 'musicgen', 'musicldm']
LISTEN = ['real', 'ma_cfg3neg_s14159265', 'ma_cfg3neg_s16180339', 'ma_cfg3neg_s27182818', 'musicgen', 'musicldm']
EXPECT_TXT = {'W': '變差', 'U': '不變', 'UW': '不變或變差', 'N': '無法判斷'}

# (key, label, expected, source, treat file token, reference file token)
T1_ROWS = [
    ('gain_p6', '音量 +6 dB', 'U', '197', 'gain_p6', 'base'),
    ('gain_m6', '音量 -6 dB', 'U', '197', 'gain_m6', 'base'),
    ('sil05', '前加 0.5 s 靜音', 'U', '197', 'sil05', 'base'),
    ('rb_id', 'rubberband 原速原調', 'U', '197', 'rb_id', 'base'),
    ('ps_p1', '移調 +1 半音', 'UW', '197', 'ps_p1', 'base'),
    ('ps_m1', '移調 -1 半音', 'UW', '197', 'ps_m1', 'base'),
    ('ps_p2', '移調 +2 半音', 'UW', '197', 'ps_p2', 'base'),
    ('ps_m2', '移調 -2 半音', 'UW', '197', 'ps_m2', 'base'),
    ('ts_090', '變速 0.9x', 'UW', '197', 'ts_090', 'base'),
    ('ts_110', '變速 1.1x', 'UW', '197', 'ts_110', 'base'),
    ('kick05_r12', '規律大鼓 /0.5 s（小 12 LU）', 'W', '197', 'kick05_r12', 'base'),
    ('kickrand_r12', '隨機大鼓（小 12 LU）', 'W', '197', 'kickrand_r12', 'base'),
    ('click05_r12', '規律 1 ms 脈衝（小 12 LU）', 'W', '197', 'click05_r12', 'base'),
    ('sq220_r18', '220 Hz 方波（小 18 LU）', 'W', '197', 'sq220_r18', 'base'),
    ('nburst05_r12', '50 ms 白噪爆（小 12 LU）', 'W', '197', 'nburst05_r12', 'base'),
    ('pno0', '同響度調內鋼琴琶音', 'W', '197', 'pno0', 'base'),
    ('lp6k', '低通 6 kHz [193]', 'UW', '193', 'lp6k', 'base'),
    ('lp4k', '低通 4 kHz [193]', 'UW', '193', 'lp4k', 'base'),
    ('lp2k', '低通 2 kHz [193]', 'UW', '193', 'lp2k', 'base'),
    ('hp200', 'high pass 200 Hz [193]', 'UW', '193', 'hp200', 'base'),
    ('eq250_p10', '250 Hz +10 dB [193]', 'UW', '193', 'eq250_p10', 'base'),
    ('ctr_snr10', '中段加白噪 SNR 10 [190]', 'W', '190', 'ctr_snr10', 'ext'),
    ('side_snr10', '頭尾（視窗外）加白噪 SNR 10 [190]', 'W', '190', 'side_snr10', 'ext'),
    ('demucs', 'htdemucs 分離再相加 [190]', 'U', '190', 'demucs', 'base'),
    ('sine1k_r18', '疊 1 kHz 正弦 r18 [189]', 'W', '189', 'sine1k_r18', 'base'),
    ('osynth_r18', '疊合成循環音 r18 [189]', 'W', '189', 'osynth_r18', 'base'),
]
T2A_ROWS = [
    ('wrong_10', '10% 音符錯音高', 'W'), ('wrong_30', '30% 音符錯音高', 'W'), ('wrong_100', '全部錯音高', 'W'),
    ('extra_harm_30', '多 30% 音符（和聲內）', 'N'), ('extra_rand_30', '多 30% 音符（隨機音高）', 'W'),
    ('jitter_50', 'onset 抖動 50 ms', 'W'), ('drop_30', '拿掉 30% 音符', 'W'),
    ('tr_p1', 'MIDI 移調 +1', 'U'), ('tr_m1', 'MIDI 移調 -1', 'U'), ('oct_p1', '高八度', 'N'),
    ('tempo_090', 'MIDI 變速 0.9x', 'U'), ('tempo_110', 'MIDI 變速 1.1x', 'U'),
]
SRC_DIR = {'193': EV / 'muq_filter_ladder_193/per_clip', '190': EV / 'muq_postproc_attack_190/per_clip',
           '189': EV / 'muq_overlay_matrix_189/per_clip'}


# ───────────────────────── io ─────────────────────────
def read_tsv(p):
    p = Path(p)
    if not p.exists():
        return None
    with open(p) as f:
        rd = csv.DictReader(f, delimiter='\t')
        rows = list(rd)
    out = {}
    for r in rows:
        d = {}
        for _, col, _ in METRICS:
            try:
                d[col] = float(r.get(col, 'nan'))
            except ValueError:
                d[col] = float('nan')
        out[r['id']] = d
    return out


def per_clip_path(src, base, tok, root):
    if src == '197':
        return root / 'per_clip' / f'T1_{base}_{tok}.tsv'
    if src == '189':
        return SRC_DIR['189'] / f'T_{base}_{tok}.tsv' if tok == 'base' else SRC_DIR['189'] / f'T_{base}_{tok}.tsv'
    return SRC_DIR[src] / f'T_{base}_{tok}.tsv'


def ref_path(src, base, tok, root):
    return per_clip_path(src, base, tok, root)


# ───────────────────────── stats ─────────────────────────
_IDX = {}


def _idx(K):
    if K not in _IDX:
        rng = np.random.default_rng(SEED + K)
        _IDX[K] = rng.integers(0, K, size=(NB, K))
    return _IDX[K]


def boot_mean(vals, clusters=None):
    """mean of vals with 95% bootstrap CI; clusters = array of cluster labels (default: each item)."""
    v = np.asarray(vals, float)
    ok = ~np.isnan(v)
    v = v[ok]
    n = len(v)
    if n == 0:
        return None
    if clusters is None:
        cl = np.arange(n)
    else:
        _, cl = np.unique(np.asarray(clusters)[ok], return_inverse=True)
    K = cl.max() + 1
    S = np.bincount(cl, weights=v, minlength=K)
    C = np.bincount(cl, minlength=K).astype(float)
    idx = _idx(K)
    bm = S[idx].sum(1) / C[idx].sum(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if n > 1 else float('nan'), 'up': float((v > 0).mean()),
            'lo': float(lo), 'hi': float(hi), 'n': int(n), 'k': int(K)}


def mark(s, m):
    if s is None:
        return ''
    t = ''
    if s['lo'] > 0 or s['hi'] < 0:
        t = '*'
    if s['lo'] > m or s['hi'] < -m:
        t = '**'
    return t


def verdict(s, m, exp):
    if s is None:
        return '待 197'
    big = s['lo'] > m or s['hi'] < -m
    inside = s['lo'] >= -m and s['hi'] <= m
    pos = s['mean'] > 0
    if exp == 'N':
        return '—'
    if exp == 'W':
        if big and not pos:
            return '抓得到'
        if big and pos:
            return '**方向錯**'
        return '沒反應'
    if exp == 'U':
        if inside:
            return '不變(對)'
        if big and not pos:
            return '過敏'
        if big and pos:
            return '**方向錯**'
        return '未定'
    if exp == 'UW':
        if big and pos:
            return '**方向錯**'
        return '可接受(不變)' if inside else ('可接受(變差)' if big else '可接受(未定)')
    return '—'


def paired_d(treat, ref, col):
    ids = [i for i in ref if i in treat]
    t = np.array([treat[i][col] for i in ids])
    r = np.array([ref[i][col] for i in ids])
    return ids, t - r


def fmt(x):
    return f'{x:+.3f}'


# ───────────────────────── table 1 ─────────────────────────
def t1_cell(root, src, tok, reft, base, col):
    t = read_tsv(per_clip_path(src, base, tok, root))
    r = read_tsv(ref_path(src, base, reft, root))
    if t is None or r is None:
        return None, None
    ids, d = paired_d(t, r, col)
    return ids, d


def table1(root, summary):
    L = ['# 197 表 1：任何音檔都能做的處理（逐段配對差 d = 處理後 − 同段 base，−30 LUFS）', '',
         f'記號：`*` CI 不跨零；`**` CI 整段在平手帶 m 之外。m：A1 0.055／PQ 0.0914／CE 0.119／CMI-RM 0.0657。'
         f'每格 = 6 個能聽的底平均 d（等權）｜同號且 `**` 的底數 k/6｜判讀。bootstrap {NB} 次、種子 {SEED}（逐 id 重抽，6 底同一組 id）。', '']
    summary['t1'] = {}
    R1 = {m[0]: 0 for m in METRICS}
    R2 = {m[0]: 0 for m in METRICS}
    L += ['| 處理 | 預期 | ' + ' | '.join(f'{m[0]}' for m in METRICS) + ' |', '|---|---|' + '---|' * len(METRICS)]
    app = {m[0]: [] for m in METRICS}
    for key, label, exp, src, tok, reft in T1_ROWS:
        cells = []
        summary['t1'][key] = {'expected': exp, 'source': src}
        for mk, col, m in METRICS:
            per = {}
            ds = {}
            for b in BASES:
                ids, d = t1_cell(root, src, tok, reft, b, col)
                if ids is None:
                    continue
                per[b] = boot_mean(d)
                ds[b] = dict(zip(ids, d))
            summary['t1'][key].setdefault(mk, {})['per_base'] = per
            app[mk].append((label, per))
            if not all(b in per for b in LISTEN):
                cells.append('待 197' if src == '197' else '待資料')
                continue
            common = sorted(set.intersection(*[set(ds[b]) for b in LISTEN]))
            M = np.array([[ds[b][i] for i in common] for b in LISTEN])
            avg_ids = np.nanmean(M, axis=0)
            s = boot_mean(avg_ids)
            s['per_base_mean'] = [float(np.nanmean(M[j])) for j in range(len(LISTEN))]
            kk = sum(1 for b in LISTEN if per[b] and np.sign(per[b]['mean']) == np.sign(s['mean'])
                     and (per[b]['lo'] > m or per[b]['hi'] < -m))
            v = verdict(s, m, exp)
            s.update({'k_same_sign_big': kk, 'verdict': v})
            summary['t1'][key][mk]['main'] = s
            if exp in ('U', 'UW') and v == '**方向錯**':
                R1[mk] += 1
            if exp == 'W' and v == '沒反應':
                R2[mk] += 1
            cells.append(f'{fmt(s["mean"])}{mark(s, m)} ｜{kk}/6｜{v}')
        L.append(f'| {label} | {EXPECT_TXT[exp]} | ' + ' | '.join(cells) + ' |')
    L += ['', '## R1／R2（預先登錄的兩條讀法，表 1 主表）', '',
          '| 指標 | R1 方向錯格數（「不變」「不變或變差」列） | R2 盲點格數（「變差」列判「沒反應」） |', '|---|---|---|']
    for mk, _, _ in METRICS:
        L.append(f'| {mk} | {R1[mk]} | {R2[mk]} |')
    summary['t1_R1'], summary['t1_R2'] = R1, R2

    # rubberband processing floor: ps_*/ts_* vs rb_id
    L += ['', '## rb_id 處理底檢查（rb_id 若超過 m，移調／變速改對 rb_id 配對再報）', '',
          '| 指標 | rb_id（6 底平均 d） | 是否超過 m |', '|---|---|---|']
    trig = {}
    for mk, col, m in METRICS:
        s = summary['t1']['rb_id'].get(mk, {}).get('main')
        if s is None:
            L.append(f'| {mk} | 待 197 | — |')
            continue
        trig[mk] = bool(s['lo'] > m or s['hi'] < -m)
        L.append(f'| {mk} | {fmt(s["mean"])}{mark(s, m)} | {"是" if trig[mk] else "否"} |')
    summary['rb_id_trigger'] = trig
    L += ['', '移調／變速對 rb_id 配對（無論是否觸發都列出）：', '',
          '| 處理 vs rb_id | ' + ' | '.join(m[0] for m in METRICS) + ' |', '|---|' + '---|' * len(METRICS)]
    for key in ['ps_p1', 'ps_m1', 'ps_p2', 'ps_m2', 'ts_090', 'ts_110']:
        row = next(r for r in T1_ROWS if r[0] == key)
        cells = []
        summary['t1'][key]['vs_rb_id'] = {}
        for mk, col, m in METRICS:
            ds = {}
            for b in LISTEN:
                ids, d = t1_cell(root, '197', key, 'rb_id', b, col)
                if ids is None:
                    break
                ds[b] = dict(zip(ids, d))
            if len(ds) < len(LISTEN):
                cells.append('待 197')
                continue
            common = sorted(set.intersection(*[set(ds[b]) for b in LISTEN]))
            avg = np.nanmean(np.array([[ds[b][i] for i in common] for b in LISTEN]), axis=0)
            s = boot_mean(avg)
            v = verdict(s, m, row[2])
            s['verdict'] = v
            summary['t1'][key]['vs_rb_id'][mk] = s
            cells.append(f'{fmt(s["mean"])}{mark(s, m)} {v}')
        L.append(f'| {row[1]} | ' + ' | '.join(cells) + ' |')

    L += ['', '# 附表：逐底（9 底；每格 mean 記號（變高比例%））；CI／SD／n 見 summary.json', '']
    for mk, col, m in METRICS:
        L += [f'## {mk}', '', '| 處理 | ' + ' | '.join(b.replace('ma_', '') for b in BASES) + ' |', '|---|' + '---|' * len(BASES)]
        for label, per in app[mk]:
            cs = []
            for b in BASES:
                s = per.get(b)
                cs.append('待' if s is None else f'{fmt(s["mean"])}{mark(s, m)} ({100 * s["up"]:.0f}%)')
            L.append(f'| {label} | ' + ' | '.join(cs) + ' |')
        L.append('')
    return L


# ───────────────────────── table 2 ─────────────────────────
def table2(root, summary):
    exc = json.loads((EV / 'midi_realpiece_123/excerpts.json').read_text())
    perf = {x['id']: x['midi'] for x in exc}
    pc = root / 'per_clip'
    L = ['# 197 表 2：MIDI 合成音樂（音符層）', '',
         f'逐段配對差 d = 處理後 − 同段 base。A 區以 MAESTRO 演奏為 cluster（bootstrap {NB}、種子 {SEED}），CMI-RM 用空 prompt。'
         '記號與判讀同表 1。', '', '## A 區（MAESTRO 150 段 × 2 SoundFont）', '']
    base = read_tsv(pc / 'T2A_base.tsv')
    summary['t2a'] = {}
    R1 = {m[0]: 0 for m in METRICS}
    R2 = {m[0]: 0 for m in METRICS}
    L += ['| 處理 | 預期 | SoundFont | ' + ' | '.join(m[0] for m in METRICS) + ' |', '|---|---|---|' + '---|' * len(METRICS)]
    for key, label, exp in T2A_ROWS:
        t = read_tsv(pc / f'T2A_{key}.tsv')
        summary['t2a'][key] = {'expected': exp}
        for sf in ('generaluser', 'fluidr3', 'combined'):
            cells = []
            for mk, col, m in METRICS:
                if t is None or base is None:
                    cells.append('待 197')
                    continue
                ids = [i for i in base if i in t and (sf == 'combined' or i.endswith('_' + sf))]
                if not ids:
                    cells.append('待 197')
                    continue
                d = np.array([t[i][col] - base[i][col] for i in ids])
                cl = [perf.get(i.rsplit('_', 1)[0], i) for i in ids]
                s = boot_mean(d, cl)
                v = verdict(s, m, exp)
                s['verdict'] = v
                summary['t2a'][key].setdefault(sf, {})[mk] = s
                if sf == 'combined':
                    if exp in ('U', 'UW') and v == '**方向錯**':
                        R1[mk] += 1
                    if exp == 'W' and v == '沒反應':
                        R2[mk] += 1
                cells.append(f'{fmt(s["mean"])}{mark(s, m)} {v}')
            L.append(f'| {label} | {EXPECT_TXT[exp]} | {sf} | ' + ' | '.join(cells) + ' |')
    # soundfont swap
    L += ['', '換 SoundFont（generaluser base → fluidr3 base，無法判斷，只報讀數）：', '',
          '| ' + ' | '.join(m[0] for m in METRICS) + ' |', '|' + '---|' * len(METRICS)]
    cells = []
    if base is not None:
        for mk, col, m in METRICS:
            ids = [i[:-len('_generaluser')] for i in base if i.endswith('_generaluser') and i[:-len('_generaluser')] + '_fluidr3' in base]
            d = np.array([base[i + '_fluidr3'][col] - base[i + '_generaluser'][col] for i in ids])
            s = boot_mean(d, [perf.get(i, i) for i in ids])
            summary['t2a'].setdefault('sf_swap', {})[mk] = s
            cells.append(f'{fmt(s["mean"])}{mark(s, m)}')
    else:
        cells = ['待 197'] * len(METRICS)
    L.append('| ' + ' | '.join(cells) + ' |')
    L += ['', f'A 區 R1（方向錯，合併列）：{R1}；R2（沒反應）：{R2}', '']
    summary['t2a_R1'], summary['t2a_R2'] = R1, R2

    # ---- B ----
    L += ['## B 區（384 段合成和弦進行；段與段之間配對）', '']
    B = read_tsv(pc / 'T2B_all.tsv')
    if B is None:
        L.append('待 197')
        return L
    meta = {}
    for i in B:
        p = i.split('_')
        meta[i] = dict(tonic=p[0], mode=p[1], oct=p[2], voc=p[3], bank=p[4])

    def key(i, **over):
        m = dict(meta[i])
        m.update(over)
        return f"{m['tonic']}_{m['mode']}_{m['oct']}_{m['voc']}_{m['bank']}"

    def comp(name, sel, mk_other, exp='N'):
        """sel: list of ids (treated); other id computed by mk_other(i)"""
        row = {}
        for mk, col, m in METRICS:
            pairs = [(i, mk_other(i)) for i in sel if mk_other(i) in B]
            d = np.array([B[a][col] - B[b][col] for a, b in pairs])
            cl = [f"{meta[a]['tonic']}_{meta[a]['mode']}" for a, b in pairs]
            row[mk] = boot_mean(d, cl)
        summary.setdefault('t2b', {})[name] = row
        return row

    comps = []
    for voc in ('up', 'down', 'rep'):
        sel = [i for i in B if meta[i]['voc'] == voc]
        comps.append((f'{voc} − block', comp(f'{voc}-block', sel, lambda i: key(i, voc='block'))))
    sel = [i for i in B if meta[i]['voc'] == 'up']
    comps.append(('up − rep', comp('up-rep', sel, lambda i: key(i, voc='rep'))))
    comps.append(('up − down', comp('up-down', sel, lambda i: key(i, voc='down'))))
    sel = [i for i in B if meta[i]['oct'] == 'o5']
    comps.append(('o5 − o4（調還是音高）', comp('o5-o4', sel, lambda i: key(i, oct='o4'))))
    sel = [i for i in B if meta[i]['mode'] == 'min']
    comps.append(('小調 − 大調（進行不同，無法判斷）', comp('min-maj', sel, lambda i: key(i, mode='maj'))))
    L += ['| 比較 | n | ' + ' | '.join(m[0] for m in METRICS) + ' |', '|---|---|' + '---|' * len(METRICS)]
    for name, row in comps:
        n = row['A1']['n'] if row.get('A1') else 0
        L.append(f'| {name} | {n} | ' + ' | '.join((f'{fmt(row[mk]["mean"])}{mark(row[mk], m)}' if row[mk] else 'n/a') for mk, _, m in METRICS) + ' |')
    # tonic curves
    tonics = []
    for i in B:
        if meta[i]['tonic'] not in tonics:
            tonics.append(meta[i]['tonic'])
    L += ['', '### 主音曲線（每個主音的平均，減 12 主音總平均；大小調、o4／o5 分開）', '']
    summary['t2b_tonic'] = {}
    for mk, col, m in METRICS:
        L += [f'**{mk}**', '', '| 調式／八度 | ' + ' | '.join(tonics) + ' | 全距 |', '|---|' + '---|' * (len(tonics) + 1)]
        curves = {}
        for mode in ('maj', 'min'):
            for oc in ('o4', 'o5'):
                vals = []
                for t in tonics:
                    v = [B[i][col] for i in B if meta[i]['tonic'] == t and meta[i]['mode'] == mode and meta[i]['oct'] == oc]
                    vals.append(float(np.nanmean(v)) if v else float('nan'))
                vals = np.array(vals)
                curves[(mode, oc)] = vals
                dev = vals - np.nanmean(vals)
                L.append(f'| {mode} {oc} | ' + ' | '.join(f'{x:+.2f}' for x in dev) + f' | {np.nanmax(vals) - np.nanmin(vals):.2f} |')
        rr = {}
        for mode in ('maj', 'min'):
            a, b = curves[(mode, 'o4')], curves[(mode, 'o5')]
            r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and not np.isnan(a).any() and not np.isnan(b).any() else float('nan')
            rr[mode] = {'pearson_o4_o5': r, 'range_o4': float(np.nanmax(a) - np.nanmin(a)), 'range_o5': float(np.nanmax(b) - np.nanmin(b)),
                        'shift_o5_minus_o4': float(np.nanmean(b) - np.nanmean(a))}
        summary['t2b_tonic'][mk] = rr
        L += ['', f'o4／o5 兩條曲線 Pearson r：大調 {rr["maj"]["pearson_o4_o5"]:.2f}、小調 {rr["min"]["pearson_o4_o5"]:.2f}；'
                  f'o5 整體平移：大調 {rr["maj"]["shift_o5_minus_o4"]:+.3f}、小調 {rr["min"]["shift_o5_minus_o4"]:+.3f}', '']
    return L


def main():
    global NB
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(EV / 'attack_metric_matrix_197'))
    ap.add_argument('--out-prefix', default='attack_metric_matrix_197')
    ap.add_argument('--nboot', type=int, default=NB)
    a = ap.parse_args()
    NB = a.nboot
    root = Path(a.root)
    RES.mkdir(parents=True, exist_ok=True)
    summary = {'root': str(root), 'seed': SEED, 'nboot': NB, 'listen': LISTEN, 'm': {m[0]: m[2] for m in METRICS}}
    t1 = table1(root, summary)
    t2 = table2(root, summary)
    (RES / f'{a.out_prefix}_table1.md').write_text('\n'.join(t1) + '\n')
    (RES / f'{a.out_prefix}_table2.md').write_text('\n'.join(t2) + '\n')
    (RES / f'{a.out_prefix}_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print('wrote', RES / f'{a.out_prefix}_table1.md')


if __name__ == '__main__':
    main()
