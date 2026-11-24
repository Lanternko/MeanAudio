#!/usr/bin/env python3
"""201b ODAQ 客觀改變量（199 的定義套到 201a 的 20 個音樂 trial）。
設計：docs/experiments/attack_change_amount_199_20261125.md、paper_closing_evidence_200_202_20261125.md §1。
x = 該 trial 的 Ref；y = 同 trial 的各處理條件（LP3.5、LP7、Q1–Q5）；另做同一批 Ref 的尺（identity、mix_r*、wn_snr*、other_clip）。
--band sr16：全部轉 16 kHz 單聲道、鎖 −30 LUFS（199 的 P.lvl30）；--band full：全部 48 kHz 單聲道、嵌入層 CLAP 直接吃 48 kHz、MERT 48→24 kHz（全頻帶補充，見 meta）；訊號層算整段，嵌入層 10 秒片段 = 一個窗（ODAQ 片段正好 10 秒）。
只 import 199 / 188 的函式，不改它們。
  python scripts/analysis/odaq_change_amount_201b.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # GPU 被 queue job 吃滿（30 GB），CLAP／MERT 在 CPU 上跑
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ.pop('MKL_NUM_THREADS', None)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import argparse
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'eval'))
import attack_change_amount_199 as A199  # noqa: E402
import attack_change_amount_tables as T199  # noqa: E402

P = A199.P
RES = HERE.parents[1] / 'docs/experiments/results'
MAN = Path('/mnt/seagate/benchmarks_r2/manifests/odaq.tsv')
MUSIC = ['LP', 'TM', 'UN', 'SH', 'PE']
SEED = 20261125
NB = 5000
SR = 16000
TREAT = ['LP3.5', 'LP7', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
R_LIST, WN_LIST = A199.R_LIST, A199.WN_LIST
MIX, WN = T199.MIX, T199.WN
QS = ['rho', 'mel', 'd_clap', 'd_mert']
EMB = ['d_clap', 'd_mert']


class EmbCPU(A199.Emb):
    def load(self):
        import torch
        self.torch = torch
        import eval_metrics as EM
        from muq_gen_steer_179_score import standard_hf_cache
        with standard_hf_cache():
            self.clap = EM.load_clap()
            import metric_win_tie_133 as W
            self.W = W
            self.feat, self.mert = W.load_mert('cpu')

    def mert_fn(self, arrs):
        return self.W.embed_mert_batch([a for a in arrs], self.feat, self.mert, 'cpu')


def embed_var(X, ws):
    """長度不一：>200000 樣本（12.5 s）取前後兩個 10 s 窗（199 規則），否則取前 10 s；依窗數分組嵌入。"""
    ws = [w if len(w) > 200000 else w[:A199.WIN] for w in ws]
    out = [None] * len(ws)
    for nw in (1, 2):
        ks = [k for k, w in enumerate(ws) if len(X.windows(w)) == nw]
        if not ks:
            continue
        e = X.embed([ws[k] for k in ks])
        for j, k in enumerate(ks):
            out[k] = {'clap': e['clap'][j], 'mert': e['mert'][j]}
    return out


def dist_var(er, ey, q):
    return np.array([float(A199.dist(a[q][None], b[q][None])[0]) for a, b in zip(er, ey)])


SR48 = 48000
LAG48 = 28800          # +-0.6 s
WIN48 = 480000


# ═════════ band configs ═════════
class Sr16:
    name = 'sr16'
    lag_ok = None

    def load(self, path):
        import librosa
        import soundfile as sf
        x, sr = sf.read(path, dtype='float32', always_2d=True)
        x = x.mean(1)
        if sr != SR:
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        return P.lvl30(x)[0]

    lvl30 = staticmethod(lambda w: P.lvl30(w)[0])
    mix = staticmethod(lambda x, o, r: P.mix(x, o, r))
    signal_layer = staticmethod(A199.signal_layer)

    def emb(self):
        return EmbCPU()


# ---- full band (48 kHz) ----
_MELFB48 = None


def melfb48():
    global _MELFB48
    if _MELFB48 is None:
        import librosa
        _MELFB48 = librosa.filters.mel(sr=SR48, n_fft=2048, n_mels=64, fmax=24000).astype(np.float64)
    return _MELFB48


def logmel48(s):
    from numpy.lib.stride_tricks import sliding_window_view
    s = np.asarray(s, np.float64)
    s = s / (math.sqrt(float(np.mean(s * s))) + 1e-12)
    fr = sliding_window_view(s, 1200)[::480] * np.hanning(1200)
    pw = np.abs(np.fft.rfft(fr, 2048, axis=1)) ** 2
    return 10 * np.log10(np.maximum(pw @ melfb48().T, 1e-10))


def signal_layer48(x, y):
    rho, lag, m0, m1 = A199.align(x, y, lag_max=LAG48)
    xs = np.asarray(x, np.float64)[m0:m1]
    ys = np.asarray(y, np.float64)[m0 + lag:m1 + lag]
    if len(xs) < 800 or float(np.mean(xs * xs)) < 1e-14 or float(np.mean(ys * ys)) < 1e-14:
        return rho, lag, float('nan')
    dx, dy = logmel48(xs), logmel48(ys)
    fl = dx.max() - 80.0
    dx, dy = np.maximum(dx, fl), np.maximum(dy, fl)
    return rho, lag, float(np.mean(np.abs(dx - dy)))


def lufs48(w):
    import pyloudnorm
    return pyloudnorm.Meter(SR48).integrated_loudness(np.asarray(w, np.float64)[:, None])


def lvl30_48(w, target=-30.0, cap=0.999):
    """與 188 的 to_lufs 相同規則，只換 48 kHz meter。"""
    w = np.asarray(w, np.float64)
    lufs = lufs48(w)
    peak = float(np.max(np.abs(w))) if w.size else 0.0
    if not math.isfinite(lufs) or peak == 0.0:
        g = 0.0
    else:
        g = target - lufs
        if peak * 10 ** (g / 20) > cap:
            g = 20 * math.log10(cap / peak)
    return np.clip(w * 10 ** (g / 20), -1.0, 1.0).astype(np.float32)


def mix48(base30, ov, r):
    ov = P.fit(ov, len(base30)) if len(ov) != len(base30) else ov
    o = np.asarray(ov, np.float64) * 10 ** ((-30.0 - r - lufs48(ov)) / 20)
    return lvl30_48(np.asarray(base30, np.float64) + o)


class EmbFull(EmbCPU):
    @staticmethod
    def windows(w):
        w = np.asarray(w, np.float32)
        if len(w) > 600000:                       # > 12.5 s @48k：前後兩個 10 s 窗
            return [w[:WIN48], w[-WIN48:]]
        return [w]

    def _prep_clap(self, w):
        x = np.asarray(w, np.float32)[:WIN48]
        return (np.round(np.clip(x, -1, 1) * 32767.0) / 32767.0).astype(np.float32)

    def _prep_mert(self, w):
        import librosa
        return librosa.resample(np.asarray(w, np.float32), orig_sr=SR48, target_sr=24000).astype(np.float32)


class Full:
    name = 'full'
    resampled = []

    def load(self, path):
        import librosa
        import soundfile as sf
        x, sr = sf.read(path, dtype='float32', always_2d=True)
        x = x.mean(1)
        if sr != SR48:
            self.resampled.append((str(path), int(sr)))
            x = librosa.resample(x, orig_sr=sr, target_sr=SR48)
        return lvl30_48(x)

    lvl30 = staticmethod(lvl30_48)
    mix = staticmethod(mix48)
    signal_layer = staticmethod(signal_layer48)

    def emb(self):
        return EmbFull()


def hf_fraction_db(w, f0=8000.0):
    """整段功率譜 f >= f0 的能量占總能量比（dB）。"""
    sp = np.abs(np.fft.rfft(np.asarray(w, np.float64))) ** 2
    fr = np.fft.rfftfreq(len(w), 1.0 / SR48)
    return float(10 * np.log10(max(sp[fr >= f0].sum(), 1e-30) / max(sp.sum(), 1e-30)))


def run_band(B, man, meta, trials, s201):
    n = len(trials)
    print(f'===== band {B.name} =====', flush=True)
    W = {(t, c): B.load(man[(t, c)]) for t in trials for c in ['Ref'] + TREAT}
    lens = {len(v) for v in W.values()}
    print('lengths', lens, flush=True)
    extra = {}
    if B.name == 'full':
        extra['resampled_to_48k'] = list(B.resampled)
        print('resampled files (not 48k):', len(B.resampled), flush=True)
        extra['hf_gt8k_energy_db'] = {c: [hf_fraction_db(W[(t, c)]) for t in trials] for c in ['Ref', 'LP7', 'LP3.5']}
        extra['hf_gt8k_energy_db_median'] = {c: float(np.median(v)) for c, v in extra['hf_gt8k_energy_db'].items()}
        print('HF>8k energy dB median', extra['hf_gt8k_energy_db_median'], flush=True)
    partner = A199.derange(n)
    xs = [W[(t, 'Ref')] for t in trials]
    cells = {}
    for c in TREAT:
        cells[c] = [W[(t, c)] for t in trials]
        assert all(len(cells[c][k]) == len(xs[k]) for k in range(n)), c
    cells['identity'] = list(xs)
    cells['other_clip'] = [B.lvl30(P.fit(xs[partner[k]], len(xs[k]))) for k in range(n)]
    for r in R_LIST:
        cells[A199.rname(r)] = [B.mix(xs[k], xs[partner[k]], r) for k in range(n)]
    for snr in WN_LIST:
        ys = []
        for k, t in enumerate(trials):
            w = xs[k]
            rng = P.rng_for('odaq', t, 'wn', snr)
            pw = float(np.mean(np.asarray(w, np.float64) ** 2))
            ys.append(B.lvl30(np.asarray(w, np.float64) + rng.standard_normal(len(w)) * math.sqrt(pw / 10 ** (snr / 10))))
        cells[f'wn_snr{snr}'] = ys

    X = B.emb()
    X.load()
    ref = embed_var(X, xs)
    full_len = len(xs[0])
    pre = [X._prep_clap(w) for w in xs[:8] if len(w) == full_len][:8]
    pf = np.concatenate([X.clap_fn([a], per_file=True) for a in pre])
    pf /= np.linalg.norm(pf, axis=1, keepdims=True)
    bt = X.clap_fn(pre)
    bt = bt / np.linalg.norm(bt, axis=1, keepdims=True)
    clap_gap = float(np.abs(pf - bt).max())
    print('CLAP batch-vs-perfile max abs diff', clap_gap, flush=True)

    per = {}
    for c, ys in cells.items():
        sig = [B.signal_layer(x, y) for x, y in zip(xs, ys)]
        e = embed_var(X, ys)
        per[c] = {'rho': np.array([s[0] for s in sig]), 'mel': np.array([s[2] for s in sig]),
                  'd_clap': dist_var(ref, e, 'clap'), 'd_mert': dist_var(ref, e, 'mert')}
        print(c, {q: round(float(np.nanmedian(per[c][q])), 4) for q in QS}, flush=True)

    idx = np.random.default_rng(SEED).integers(0, n, size=(NB, n))

    def stat(v):
        v = np.asarray(v, float)
        b = np.nanmedian(v[idx], axis=1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        return {'median': float(np.nanmedian(v)), 'lo': float(lo), 'hi': float(hi)}

    med = {c: {q: float(np.nanmedian(per[c][q])) for q in QS} for c in per}
    full = {c: {q: stat(per[c][q]) for q in QS} for c in per}
    gates = T199.gates({('odaq', c): med[c] for c in med}, ['odaq'])
    ok_q = {q: gates[q]['pass'] for q in QS}
    ladder_meds = {q: [med[c][q] for c in MIX] for q in QS}
    other = med['other_clip']
    m12, m0 = med['mix_r12'], med['mix_r0']

    cond_out = {}
    for c in TREAT + ['identity', 'other_clip']:
        d = {'n': n, 'stats': full[c], 'C': {q: med[c][q] / other[q] for q in QS}, 'equiv_mix': {}}
        for q in QS:
            v, why = T199.interp_mix(med[c][q], ladder_meds[q])
            d['equiv_mix'][q] = {'r': v, 'note': why}
        qs = [q for q in EMB if ok_q[q]]
        if not qs:
            d['class'] = '無法分類'
        elif any(med[c][q] >= m0[q] for q in qs):
            d['class'] = '換成另一段音樂'
        elif all(med[c][q] <= m12[q] for q in qs):
            d['class'] = '小改變'
        else:
            d['class'] = '中等'
        cond_out[c] = d
    by_method = {}
    for m in MUSIC:
        ks = [k for k, t in enumerate(trials) if meta[t] == m]
        by_method[m] = {c: {q: float(np.median(per[c][q][ks])) for q in QS} for c in TREAT}

    gname = {'LP3.5': 'LP35 錨點（3.5 kHz）', 'LP7': 'LP70 錨點（7 kHz）'}
    for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
        gname[q] = f'音樂池 {q}（描述）'
    for c in TREAT:
        g = s201['groups'].get(gname[c])
        if g:
            a1 = g['scorers']['A1 MI']['norm']
            cond_out[c]['mushra'] = {'mean': g['human']['mean'], 'lo': g['human']['lo'], 'hi': g['human']['hi']}
            cond_out[c]['a1_mi'] = {'delta': a1['delta'], 'lo': a1['lo'], 'hi': a1['hi'], 'class': a1['class']}

    out = {'seed': SEED, 'nboot': NB, 'n_trials': n, 'trials': trials, 'clap_batch_vs_perfile_maxdiff': clap_gap,
           'acceptance': gates, 'ladders': {c: full[c] for c in MIX + WN + ['identity', 'other_clip']},
           'conditions': cond_out, 'q_by_method_median': by_method}
    out.update(extra) if B.name == 'full' else None
    return out, full, gates, cond_out, by_method, m12, m0


def fm(s):
    v = s['median']
    t = f'{v:.3f}' if abs(v) >= 0.01 or v == 0 else f'{v:.4f}'
    return f'{t} [{s["lo"]:.3f},{s["hi"]:.3f}]'


def eq(d):
    def one(q):
        r = d['equiv_mix'][q]
        return f'{r["r"]:.1f}' if r['r'] is not None else r['note']
    return f'{one("d_clap")}／{one("d_mert")}'


def md_band(label, desc, res):
    out, full, gates, cond_out, by_method, m12, m0 = res
    n = out['n_trials']
    L = [f'## 版本：{label}', '', desc, '',
         '### 驗收（199 §4）', '', '| 量 | identity ≤ 1/10 mix_r24 | 混音階梯嚴格單調 | 白噪階梯嚴格單調 | other_clip > mix_r0 > mix_r12 | 結論 |', '|---|---|---|---|---|---|']
    for q in QS:
        x = gates[q]
        L.append(f'| {q} | {"過" if x["identity"] else "不過"} | {"過" if x["mix_mono"] else "不過"} | {"過" if x["wn_mono"] else "不過"} | '
                 f'{"過" if x["order"] else "不過"} | {"通過" if x["pass"] else "整欄剔除"} |')
    L += ['', '### 階梯表', '', '| 階梯點 | rho | mel (dB) | d_clap | d_mert |', '|---|---|---|---|---|']
    for c in ['identity'] + MIX + WN + ['other_clip']:
        L.append(f'| {c} | ' + ' | '.join(fm(full[c][q]) for q in QS) + ' |')
    L += ['', '### 逐條件表（20 trial 合併）', '',
          'C＝中位數 ÷ other_clip 中位數；等效混入＝在 mix 階梯上內插的 r（LU，小於原版多少的無關音樂；數字越小越像換掉）。'
          'MUSHRA Δ＝201a 人類 條件−Ref；A1 MI Δ＝201a 的 native_norm。Q1–Q5 是各方法自己的劑量（不同方法的 Qk 不同義），人類／A1 列為 201a 的音樂池描述值。', '',
          '| 條件 | rho | mel | d_clap (C) | d_mert (C) | 等效混入 clap／mert | 分類 | MUSHRA Δ [CI] | A1 MI Δ [CI] |', '|---|---|---|---|---|---|---|---|---|']
    for c in TREAT + ['other_clip']:
        d = cond_out[c]
        s = d['stats']
        hum = d.get('mushra')
        a1 = d.get('a1_mi')
        L.append(f'| {c} | {fm(s["rho"])} | {fm(s["mel"])} | {fm(s["d_clap"])} ({d["C"]["d_clap"]:.2f}) | {fm(s["d_mert"])} ({d["C"]["d_mert"]:.2f}) | '
                 f'{eq(d)} | {d["class"]} | '
                 + (f'{hum["mean"]:+.1f} [{hum["lo"]:+.1f},{hum["hi"]:+.1f}]' if hum else '—') + ' | '
                 + (f'{a1["delta"]:+.3f} [{a1["lo"]:+.3f},{a1["hi"]:+.3f}]' if a1 else '—') + ' |')
    L += ['', '### 逐方法 Q（描述，每格 4 trial 的中位數；d_clap／d_mert）', '',
          '| 方法 | ' + ' | '.join(TREAT) + ' |', '|---|' + '---|' * len(TREAT)]
    for m in MUSIC:
        L.append(f'| {m} | ' + ' | '.join(f'{by_method[m][c]["d_clap"]:.3f}／{by_method[m][c]["d_mert"]:.3f}' for c in TREAT) + ' |')
    L += ['', f'分類門檻（中位數）：mix_r12＝clap {m12["d_clap"]:.3f}／mert {m12["d_mert"]:.3f}；mix_r0＝clap {m0["d_clap"]:.3f}／mert {m0["d_mert"]:.3f}。', '']
    return L


def md_compare(R):
    L = ['## 對照：sr16 vs full（中位數；等效混入 r 為 clap／mert，LU）', '',
         '| 條件 | 版本 | rho | mel (dB) | d_clap | d_mert | 等效混入 r | 分類 | MUSHRA Δ | A1 MI Δ |', '|---|---|---|---|---|---|---|---|---|---|']
    for c in TREAT:
        for b in ('sr16', 'full'):
            d = R[b][3][c]
            s = d['stats']
            hum, a1 = d.get('mushra'), d.get('a1_mi')
            L.append(f'| {c} | {b} | {s["rho"]["median"]:.4f} | {s["mel"]["median"]:.3f} | {s["d_clap"]["median"]:.4f} | {s["d_mert"]["median"]:.4f} | '
                     f'{eq(d)} | {d["class"]} | '
                     + (f'{hum["mean"]:+.1f}' if hum and b == 'sr16' else (f'{hum["mean"]:+.1f}' if hum else '—')) + ' | '
                     + (f'{a1["delta"]:+.3f}' if a1 else '—') + ' |')
    L.append('')
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--band', choices=['sr16', 'full', 'both'], default='both')
    args = ap.parse_args()
    man, meta = {}, {}
    with open(MAN, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            e = json.loads(r['extra'])
            man[(e['trial'], e['condition'])] = r['src']
            meta[e['trial']] = e['method']
    trials = [t for m in MUSIC for t in sorted(t for t in meta if meta[t] == m)]
    assert len(trials) == 20
    print('trials', len(trials), flush=True)
    P.POOL = ThreadPoolExecutor(8)
    A199.P.POOL = P.POOL
    s201 = json.loads((RES / 'odaq_human_anchor_201_summary.json').read_text())
    bands = ['sr16', 'full'] if args.band == 'both' else [args.band]
    R = {}
    for b in bands:
        R[b] = run_band(Sr16() if b == 'sr16' else Full(), man, meta, trials, s201)
    jp = RES / 'odaq_change_amount_201b_summary.json'
    summ = {}
    if jp.exists():
        old = json.loads(jp.read_text())
        if 'meta' in old:
            summ = old
    for b in bands:
        summ[b] = R[b][0]
    meta_o = summ.get('meta', {})
    if 'full' in R:
        o = R['full'][0]
        meta_o.update({'hf_gt8k_energy_db_median': o['hf_gt8k_energy_db_median'],
                       'hf_gt8k_energy_db_per_trial': o['hf_gt8k_energy_db'],
                       'hf_note': 'LP7 與 Ref 在 48 kHz −30 LUFS 波形上 ≥8 kHz 的功率占整段總功率比（dB，20 段中位數）；LP3.5 為額外對照',
                       'full_resampled_to_48k': o['resampled_to_48k']})
    summ['meta'] = meta_o
    jp.write_text(json.dumps(summ, indent=1, default=float, ensure_ascii=False))

    L = ['# 201b ODAQ 客觀改變量（2026-11-25）', '',
         '20 個音樂 trial（ODAQ，10 秒）；x＝Ref，y＝處理條件；每格＝逐 trial 中位數［95% CI，trial bootstrap '
         f'{NB} 次、種子 {SEED}］。定義與分類同 199 設計 §2–§5。單一底（ODAQ），不做多數決。兩個版本並列：sr16＝16 kHz（Nyquist 8 kHz）流程；full＝48 kHz 全頻帶流程。', '']
    if 'full' in R:
        h = meta_o['hf_gt8k_energy_db_median']
        L += [f'8 kHz 以上能量占比（48 kHz 波形，中位數）：Ref {h["Ref"]:.1f} dB、LP7 {h["LP7"]:.1f} dB、LP3.5 {h["LP3.5"]:.1f} dB。', '']
    if len(R) == 2:
        L += md_compare(R)
    descs = {'sr16': '16 kHz 單聲道、−30 LUFS；嵌入層沿用 199。', 'full': '48 kHz 單聲道、−30 LUFS（48 kHz meter）；rho／mel 在 48 kHz（lag ±28800、mel 64 頻帶 1200/480、n_fft 2048、fmax 24 kHz）；CLAP 直接吃 48 kHz；MERT 48→24 kHz。'}
    for b in bands:
        L += md_band(b, descs[b], R[b])
    (RES / 'odaq_change_amount_201b_table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
