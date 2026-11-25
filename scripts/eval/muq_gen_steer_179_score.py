#!/usr/bin/env python
"""179 scoring + analysis: manipulation check, MuQ-Eval, AES, CLAP, level, and the
post-hoc lowpass reference curve; writes <root>/summary.json and the results docs.

  python scripts/eval/muq_gen_steer_179_score.py --root <arms root> --tsv <tsv>
"""
import argparse
import contextlib
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/eval'), str(HERE)]
import muq_gen_steer_179_common as C  # noqa: E402
import eval_metrics as em  # noqa: E402
import muq_lowpass_probe_20261028 as lpbase  # noqa: E402
import newmetrics_common as NC  # noqa: E402

# One HF cache for every scorer. newmetrics_common.setup_env() (run by the MuQ loader)
# would otherwise point HF_HOME at the shared Seagate cache, while AES and the CLAP
# text encoder are cached in the default one; huggingface_hub and transformers each
# freeze the cache path at import time, so scope it once, before either import.
# models--zhudi2825--MuQ-Eval (A3a) is symlinked into the default cache.
NC.HF_HOME = Path.home() / '.cache/huggingface'
from newmetrics_muqeval import score_muqeval  # noqa: E402

CUTOFFS = (4000, 5000, 6000)
NFFT, HOP = 1024, 512


# ── manipulation features ────────────────────────────────
def band_from_array(m, sr):
    m = np.asarray(m, dtype=np.float64)
    if m.ndim > 1:
        m = m.mean(axis=1)
    if m.size < NFFT:
        m = np.pad(m, (0, NFFT - m.size))
    f, p = welch(m, fs=sr, window='hann', nperseg=NFFT, noverlap=HOP,
                 detrend=False, scaling='density')
    lo = f < 8000
    band = (f >= 4000) & (f < 8000)
    e_tot = float(p[lo].sum())
    e_band = float(p[band].sum())
    return {
        'band_ratio_db': 10.0 * math.log10((e_band + 1e-30) / (e_tot + 1e-30)),
        'centroid_hz': float((f[lo] * p[lo]).sum() / (e_tot + 1e-30)),
    }


def band_features(path):
    x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    return band_from_array(x, sr)


# ── per-arm scoring ──────────────────────────────────────
# The two scorer families read different HF caches: the MuQ weights live in the shared
# Seagate cache (newmetrics_common.HF_HOME), facebook/audiobox-aesthetics and the CLAP
# text encoder in the default one. huggingface_hub reads its cache constants at call
# time, so scope them per scorer instead of relying on whichever import ran first
# (that is what made AES fail with LocalEntryNotFoundError after MuQ's setup_env()).
@contextlib.contextmanager
def hf_cache(root):
    import huggingface_hub.constants as K
    root = Path(root)
    old = (K.HF_HOME, K.HF_HUB_CACHE)
    K.HF_HOME, K.HF_HUB_CACHE = str(root), str(root / 'hub')
    try:
        yield
    finally:
        K.HF_HOME, K.HF_HUB_CACHE = old


def standard_hf_cache():
    return hf_cache(Path.home() / '.cache/huggingface')


def muq_hf_cache():
    import newmetrics_common as NC
    return hf_cache(NC.HF_HOME)


def score_arm(gen_dir, rows, cache):
    cache = Path(cache)
    if cache.is_file():
        rec = C.read_json(cache)
        if rec and len(rec) == len(rows) and {r['id'] for r in rec} == {i for i, _ in rows}:
            return rec
    gen_dir = Path(gen_dir)
    paths = [gen_dir / f'{i}.flac' for i, _ in rows]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f'[FAIL] {len(missing)} missing clips in {gen_dir} (first {missing[0]})')

    recs = {i: {'id': i} for i, _ in rows}

    # CPU level first: it forks a pool, keep it away from CUDA-initialised state
    lv = em.score_level(paths)
    for p, d in lv.items():
        i = Path(p).stem
        recs[i]['lufs'] = d['lufs']
        recs[i]['silent'] = d['silent']

    with muq_hf_cache():
        for v in ('A1', 'A3a'):
            for (i, _), p in zip(rows, paths):
                s = score_muqeval(str(p), None, variant=v)
                recs[i][f'muq_{v}_MI'] = s['MI']
                recs[i][f'muq_{v}_TA'] = s['TA']
            print(f'[score] {gen_dir.name} muq {v} done', flush=True)

    with standard_hf_cache():
        aes, failed = em.score_aes(paths, batch_size=32)
        if failed:
            raise SystemExit(f'[FAIL] AES failed on {len(failed)} clips (first {next(iter(failed))})')
        for p, d in aes.items():
            i = Path(p).stem
            for k in ('PQ', 'CE', 'CU', 'PC'):
                recs[i][f'aes_{k}'] = d[k]
        print(f'[score] {gen_dir.name} aes done', flush=True)

        clap = em.score_clap(rows, gen_dir)
    for i, v in clap.items():
        recs[i]['clap'] = v
    print(f'[score] {gen_dir.name} clap done', flush=True)

    for (i, _), p in zip(rows, paths):
        recs[i].update(band_features(p))

    out = [recs[i] for i, _ in rows]
    bad = [r['id'] for r in out if any(isinstance(r.get(k), float) and not math.isfinite(r[k])
                                       for k in C.METRICS)]
    if bad:
        raise SystemExit(f'[FAIL] non-finite metrics on {len(bad)} clips (first {bad[0]})')
    C.write_json(cache, out)
    return out


def score_lowpass(a0_dir, rows, cache):
    cache = Path(cache)
    if cache.is_file():
        rec = C.read_json(cache)
        if rec and len(rec) == len(rows) * len(CUTOFFS):
            return rec
    out = []
    with muq_hf_cache():
        for fc in CUTOFFS:
            for i, _ in rows:
                x, sr = sf.read(str(Path(a0_dir) / f'{i}.flac'), dtype='float32', always_2d=True)
                y = lpbase.lp(x, sr, float(fc)).astype(np.float32)
                s1 = score_muqeval(y, sr, variant='A1')
                s3 = score_muqeval(y, sr, variant='A3a')
                r = {'id': i, 'cutoff': fc, 'muq_A1_MI': s1['MI'], 'muq_A1_TA': s1['TA'],
                     'muq_A3a_MI': s3['MI'], 'muq_A3a_TA': s3['TA'], **band_from_array(y, sr)}
                out.append(r)
            print(f'[score] lowpass {fc} done ({len(out)})', flush=True)
    C.write_json(cache, out)
    return out


# ── statistics ───────────────────────────────────────────
def boot_ci(d, n=2000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def boot_p_gt0(d, n=2000, seed=0):
    """One-sided bootstrap p for H0: mean <= 0 (p = fraction of resample means <= 0)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return max(float((m <= 0).mean()), 1.0 / n)


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, prev, adj = len(items), 0.0, {}
    for rank, (k, p) in enumerate(items):
        val = min(1.0, max(prev, p * (m - rank)))
        adj[k] = val
        prev = val
    return adj


def paired(ref, arm, key):
    idx = {r['id']: r for r in arm}
    a, b = [], []
    for r in ref:
        q = idx.get(r['id'])
        if q is None or r.get(key) is None or q.get(key) is None:
            continue
        a.append(float(q[key]))
        b.append(float(r[key]))
    return np.asarray(a) - np.asarray(b)


def delta_summary(d):
    return {'n': int(len(d)), 'mean': float(np.mean(d)), 'ci95': boot_ci(d),
            'n_up': int((d > 0).sum()), 'n_down': int((d < 0).sum()),
            'frac_up': float((d > 0).mean())}


# ── summary + write-up ───────────────────────────────────
def analyse(ref, arms, lowpass, meta):
    S = {'meta': meta, 'arms': {}, 'criteria': {}, 'holm': {}}
    S['reference'] = {'n': len(ref)}
    for k in ('band_ratio_db', 'centroid_hz', 'muq_A1_MI', 'aes_PQ'):
        vals = [float(r[k]) for r in ref if r.get(k) is not None]
        S['reference'][k] = float(np.mean(vals))
    for arm, recs in arms.items():
        S['arms'][arm] = {k: delta_summary(paired(ref, recs, k)) for k in C.METRICS}
        S['arms'][arm]['silent_n'] = int(sum(int(r['silent']) for r in recs))
        S['arms'][arm]['muq_A1_MI_p_gt0'] = boot_p_gt0(paired(ref, recs, 'muq_A1_MI'))

    for arm, recs in arms.items():
        if arm == 'A0':
            continue
        d_band = paired(ref, recs, 'band_ratio_db')
        d_muq = paired(ref, recs, 'muq_A1_MI')
        d_pq = paired(ref, recs, 'aes_PQ')
        db, dq, dp = delta_summary(d_band), delta_summary(d_muq), delta_summary(d_pq)
        manip = bool(db['mean'] <= -3.0 and db['ci95'][1] < 0.0)
        attack = bool(manip and dq['mean'] >= 0.10 and dq['ci95'][0] > 0.0
                      and dp['mean'] <= 0.0 and dp['ci95'][1] < 0.05)
        S['criteria'][arm] = {'manipulation': manip, 'attack': attack,
                              'band_mean': db['mean'], 'band_ci': db['ci95'],
                              'muq_mean': dq['mean'], 'muq_ci': dq['ci95'],
                              'pq_mean': dp['mean'], 'pq_ci': dp['ci95']}

    S['holm'] = holm({a: S['arms'][a]['muq_A1_MI_p_gt0'] for a in C.FAMILY if a in arms})

    lp = {}
    ref_idx = {r['id']: r for r in ref}
    for fc in CUTOFFS:
        rows = [r for r in lowpass if r['cutoff'] == fc]
        lp[f'lp{fc // 1000}k'] = {k: delta_summary(paired(ref, rows, k))
                                  for k in ('muq_A1_MI', 'muq_A1_TA', 'muq_A3a_MI',
                                            'muq_A3a_TA', 'band_ratio_db', 'centroid_hz')}
        lp[f'lp{fc // 1000}k']['n'] = len(rows)
    S['lowpass'] = lp
    S['verdict'] = {
        'any_attack': any(c['attack'] for c in S['criteria'].values()),
        'attack_arms': [a for a, c in S['criteria'].items() if c['attack']],
        'any_manipulation': any(c['manipulation'] for c in S['criteria'].values()),
        'manipulation_arms': [a for a, c in S['criteria'].items() if c['manipulation']],
    }
    return S


def fmt(v, nd=3, sign=True):
    return f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'


def write_results(S, path):
    arms = S['arms']
    crit = S['criteria']
    lp = S['lowpass']
    any_attack = S['verdict']['any_attack']
    any_manip = S['verdict']['any_manipulation']
    grade = 5 if any_attack else (3 if any_manip else 1)
    grade_txt = {5: '5/5 重大達成（建議採用為生成端 hack 證據）',
                 3: '3/5 一半一半（頻段推得動，但沒換到 MuQ）',
                 1: '1/5 失敗（生成端推不動頻段）'}[grade]
    refmean = arms.get('A0', {}).get('band_ratio_db')
    lines = ['# 179：生成端能不能把 MuQ 偏誤頻段（4–8 kHz）推下去、刷高 MuQ（2026-11-03）', '',
             '設計與預登錄判準：`docs/experiments/muq_gen_steer_179_20261103.md`。', '',
             '# 總評', '', f'**{grade_txt}**', '']
    if any_attack:
        a = S['verdict']['attack_arms']
        lines.append(f'核心結論：生成端攻擊成立——{", ".join(a)} 同時通過操弄、MuQ 與 PQ 三條，'
                     '只靠推論期引導就能用 MuQ 的分數換掉音質。')
    elif any_manip:
        a = S['verdict']['manipulation_arms']
        lines.append(f'核心結論：操弄成立（{", ".join(a)} 的 4–8 kHz 真的被推下去 ≥3 dB），'
                     '但 MuQ A1 MI 沒有跟著漲到 +0.10，生成端 hack 不成立。')
    else:
        lines.append('核心結論：沒有任何 arm 的 4–8 kHz 占比比 A0 低 ≥3 dB——文字與差值向量都推不動'
                     '這個頻段（與 D1 同類），生成端 hack 不成立。')
    lines += ['', '**關鍵數據與例子**', '',
              f'A0 絕對值（n={S["reference"]["n"]}）：4–8k 占比 '
              f'{fmt(S["reference"]["band_ratio_db"], 2)} dB、頻譜重心 '
              f'{S["reference"]["centroid_hz"]:.0f} Hz、MuQ A1 MI '
              f'{S["reference"]["muq_A1_MI"]:.3f}、AES PQ {S["reference"]["aes_PQ"]:.3f}。', '',
              '| arm | Δ 4–8k 占比 (dB) | Δ MuQ A1 MI | Δ AES PQ | 操弄 | 攻擊 |',
              '|---|---|---|---|---|---|']
    for arm in C.MAIN_ARMS:
        if arm not in arms:
            continue
        b, m, p = arms[arm]['band_ratio_db'], arms[arm]['muq_A1_MI'], arms[arm]['aes_PQ']
        c = crit.get(arm, {})
        lines.append(f'| {arm} | {fmt(b["mean"], 2)} [{fmt(b["ci95"][0], 2)}, {fmt(b["ci95"][1], 2)}] '
                     f'| {fmt(m["mean"], 3)} [{fmt(m["ci95"][0], 3)}, {fmt(m["ci95"][1], 3)}] '
                     f'| {fmt(p["mean"], 3)} [{fmt(p["ci95"][0], 3)}, {fmt(p["ci95"][1], 3)}] '
                     f'| {"✔" if c.get("manipulation") else "✘"} | {"✔" if c.get("attack") else "✘"} |')
    lines += ['', '後處理參考（A0 音檔直接 lowpass，不落地）：', '',
              '| 介入 | Δ 4–8k 占比 (dB) | Δ MuQ A1 MI | Δ AES PQ |', '|---|---|---|---|']
    for k, v in lp.items():
        lines.append(f'| {k} | {fmt(v["band_ratio_db"]["mean"], 2)} | '
                     f'{fmt(v["muq_A1_MI"]["mean"], 3)} | — |')
    holm = ', '.join(f'{a} p={S["holm"][a]:.3f}' for a in C.FAMILY if a in S['holm'])
    lines += ['', f'Holm（A1–A5，MuQ A1 MI 單尾 p）：{holm}。', '']
    if any_manip:
        best = max(S['verdict']['manipulation_arms'],
                   key=lambda a: arms[a]['muq_A1_MI']['mean'])
        lines.append(f'機制：推得最遠的 arm 是 {best}，Δ 頻段 '
                     f'{fmt(arms[best]["band_ratio_db"]["mean"], 2)} dB 換到 Δ MuQ '
                     f'{fmt(arms[best]["muq_A1_MI"]["mean"], 3)}；對照後處理曲線，'
                     '同樣的 dB 移動在 lowpass 上換到的分數見上表。')
    lines += ['', '**要你決定的事**', '']
    if any_attack:
        lines += ['1. 是否把這次的生成端證據寫進指標分析（MuQ 當 reward 會被 hack）。',
                  '2. 下一步：擴大 λ 網格或換頻段描述詞，確認劑量反應；或直接做 mel 梯度引導當上界。']
    elif any_manip:
        lines += ['1. 是否接受「頻段可推、但 MuQ 沒反應」為這次的結論（不做梯度引導）。',
                  '2. 若要追 MuQ 增益，下一步建議直接做 mel 域梯度引導，量它與 lowpass 曲線的距離。']
    else:
        lines += ['1. 是否把這條線收成「文字推不動頻段（同 D1）」，保留為負面結果。',
                  '2. 若要繼續，下一步建議 mel 梯度引導（唯一還沒測的生成端槓桿）。']
    lines += ['', '不可寫：只有一個 checkpoint／seed／前 1000 個 prompt、16 kHz 輸出；'
              '後處理曲線不是生成端能力；無本次刺激的人類評分。', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--tsv', required=True)
    ap.add_argument('--arms', default=None)
    ap.add_argument('--no-lowpass', action='store_true')
    ap.add_argument('--no-docs', action='store_true', help='smoke: skip the results docs in the repo')
    a = ap.parse_args()
    root = Path(a.root)
    rows = C.read_rows(a.tsv)
    arms = [x.strip() for x in (a.arms.split(',') if a.arms else C.FULL_ARMS) if x.strip()]
    score_dir = root / 'scores'
    score_dir.mkdir(parents=True, exist_ok=True)

    scored = {}
    for arm in arms:
        scored[arm] = score_arm(root / arm / 'audio', rows, score_dir / f'{arm}.json')

    lowpass = score_lowpass(root / 'A0' / 'audio', rows, score_dir / 'lowpass_ref.json') \
        if (not a.no_lowpass and 'A0' in scored) else []

    # harness consistency: lambda=0 cell vs A0, per-clip max abs difference
    harness = {}
    l0 = root / 'A4_L0' / 'audio'
    if l0.is_dir() and 'A0' in scored:
        diffs = []
        for i, _ in rows:
            p0, p1 = root / 'A0' / 'audio' / f'{i}.flac', l0 / f'{i}.flac'
            if not (p0.is_file() and p1.is_file()):
                continue
            x0, _ = sf.read(str(p0), dtype='float64')
            x1, _ = sf.read(str(p1), dtype='float64')
            if x0.shape != x1.shape:
                diffs.append(float('inf'))
                continue
            diffs.append(float(np.max(np.abs(x0 - x1))))
        if diffs:
            harness['lambda0_vs_A0'] = {'n': len(diffs), 'max_abs_diff': max(diffs),
                                        'mean_abs_diff': float(np.mean(diffs))}
            print(f'[harness] lambda0 vs A0 max_abs_diff={max(diffs):.3e} n={len(diffs)}',
                  flush=True)

    meta = {'root': str(root), 'tsv': str(Path(a.tsv).resolve()), 'tsv_sha256': C.sha256(a.tsv),
            'n_prompts': len(rows), 'arms': arms, 'cfg': C.CFG, 'seed': C.SEED,
            'ckpt': str(C.CKPT), 'ckpt_sha256': C.sha256(C.CKPT),
            'harness': harness, 'prompts': {
                'fid': C.FID, 'bright_neg': C.BRIGHT_NEG, 'bright_vec': C.BRIGHT_VEC, 'dark': C.DARK}}
    S = analyse(scored['A0'], {k: v for k, v in scored.items()
                               if k in C.MAIN_ARMS and k != 'A0'}, lowpass, meta)
    S['meta']['harness'] = harness
    C.write_json(root / 'summary.json', S)
    if not a.no_docs:
        C.write_json(C.RESULTS_DIR / 'muq_gen_steer_179_summary.json', S)
        write_results(S, C.RESULTS_DIR / 'muq_gen_steer_179_results.md')
    print('[score] verdict', S['verdict'], flush=True)


if __name__ == '__main__':
    main()
