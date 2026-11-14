#!/usr/bin/env python3
"""Analyse the extra matrix cells: PAM400 / MusicEval (Spearman, round-1 protocol) and
MusicPrefs (pairwise agreement, round-2 G3 protocol).

PAM400 / ME reuse the round-1 protocol verbatim (research/eval/newmodels_20261020/analyze.py):
cluster bootstrap B=2000 seed 20261020 over prompts, within-system Spearman, and pair_all
(within-prompt item pairs, credit 0.5 when the metric difference is exactly 0).  Only the
new columns (SongEval / SingMOS / CLAP) are added to the round-1 rows.

MusicPrefs reuses the round-2 G3 protocol (analyze_r2.analyze_pairwise): decisive human
pairs only, metric tie band m = 0.10 * (max-min of the metric over the scored items),
cluster bootstrap by prompt, plus the same Bradley-Terry held-out ceiling as
human_ceiling_r2.bt_half.

  python -I analyze_extra_cells.py --datasets pam400,me,musicprefs --nboot 2000 \
      --out /mnt/seagate/benchmarks_r2/analysis_extra
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402

_sp = importlib.util.spec_from_file_location('bench_r2_datasets', HERE / 'bench_datasets.py')
D = importlib.util.module_from_spec(_sp)
sys.modules['bench_r2_datasets'] = D
_sp.loader.exec_module(D)
_sa = importlib.util.spec_from_file_location('analyze_r2_mod', HERE / 'analyze_r2.py')
A = importlib.util.module_from_spec(_sa)
sys.modules['analyze_r2_mod'] = A
_sa.loader.exec_module(A)

R1 = Path('/home/kojiek/MeanAudio/research/eval/newmodels_20261020')
MP_CSV = D.DL / 'musicprefs' / 'human_preference.csv'
SEED_R1 = 20261020
SEED_R2 = 20261025


def log(*a):
    print(*a, flush=True)


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t')
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})


# ───────────────────────── PAM400 / MusicEval (round-1 protocol) ─────────────────────────
R1_SCORES = {'muqeval_A1': ['muqeval_A1_MI', 'muqeval_A1_TA'],
             'muqeval_A3a': ['muqeval_A3a_MI', 'muqeval_A3a_TA'],
             'cmirm': ['cmi_empty_quality', 'cmi_empty_alignment', 'cmi_caption_quality',
                       'cmi_caption_alignment'],
             'pam': ['pam']}
R1_FILES = {('pam400', m): f'scores_{m}_pam500.tsv' for m in R1_SCORES}
R1_FILES.update({('me', m): f'scores_{m}_me2748.tsv' for m in R1_SCORES})
# AES and MEva never needed rescoring: the round-1 manifests already carry them
R1_INLINE = {'pam400': ('bench_pam500.tsv', ['aes_PQ', 'aes_CE', 'aes_CU', 'aes_PC', 'meva']),
             'me': ('bench_musiceval2748.tsv', ['aes_PQ', 'aes_CE', 'aes_CU', 'aes_PC', 'meva'])}


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_extra_rows(ds):
    """Manifest row + extra JSON (human targets) + round-1 score tsvs + the new scores/<ds> tsvs."""
    items = D.read_manifest(ds)
    rows = []
    for it in items:
        r = dict(it)
        r.update(json.loads(it['extra'] or '{}'))
        rows.append(r)
    by_key = {r['item_id']: r for r in rows}
    src, cols_inline = R1_INLINE[ds]
    for rr in csv.DictReader(open(R1 / src), delimiter='\t'):
        target = by_key.get(rr['key'])
        if target is not None:
            target.update({k: rr[k] for k in cols_inline if k in rr})
    for model, cols in R1_SCORES.items():
        p = R1 / R1_FILES[(ds, model)]
        if not p.exists():
            raise SystemExit(f'missing round-1 scores {p}')
        for rr in csv.DictReader(open(p), delimiter='\t'):
            target = by_key.get(rr['key'])
            if target is not None:
                target.update({k: rr[k] for k in cols if k in rr})
    sc_dir = D.SCORES / ds
    for p in sorted(sc_dir.glob('*.tsv')):
        if p.name.endswith('.err.tsv'):
            continue
        for rr in csv.DictReader(open(p), delimiter='\t'):
            target = by_key.get(rr['item_id'])
            if target is not None:
                target.update({k: v for k, v in rr.items() if k not in ('unit_id', 'item_id', 'n_win')})
    return rows


def col(rows, c):
    return np.array([_float(r.get(c)) for r in rows])


def wcorr(kind, x, y):
    if kind == 'spearman':
        return spearmanr(x, y)[0]
    if kind == 'pearson':
        return pearsonr(x, y)[0]
    return kendalltau(x, y)[0]


def demean(v, groups):
    out = v.copy()
    for s in np.unique(groups):
        out[groups == s] -= v[groups == s].mean()
    return out


def analyse_r1(tag, rows, metrics, targets, cluster, groups, pair_mask, nboot=2000):
    """Copy of round-1 analyse(); same seed, same pair_all / within definitions."""
    n = len(rows)
    cl = np.array([r[cluster] for r in rows])
    ucl, cinv = np.unique(cl, return_inverse=True)
    rng = np.random.default_rng(SEED_R1)
    cnt = rng.multinomial(len(ucl), np.ones(len(ucl)) / len(ucl), size=nboot)
    idx_by_cl = [np.where(cinv == i)[0] for i in range(len(ucl))]
    boots = [np.concatenate([np.repeat(idx_by_cl[i], c) for i, c in enumerate(cnt[b]) if c])
             for b in range(nboot)]
    pairs = []
    if pair_mask is not None:
        for i in range(len(ucl)):
            ix = [j for j in idx_by_cl[i] if pair_mask(rows[j])]
            for x, y in itertools.combinations(ix, 2):
                pairs.append((i, x, y))
    P = np.array(pairs) if pairs else np.zeros((0, 3), int)
    out = []
    H = {t: col(rows, t) for t in targets}
    M = {m: col(rows, m) for m in metrics}
    for t in targets:
        h = H[t]
        for m in metrics:
            x = M[m]
            ok = np.isfinite(x) & np.isfinite(h)
            if ok.sum() < 30:
                continue
            cell = dict(bench=tag, target=t, metric=m, n=int(ok.sum()))
            for kind in ('spearman', 'pearson', 'kendall'):
                pt = wcorr(kind, x[ok], h[ok])
                nb = nboot if kind != 'kendall' or n < 1500 else 500
                bs = [wcorr(kind, x[bi][ok[bi]], h[bi][ok[bi]]) for bi in boots[:nb]]
                cell[kind] = [float(pt), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
            if groups is not None:
                xs, hs = demean(x, groups), demean(h, groups)
                pt = spearmanr(xs[ok], hs[ok])[0]
                bs = []
                for bi in boots:
                    bi = bi[ok[bi]]
                    bs.append(spearmanr(demean(x[bi], groups[bi]), demean(h[bi], groups[bi]))[0])
                cell['within'] = [float(pt), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
            if len(P) >= 200:
                a, b_ = P[:, 1], P[:, 2]
                hd, md = h[a] - h[b_], x[a] - x[b_]
                valid = np.isfinite(hd) & np.isfinite(md) & (hd != 0)
                credit = np.where(md == 0, 0.5, (np.sign(md) == np.sign(hd)).astype(float))
                bins = {'all': valid, 'lt05': valid & (np.abs(hd) < 0.5),
                        '05_1': valid & (np.abs(hd) >= 0.5) & (np.abs(hd) < 1),
                        'ge1': valid & (np.abs(hd) >= 1)}
                for k, mk in bins.items():
                    num = np.bincount(P[:, 0], weights=credit * mk, minlength=len(ucl))
                    den = np.bincount(P[:, 0], weights=mk.astype(float), minlength=len(ucl))
                    pt = num.sum() / den.sum() if den.sum() else np.nan
                    bs = (cnt @ num) / np.maximum(cnt @ den, 1e-9)
                    cell['pair_' + k] = [float(pt), float(np.nanpercentile(bs, 2.5)),
                                         float(np.nanpercentile(bs, 97.5)), int(den.sum())]
            out.append(cell)
    return out


def pam400_me(nboot, want):
    cells = {}
    rows = load_extra_rows('pam400')
    metrics = ['aes_PQ', 'aes_CE', 'aes_CU', 'aes_PC', 'meva',
               'muqeval_A1_MI', 'muqeval_A1_TA', 'muqeval_A3a_MI', 'muqeval_A3a_TA',
               'cmi_empty_quality', 'cmi_empty_alignment', 'cmi_caption_quality',
               'cmi_caption_alignment', 'pam', 'se_coh', 'se_mem', 'se_nat', 'se_cla', 'se_mus',
               'singmos', 'clap']
    if 'pam400' in want:
        cells['pam400'] = analyse_r1('PAM400', rows, metrics, ['pam_OVL', 'pam_REL'], 'prompt',
                                     np.array([r['group'] for r in rows]),
                                     lambda _: True, nboot)
    if 'me' in want:
        rows = load_extra_rows('me')
        cells['me'] = analyse_r1('ME', rows, metrics, ['human_OVL', 'human_REL'], 'prompt',
                                 np.array([r['group'] for r in rows]),
                                 lambda x: x['shared'] == 'True', nboot)
    return cells


# ───────────────────────── MusicPrefs (round-2 G3 protocol) ─────────────────────────
def bt_half(win_pairs, systems, rng, nboot):
    """BT on a random half of the votes, accuracy on the other half (human_ceiling_r2.bt_half)."""
    n = len(win_pairs)
    accs = []
    for _ in range(nboot):
        idx = rng.permutation(n)
        h = n // 2
        bt = A.bradley_terry(systems, [(win_pairs[i][0], win_pairs[i][1]) for i in idx[:h]], iters=60)
        ok = []
        for w, l in (win_pairs[i] for i in idx[h:]):
            pw, pl = bt.get(w, 0.0), bt.get(l, 0.0)
            ok.append(0.5 if pw == pl else float((w if pw > pl else l) == w))
        accs.append(float(np.mean(ok)))
    return np.array(accs)


def load_scores_items(ds):
    out = {}
    for p in sorted((D.SCORES / ds).glob('*.tsv')):
        if p.name.endswith('.err.tsv'):
            continue
        model, rest = p.stem.split('__', 1)
        variant, loud = rest.rsplit('_', 1)
        if (variant, loud) != ('native', 'raw'):
            continue
        rows = {}
        for rr in csv.DictReader(open(p), delimiter='\t'):
            v = {}
            for k, x in rr.items():
                if k in ('unit_id', 'item_id', 'n_win') or x is None:
                    continue
                try:
                    fx = float(x)
                except ValueError:
                    continue
                if np.isfinite(fx):
                    v[k] = fx
            if v:
                rows[rr['item_id']] = v
        out[(model, variant, loud)] = rows
    return out


def musicprefs(nboot):
    pairs = list(csv.DictReader(open(MP_CSV)))
    sc = load_scores_items('musicprefs')
    if not sc:
        raise SystemExit('no musicprefs scores yet')
    rng = np.random.default_rng(SEED_R2)
    rows, ceilings = [], {}
    for axis in ('musicality', 'fidelity'):
        dec = [p for p in pairs if p[axis] in ('a_wins', 'b_wins')]
        systems = sorted({p['system_a'] for p in dec} | {p['system_b'] for p in dec})
        wp = [((p['system_a'], p['system_b']) if p[axis] == 'a_wins' else (p['system_b'], p['system_a']))
              for p in dec]
        acc = bt_half(wp, systems, rng, nboot)
        ceilings[axis] = dict(n_pairs=len(dec), n_systems=len(systems),
                              bt_median=float(np.nanmedian(acc)),
                              bt_ci=[float(np.nanpercentile(acc, 2.5)), float(np.nanpercentile(acc, 97.5))],
                              floor=0.5)
        log(f'musicprefs {axis}: {len(dec)} decisive pairs, {len(systems)} systems, '
            f'BT held-out {ceilings[axis]["bt_median"]:.3f}')
        for (model, _, _), vals in sorted(sc.items()):
            for mc in sorted({k for v in vals.values() for k in v}):
                if mc in ('n_win',):
                    continue
                vv = {i: v[mc] for i, v in vals.items() if mc in v}
                use = [p for p in dec if p['audio_a'].removesuffix('.wav') in vv
                       and p['audio_b'].removesuffix('.wav') in vv]
                if len(use) < 50:
                    continue
                S = max(vv.values()) - min(vv.values())
                m = 0.10 * S
                d = np.array([vv[p['audio_a'].removesuffix('.wav')] -
                              vv[p['audio_b'].removesuffix('.wav')] for p in use])
                h = np.array([1.0 if p[axis] == 'a_wins' else -1.0 for p in use])
                score_m = np.where(np.abs(d) < m, 0.5, (np.sign(d) == h).astype(float))
                score_0 = np.where(d == 0, 0.5, (np.sign(d) == h).astype(float))
                clus = {}
                for k, p in enumerate(use):
                    clus.setdefault(p['prompt'], []).append(k)
                keys = list(clus)
                bs = []
                for _ in range(nboot):
                    pick = rng.integers(0, len(keys), len(keys))
                    ids = np.concatenate([clus[keys[q]] for q in pick])
                    bs.append(score_m[ids].mean())
                # item-level target = win rate over the decisive pairs the item appears in
                wins = {}
                for p in use:
                    ia, ib = p['audio_a'].removesuffix('.wav'), p['audio_b'].removesuffix('.wav')
                    wins.setdefault(ia, []).append(1.0 if p[axis] == 'a_wins' else 0.0)
                    wins.setdefault(ib, []).append(0.0 if p[axis] == 'a_wins' else 1.0)
                items = sorted(wins)
                x = np.array([vv[i] for i in items])
                y = np.array([np.mean(wins[i]) for i in items])
                idx = A.boot_idx(len(items), nboot, seed=SEED_R2)
                st = A.corr_stats(x, y, idx, kendall=True)
                # system-level: BT (all decisive) vs metric system mean
                sysm = {}
                for i in items:
                    s = next(p['system_a'] if p['audio_a'].removesuffix('.wav') == i else p['system_b']
                             for p in use if i in (p['audio_a'].removesuffix('.wav'),
                                                   p['audio_b'].removesuffix('.wav')))
                    sysm.setdefault(s, []).append(vv[i])
                sysm = {k: float(np.mean(v)) for k, v in sysm.items()}
                bt = A.bradley_terry(systems, wp, iters=60)
                common = [s for s in sysm if s in bt]
                row = dict(dataset='musicprefs', model=model, col=mc, axis=axis, n_pairs=len(use),
                           m=float(m), agree_m=float(score_m.mean()),
                           agree_m_ci=[float(np.nanpercentile(bs, 2.5)),
                                       float(np.nanpercentile(bs, 97.5))],
                           agree_forced=float(score_0.mean()), tie_frac=float((np.abs(d) < m).mean()),
                           n_items=len(items), within_sys=None)
                if 'spearman' in st:
                    row.update(spearman=st['spearman'], spearman_ci=st['spearman_ci'],
                               pearson=st['pearson'], kendall=st['kendall'])
                if len(common) >= 5:
                    row['kendall_tau_sys'] = float(kendalltau([sysm[s] for s in common],
                                                              [bt[s] for s in common])[0])
                rows.append(row)
    write_tsv(D.ROOT / 'analysis_extra' / 'musicprefs_pairwise.tsv', rows)
    log(f'musicprefs: {len(rows)} rows')
    return ceilings, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='pam400,me,musicprefs')
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--out', default=str(D.ROOT / 'analysis_extra'))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want = set(a.datasets.split(','))
    result = {}
    if want & {'pam400', 'me'}:
        cells = pam400_me(a.nboot, want)
        for ds, rows in cells.items():
            write_tsv(out / f'{ds}_extra_cells.tsv', rows)
            result[ds] = rows
            log(f'{ds}: {len(rows)} cells')
    if 'musicprefs' in want:
        ceilings, rows = musicprefs(a.nboot)
        result['musicprefs'] = rows
        result['musicprefs_ceiling'] = ceilings
    # merge into any earlier partial (J6 writes pam400/me, J7 writes musicprefs; the queue
    # may run them in either order and each must not clobber the other)
    dst = out / 'extra_cells.json'
    if dst.exists():
        try:
            old = json.loads(dst.read_text())
        except ValueError:
            old = {}
        old.update(result)
        result = old
    dst.write_text(json.dumps(result, indent=1, default=float))
    log(f'wrote {dst}')


if __name__ == '__main__':
    main()
