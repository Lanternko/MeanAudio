#!/usr/bin/env python3
"""Round-2 benchmark screening: human ceiling / floor, plus a descriptive SingMOS x Arena readout.

Design: docs/experiments/benchmark_screening_r2_20261025.md §6.2 item 4 and §7 (preregistered).
Nothing here changes a round-2 gate: the numbers are reference/ descriptive only.

  ceiling   split-half reliability of the human panel (2,000 draws, seed 20261025), the
            "one rater vs. the mean of the others" Spearman, floors (rho=0 / 0.50) and a
            permutation 95% upper bound.  ODAQ additionally gets the within-trial version
            (and the 16 kHz / 24 kHz bandwidth-visible subsets that its G6a uses).
  pairwise  AIME / Music Arena: vote-level ceiling via system-level Bradley-Terry fitted on
            half of the battles and predicting the other half (Arena has one vote per battle;
            AIME has one vote per unique pair).  AIME also gets the split-half of the snippet
            win rate, which is the reliability of its G1 human target.
  singmos   SingMOS predictor on Arena native raw (queue job 152) -> G1 / G3 / within-system,
            computed with analyze_r2.py's own helpers.  Skipped (marked pending) until the
            score column exists.

Usage:
  ~/venvs/dac/bin/python -I human_ceiling_r2.py \
      --out-md  /home/kojiek/MeanAudio/docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.md \
      --out-json /home/kojiek/MeanAudio/docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.json
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

SEED = 20261025
NBOOT = 2000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


D = _load('bench_r2_datasets', HERE / 'bench_datasets.py')
A = _load('bench_r2_analyze', HERE / 'analyze_r2.py')   # read-only use of its helpers

RNG_TAG = {'odaq': 1, 'mc522': 2, 'songeval': 3, 'singmos': 4, 'aime': 5, 'arena': 6, 'singmos_arena': 7}


def rng_for(tag):
    return np.random.default_rng([SEED, RNG_TAG[tag]])


def ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return [float('nan'), float('nan')]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(stats.spearmanr(a, b)[0])


def sb_curve(r):
    r = np.asarray(r, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 2 * r / (1 + r)


def split_half(M, rng, B=NBOOT, per_item=True, null=False):
    """M: items x raters.  Even split floor(k/2)/floor(k/2) (an odd last rater is dropped).
    per_item=True randomises the rater order separately per item (rater slots are anonymous
    across items: MC522 / SingMOS); False uses one shared split of the rater panel (ODAQ)."""
    n, k = M.shape
    h = k // 2
    out = []
    for _ in range(B):
        if per_item:
            perm = np.argsort(rng.random((n, k)), axis=1)
        else:
            perm = np.broadcast_to(rng.permutation(k), (n, k))
        X = np.take_along_axis(M, perm, axis=1)
        a, b = X[:, :h].mean(1), X[:, h:2 * h].mean(1)
        if null:
            b = b[rng.permutation(n)]
        out.append(spearman(a, b))
    return np.array(out)


def split_half_lists(L, rng, B=NBOOT, null=False):
    """Variable-length per-item rating lists (AIME votes per snippet): balanced floor(n/2) halves."""
    out = []
    for _ in range(B):
        aa, bb = [], []
        for v in L:
            v = np.asarray(v, float)
            h = len(v) // 2
            p = rng.permutation(len(v))
            aa.append(v[p[:h]].mean())
            bb.append(v[p[h:2 * h]].mean())
        if null:
            bb = np.asarray(bb)[rng.permutation(len(bb))]
        out.append(spearman(aa, bb))
    return np.array(out)


def summarize_split(rs, null_rs):
    sb = sb_curve(rs)
    sq = np.sqrt(np.clip(sb, 0, None))
    return {
        'nboot': int(len(rs)),
        'r_half_median': float(np.nanmedian(rs)),
        'r_half_ci': ci(rs),
        'sb_median': float(np.nanmedian(sb)),
        'sb_ci': ci(sb),
        'sqrt_sb': float(np.sqrt(max(np.nanmedian(sb), 0.0))),
        'sqrt_sb_ci': ci(sq),
        'perm95_r_half': float(np.nanpercentile(null_rs, 95)),
        'perm95_sb': float(np.nanpercentile(sb_curve(null_rs), 95)),
    }


def single_rest_complete(M, rng, B=NBOOT):
    """One rater vs. the mean of the other raters, across items; median over rater slots,
    CI by resampling items."""
    n, k = M.shape
    vals = np.array([spearman(M[:, j], np.delete(M, j, 1).mean(1)) for j in range(k)])
    meds = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v = [spearman(M[i, j], np.delete(M[i], j, 1).mean(1)) for j in range(k)]
        meds.append(np.nanmedian(v))
    return {'median': float(np.nanmedian(vals)), 'ci': ci(meds),
            'min': float(np.nanmin(vals)), 'max': float(np.nanmax(vals)),
            'n_raters': int(k), 'per_rater': [float(x) for x in vals]}


# ───────────────────────── ODAQ ─────────────────────────
def odaq():
    import pandas as pd
    d = pd.read_csv(D.DL / 'ODAQ/ODAQ/ODAQ_listening_test/ODAQ_results.csv')
    subs = sorted(d.subject.unique())
    trials = sorted(d.item.unique())
    rng = rng_for('odaq')

    # item level: 240 (trial x process) x 26 subjects
    piv = d.pivot_table(index=['item', 'process'], columns='subject', values='score')
    M = piv[subs].to_numpy(float)
    rs = split_half(M, rng, per_item=False)
    nul = split_half(M, rng, per_item=False, null=True)
    item_level = summarize_split(rs, nul)
    item_level['single_vs_rest'] = single_rest_complete(M, rng)

    # within trial: 8 conditions per trial; human has no bandwidth limit, but a metric does.
    # G6a (analyze_r2) ranks only the Q1-Q5 conditions inside a trial, dropping the Q conditions a
    # band-limited model cannot see (lp_visible); the human ceiling must use the same unit.
    VIS = {16: {'Q1'}, 24: {'Q1', 'Q2', 'Q3'}, 44: {'Q1', 'Q2', 'Q3', 'Q4', 'Q5'}}

    def vis(method, cond, path):
        if not cond.startswith('Q'):
            return False                               # G6a never ranks LP3.5/LP7/Ref
        return True if method != 'LP' else cond in VIS[path]

    def trial_matrices(path=None):
        out = {}
        for t in trials:
            g = d[d.item == t].sort_values('process')
            gp = g.drop_duplicates('process')          # 8 conditions (strategy x condition), not 8 x 26 subjects
            if path is None:
                mask = np.ones(len(gp), bool)          # all 8 conditions (descriptive)
            elif path == 'allQ':
                mask = np.array([r.condition.startswith('Q') for _, r in gp.iterrows()])
            else:
                mask = np.array([vis(r.method, r.condition, path) for _, r in gp.iterrows()])
            gm = g.pivot(index='process', columns='subject', values='score').loc[:, subs]
            out[t] = (gm.to_numpy(float), mask)
        return out

    def wt_split(mats, rng, B=NBOOT, null=False, agg=np.nanmedian):
        # G6a in analyze_r2 is the *mean* over trials of the per-trial Spearman, so the ceiling
        # for that statistic aggregates trials with nanmean; the median saturates at 1.0 here.
        meds = []
        for _ in range(B):
            perm = rng.permutation(len(subs))
            a_i, b_i = perm[:len(subs) // 2], perm[len(subs) // 2:2 * (len(subs) // 2)]
            rhos = []
            for M, mask in mats.values():
                if mask.sum() < 3:
                    continue
                a = M[mask][:, a_i].mean(1)
                b = M[mask][:, b_i].mean(1)
                if null:
                    b = b[rng.permutation(len(b))]
                r = spearman(a, b)
                if np.isfinite(r):
                    rhos.append(r)
            meds.append(agg(rhos) if rhos else np.nan)
        return np.array(meds)

    def wt_single(mats, agg=np.nanmedian):
        per_trial = {}
        for t, (M, mask) in mats.items():
            v = [spearman(M[mask][:, j], np.delete(M[mask], j, 1).mean(1)) for j in range(len(subs))]
            per_trial[t] = np.array(v)
        vals = np.concatenate([per_trial[t] for t in mats if (mats[t][1].sum() >= 3)])
        meds = []
        for _ in range(NBOOT):
            pick = rng.integers(0, len(trials), len(trials))
            v = np.concatenate([per_trial[trials[i]] for i in pick])
            meds.append(agg(v))
        return {'median': float(agg(vals)), 'ci': ci(meds),
                'min': float(np.nanmin(vals)), 'max': float(np.nanmax(vals)),
                'n_raters': len(subs)}

    out = {'target': 'MUSHRA score (BAQ)', 'unit': 'item = trial x process (240); trial=30',
           'n_units': 240, 'n_trials': len(trials), 'n_raters': len(subs),
           'item_level': item_level}
    for label, path in (('within_trial_all8', None), ('within_trial_allQ', 'allQ'),
                        ('within_trial_g6a16', 16), ('within_trial_g6a24', 24)):
        mats = trial_matrices(path)
        rs = wt_split(mats, rng, agg=np.nanmean)
        nul = wt_split(mats, rng, null=True, agg=np.nanmean)
        s = summarize_split(rs, nul)
        s['single_vs_rest'] = wt_single(mats, agg=np.nanmean)
        s['n_trials_used'] = int(sum(1 for _, m in mats.values() if m.sum() >= 3))
        out[label] = s
    return out


# ───────────────────────── MC522 ─────────────────────────
MC522_HUMAN = Path('/home/kojiek/MeanAudio/research/eval/aes_human_ratings/AES_natural_music.jsonl')
AXES = {'PQ': 'Production_Quality', 'PC': 'Production_Complexity',
        'CE': 'Content_Enjoyment', 'CU': 'Content_Usefulness'}


def mc522():
    human = {}
    with open(MC522_HUMAN) as f:
        for line in f:
            r = json.loads(line)
            if '/musiccaps/' not in r['data_path']:
                continue
            y = Path(r['data_path']).stem
            human[y] = {k: np.array(r[v], float) for k, v in AXES.items()}
    items = [it['item_id'] for it in D.read_manifest('mc522')]
    have = [i for i in items if i in human]
    rng = rng_for('mc522')
    out = {'target': 'human PQ (Production_Quality) = 10-rater mean',
           'unit': 'clip (522)', 'n_units': len(have),
           'human_source': str(MC522_HUMAN), 'axes': {}}
    for axis, key in AXES.items():
        M = np.stack([human[i][axis] for i in have])
        rs = split_half(M, rng, per_item=True)
        nul = split_half(M, rng, per_item=True, null=True)
        s = summarize_split(rs, nul)
        s['single_vs_rest'] = single_rest_complete(M, rng)
        out['axes'][axis] = s
    # check that the manifest aggregate is the 10-rater mean of this source
    man = {it['item_id']: json.loads(it['extra'])['human_PQ'] for it in D.read_manifest('mc522')}
    diff = np.array([abs(man[i] - human[i]['PQ'].mean()) for i in have])
    out['manifest_mean_check_max_abs_diff'] = float(diff.max())
    return out


# ───────────────────────── SongEval ─────────────────────────
DIMS = ('Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness')


def songeval():
    rows = [json.loads(l) for l in open(D.META / 'songeval/metadata.jsonl')]
    s800 = {it['item_id'] for it in D.read_manifest('songeval') if 's800' in it['subset'].split(';')}
    rows = [r for r in rows if Path(r['file_name']).stem in s800]
    rows.sort(key=lambda r: r['file_name'])
    rng = rng_for('songeval')
    out = {'target': 'Musicality (primary)', 'unit': 'song (S800, n=%d)' % len(rows),
           'n_units': len(rows), 'n_ann_per_song': 4, 'dims': {}}
    for dim in DIMS:
        ann = [np.array([a[dim] for a in r['annotation']], float) for r in rows]
        rs, nul_vals = [], []
        for null in (False, True):
            vals = []
            for _ in range(NBOOT):
                a, b = [], []
                for v in ann:
                    p = rng.permutation(4)
                    a.append(v[p[:2]].mean())
                    b.append(v[p[2:]].mean())
                if null:
                    b = np.asarray(b)[rng.permutation(len(b))]
                vals.append(spearman(a, b))
            if null:
                nul_vals = vals
            else:
                rs = vals
        # single random annotator per song vs. mean of the other three
        single_draws = []
        for _ in range(NBOOT):
            a, b = [], []
            for v in ann:
                j = rng.integers(0, 4)
                a.append(v[j])
                b.append(np.delete(v, j).mean())
            single_draws.append(spearman(a, b))
        s = summarize_split(np.array(rs), np.array(nul_vals))
        s['single_vs_rest'] = {'median': float(np.nanmedian(single_draws)),
                               'ci': ci(single_draws),
                               'n_raters': 4,
                               'note': 'one random annotator per song vs. mean of the other three'}
        # per-annotator-identity robustness (annotators rate disjoint song blocks)
        per_id = {}
        for r in rows:
            ids = [a['annotator'] for a in r['annotation']]
            for i, a in enumerate(r['annotation']):
                per_id.setdefault(a['annotator'], []).append(
                    (a[dim], float(np.mean([x[dim] for j, x in enumerate(r['annotation']) if j != i]))))
        id_rhos = {}
        for k, v in per_id.items():
            x = np.array([p[0] for p in v]); y = np.array([p[1] for p in v])
            id_rhos[k] = spearman(x, y)
        s['single_vs_rest_per_annotator_id'] = {k: float(v) for k, v in sorted(id_rhos.items())}
        out['dims'][dim] = s
    out['primary'] = 'Musicality'
    return out


# ───────────────────────── SingMOS-Pro ─────────────────────────
def singmos():
    meta = json.loads((D.META / 'singmos_pro/metadata.json').read_text())
    rows = [r for r in meta if r['split'] == 'test']
    rng = rng_for('singmos')
    L = [np.array(r['judge_score'], float) for r in rows]
    n_judge = np.array([len(v) for v in L])
    rs = split_half_lists(L, rng)
    nul = split_half_lists(L, rng, null=True)
    s = summarize_split(rs, nul)
    single = []
    for _ in range(NBOOT):
        a, b = [], []
        for v in L:
            j = rng.integers(0, len(v))
            a.append(v[j])
            b.append(np.delete(v, j).mean())
        single.append(spearman(a, b))
    s['single_vs_rest'] = {'median': float(np.nanmedian(single)), 'ci': ci(single),
                           'n_raters': 'varies', 'note': 'one random judge per clip vs. mean of the rest'}
    return {'target': 'per-clip MOS (official test split)', 'unit': 'clip (test, n=%d)' % len(rows),
            'n_units': len(rows), 'n_judge_counts': {str(int(k)): int((n_judge == k).sum())
                                                     for k in sorted(set(n_judge))},
            'split_half': s}


# ───────────────────────── AIME ─────────────────────────
def aime():
    import pandas as pd
    s = pd.read_parquet(D.META / 'aime_survey/train.parquet')
    s = s[s['question-type'] == 'Music Quality'].reset_index(drop=True)
    pairs, votes = [], {}
    for r in s.itertuples(index=False):
        # fields: _0 question-type, description, _2 model-1, _3 track-1-id, _4 begin, _5 end,
        #         _6 model-2, _7 track-2-id, _8 begin, _9 end, answer
        a = f"{r._2}|{int(r._3):05d}|{r._4}-{r._5}"       # model-1, track-1, begin, end
        b = f"{r._6}|{int(r._7):05d}|{r._8}-{r._9}"
        ans = int(r.answer)
        pa = 1 if ans == 1 else 0                          # +1 = snippet a was preferred
        pairs.append(dict(a=a, b=b, sys_a=r._2, sys_b=r._6, pa=pa))
        votes.setdefault(a, []).append(pa)
        votes.setdefault(b, []).append(1 - pa)
    items = sorted(votes)
    L = [votes[i] for i in items]
    rng = rng_for('aime')
    rs = split_half_lists(L, rng)
    nul = split_half_lists(L, rng, null=True)
    g1 = summarize_split(rs, nul)
    single = []
    for _ in range(NBOOT):
        a, b = [], []
        for v in L:
            j = rng.integers(0, len(v))
            a.append(v[j]); b.append(np.delete(v, j).mean())
        single.append(spearman(a, b))
    g1['single_vs_rest'] = {'median': float(np.nanmedian(single)), 'ci': ci(single),
                            'n_raters': 'votes per snippet (12 for all)',
                            'note': 'one random vote vs. the mean of the other 11'}
    # vote-level ceiling: BT on half of the unique-pair votes, predict the other half
    systems = sorted({p['sys_a'] for p in pairs} | {p['sys_b'] for p in pairs})
    wp = [(p['sys_a'] if p['pa'] == 1 else p['sys_b'],
           p['sys_b'] if p['pa'] == 1 else p['sys_a']) for p in pairs]
    acc = bt_half(wp, systems, rng)
    acc_null = bt_half(wp, systems, rng, null=True)
    g3 = {'n_votes': len(pairs), 'n_unique_pairs': len({tuple(sorted((p['a'], p['b']))) for p in pairs}),
          'n_systems': len(systems), 'nboot': NBOOT,
          'bt_half_accuracy_median': float(np.nanmedian(acc)),
          'bt_half_accuracy_ci': ci(acc),
          'perm95_accuracy': float(np.nanpercentile(acc_null, 95)),
          'floor': 0.5}
    g3['repeated_votes_per_pair'] = 'none (7,800 unique pairs; each snippet gets 12 votes vs different opponents)'
    return {'target': 'pairwise preference (Music Quality); G1 target = snippet win rate',
            'unit': 'snippet (1,300) / vote (7,800)', 'n_units': len(items),
            'n_votes': len(pairs), 'G1': g1, 'G3': g3}


def bt_half(win_pairs, systems, rng, null=False):
    """Bradley-Terry fitted on a random half of the votes, accuracy on the other half."""
    n = len(win_pairs)
    accs = []
    for _ in range(NBOOT):
        p = list(win_pairs)
        if null:
            p = [(w, l) if rng.random() < 0.5 else (l, w) for w, l in p]
        idx = rng.permutation(n)
        h = n // 2
        bt = A.bradley_terry(systems, [(p[i][0], p[i][1]) for i in idx[:h]], iters=60)
        ok = []
        for w, l in (p[i] for i in idx[h:]):
            pw, pl = bt.get(w, 0.0), bt.get(l, 0.0)
            ok.append(0.5 if pw == pl else float((w if pw > pl else l) == w))
        accs.append(float(np.mean(ok)))
    return np.array(accs)


# ───────────────────────── Arena ─────────────────────────
def arena():
    pairs = [p for p in A.arena_pairs()]
    dec = [p for p in pairs if p['win'] in (0, 1)]
    systems = sorted({p['sys_a'] for p in dec} | {p['sys_b'] for p in dec})
    wp = [((p['sys_a'] if p['win'] == 0 else p['sys_b']),
           (p['sys_b'] if p['win'] == 0 else p['sys_a'])) for p in dec]
    rng = rng_for('arena')
    acc = bt_half(wp, systems, rng)
    acc_null = bt_half(wp, systems, rng, null=True)
    # battles per audio item: split-half of an item win rate is impossible if this is 1
    cnt = {}
    for p in dec:
        cnt[p['a']] = cnt.get(p['a'], 0) + 1
        cnt[p['b']] = cnt.get(p['b'], 0) + 1
    out = {'target': 'pairwise preference (TIE / BOTH_BAD excluded)',
           'n_battles_total': len(pairs), 'n_battles_decided': len(dec),
           'n_tie': sum(1 for p in pairs if p['pref'] == 'TIE'),
           'n_both_bad': sum(1 for p in pairs if p['pref'] == 'BOTH_BAD'),
           'decision_rate': len(dec) / len(pairs), 'n_systems': len(systems),
           'n_items': len(cnt), 'max_battles_per_item': int(max(cnt.values())),
           'G3': {'nboot': NBOOT, 'bt_half_accuracy_median': float(np.nanmedian(acc)),
                  'bt_half_accuracy_ci': ci(acc),
                  'perm95_accuracy': float(np.nanpercentile(acc_null, 95)), 'floor': 0.5},
           'G1_split_half': 'not computable: at most %d battle per audio item' % max(cnt.values())}
    return out


# ───────────────────────── SingMOS x Arena ─────────────────────────
def singmos_arena(force=False):
    """force=True is only for exercising the analysis path before job 152 finishes; the report never uses it."""
    sc = A.load_scores('arena')
    key = ('singmos', 'native', 'raw')
    vals = {i: d['singmos'] for i, d in sc[key].items() if 'singmos' in d} if key in sc else {}
    contract = '/home/kojiek/MeanAudio/docs/experiments/bench_r2_J5c_singmos_contract.json'
    summ = Path('/home/kojiek/MeanAudio/docs/experiments/results/bench_r2_J5c_singmos_summary.json')
    planned = len(D.read_manifest('arena'))
    if not (summ.exists() or force):
        # the run_job summary is written only after --validate-only passes, so partial rows are not analysed
        return {'status': 'running' if vals else 'pending', 'queue_job': 152, 'contract': contract,
                'n_scored': len(vals), 'planned': planned,
                'note': ('p2 job 152 is still running; rows are partial and are NOT analysed'
                         if vals else 'waiting for the p2 job to write scores/arena/singmos__native_raw.tsv')}
    pairs = A.arena_pairs()
    dec = [p for p in pairs if p['win'] in (0, 1)]
    wins = {}
    for p in dec:
        for it, w in ((p['a'], 1 - p['win']), (p['b'], p['win'])):
            wins.setdefault(it, []).append(w)
    hw = {i: float(np.mean(v)) for i, v in wins.items()}
    isys = {it['item_id']: it['group'] for it in D.read_manifest('arena')}
    items = [i for i in vals if i in hw]
    x = np.array([vals[i] for i in items]); y = np.array([hw[i] for i in items])
    idx = A.boot_idx(len(items), NBOOT)
    st = A.corr_stats(x, y, idx, kendall=True)
    sysn = [isys.get(i, '') for i in items]
    sw = A.corr_stats(A.center_by(x, sysn), A.center_by(y, sysn), idx, kendall=False)
    S = max(vals.values()) - min(vals.values())
    m = 0.10 * S
    use = [p for p in dec if p['a'] in vals and p['b'] in vals]
    d = np.array([vals[p['a']] - vals[p['b']] for p in use])
    h = np.array([1.0 if p['win'] == 0 else -1.0 for p in use])
    score_m = np.where(np.abs(d) < m, 0.5, (np.sign(d) == h).astype(float))
    score_0 = np.where(d == 0, 0.5, (np.sign(d) == h).astype(float))
    rng = rng_for('singmos_arena')
    clus = {}
    for k, p in enumerate(use):
        clus.setdefault(p.get('uuid') or p['a'], []).append(k)
    keys = list(clus)
    bs_m, bs_0 = [], []
    for _ in range(NBOOT):
        pick = rng.integers(0, len(keys), len(keys))
        ids = np.concatenate([clus[keys[q]] for q in pick])
        bs_m.append(score_m[ids].mean()); bs_0.append(score_0[ids].mean())
    sysm = {}
    for i, v in vals.items():
        sysm.setdefault(isys.get(i, ''), []).append(v)
    sysm = {k: float(np.mean(v)) for k, v in sysm.items() if k}
    bt = A.bradley_terry(list(sysm), [((p['sys_a'], p['sys_b']) if p['win'] == 0 else (p['sys_b'], p['sys_a']))
                                      for p in dec])
    common = [s for s in sysm if s in bt]
    tau = float(stats.kendalltau([sysm[s] for s in common], [bt[s] for s in common])[0]) if len(common) >= 5 else float('nan')
    tie = [p for p in pairs if p['pref'] == 'TIE' and p['a'] in vals and p['b'] in vals]
    return {'status': 'done', 'queue_job': 152,
            'G1': {'spearman': st.get('spearman'), 'ci': st.get('spearman_ci'),
                   'kendall': st.get('kendall'), 'n_items': len(items)},
            'G2_within_system': {'spearman': sw.get('spearman'), 'ci': sw.get('spearman_ci')},
            'G3': {'n_pairs': len(use), 'm': float(m),
                   'agree_m': float(score_m.mean()), 'agree_m_ci': ci(bs_m),
                   'agree_forced': float(score_0.mean()), 'agree_forced_ci': ci(bs_0),
                   'tie_frac': float((np.abs(d) < m).mean()),
                   'n_tie_battles': len(tie),
                   'tie_small_diff_frac': float(np.mean([abs(vals[p['a']] - vals[p['b']]) < m for p in tie])) if tie else None},
            'system_rank_kendall_tau': tau, 'n_systems': len(common),
            'note': 'descriptive only; prereg §148 gates SingMOS on B4 only, so this cell never enters the r2 decision'}


# ───────────────────────── r2 reference numbers ─────────────────────────
def r2_numbers():
    p = Path('/home/kojiek/MeanAudio/docs/experiments/results/benchmark_screening_r2_20261025_summary.json')
    dec = json.loads(p.read_text())['decision']

    ci_key = {'G1_stat': 'G1_ci', 'G3_stat': 'G3_ci', 'G6a': 'G6a_ci'}

    def best(ds, key):
        cand = [(fam, per) for fam, c in dec.items()
                for per in ((c.get('per_dataset') or {}).get(ds),)
                if per and key in per and ds in A.INDEP.get(fam, set())]
        if not cand:
            return (None, None, None)
        fam, per = max(cand, key=lambda kv: kv[1][key])
        return (fam, per[key], per.get(ci_key[key]))

    out = {}
    for ds in ('odaq', 'aime', 'singmos', 'mc522', 'songeval', 'arena'):
        out[ds] = {'G1': best(ds, 'G1_stat'), 'G3': best(ds, 'G3_stat'), 'G6a': best(ds, 'G6a')}
    return out


# ───────────────────────── report ─────────────────────────
def f3(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:.3f}'


def c3(c):
    if c is None or any(not np.isfinite(x) for x in c):
        return 'n/a'
    return f'[{c[0]:.3f}, {c[1]:.3f}]'


def pct(x):
    return f'{100 * x:.0f}%'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-md', default='/home/kojiek/MeanAudio/docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.md')
    ap.add_argument('--out-json', default='/home/kojiek/MeanAudio/docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.json')
    ap.add_argument('--from-json', help='skip the bootstrap; re-render from an existing result json (r2_reference is recomputed)')
    a = ap.parse_args()
    t0 = time.time()
    if a.from_json:
        res = json.loads(Path(a.from_json).read_text())
        print(f'loaded {a.from_json}', flush=True)
        res['singmos_arena'] = singmos_arena()  # cheap; reflects queue progress or the finished job
    else:
        res = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'seed': SEED, 'nboot': NBOOT,
               'script': str(Path(__file__).resolve()), 'datasets': {}}
        for name, fn in (('odaq', odaq), ('mc522', mc522), ('songeval', songeval),
                         ('singmos', singmos), ('aime', aime), ('arena', arena)):
            print('computing', name, flush=True)
            res['datasets'][name] = fn()
        print('computing singmos x arena', flush=True)
        res['singmos_arena'] = singmos_arena()
    res['r2_reference'] = r2_numbers()
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    Path(a.out_md).write_text(render_md(res), encoding='utf-8')
    print(f'wrote {a.out_md}\nwrote {a.out_json}  ({time.time() - t0:.0f}s)', flush=True)


def render_md(res):
    R = res['datasets']
    o, mc, se, sm, ai, ar = R['odaq'], R['mc522'], R['songeval'], R['singmos'], R['aime'], R['arena']
    ref = res['r2_reference']
    sa = res['singmos_arena']

    def split_row(ds, s, best_pair, gate):
        ceil = s['sqrt_sb']
        val = best_pair[1]
        ratio = f'{val / ceil:.2f}' if (val is not None and ceil and np.isfinite(ceil) and ceil > 0) else 'n/a'
        return (f'| {ds} | {s["sb_median"]:.3f} {c3(s["sb_ci"])} | {ceil:.3f} | '
                f'{s["single_vs_rest"]["median"]:.3f} {c3(s["single_vs_rest"]["ci"])} | 0 | '
                f'{s["perm95_r_half"]:.3f} | {gate} | {best_pair[0] or "—"} {f3(val)} | {ratio} |')

    o_s = o['within_trial_allQ']
    mc_s = mc['axes']['PQ']
    se_s = se['dims']['Musicality']
    sm_s = sm['split_half']
    a_g1 = ai['G1']
    a_g3 = ai['G3']
    ar_g3 = ar['G3']
    b_o = ref['odaq']['G6a']
    b_mc = ref['mc522']['G1']
    b_sm = ref['singmos']['G1']
    b_a3 = ref['aime']['G3']
    b_ar = ref['arena']['G3']

    g3_rows = []
    for name, g, b in (('AIME', a_g3, b_a3), ('Music Arena', ar_g3, b_ar)):
        ceil = g['bt_half_accuracy_median']
        ratio = f'{b[1] / ceil:.2f}' if (b[1] is not None and ceil) else 'n/a'
        g3_rows.append(f'| {name} | {ceil:.3f} {c3(g["bt_half_accuracy_ci"])} | 0.500 | '
                       f'{g["perm95_accuracy"]:.3f} | G3 0.55 | {b[0] or "—"} {f3(b[1])} | {ratio} |')

    g1_rows = [
        split_row('ODAQ（G6a within-trial, Q1–Q5；對應 pam 0.603）', o_s, b_o, 'G6a 0.50'),
        split_row('MC522（PQ, 10 raters）', mc_s, b_mc, 'G1 0.50'),
        split_row('SongEval S800（Musicality, 4 annotators）', se_s, (None, None), 'G1 0.50（無獨立指標）'),
        split_row('SingMOS-Pro test（MOS）', sm_s, b_sm, 'G1 0.50'),
        split_row('AIME（G1：snippet 勝率, 12 votes）', a_g1, ref['aime']['G1'], 'G1 0.50'),
        split_row('ODAQ（G6a, 16 kHz 規則：AES／SingMOS，26 trials）', o['within_trial_g6a16'], (None, None), '參考'),
        split_row('ODAQ（G6a, 24 kHz 規則：MuQ-Eval／CMI-RM／SongEval，30 trials）', o['within_trial_g6a24'], (None, None), '參考'),
        split_row('ODAQ（item level, 參考）', o['item_level'], (None, None), '—'),
        split_row('ODAQ（within-trial, all 8 conditions）', o['within_trial_all8'], (None, None), '—'),
    ]

    if sa.get('status') == 'done':
        sa_block = f"""| 讀數 | 值 |
|---|---|
| G1 Spearman（item = audio vs. 該 audio 的 battle 勝率） | **{f3(sa['G1']['spearman'])} {c3(sa['G1']['ci'])}**（n_items={sa['G1']['n_items']}） |
| G2 系統內 Spearman | {f3(sa['G2_within_system']['spearman'])} {c3(sa['G2_within_system']['ci'])} |
| G3 一致率 m=0.10S（S={sa['G3']['m'] / 0.10:.3f}, m={sa['G3']['m']:.3f}） | **{f3(sa['G3']['agree_m'])} {c3(sa['G3']['agree_m_ci'])}**（n_pairs={sa['G3']['n_pairs']}） |
| G3 forced-choice（m=0） | {f3(sa['G3']['agree_forced'])} {c3(sa['G3']['agree_forced_ci'])} |
| G3 tie_frac（\\|Δ\\| < m） | {f3(sa['G3']['tie_frac'])} |
| 系統排名 Kendall τ（人類 BT vs. SingMOS 系統平均） | {f3(sa['system_rank_kendall_tau'])}（{sa['n_systems']} systems） |

**描述性，不進判定。** 預登錄 §148 只讓 SingMOS predictor 在 B4（SingMOS-Pro）進閘門；Arena 這一格永遠不計入 round-2 判定，也不改 round-2 的「未定」。
"""
    else:
        state = ('**進行中（job 152）**' if sa.get('status') == 'running' else '**待 job 152。**')
        sa_block = f"""{state} 已排入 `~/gpu_queue/p2/pending/152_bench_r2_J5c_singmos.sh`（plan key J5：`native_raw` 3,998 + `native_norm` norm800），contract＝`{sa['contract']}`。目前 native raw 已寫 {sa.get('n_scored', 0):,}／{sa.get('planned', 3998):,} 列；**部分分數不分析**，`run_job.py` 的 summary 通過 `--validate-only` 後本節才會自動填數字。
分數會寫到 `/mnt/seagate/benchmarks_r2/scores/arena/singmos__native_raw.tsv` 與 `singmos__native_norm.tsv`，格式與其他模型相同；job 完成後重跑本腳本即會自動填上 G1／G2／G3：
```bash
~/venvs/dac/bin/python -I scripts/eval/bench_r2/human_ceiling_r2.py
```
"""

    md = f"""# Round-2 benchmark screening：人類天花板與地板（2026-10-27）

預登錄：`docs/experiments/benchmark_screening_r2_20261025.md` §6.2 第 4 點（人類天花板只當參考、不進門檻）與 §7（門檻）。**本檔不改變 round-2 的任何判定、門檻或 summary**；所有數字都是描述性參考。
腳本：`scripts/eval/bench_r2/human_ceiling_r2.py`（新檔；重用 `analyze_r2.py` 的 `corr_stats`／`boot_idx`／`center_by`／`bradley_terry` 與 `bench_datasets.py` 的 manifest／PLAN，不改這兩支）。
抽樣：split-half 每資料集 **2,000 次、seed `20261025`**；報中位數與 percentile 95% 區間。地板＝ρ 0／一致率 0.50；「perm 95%」是把 item 對齊或票的極性打亂後同一統計量的 95th percentile（＝在這個樣本大小下，單靠雜訊可以達到的上界）。

**怎麼讀 split-half 與 SB**：`r_half` 是兩位「半panel」評分之間的相關；`SB = 2r/(1+r)`（Spearman-Brown）把它外推到完整 panel 的 reliability。指標是對**觀測到的人類平均**算相關，古典測量理論下它的上限是 **√SB**，不是 SB 本身；因此「佔天花板比例」用 `指標值 / √SB`。表中兩者都列，避免把 reliability 當成 correlation 直接比。
未做 SB 以外的不確定性修正；每個資料集的 rater 結構不同（ODAQ 26 位固定聽眾、MC522 10 個匿名 slot／clip、SongEval 每首 4 位、SingMOS-Pro 每 clip 5 位、AIME 每 snippet 12 票），細節見各節。

## 1. 對照表（每個 dataset 一列；主要統計量）

| dataset | split-half SB（中位數 [95%]） | √SB | 單一評分者 vs 其餘（中位數 [95%]） | 地板 | perm 95% | 現行門檻 | r2 最佳獨立指標（值） | 佔天花板 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(g1_rows)}

pairwise 的 G3 天花板（人類共識能預測單票的上限；BT 配適一半、預測另一半）：

| dataset | BT held-out 準確率（中位數 [95%]） | 地板 | perm 95% | 現行門檻 | r2 最佳獨立指標（值） | 佔天花板 |
|---|---|---|---|---|---|---|
{chr(10).join(g3_rows)}

- 「r2 最佳獨立指標」取自 `results/benchmark_screening_r2_20261025_summary.json` 的 `decision`，只考慮 `analyze_r2.INDEP` 標為獨立的指標與 dataset 配對；ODAQ 用 G6a、pairwise 用 G3、其餘用 G1。SongEval 依 §5 沒有獨立指標（B3 重疊），故為 n/a。
- Arena 的 G1 無法算 split-half：3,998 個音檔、2,000 個 battle；扣掉 TIE／BOTH_BAD 後 {ar['n_items']:,} 個音檔各只有 {ar['max_battles_per_item']} 張決定票。
- AIME 每個 unique pair 只有 1 票，沒有 pair 內重複票；它的 vote-level 上限用系統層級 BT（同 Arena），另加 snippet 勝率的 split-half（見 §3）。
- SingMOS × Arena：{'已完成，見 §4' if sa.get('status') == 'done' else '待 job 152，見 §4'}。

## 2. 各資料集（item-level）

### 2.1 B1 ODAQ（MUSHRA；26 位聽眾 × 30 trials × 8 conditions）

G1/G2/G3 不適用，round-2 用 G6a（trial 內 Spearman）。`analyze_r2` 的 G6a **只在 trial 內對 Q1–Q5 排名**（LP3.5／LP7／Ref 不進排名），並用 `lp_visible` 把頻寬看不到的 Q 條件拿掉（LP trials 才有這個限制；16 kHz 模型只剩 Q1，24 kHz 模型剩 Q1–Q3，44 kHz 以上全可見）。人類沒有頻寬限制，所以主列用**全部 Q1–Q5、30/30 trials**（對應最好的 r2 獨立指標 pam，path 44）；另列 16 kHz（AES／SingMOS；4 個 LP trials 只剩 1 個 Q 條件、不足 3 條件，實際 26 trials）與 24 kHz（MuQ-Eval／CMI-RM／SongEval；30 trials）規則下的子集，及 item level／全 8 條件當對照。

| 層級 | split-half r（trial 平均） | SB [95%] | √SB | 單人 vs 其餘 | perm 95% |
|---|---|---|---|---|---|
| **within-trial, Q1–Q5（G6a 主列；30 trials）** | {f3(o['within_trial_allQ']['r_half_median'])} | {f3(o['within_trial_allQ']['sb_median'])} | {f3(o['within_trial_allQ']['sqrt_sb'])} | {f3(o['within_trial_allQ']['single_vs_rest']['median'])} {c3(o['within_trial_allQ']['single_vs_rest']['ci'])} | {f3(o['within_trial_allQ']['perm95_r_half'])} |
| within-trial, 16 kHz 規則（{o['within_trial_g6a16']['n_trials_used']} trials） | {f3(o['within_trial_g6a16']['r_half_median'])} | {f3(o['within_trial_g6a16']['sb_median'])} | {f3(o['within_trial_g6a16']['sqrt_sb'])} | {f3(o['within_trial_g6a16']['single_vs_rest']['median'])} {c3(o['within_trial_g6a16']['single_vs_rest']['ci'])} | {f3(o['within_trial_g6a16']['perm95_r_half'])} |
| within-trial, 24 kHz 規則（{o['within_trial_g6a24']['n_trials_used']} trials） | {f3(o['within_trial_g6a24']['r_half_median'])} | {f3(o['within_trial_g6a24']['sb_median'])} | {f3(o['within_trial_g6a24']['sqrt_sb'])} | {f3(o['within_trial_g6a24']['single_vs_rest']['median'])} {c3(o['within_trial_g6a24']['single_vs_rest']['ci'])} | {f3(o['within_trial_g6a24']['perm95_r_half'])} |
| within-trial, all 8 conditions | {f3(o['within_trial_all8']['r_half_median'])} | {f3(o['within_trial_all8']['sb_median'])} | {f3(o['within_trial_all8']['sqrt_sb'])} | {f3(o['within_trial_all8']['single_vs_rest']['median'])} {c3(o['within_trial_all8']['single_vs_rest']['ci'])} | {f3(o['within_trial_all8']['perm95_r_half'])} |
| item level（240 檔，參考；不是 G6a 的單位） | {f3(o['item_level']['r_half_median'])} | {f3(o['item_level']['sb_median'])} | {f3(o['item_level']['sqrt_sb'])} | {f3(o['item_level']['single_vs_rest']['median'])} {c3(o['item_level']['single_vs_rest']['ci'])} | {f3(o['item_level']['perm95_r_half'])} |

trials 數：Q1–Q5＝{o['within_trial_allQ']['n_trials_used']}、16 kHz 規則＝{o['within_trial_g6a16']['n_trials_used']}、24 kHz 規則＝{o['within_trial_g6a24']['n_trials_used']}、all8＝{o['within_trial_all8']['n_trials_used']}。單人 vs 其餘在 within-trial 版是「單一聽眾在該 trial 的可見 Q 條件上排名」對「其餘 25 人的平均排名」，對該子集所有 trial × 26 人取平均。within-trial 的 split-half r 用 trial **平均**（G6a 本身就是 30 個 trial Spearman 的平均；這裡中位數會飽和在 1.0，已改報平均）。

### 2.2 B5 MC522（522 首真實錄音；10 位評分者 × 4 軸）

人類逐評分者分數在 `research/eval/aes_human_ratings/AES_natural_music.jsonl`（`/musiccaps/` 那 549 筆；與 522 個 manifest item 的交集＝{mc['n_units']}）。manifest 的 `human_PQ` 與這份 10 人平均的最大絕對差＝{mc['manifest_mean_check_max_abs_diff']:.2e}（同一來源）。這份檔案只給每個 clip 的 10 個分數（沒有跨 clip 可追蹤的評分者 ID），因此無法確認 slot 是否同一人；每個 item 獨立隨機對半（5/5）。

| 軸 | split-half r | SB [95%] | √SB | 單人 vs 其餘 |
|---|---|---|---|---|
| **PQ（primary）** | {f3(mc['axes']['PQ']['r_half_median'])} | {f3(mc['axes']['PQ']['sb_median'])} {c3(mc['axes']['PQ']['sb_ci'])} | {f3(mc['axes']['PQ']['sqrt_sb'])} | {f3(mc['axes']['PQ']['single_vs_rest']['median'])} {c3(mc['axes']['PQ']['single_vs_rest']['ci'])} |
| PC | {f3(mc['axes']['PC']['r_half_median'])} | {f3(mc['axes']['PC']['sb_median'])} {c3(mc['axes']['PC']['sb_ci'])} | {f3(mc['axes']['PC']['sqrt_sb'])} | {f3(mc['axes']['PC']['single_vs_rest']['median'])} |
| CE | {f3(mc['axes']['CE']['r_half_median'])} | {f3(mc['axes']['CE']['sb_median'])} {c3(mc['axes']['CE']['sb_ci'])} | {f3(mc['axes']['CE']['sqrt_sb'])} | {f3(mc['axes']['CE']['single_vs_rest']['median'])} |
| CU | {f3(mc['axes']['CU']['r_half_median'])} | {f3(mc['axes']['CU']['sb_median'])} {c3(mc['axes']['CU']['sb_ci'])} | {f3(mc['axes']['CU']['sqrt_sb'])} | {f3(mc['axes']['CU']['single_vs_rest']['median'])} |

### 2.3 B3 SongEval（S800；每首 4 位標註者，5 維取 4 位平均）

每首獨立隨機分 2/2；primary＝Musicality。標註者有 4 個不相交的 block（a01–04、a05–08、a09–12、a13–16），所以「單人 vs 其餘」主報每首隨機抽一位的 2,000 次分布，另列同一位標註者在其負責歌曲上的 ρ 中位數當穩健性檢查（per-ID 見 JSON）。

| 維度 | split-half r | SB [95%] | √SB | 單人 vs 其餘 |
|---|---|---|---|---|
| **Musicality（primary）** | {f3(se['dims']['Musicality']['r_half_median'])} | {f3(se['dims']['Musicality']['sb_median'])} {c3(se['dims']['Musicality']['sb_ci'])} | {f3(se['dims']['Musicality']['sqrt_sb'])} | {f3(se['dims']['Musicality']['single_vs_rest']['median'])} {c3(se['dims']['Musicality']['single_vs_rest']['ci'])} |
| Coherence | {f3(se['dims']['Coherence']['r_half_median'])} | {f3(se['dims']['Coherence']['sb_median'])} {c3(se['dims']['Coherence']['sb_ci'])} | {f3(se['dims']['Coherence']['sqrt_sb'])} | {f3(se['dims']['Coherence']['single_vs_rest']['median'])} |
| Memorability | {f3(se['dims']['Memorability']['r_half_median'])} | {f3(se['dims']['Memorability']['sb_median'])} {c3(se['dims']['Memorability']['sb_ci'])} | {f3(se['dims']['Memorability']['sqrt_sb'])} | {f3(se['dims']['Memorability']['single_vs_rest']['median'])} |
| Clarity | {f3(se['dims']['Clarity']['r_half_median'])} | {f3(se['dims']['Clarity']['sb_median'])} {c3(se['dims']['Clarity']['sb_ci'])} | {f3(se['dims']['Clarity']['sqrt_sb'])} | {f3(se['dims']['Clarity']['single_vs_rest']['median'])} |
| Naturalness | {f3(se['dims']['Naturalness']['r_half_median'])} | {f3(se['dims']['Naturalness']['sb_median'])} {c3(se['dims']['Naturalness']['sb_ci'])} | {f3(se['dims']['Naturalness']['sqrt_sb'])} | {f3(se['dims']['Naturalness']['single_vs_rest']['median'])} |

### 2.4 B4 SingMOS-Pro test（3,101 clips；每 clip 5 位評審）

rater 用匿名 slot（`judge_id` 跨 clip 不保證同一人），每 clip 獨立隨機、平衡對半（5 位者用 2/2；若 n 為奇數會丟 1 位，本 test split 沒有這種 clip）。n_judge 分布：{', '.join(f'{k} 位 × {v}' for k, v in sm['n_judge_counts'].items())}。

- split-half r＝{f3(sm['split_half']['r_half_median'])}、SB＝{f3(sm['split_half']['sb_median'])} {c3(sm['split_half']['sb_ci'])}、**√SB＝{f3(sm['split_half']['sqrt_sb'])}**
- 單一評審 vs 其餘＝{f3(sm['split_half']['single_vs_rest']['median'])} {c3(sm['split_half']['single_vs_rest']['ci'])}
- perm 95% r＝{f3(sm['split_half']['perm95_r_half'])}
- 對照：SingMOS predictor 在 B4 的 G1＝{f3(b_sm[1])} {c3(b_sm[2])}，約為 √SB 的 {b_sm[1] / sm['split_half']['sqrt_sb']:.2f}。

## 3. 成對資料集：vote-level 天花板

AIME：7,800 張 Music Quality 票、**7,800 個 unique pair（每 pair 恰 1 票）**；每 snippet 有 12 票（對 12 個不同對手）。所以 pair 內沒有重複票可算一致率；vote-level 上限用「系統層級 BT 配適一半票、預測另一半」（同 Arena 的作法），另用 12 票對半算 snippet 勝率的 split-half（這是 AIME G1 人類 target 的 reliability）。

Music Arena：2,000 battles、1,412 張決定票（A 670／B 742／TIE 155／BOTH_BAD 433；決定率 {pct(ar['decision_rate'])}）。每 battle 恰 1 票且沒有重複 pair，因此同樣用系統層級 BT。**BT 上限只利用「系統共識」，不含 prompt 或音檔內容資訊；它是任何指標能達到的系統層級共識上限，不是任何單一 audio-based metric 的硬上限。** TIE／BOTH_BAD 佔 {pct(1 - ar['decision_rate'])}，G3 只評決定票，這一點本身限制了 G3 的意義。

| | AIME | Music Arena |
|---|---|---|
| 可用的票 | {ai['n_votes']:,}（13 systems） | {ar['n_battles_decided']:,}（{ar['n_systems']} systems） |
| BT held-out 準確率（中位數 [95%]） | {f3(a_g3['bt_half_accuracy_median'])} {c3(a_g3['bt_half_accuracy_ci'])} | {f3(ar_g3['bt_half_accuracy_median'])} {c3(ar_g3['bt_half_accuracy_ci'])} |
| 地板／perm 95% | 0.500／{f3(a_g3['perm95_accuracy'])} | 0.500／{f3(ar_g3['perm95_accuracy'])} |
| G1 人類 target split-half SB（√SB） | {f3(ai['G1']['sb_median'])}（{f3(ai['G1']['sqrt_sb'])}） | 不可算（≤{ar['max_battles_per_item']} 票／音檔） |
| G1 單票 vs 其餘 | {f3(ai['G1']['single_vs_rest']['median'])} {c3(ai['G1']['single_vs_rest']['ci'])} | 不可算 |
| r2 最佳獨立指標 | G3 {b_a3[0]} {f3(b_a3[1])}；G1 {ref['aime']['G1'][0]} {f3(ref['aime']['G1'][1])} | G3 {b_ar[0]} {f3(b_ar[1])}；G1 {ref['arena']['G1'][0]} {f3(ref['arena']['G1'][1])} |

## 4. SingMOS × Music Arena（B6；描述性，不進判定）

{sa_block}

## 5. 讀法與限制

1. **天花板不是門檻**：G1 0.50／G3 0.55 是預登錄門檻；本檔只回答「人類評分本身有多穩定」，不改變任何判定。
2. **√SB 才是可比的相關上限**：SB 是完整 panel 的 reliability。若把 SB 直接和指標的 ρ 比，會把上限高估（例：MC522 PQ SB={f3(mc_s['sb_median'])}，但觀測平均的可預測上限是 √SB={f3(mc_s['sqrt_sb'])}）。表中「佔天花板」一律用 √SB（G3 用 BT held-out 準確率）。
3. **split-half 的 rater 結構不同**：ODAQ 用固定的 26 人 panel 對半（同一群人跨 trial）；MC522／SingMOS 只有匿名 slot、無法跨 item 追蹤同一人，所以每個 item 獨立隨機對半；SongEval 每首 4 位、每首獨立分半。這些都是標準 split-half 的合理延伸，但數字不能跨資料集當成同一種「rater 一致性」直接排序。
4. **AIME／Arena 沒有重複 pair**：兩者的 vote-level 上限是系統層級 BT 的 held-out 準確率。它忽略 prompt／音檔內容，所以對「看得到音檔的指標」是一個偏保守（只給系統共識）的上限；反過來，它也可能是任何只依賴系統身份的模型的樂觀值。
5. **Arena 的 TIE／BOTH_BAD**：{pct(1 - ar['decision_rate'])} 的 battle 不是 A/B 二選一，G3 的 0.55 只在決定票上有意義；人類自己在這 {pct(1 - ar['decision_rate'])} 上沒有可用的 forced-choice 答案。
6. **SingMOS 的域外警告**：預登錄 §148 只讓 SingMOS predictor 在 B4 進閘門。Arena 這格只描述，不能拿來推翻或支持「SingMOS 可用於純器樂」。
7. **沒有針對我們 arm 的人類評分**：本檔全部是公開 benchmark 的人類評分，與 round-1／round-2 的「未定」判定無關。

## 6. 重現

```bash
cd /home/kojiek/MeanAudio
~/venvs/dac/bin/python -I scripts/eval/bench_r2/human_ceiling_r2.py \\
  --out-md docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.md \\
  --out-json docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.json
# SingMOS x Arena（job 152）完成後同一條指令重跑即會補上 §4
```

輸出 JSON：`docs/experiments/results/benchmark_screening_r2_human_ceiling_20261027.json`（每個資料集的逐項數字、per-rater/per-annotator 值、permutation 上界）。
"""
    return md


if __name__ == '__main__':
    main()
