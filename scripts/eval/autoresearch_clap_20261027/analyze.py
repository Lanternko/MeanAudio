#!/usr/bin/env python
"""Paired analysis of one round: per-trial deltas, bootstrap CIs, promotion list.

    analyze.py --configs rounds/r1.json --scratch SCRATCH [--round r1] [--out PATH]

Statistics (prereg 2.3-2.5): delta = mean over the split ids of
(trial lvl30 CLAP - baseline lvl30 CLAP), paired by id; 95% CI from 10000 id
resamples with seed 20261027. sigma_used = max(sigma_seed(n), |dC1|, |dC2|) with
the C1/C2 null controls taken from the same round when present.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                   # noqa: E402

BOOT = 10000
BOOT_SEED = 20261027


def _cell_per_clip(path):
    if not Path(path).is_file():
        return {}
    return {r['id']: r['clap'] for r in common.read_per_clip(path)
            if common.finite(r.get('clap'))}


def baseline_clap(scratch, split, ckpt, cfgkey, cache={}):
    key = (str(scratch), split, ckpt, cfgkey)
    if key not in cache:
        cell = common.canonical_cell(ckpt, cfgkey, 'lvl30')
        cache[key] = _cell_per_clip(Path(cell) / Path(cell).name / 'per_clip.tsv')
    return cache[key]


def trial_clap(t):
    cell = trial_mod_level30(t)
    return _cell_per_clip(cell / cell.name / 'per_clip.tsv')


def trial_mod_level30(t):
    cfg_id = json.loads((Path(t) / 'config.json').read_text())['config']['id']
    return Path(t) / 'lvl30' / f'{cfg_id}_raw_lvl30'


def paired_stats(trial_map, base_map, ids, seed=BOOT_SEED, nboot=BOOT):
    pairs = [(trial_map[i], base_map[i]) for i in ids if i in trial_map and i in base_map]
    if len(pairs) < 2:
        return None
    tv = np.array([p[0] for p in pairs], dtype=np.float64)
    bv = np.array([p[1] for p in pairs], dtype=np.float64)
    d = tv - bv
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    means = d[idx].mean(axis=1)
    return {'delta': float(d.mean()), 'ci_low': float(np.percentile(means, 2.5)),
            'ci_high': float(np.percentile(means, 97.5)), 'n_pairs': int(len(d)),
            'n_trial': int(len(trial_map)), 'n_base': int(len(base_map))}


def load_trials(scratch, plan):
    root = Path(scratch) / 'trials' / plan['round']
    out = []
    for sd in sorted(root.glob('*/summary.json')):
        rec = common.read_json(sd)
        if rec:
            rec['_dir'] = str(sd.parent)
            out.append(rec)
    return out


def sigma_parts(trials, plan):
    n = len(common.split_ids(Path(plan.get('_scratch', '.')), plan['split'])) \
        if plan.get('_scratch') else plan.get('n', 0)
    floor = common.sigma_floor(n) if n else None
    nulls = {}
    for t in trials:
        cid = t.get('config_id')
        if cid in ('C1_null_seed7', 'C2_null_seed161803'):
            nulls[cid] = abs((t.get('delta_vs_baseline') or {}).get('lvl30_clap') or 0.0)
    vals = [v for v in (floor, *nulls.values()) if v is not None]
    return {'sigma_seed_n': floor, 'null_abs': nulls,
            'sigma_used': max(vals) if vals else None}


def promote_r1(trials, sigma, split_n):
    thr = common.THRESHOLDS['dev']
    for t in trials:
        d = (t.get('delta_vs_baseline') or {}).get('lvl30_clap')
        t['_delta'] = d
        t['_promotable'] = bool(d is not None and d >= thr
                                and t.get('status') in ('pass', 'fad_pending'))
        t['_fad_only'] = t.get('status') == 'fad_pending'
    cands = sorted([t for t in trials if t['_promotable'] and not t['_fad_only']],
                   key=lambda t: -t['_delta'])
    picked, per_family = [], {}
    for t in cands:
        fam = t.get('family')
        if len(picked) >= 8:
            break
        if per_family.get(fam, 0) >= 2:
            continue
        picked.append(t)
        per_family[fam] = per_family.get(fam, 0) + 1
    pending = sorted([t for t in trials if t['_promotable'] and t['_fad_only']
                      and t not in picked], key=lambda t: -t['_delta'])[:2]
    underpowered = len(picked) < 3
    if underpowered:
        extra = [t for t in sorted(trials, key=lambda t: -(t['_delta'] or -1e9))
                 if t not in picked][:3 - len(picked)]
        picked += extra
    return {'promoted': [t['config_id'] for t in picked],
            'fad_pending_carried': [t['config_id'] for t in pending],
            'underpowered': underpowered,
            'threshold': thr, 'sigma_used': sigma['sigma_used'],
            'expected_max_null': (sigma['sigma_used'] or 0.0) * 2.2 if sigma['sigma_used'] else None}


def promote_r2(trials, sigma):
    thr = common.THRESHOLDS['dev']
    out = []
    for t in trials:
        st = t.get('_stats') or {}
        d = st.get('delta', (t.get('delta_vs_baseline') or {}).get('lvl30_clap'))
        ok = (d is not None and d >= thr and (st.get('ci_low') or -1) > 0
              and t.get('status') in ('pass', 'fad_pending'))
        out.append(t['config_id'] if ok else None)
        t['_promotable'] = bool(ok)
    ordered = sorted([t for t in trials if t.get('_promotable')],
                     key=lambda t: -(t['_stats'] or {}).get('delta', 0.0))
    return {'promoted': [t['config_id'] for t in ordered], 'threshold': thr,
            'sigma_used': sigma['sigma_used'], 'stopping_rule_E2': not ordered}


def analyze_round(scratch, plan, out_path=None):
    scratch = Path(scratch)
    split = plan['split']
    ids = common.split_ids(scratch, split)
    trials = load_trials(scratch, plan)
    base_cache = {}
    for t in trials:
        ckpt = t['checkpoint']['name']
        cfg = json.loads((Path(t['_dir']) / 'config.json').read_text())['config']
        base = baseline_clap(scratch, split, ckpt, common.baseline_cfgkey(cfg), base_cache)
        t['_stats'] = paired_stats(trial_clap(t['_dir']), base, ids)
    plan_sigma = dict(plan, _scratch=str(scratch), n=len(ids))
    sigma = sigma_parts(trials, plan_sigma)
    round_id = plan['round']
    if round_id == 'r1':
        promotion = promote_r1(trials, sigma, len(ids))
    elif round_id == 'r2':
        promotion = promote_r2(trials, sigma)
    elif round_id == 'r3':
        singles = {t['config_id']: (t['_stats'] or {}).get('delta', -1e9) for t in trials}
        best = max(singles, key=singles.get) if singles else None
        combos = [t for t in trials if t.get('family') == 'combo']
        best_combo = max(combos, key=lambda t: (t['_stats'] or {}).get('delta', -1e9),
                         default=None)
        combo_ok = bool(best_combo and best and
                        (best_combo['_stats'] or {}).get('delta', -1e9)
                        - singles[best] >= common.THRESHOLDS['combo_over_single'])
        promotion = {'best_single': best, 'best_combo': (best_combo or {}).get('config_id'),
                     'combo_adopted': combo_ok,
                     'winner': (best_combo['config_id'] if combo_ok else best),
                     'stopping_rule_E3': not combo_ok}
    else:
        promotion = evaluate_final(trials)
    out = {
        'round': round_id, 'split': split, 'n': len(ids), 'sigma': sigma,
        'promotion': promotion,
        'trials': [{'config_id': t.get('config_id'), 'family': t.get('family'),
                    'checkpoint': t['checkpoint']['name'], 'seed': t.get('seed'),
                    'status': t.get('status'),
                    'delta_lvl30_clap': (t.get('_stats') or {}).get('delta'),
                    'ci': [(t.get('_stats') or {}).get('ci_low'),
                           (t.get('_stats') or {}).get('ci_high')],
                    'delta_lvl30_pq': (t.get('delta_vs_baseline') or {}).get('lvl30_pq'),
                    'delta_fad': (t.get('delta_vs_baseline') or {}).get('fad'),
                    'gates': t.get('gates')} for t in trials],
    }
    if out_path:
        common.atomic_json(out_path, out)
    return out


def evaluate_final(trials):
    """Rung 4/5 win criterion on the collected cells."""
    cells = [t for t in trials if (t.get('_stats') or {}).get('delta') is not None]
    deltas = [(t['_stats']['delta'], t['checkpoint']['name'], int(t.get('seed') or 0))
              for t in cells]
    if not deltas:
        return {'win': False, 'reason': 'no scored cells'}
    mean = sum(d for d, _, _ in deltas) / len(deltas)
    by_ck = {}
    by_seed = {}
    for d, ck, sd in deltas:
        by_ck.setdefault(ck, []).append(d)
        by_seed.setdefault(sd, []).append(d)
    ck_means = {k: sum(v) / len(v) for k, v in by_ck.items()}
    seed_means = {k: sum(v) / len(v) for k, v in by_seed.items()}
    gates_ok = all(all(t['gates'].get(g) is not False for g in ('G2', 'G3', 'G4', 'G5', 'G6', 'G7'))
                   for t in cells)
    n_ck = len(ck_means)
    positive_ck = sum(v > 0 for v in ck_means.values())
    if n_ck >= 3:
        thr = common.THRESHOLDS['heldout']
        win = (mean >= thr and positive_ck >= 2
               and all(v > 0 for v in seed_means.values()) and gates_ok)
        label = 'win' if win else ('checkpoint_specific' if mean >= thr and positive_ck < 2
                                   else 'not_replicated')
    else:
        thr = common.THRESHOLDS['dev']
        win = mean >= thr and positive_ck >= 2 and gates_ok
        label = 'win' if win else 'not_adopted'
    return {'mean_delta': mean, 'checkpoint_means': ck_means, 'seed_means': seed_means,
            'gates_ok': gates_ok, 'threshold': thr, 'win': win, 'label': label}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--configs', required=True)
    p.add_argument('--scratch', default=str(common.SCRATCH))
    p.add_argument('--out', default=None)
    a = p.parse_args(argv)
    plan = common.load_plan(a.configs)
    out = analyze_round(a.scratch, plan,
                        a.out or Path(a.scratch) / 'rounds_out' / f"{plan['round']}_summary.json")
    print(json.dumps({'round': out['round'], 'sigma': out['sigma'],
                      'promotion': out['promotion'],
                      'top': sorted([t for t in out['trials'] if t['delta_lvl30_clap'] is not None],
                                    key=lambda t: -t['delta_lvl30_clap'])[:8]},
                     ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
