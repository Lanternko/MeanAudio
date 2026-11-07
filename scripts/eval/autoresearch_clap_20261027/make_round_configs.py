#!/usr/bin/env python
"""Build the next round's plan from the prereg rules (doc 3.3), never by hand.

    make_round_configs.py --round r1 --out SCRATCH/rounds/r1.json
    make_round_configs.py --round r2 --summary SCRATCH/rounds_out/r1_summary.json \
                          --prev SCRATCH/rounds/r1.json --out SCRATCH/rounds/r2.json
    make_round_configs.py --round r3 --summary ..._r2_summary.json --prev .../r2.json ...
    make_round_configs.py --round r4 --winner-id F2_short --prev .../r3.json ...
    make_round_configs.py --round r5 --winner-json '<config json>' ...

Rules: rung 2 = top 8 (<=2 per family) + <=2 FAD-pending, dev1024, one checkpoint,
seed 42; rung 3 = pairwise combos over the six registered complementary family
pairs, dev1024; rung 4 = heldout1024 x 3 checkpoints x 3 seeds; rung 5 = full5521
x 3 checkpoints, seed 42.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                   # noqa: E402

PAIRS = (('F2', 'F3'), ('F2', 'F4'), ('F2', 'F5'), ('F1', 'F2'), ('F1', 'F3'), ('F3', 'F5'))
RESERVED = ('interval', 'schedule', 'geometry', 'adg_gamma', 'apg_eta')


def r1_plan():
    space = common.load_prereg()['search_space_rung1']
    return {'round': 'r1', 'split': space['split'], 'checkpoints': [common.SEEDS[0]],
            'seeds': [space['seed']], 'configs': [dict(c) for c in space['configs']]}


def _by_id(path):
    plan = common.load_plan(path)
    return {c['id']: dict(c) for c in plan['configs']}


def r2_plan(summary_path, prev_path):
    summary = common.read_json(summary_path)
    if not summary:
        raise SystemExit(f'[FAIL] unreadable summary {summary_path}')
    prom = summary['promotion']
    ids = list(prom.get('promoted') or []) + list(prom.get('fad_pending_carried') or [])
    if len(ids) > 10:
        raise SystemExit(f'[FAIL] rung 2 takes at most 10 configs, got {len(ids)}')
    cfg_by_id = _by_id(prev_path)
    missing = [i for i in ids if i not in cfg_by_id]
    if missing:
        raise SystemExit(f'[FAIL] promoted ids not in {prev_path}: {missing}')
    return {'round': 'r2', 'split': 'dev1024', 'checkpoints': [common.SEEDS[0]],
            'seeds': [42], 'configs': [cfg_by_id[i] for i in ids],
            'source': {'summary': str(summary_path), 'underpowered': prom.get('underpowered'),
                       'sigma_used': prom.get('sigma_used')}}


def _merge(base, add):
    out = dict(base)
    for key, val in add.items():
        if key in ('id', 'family', 'control', 'is_numeric_control', 'seed', 'prompt_text'):
            continue
        if key == 'prompt':
            if val != 'stock':
                out['prompt'] = val
            continue
        if key in RESERVED or key == 'cfg':
            out[key] = val
        elif key in ('steps', 'grid'):
            continue                                # rung 3 keeps 25/ uniform
        else:
            out[key] = val
    return out


def r3_plan(summary_path, prev_path):
    summary = common.read_json(summary_path)
    if not summary:
        raise SystemExit(f'[FAIL] unreadable summary {summary_path}')
    cfg_by_id = _by_id(prev_path)
    scored = [t for t in summary['trials'] if t.get('delta_lvl30_clap') is not None]
    best = {}
    for t in scored:
        cfg = cfg_by_id.get(t['config_id'])
        if not cfg:
            continue
        fam = cfg.get('family')
        if fam not in ('F1', 'F2', 'F3', 'F4', 'F5'):
            continue
        if fam not in best or t['delta_lvl30_clap'] > best[fam][0]:
            best[fam] = (t['delta_lvl30_clap'], cfg)
    combos = []
    for a, b in PAIRS:
        if a not in best or b not in best:
            continue
        ca, cb = best[a][1], best[b][1]
        merged = _merge(ca, cb)
        merged['id'] = f'X_{a}_{ca["id"]}__{b}_{cb["id"]}'
        merged['family'] = 'combo'
        merged['components'] = [ca['id'], cb['id']]
        combos.append(merged)
    if not combos:
        raise SystemExit('[FAIL] rung 3 needs at least one pair with both families promoted')
    return {'round': 'r3', 'split': 'dev1024', 'checkpoints': [common.SEEDS[0]],
            'seeds': [42], 'configs': combos,
            'source': {'summary': str(summary_path),
                       'bases': {k: v[1]['id'] for k, v in best.items()}}}


def single_plan(round_id, cfg, split, checkpoints, seeds):
    cfg = dict(cfg)
    cfg['id'] = f'WIN_{cfg["id"]}' if not str(cfg['id']).startswith('WIN_') else cfg['id']
    cfg.pop('components', None)
    return {'round': round_id, 'split': split, 'checkpoints': checkpoints,
            'seeds': seeds, 'configs': [cfg]}


def resolve_winner(args):
    if args.winner_json:
        return json.loads(args.winner_json)
    if not args.winner_id or not args.prev:
        raise SystemExit('[FAIL] give --winner-json or --winner-id with --prev')
    cfg = _by_id(args.prev).get(args.winner_id)
    if cfg is None:
        raise SystemExit(f'[FAIL] winner {args.winner_id} not in {args.prev}')
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--round', required=True, choices=('r1', 'r2', 'r3', 'r4', 'r5'))
    p.add_argument('--out', required=True)
    p.add_argument('--summary', help='previous round summary.json')
    p.add_argument('--prev', help='previous round plan (to resolve config dicts)')
    p.add_argument('--winner-id')
    p.add_argument('--winner-json')
    a = p.parse_args(argv)
    if a.round == 'r1':
        plan = r1_plan()
    elif a.round == 'r2':
        plan = r2_plan(a.summary, a.prev)
    elif a.round == 'r3':
        plan = r3_plan(a.summary, a.prev)
    elif a.round == 'r4':
        plan = single_plan('r4', resolve_winner(a), 'heldout1024', list(common.SEEDS), [42, 7, 161803])
    else:
        plan = single_plan('r5', resolve_winner(a), 'full5521', list(common.SEEDS), [42])
    plan['written_at'] = common.now_iso()
    common.atomic_json(a.out, plan)
    print(json.dumps({'round': plan['round'], 'split': plan['split'],
                      'checkpoints': plan['checkpoints'], 'seeds': plan['seeds'],
                      'n_configs': len(plan['configs']),
                      'ids': [c['id'] for c in plan['configs']]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
