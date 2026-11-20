#!/usr/bin/env python
"""Run every trial of one round, skipping trials whose summary.json already matches.

    driver.py --configs SCRATCH/rounds/r1.json --scratch SCRATCH [--split dev256]
              [--checkpoints s14159265,...] [--seeds 42] [--no-fad] [--limit N]

Trial-level resume is the unit of work recovery: a killed job restarts the same
configs.json and only the missing/failed trials cost GPU again. Writes
rounds_out/<round>_trials.jsonl (one line per trial) and prints a compact table.
Exit 4 if any trial ended in status "error", 3 if the disk guard fires.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                   # noqa: E402
import trial as trial_mod                       # noqa: E402

DISK_REFUSE = 20_000_000_000


def seeds_for(plan, cfg, seeds):
    if len(seeds) == 1 and cfg.get('seed') is not None:
        return [int(cfg['seed'])]
    return [int(s) for s in seeds]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--configs', required=True)
    p.add_argument('--scratch', default=str(common.SCRATCH))
    p.add_argument('--split', default=None)
    p.add_argument('--checkpoints', default=None)
    p.add_argument('--seeds', default=None)
    p.add_argument('--no-fad', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--keep-audio', action='store_true')
    a = p.parse_args(argv)
    plan = common.load_plan(a.configs)
    split = a.split or plan['split']
    ckpts = ([s for s in a.checkpoints.split(',') if s] if a.checkpoints else list(plan['checkpoints']))
    seeds = ([s for s in a.seeds.split(',') if s] if a.seeds else [str(s) for s in plan['seeds']])
    jobs = []
    for cfg in plan['configs']:
        for ckpt in ckpts:
            for seed in seeds_for(plan, cfg, seeds):
                jobs.append((cfg, ckpt, seed))
    if a.limit:
        jobs = jobs[:a.limit]
    out_dir = Path(a.scratch) / 'rounds_out'
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{plan['round']}_trials.jsonl"
    t_start = time.time()
    counts = {}
    errors = []
    with open(jsonl, 'a', encoding='utf-8') as jf:
        for k, (cfg, ckpt, seed) in enumerate(jobs, 1):
            free = common.free_bytes(Path(a.scratch))
            if free < DISK_REFUSE:
                raise SystemExit(f'[FAIL] disk guard: {free / 1e9:.1f} GB free < 20 GB')
            t0 = time.time()
            out = trial_mod.run_trial(a.configs, cfg, ckpt, seed, a.scratch, split,
                                      do_fad=not a.no_fad, keep_audio=a.keep_audio)
            status = out.get('status')
            counts[status] = counts.get(status, 0) + 1
            rec = {'at': common.now_iso(), 'config_id': cfg['id'], 'checkpoint': ckpt,
                   'seed': seed, 'status': status, 'skipped': bool(out.get('skipped')),
                   'delta_lvl30_clap': (out.get('delta_vs_baseline') or {}).get('lvl30_clap'),
                   'delta_lvl30_pq': (out.get('delta_vs_baseline') or {}).get('lvl30_pq'),
                   'delta_fad': (out.get('delta_vs_baseline') or {}).get('fad'),
                   'gates': out.get('gates'), 'seconds': round(time.time() - t0, 1),
                   'error': out.get('error')}
            jf.write(json.dumps(rec, ensure_ascii=False) + '\n')
            jf.flush()
            if status == 'error':
                errors.append((cfg['id'], ckpt, seed, out.get('error')))
            print(f"[driver {plan['round']}] {k}/{len(jobs)} {cfg['id']} {ckpt} s{seed} "
                  f"{status} dCLAP={rec['delta_lvl30_clap']} dPQ={rec['delta_lvl30_pq']} "
                  f"dFAD={rec['delta_fad']} {rec['seconds']}s", flush=True)
    summary = {'round': plan['round'], 'split': split, 'n_jobs': len(jobs), 'counts': counts,
               'errors': errors, 'wall_s': round(time.time() - t_start, 1),
               'finished_at': common.now_iso()}
    common.atomic_json(out_dir / f"{plan['round']}_driver.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    return 4 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
