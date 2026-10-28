#!/usr/bin/env python3
"""Queue action for one autoresearch-clap job (phase 0 or one rung).

    --preflight      prereg / checkpoint / split / plan sha, disk floor; rc 75 = retry later
    --validate-only  every planned trial has a valid summary (or p0_summary passed)
    (default)        mode phase0 -> phase0.py all; mode round -> driver.py then analyze.py

The contract (`GPU_QUEUE_CONTRACT`) fixes mode, scratch, round and plan; nothing is
decided here. Run with -I, so this file bootstraps its own sys.path.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import common                                   # noqa: E402

DISK_REFUSE = 20_000_000_000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contract():
    return json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())


def fixed(c):
    fp = c['fixed_protocol']
    return fp['mode'], Path(fp['scratch']), fp['round'], fp.get('plan')


def check_prereg(c):
    fp = c['fixed_protocol']
    got = sha(common.PREREG)
    if got != fp['prereg_sha256']:
        return f'prereg sha changed: {got} != {fp["prereg_sha256"]}'
    return None


def check_checkpoints():
    bad = [s for s in common.SEEDS
           if not Path(common.CHECKPOINTS[s]).is_file()
           or common.sha256(common.CHECKPOINTS[s]) != common.CHECKPOINT_SHA256[s]]
    return f'checkpoint sha mismatch: {bad}' if bad else None


def check_splits(scratch):
    man = common.read_json(Path(scratch) / 'tsv' / 'manifest.json')
    if not man:
        return f'missing split manifest under {scratch}/tsv (phase 0 step verify)'
    prereg = common.load_prereg()['splits']
    for name in ('dev256', 'dev1024', 'heldout1024'):
        got = man['splits'].get(name) or {}
        want = prereg.get(name, {}).get('sha256')
        if want and got.get('sha256') != want:
            return f'{name} split sha differs from prereg'
    return None


def check_plan(scratch, round_id, plan_path):
    if is_phase0(round_id):
        return None
    if not plan_path:
        return 'round mode without a plan'
    p = Path(plan_path)
    if not p.is_file():
        return f'missing plan {p}'
    try:
        plan = common.load_plan(p)
    except SystemExit as exc:
        return str(exc)
    if plan['round'] != round_id:
        return f'plan round {plan["round"]} != contract round {round_id}'
    got = sha(p)
    want = common.read_json(Path(scratch) / 'rounds' / f'{round_id}.sha256')
    if want and want.get('sha256') != got:
        return f'plan sha changed since registration ({got})'
    return None


def is_phase0(round_id):
    return str(round_id).lower() in ('p0', 'phase0')


def preflight(c):
    mode, scratch, round_id, plan = fixed(c)
    checks = [lambda: check_prereg(c), check_checkpoints]
    if mode == 'phase0':
        checks.append(lambda: None if common.MC_TSV.is_file() else f'missing {common.MC_TSV}')
    else:
        checks += [lambda: check_splits(scratch),
                   lambda: check_plan(scratch, round_id, plan)]
    for fn in checks:
        err = fn()
        if err:
            print(f'[preflight] FAIL {err}', flush=True)
            return 2
    free = common.free_bytes(scratch)
    if free < DISK_REFUSE:
        print(f'[preflight] WAIT disk {free / 1e9:.1f} GB < 20 GB', flush=True)
        return 75
    print(f'[preflight] ok mode={mode} round={round_id} free={free / 1e9:.1f} GB', flush=True)
    return 0


def run_phase0(scratch):
    cmd = [common.PY, str(HERE / 'phase0.py'), 'all', '--scratch', str(scratch)]
    print(f'[run] {" ".join(cmd)}', flush=True)
    return subprocess.run(cmd, cwd=str(common.ROOT)).returncode


def run_round(scratch, round_id, plan):
    out = Path(scratch) / 'rounds_out'
    out.mkdir(parents=True, exist_ok=True)
    cmd = [common.PY, str(HERE / 'driver.py'), '--configs', str(plan),
           '--scratch', str(scratch)]
    print(f'[run] {" ".join(cmd)}', flush=True)
    rc = subprocess.run(cmd, cwd=str(common.ROOT)).returncode
    rec = {'round': round_id, 'driver_rc': rc, 'finished_at': common.now_iso()}
    if rc == 0:
        cmd2 = [common.PY, str(HERE / 'analyze.py'), '--configs', str(plan),
                '--scratch', str(scratch),
                '--out', str(out / f'{round_id}_summary.json')]
        print(f'[run] {" ".join(cmd2)}', flush=True)
        rc2 = subprocess.run(cmd2, cwd=str(common.ROOT)).returncode
        rec['analyze_rc'] = rc2
        rc = rc2
    common.atomic_json(out / f'{round_id}_run.json', rec)
    return rc


def validate(c):
    mode, scratch, round_id, plan = fixed(c)
    if mode == 'phase0':
        p0 = common.read_json(Path(scratch) / 'phase0' / 'p0_summary.json')
        if not p0 or p0.get('status') != 'pass':
            print(f'[validate] FAIL phase0 status={(p0 or {}).get("status")}', flush=True)
            return 2
        print('[validate] ok phase0 pass', flush=True)
        return 0
    p = common.load_plan(plan)
    missing, bad = [], []
    for cfg in p['configs']:
        for ckpt in p['checkpoints']:
            seed = (int(cfg['seed']) if len(p['seeds']) == 1 and cfg.get('seed') is not None
                    else int(p['seeds'][0]))
            t = common.trial_dir(scratch, p, cfg, ckpt, seed)
            rec = common.read_json(t / 'summary.json') or {}
            want = common.config_sha(cfg, ckpt, seed, p['split'])
            if not rec:
                missing.append(str(t))
            elif rec.get('config_sha') != want:
                bad.append(str(t))
    if missing or bad:
        print(f'[validate] FAIL missing={len(missing)} bad={len(bad)} '
              f'first={(missing + bad)[0]}', flush=True)
        return 2
    print(f'[validate] ok {sum(len(p["configs"]) for _ in [0]) * len(p["checkpoints"])} trials',
          flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    c = contract()
    if a.preflight:
        return preflight(c)
    if a.validate_only:
        return validate(c)
    mode, scratch, round_id, plan = fixed(c)
    if mode == 'phase0':
        return run_phase0(scratch)
    return run_round(scratch, round_id, plan)


if __name__ == '__main__':
    raise SystemExit(main())
