#!/usr/bin/env python3
"""Queue action for one round-2 benchmark-screening job (contract fixed_protocol: job, datasets, models).

  --preflight      sha check, disk floor, manifests exist; rc 75 while another decode_cache.py is still writing
  (default)        decode caches (idempotent) -> score_r2.py per dataset/model -> summary json
  --validate-only  every planned unit has a score row or an error row; summary json present
Run with -I; the MEva model runs in its own venv (score_r2 is lazy-import).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DAC = '/home/kojiek/venvs/dac/bin/python'
MEVA = '/home/kojiek/MeanAudio/runtime/meva_20261003/venv/bin/python'
sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location('bench_r2_datasets', HERE / 'bench_datasets.py')
D = importlib.util.module_from_spec(_sp)
sys.modules['bench_r2_datasets'] = D
_sp.loader.exec_module(D)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contract():
    return json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())


def decoders_running():
    me = os.getpid()
    for p in Path('/proc').iterdir():
        if p.name.isdigit() and int(p.name) != me:
            try:
                cmd = (p / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            if any(x.endswith(b'decode_cache.py') for x in cmd[:4]):
                return True
    return False


def preflight(c):
    for k, v in c['bindings'].items():
        if k.endswith('_sha256') and not k.startswith('launcher'):
            f = c['bindings'][k[:-7]]
            if sha(f) != v:
                sys.exit(f'{k[:-7]} sha mismatch')
    if shutil.disk_usage(D.ROOT).free < c['fixed_protocol']['min_free_bytes']:
        sys.exit('free space below the contract floor')
    for ds in c['fixed_protocol']['datasets']:
        if not (D.MAN / f'{ds}.tsv').exists() and not list(D.MAN.glob(f'{ds}*')):
            sys.exit(f'manifest missing: {ds}')
    if decoders_running():
        print('another decode_cache.py is running; wait')
        return 75
    print('preflight ok')
    return 0


def sh(cmd):
    print('+', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}).returncode


def run(c):
    fp = c['fixed_protocol']
    job, rc_all = fp['job'], 0
    for ds in fp['datasets']:
        rc = sh([DAC, '-I', str(HERE / 'decode_cache.py'), ds, '--job', job, '--workers', '6'])
        if rc:
            return rc
    for ds in fp['datasets']:
        for m in fp['models'][ds]:
            py = MEVA if m == 'meva' else DAC
            rc = sh([py, '-I', str(HERE / 'score_r2.py'), '--dataset', ds, '--job', job, '--models', m])
            if rc:
                print(f'model {m} on {ds} failed rc={rc}; continuing', flush=True)
                rc_all = rc_all or rc
    return rc_all


def read_ids(path):
    if not path.exists():
        return set()
    with open(path) as f:
        next(f, None)
        return {ln.split('\t', 1)[0] for ln in f}


def summary(c):
    fp = c['fixed_protocol']
    out = {'job': fp['job'], 'datasets': {}}
    missing = 0
    for ds in fp['datasets']:
        items = D.read_manifest(ds)
        d = out['datasets'][ds] = {}
        for m in fp['models'][ds]:
            for variant, loud, subset in D.PLAN[ds][fp['job']]:
                units = {D.unit_id(ds, it['item_id'], variant, loud) for it in items if D.in_subset(it, subset)}
                base = D.SCORES / ds / f'{m}__{variant}_{loud}'
                ok = read_ids(Path(str(base) + '.tsv'))
                er = read_ids(Path(str(base) + '.err.tsv'))
                miss = len(units - ok - er)
                missing += miss
                d[f'{m}__{variant}_{loud}'] = dict(planned=len(units), ok=len(units & ok), err=len(units & er), missing=miss)
    out['missing_total'] = missing
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    c = contract()
    rep = Path(c['reports'][0]['path'])
    if a.preflight:
        return preflight(c)
    if a.validate_only:
        s = json.loads(rep.read_text())
        if s['missing_total']:
            sys.exit(f"{s['missing_total']} units neither scored nor recorded as errors")
        print('validate ok')
        return 0
    rc = run(c)
    s = summary(c)
    s['run_rc'] = rc
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(s, indent=1))
    return 0 if s['missing_total'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
