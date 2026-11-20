#!/usr/bin/env python3
"""Queue action for the extra matrix jobs (J6: missing round-2 cells + PAM400/ME; J7: MusicPrefs).

  --preflight      sha bindings, disk floor, source audio present, index present for r2 datasets
  (default)        prep extra datasets -> score_extra.py per (dataset, model) -> analyse -> summary
  --validate-only  every unit scored or errored, analysis outputs present, summary present

Contract fixed_protocol:
  {"job": "J6"|"J7", "cells": [{"dataset","variant","loud","models":[...]}],
   "prep": ["pam400","me","musicprefs"], "analyze_datasets": [...], "nboot": 2000,
   "min_free_bytes": 21474836480}
"""
import argparse
import csv
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
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import importlib.util  # noqa: E402

_sp = importlib.util.spec_from_file_location('bench_r2_datasets', HERE / 'bench_datasets.py')
D = importlib.util.module_from_spec(_sp)
sys.modules['bench_r2_datasets'] = D
_sp.loader.exec_module(D)
EXTRA = ('pam400', 'me', 'musicprefs')
R2 = ('odaq', 'mc522', 'aime', 'singmos', 'songeval', 'arena')
PRIMARY = {'odaq': ('native', 'raw'), 'mc522': ('V10', 'raw'), 'singmos': ('native', 'raw'),
           'songeval': ('VFULL', 'raw'), 'aime': ('native', 'raw'), 'arena': ('native', 'raw')}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def contract():
    return json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())


def log(*a):
    print(*a, flush=True)


def preflight(c):
    for k, v in c['bindings'].items():
        if k.endswith('_sha256') and not k.startswith('launcher'):
            f = c['bindings'][k[:-7]]
            if sha(f) != v:
                sys.exit(f'{k[:-7]} sha mismatch')
    fp = c['fixed_protocol']
    if shutil.disk_usage(D.ROOT).free < fp['min_free_bytes']:
        sys.exit('free space below the contract floor')
    for ds in fp.get('prep', []):
        if ds == 'musicprefs':
            if not (D.DL / 'musicprefs' / 'human_preference.csv').exists():
                sys.exit('musicprefs csv missing')
            if len(list((D.DL / 'musicprefs').glob('train-*-of-*.parquet'))) != 12:
                sys.exit('musicprefs parquet shards incomplete (expected 12)')
        else:
            if not (HERE / 'extra_datasets.py').exists():
                sys.exit('extra_datasets.py missing')
    for cell in fp['cells']:
        ds = cell['dataset']
        if ds in EXTRA:
            if not (D.MAN / f'{ds}.tsv').exists() and ds in ('pam400', 'me'):
                sys.exit(f'extra manifest missing: {ds}')
        else:
            if not (D.MAN / f'{ds}.tsv').exists():
                sys.exit(f'manifest missing: {ds}')
            if not (D.DER / ds / 'index.tsv').exists():
                sys.exit(f'derived index missing: {ds}')
    print('preflight ok')
    return 0


def sh(cmd, offline=True):
    log('+', ' '.join(cmd))
    env = dict(os.environ)
    if not offline:  # cmirm tokenizer and singmos torch.hub call the hub API even with a warm cache
        for k in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
            env.pop(k, None)
    return subprocess.run(cmd, env=env).returncode


def scored_ids(path):
    if not Path(path).exists():
        return set()
    with open(path) as f:
        next(f, None)
        return {ln.split('\t', 1)[0] for ln in f}


def index_units(ds, variant, loud):
    out = []
    with open(D.DER / ds / 'index.tsv', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            # unit_id = <ds>|<item...>|<variant>|<loud>; AIME item ids themselves contain
            # '|' (aime|AudioLDM 2 Large|01509|00:00:00-00:00:10|native|raw), so only the
            # last two fields are variant/loud.
            v, l = r['unit_id'].split('|')[-2:]
            if (v, l) == (variant, loud) and not int(r['invalid'] or 0):
                out.append(r['unit_id'])
    return out


def run(c):
    fp = c['fixed_protocol']
    rc_all = 0
    for ds in fp.get('prep', []):
        rc = sh([DAC, '-I', str(HERE / 'extra_datasets.py'), 'prep', ds], offline=False)
        if rc:
            return rc
    for cell in fp['cells']:
        ds, variant, loud = cell['dataset'], cell['variant'], cell['loud']
        for model in cell['models']:
            py = MEVA if model == 'meva' else DAC
            rc = sh([py, '-I', str(HERE / 'score_extra.py'), '--dataset', ds, '--models', model,
                     '--variant', variant, '--loud', loud], offline=model not in ('cmirm', 'singmos'))
            if rc:
                log(f'model {model} on {ds} failed rc={rc}; continuing')
                rc_all = rc_all or rc
    r2 = [d for d in fp.get('analyze_datasets', []) if d in R2]
    if r2:
        rc = sh([DAC, '-I', str(HERE / 'analyze_r2.py'), '--datasets', ','.join(r2),
                 '--nboot', str(fp.get('nboot', 2000)), '--out', str(D.ROOT / 'analysis_extra')])
        rc_all = rc_all or rc
    extra = [d for d in fp.get('analyze_datasets', []) if d in ('pam400', 'me')]
    if extra:
        rc = sh([DAC, '-I', str(HERE / 'analyze_extra_cells.py'), '--datasets', ','.join(extra),
                 '--nboot', str(fp.get('nboot', 2000)), '--out', str(D.ROOT / 'analysis_extra')])
        rc_all = rc_all or rc
    if 'musicprefs' in fp.get('analyze_datasets', []):
        rc = sh([DAC, '-I', str(HERE / 'analyze_extra_cells.py'), '--datasets', 'musicprefs',
                 '--nboot', str(fp.get('nboot', 2000)), '--out', str(D.ROOT / 'analysis_extra')])
        rc_all = rc_all or rc
    if rc_all == 0:
        summarize(c)
    return rc_all


def read_tsv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def summarize(c):
    fp = c['fixed_protocol']
    out = {'job': fp['job'], 'seed_r2': 20261025, 'seed_r1': 20261020,
           'nboot': fp.get('nboot', 2000), 'r2_cells': [], 'extra_cells': [], 'musicprefs': [],
           'musicprefs_ceiling': {}}
    ana = D.ROOT / 'analysis_extra'
    for ds in [d for d in fp.get('analyze_datasets', []) if d in R2]:
        if ds == 'odaq':
            p = ana / 'odaq_g6.tsv'
            if p.exists():
                out['r2_cells'] += [dict(dataset=ds, **r) for r in read_tsv(p)]
        else:
            p = ana / f'{ds}_corr.tsv'
            if p.exists():
                out['r2_cells'] += [dict(dataset=ds, **r) for r in read_tsv(p)]
    x = ana / 'extra_cells.json'
    if x.exists():
        d = json.loads(x.read_text())
        for ds in ('pam400', 'me'):
            if ds in d:
                out['extra_cells'] += d[ds]
        if 'musicprefs' in d:
            out['musicprefs'] = d['musicprefs']
            out['musicprefs_ceiling'] = d.get('musicprefs_ceiling', {})
    path = Path(c['reports'][0]['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=str))
    log(f'summary -> {path}')


def validate(c):
    fp = c['fixed_protocol']
    bad = []
    for cell in fp['cells']:
        ds, variant, loud = cell['dataset'], cell['variant'], cell['loud']
        want = set(index_units(ds, variant, loud))
        for model in cell['models']:
            label = f'{model}__{variant}_{loud}'
            got = scored_ids(D.SCORES / ds / f'{label}.tsv') | scored_ids(D.SCORES / ds / f'{label}.err.tsv')
            miss = want - got
            if miss:
                bad.append(f'{ds}/{label}: {len(miss)}/{len(want)} units missing')
    if not (D.ROOT / 'analysis_extra' / 'extra_cells.json').exists():
        bad.append('analysis_extra/extra_cells.json missing')
    if not Path(c['reports'][0]['path']).exists():
        bad.append('summary json missing')
    if bad:
        for b in bad:
            log('INVALID:', b)
        return 1
    log('validate ok')
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
    return run(c)


if __name__ == '__main__':
    raise SystemExit(main())
