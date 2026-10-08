#!/usr/bin/env python3
"""p2 113 action: MusicCaps 5521 generation for the external TTM baselines.

Contract: docs/experiments/ttm_external_baselines_113_20261008_contract.json (read from
GPU_QUEUE_CONTRACT). Generation only; lvl30 processing and metrics are a later job.

  (no flag)        run scripts/eval/ttm_external_generate.py for each registered system
                   (the generator resumes by itself: finished batches are skipped)
  --preflight      inputs, model snapshots, generator sha, free disk
  --validate-only  every TSV id has a finite, full-length clip at the registered sr in every
                   system; writes the registered summary.json
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SILENT_DBFS = -45.0  # same rule as eval_metrics.py


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())


def tsv_ids(tsv):
    with open(tsv, newline='') as f:
        return [r['id'] for r in csv.DictReader(f, delimiter='\t')]


def preflight(c):
    p = c['fixed_protocol']
    ids = tsv_ids(p['tsv'])
    if len(ids) != p['n_prompts']:
        sys.exit(f'TSV has {len(ids)} rows, expected {p["n_prompts"]}')
    if sha256(c['bindings']['generator']) != c['bindings']['generator_sha256']:
        sys.exit('generator sha mismatch')
    for s in c['systems']:
        snap = Path(s['snapshot'])
        if not snap.is_dir():
            sys.exit(f'missing snapshot {snap}')
    out = Path(c['storage']['path'])
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < c['storage']['min_free_bytes']:
        sys.exit('not enough free disk')
    print('preflight ok')


def run(c):
    p = c['fixed_protocol']
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONUNBUFFERED': '1'}
    for s in c['systems']:
        out = Path(c['storage']['path']) / s['name']
        out.mkdir(parents=True, exist_ok=True)
        cmd = [p['python'], c['bindings']['generator'], '--system', s['name'],
               '--tsv', p['tsv'], '--out_dir', str(out), '--batch_size', str(s['batch_size'])]
        with open(out / 'gen.log', 'a') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        if rc:
            sys.exit(rc)


def validate(c):
    p = c['fixed_protocol']
    ids = tsv_ids(p['tsv'])
    summary = {'document_kind': 'ttm_external_baselines_113_summary_v1',
               'tsv': p['tsv'], 'n_prompts': len(ids), 'systems': {}}
    bad = []
    for s in c['systems']:
        out = Path(c['storage']['path']) / s['name']
        man = json.loads((out / 'manifest.json').read_text())
        if not man.get('complete') or man['revision'] != s['revision'] or man['batch_size'] != s['batch_size']:
            bad.append(f'{s["name"]}: manifest incomplete or disagrees with contract')
            continue
        peaks = []
        n_silent = n_short = n_bad = 0
        for i in ids:
            f = out / 'audio' / f'{i}.flac'
            if not f.is_file():
                n_bad += 1
                continue
            x, sr = sf.read(str(f), dtype='float32')
            if sr != s['sampling_rate'] or x.ndim != 1 or not np.isfinite(x).all():
                n_bad += 1
                continue
            if len(x) < p['min_seconds'] * sr:
                n_short += 1
            rms = 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)
            n_silent += int(rms < SILENT_DBFS)
            peaks.append(float(np.abs(x).max()))
        if n_bad or n_short:
            bad.append(f'{s["name"]}: {n_bad} missing/invalid, {n_short} short')
        summary['systems'][s['name']] = {
            'repo': s['repo'], 'revision': s['revision'], 'sampling_rate': s['sampling_rate'],
            'batch_size': s['batch_size'], 'n_valid': len(ids) - n_bad - n_short,
            'n_silent_rms_lt_-45dbfs': n_silent,
            'peak_max': max(peaks, default=None),
            'generation_seconds': round(sum(b['seconds'] for b in man['batches']), 1),
            'peak_vram_gb': max((b['peak_vram_gb'] for b in man['batches']), default=None),
            'manifest_sha256': sha256(out / 'manifest.json')}
    if bad:
        sys.exit('; '.join(bad))
    report = Path(c['reports'][0]['path'])
    report.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    c = contract()
    if '--preflight' in sys.argv:
        preflight(c)
    elif '--validate-only' in sys.argv:
        validate(c)
    else:
        run(c)
