"""Shared loaders for metric_std_20261020 (read-only on 126a/b/c outputs)."""
import csv
import json
from pathlib import Path

import numpy as np

EV = Path('/home/kojiek/eval_output_nvme')
RES = Path('/home/kojiek/MeanAudio/docs/experiments/results')
NEURAL = ('PQ', 'CE', 'CU', 'PC', 'meva', 'clap')
MIR = {'pulse_clarity': 1, 'key_cnn_conf': 1, 'ibi_cv': -1, 'chroma_entropy': -1}
DIRECTED = ('PQ', 'CE', 'CU', 'meva', 'clap', *MIR)
PASS = 0.8
SYSTEMS_126A = ['real', 'ours_cfg3neg', 'ours_cfg0', 'musicgen', 'musicldm']


def S_of(name):
    return json.loads((RES / f'metric_win_tie_{name}_summary.json').read_text())['protocol']['S']


def load_126a():
    """-> {variant: {system: {'kind','ids','m':{metric: array}}}} (variant incl. 'clean')"""
    root = EV / 'metric_win_tie_126a_20261016/units'
    summ = json.loads((RES / 'metric_win_tie_126a_summary.json').read_text())
    kinds = {v['name']: v['kind'] for v in summ['protocol']['variants']}
    out = {}
    for vd in sorted(root.iterdir()):
        for sd in sorted(vd.iterdir()):
            rows = list(csv.DictReader(open(sd / 'scores.tsv'), delimiter='\t'))
            out.setdefault(vd.name, {})[sd.name] = {
                'kind': kinds[vd.name], 'ids': [r['id'] for r in rows],
                'm': {m: np.array([float(r[m]) for r in rows]) for m in NEURAL}}
    return out


def load_keyed(root, items_path, mir_path):
    """126b/126c: per-key scores (neural from units/*/scores.tsv, MIR from mir.json sign-flipped)."""
    sc = {}
    for ud in sorted((root / 'units').iterdir()):
        for r in csv.DictReader(open(ud / 'scores.tsv'), delimiter='\t'):
            sc[r['key']] = {m: float(r[m]) for m in NEURAL}
    mir = json.loads(Path(mir_path).read_text())
    for k, v in mir.items():
        if k not in sc:
            continue
        for m, sign in MIR.items():
            x = v.get(m, float('nan'))
            sc[k][m] = sign * x if np.isfinite(x) else float('nan')
    rows = json.loads(Path(items_path).read_text())
    return sc, rows


def judge_rates(d, m):
    d = d[np.isfinite(d)]
    if not len(d):
        return None
    return {'tie': float(np.mean(np.abs(d) <= m)), 'worse': float(np.mean(d < -m)),
            'better': float(np.mean(d > m)), 'n': int(len(d))}


def passes(rates, kind, metric):
    """None when the requirement is not defined (soft/tie = tie rate; lose = worse rate, directed metrics only)."""
    if rates is None:
        return None
    if kind in ('tie', 'soft'):
        return rates['tie'] >= PASS
    if kind == 'lose' and metric in DIRECTED:
        return rates['worse'] >= PASS
    return None
