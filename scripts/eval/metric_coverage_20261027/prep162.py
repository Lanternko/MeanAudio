#!/usr/bin/env python3
"""Job 162 (GPU): clean S for the new models on the 5 TTM clean systems.

Design: docs/experiments/metric_coverage_completion_20261027.md §5.1, §8.1.

S is the 126a definition for every model — range of the 5 clean-system means — except
MuQ-embed, whose S is the 133 centroid definition (mean cosim(clean_i, mu_system),
range over systems) so that `S_fallback` from 133 is the same quantity (§8.1).

Resume: one append-only TSV per model under clean_s/<model>/, then s_values_<model>.json
is written only after every clip of every system has a finite score.
"""
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import common as C  # noqa: E402
import adapters  # noqa: E402

CLEAN = C.OUT_ROOT / 'clean_s'
MODELS = ['muqeval_A1', 'muqeval_A3a', 'cmirm', 'pam', 'singmos', 'muq']


def log(*a):
    C.log('[162]', *a)


def clean_items():
    clips = [cid for cid, _ in C.A.items()]
    items = []
    for s in C.SYSTEMS:
        for cid in clips:
            p = C.R115 / C.SYSTEM_DIRS[s] / 'lvl30' / 'audio' / f'{cid}.flac'
            items.append({'key': f'{s}__{cid}', 'base': '', 'system': s, 'clip_id': cid,
                          'variant': 'clean', 'kind': 'ref', 'source': s, 'case': 'clean',
                          'axis': '', 'impl': '', 'value': None, 'over_fs': 0,
                          'path': str(p), 'src_path': str(p)})
    return items


def tsfile(model):
    return CLEAN / model / f'clean_{model}.tsv'


def score_clean(model, items, dry_run=False):
    path = tsfile(model)
    out_cols = C.MODELS[model]['cols']
    ref = bool(C.MODELS[model].get('ref'))
    emb_path = CLEAN / model / f'emb_{model}.tsv'
    done = C.read_tsv(emb_path if ref else path) if not dry_run else {}
    todo = [it for it in items if it['key'] not in done]
    log(f'{model}: {len(items) - len(todo)}/{len(items)} done, {len(todo)} to score')
    if dry_run or not todo:
        return {'n_todo': len(todo), 'dry_run': dry_run}
    path.parent.mkdir(parents=True, exist_ok=True)
    scorer = adapters.build(model)
    t0 = time.time()
    n_ok = n_bad = 0
    for i in range(0, len(todo), scorer.batch):
        batch = todo[i:i + scorer.batch]
        res = scorer.score_batch(batch)
        if ref:
            col = scorer.cols[0]
            rows = [{'key': it['key'], col: r[col]} for it, r in zip(batch, res)
                    if not isinstance(r, Exception) and col in r]
            bad = len(batch) - len(rows)
            if rows:
                C.append_rows(emb_path, ['key', col], rows)
        else:
            rows, errs = [], []
            for it, r in zip(batch, res):
                if isinstance(r, Exception):
                    errs.append({'key': it['key'], 'error': repr(r)[:200]})
                    continue
                vals = {c: r.get(c) for c in out_cols}
                if not all(isinstance(v, float) and math.isfinite(v) for v in vals.values()):
                    errs.append({'key': it['key'], 'error': 'missing or nonfinite'})
                    continue
                rows.append({'key': it['key'], **{c: C.fmt(vals[c]) for c in out_cols}})
            bad = len(errs)
            if rows:
                C.append_rows(path, ['key'] + out_cols, rows)
        n_ok += len(rows)
        n_bad += bad
        if (i // scorer.batch) % 20 == 0:
            log(f'  {model} {min(i + scorer.batch, len(todo))}/{len(todo)} ok={n_ok} '
                f'err={n_bad} {time.time() - t0:.0f}s')
    if not ref:
        C.compact_shard(path)
    return {'n_ok': n_ok, 'n_err': n_bad, 'elapsed_s': round(time.time() - t0, 1),
            'clips_per_s': round(n_ok / max(time.time() - t0, 1e-9), 2)}


def compute_s(model, items):
    """Range of the 5 clean-system means (MuQ-embed: 133 centroid definition)."""
    if model == 'muq':
        embs = C.read_tsv(tsfile(model).with_name(f'emb_{model}.tsv'))
        if not embs:
            return {}, {'reason': 'embeddings missing'}
        vecs = {k: np.frombuffer(__import__('base64').b64decode(r['muq_emb']),
                                 dtype=np.float16).astype(np.float32)[:1024]
                for k, r in embs.items()}
        dim = 1024
        means = {}
        for s in C.SYSTEMS:
            keys = [it['key'] for it in items if it['system'] == s and it['key'] in vecs]
            if not keys:
                return {}, {'reason': f'{s}: no embeddings'}
            mu = np.mean([vecs[k][:dim] / max(np.linalg.norm(vecs[k][:dim]), 1e-8)
                          for k in keys], axis=0)
            mu = mu / max(np.linalg.norm(mu), 1e-8)
            cs = [float(np.dot(vecs[k][:dim] / max(np.linalg.norm(vecs[k][:dim]), 1e-8), mu))
                  for k in keys]
            means[s] = float(np.mean(cs))
        S = {'muq_cosim': float(max(means.values()) - min(means.values()))}
        return S, {'means': means, 'n': {s: sum(1 for it in items
                                                if it['system'] == s and it['key'] in vecs)
                                        for s in C.SYSTEMS},
                   'S_definition': '133 centroid: mean cosim(clean_i, mu_system), range over 5'}
    cols = C.MODELS[model]['cols']
    scores = C.parse_score_tsv(tsfile(model), cols)
    means = {}
    for s in C.SYSTEMS:
        keys = [it['key'] for it in items if it['system'] == s]
        vals = {c: [] for c in cols}
        for k in keys:
            for c in cols:
                if k in scores and c in scores[k]:
                    vals[c].append(scores[k][c])
        means[s] = {c: float(np.mean(vals[c])) if len(vals[c]) == len(keys) and vals[c]
                    else None for c in cols}
    S = {}
    for c in cols:
        v = [means[s][c] for s in C.SYSTEMS]
        S[c] = (max(v) - min(v)) if all(isinstance(x, float) and math.isfinite(x)
                                        for x in v) else None
    return S, {'means': means, 'n': {s: len([it for it in items if it['system'] == s])
                                     for s in C.SYSTEMS},
               'S_definition': 'range of the 5 clean-system means (§5.1)'}


def run(dry_run=False):
    t0 = time.time()
    CLEAN.mkdir(parents=True, exist_ok=True)
    C.S_FILE.mkdir(parents=True, exist_ok=True)
    items = clean_items()
    assert len(items) == 5 * 5131, len(items)
    summary = {}
    for model in MODELS:
        s_path = C.S_FILE / f's_values_{model}.json'
        if s_path.is_file() and not dry_run:
            log(f'{model}: S already written, skipping ({s_path})')
            summary[model] = {'skipped': True, 'S': C.read_json(s_path, {}).get('S')}
            continue
        info = score_clean(model, items, dry_run=dry_run)
        if dry_run:
            summary[model] = info
            continue
        S, extra = compute_s(model, items)
        usable = bool(S) and all(isinstance(v, float) and math.isfinite(v) and v > 0
                                 for v in S.values())
        doc = {'document_kind': 'metric_coverage_s_values_v1', 'job': 162,
               'written_at': C.written_at(), 'model': model,
               'label': C.MODELS[model]['label'], 'S': S, 'S_usable': usable,
               'S_source': '162', 'S_fallback': False,
               'protocols': ['126a', '126b', '126c', '134'],
               'S_by_protocol': {},
               'clean_tsv': str(tsfile(model)),
               'clean_tsv_sha256': (C.sha256_file(tsfile(model))
                                    if tsfile(model).is_file() else None),
               'n_clips': len(items),
               'models': info, **extra}
        if model == 'muq':
            doc['fallback_rule'] = ('if unusable, judge may borrow 133 S_muq=0.108679 and '
                                    'must set S_source="133", S_fallback=true, matrix '
                                    'inconclusive (§8.1)')
        C.write_json_atomic(s_path, doc)
        summary[model] = {'S': S, 'usable': usable, **info}
        log(f'{model}: S={S} usable={usable}')
    C.write_json_atomic(C.OUT_ROOT / 'jobs' / 'job_done_162.json',
                        {'job': 162, 'written_at': C.written_at(), 'models': summary,
                         'elapsed_s': round(time.time() - t0, 1)})
    return 0


def validate():
    errs = []
    for m in MODELS:
        p = C.S_FILE / f's_values_{m}.json'
        if not p.is_file():
            errs.append(f'missing {p}')
            continue
        d = C.read_json(p, {})
        if not d.get('S'):
            errs.append(f'{p}: no S block')
    return errs
