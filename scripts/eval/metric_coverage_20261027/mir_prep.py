#!/usr/bin/env python3
"""Job 176 (CPU): MIR four readings for the missing cells.

Design: docs/experiments/metric_coverage_completion_20261027.md §1.2 note, §5.1, §8.2.

  * 126a sampled (265 units x 512 clips, the 161 audio cache) -> the four ○ cells
  * clean S: the full 5 x 5,131 clean clips -> S_126a = SD of all clean clips
  * 134: all 49,587 clips -> key_cnn_conf / chroma_entropy (pulse/ibi recomputed in the
    same pass for the same mir_metrics.py revision) -> S_134_extra = SD of clean clips

Nothing here re-implements MIR: scripts/eval/mir_metrics.score_mir runs its madmom and
librosa workers exactly as the standard eval does.
"""
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
import score_coverage as SC  # noqa: E402

MIR = C.OUT_ROOT / 'mir'
S_MIR = C.S_FILE / 's_values_mir.json'
COLS = C.MODELS['mir']['cols']


def log(*a):
    C.log('[176]', *a)


def clean_items():
    clips = [cid for cid, _ in C.A.items()]
    out = []
    for s in C.SYSTEMS:
        for cid in clips:
            p = C.R115 / C.SYSTEM_DIRS[s] / 'lvl30' / 'audio' / f'{cid}.flac'
            out.append({'key': f'{s}__{cid}', 'base': '', 'system': s, 'clip_id': cid,
                        'variant': 'clean', 'kind': 'ref', 'source': s, 'case': 'clean',
                        'axis': '', 'impl': '', 'value': None, 'over_fs': 0,
                        'path': str(p), 'src_path': str(p)})
    return out


def sd_s(items, scores_path, cols):
    scores = C.parse_score_tsv(scores_path, cols)
    out = {}
    for c in cols:
        v = np.array([r[c] for r in scores.values() if c in r], dtype=float)
        v = v[np.isfinite(v)]
        out[c] = float(np.std(v)) if v.size > 1 else None
    return out


def run(dry_run=False):
    t0 = time.time()
    MIR.mkdir(parents=True, exist_ok=True)
    C.S_FILE.mkdir(parents=True, exist_ok=True)
    summary = {'job': 176, 'written_at': C.written_at()}
    scorer = None if dry_run else adapters.build('mir')

    items = C.items_126a('mir', C.SAMPLE_N)
    if dry_run:
        log(f'dry-run: 126a {len(items)} clips, 134 {len(C.load_items_file("134"))} clips, '
            f'clean {5 * 5131} clips')
        return 0

    # 1) clean S first (full 5 x 5,131): the ambiguous full rerun in step 3 needs
    #    S on disk before it can judge which 126a units sit on the 0.80 band.
    ci = clean_items()
    clean = SC.run_protocol('mir', '126b', ci, MIR / 'clean_s', scorer, '0/1')
    # protocol name only selects shard naming; key/base come from the items
    clean_path = C.shard_file(MIR / 'clean_s', '126b', 'mir', 0, '000000')
    S_126a = sd_s(ci, clean_path, COLS)
    summary['clean'] = {**clean, 'S_126a': S_126a, 'tsv': str(clean_path)}
    C.write_json_atomic(S_MIR, {'document_kind': 'metric_coverage_s_values_mir_v1',
                                'job': 176, 'written_at': C.written_at(),
                                'S_126a': S_126a, 'S_134_extra': {},
                                'S_by_protocol': {'126a': S_126a},
                                'interim': True})
    log(f'clean S_126a={S_126a}')

    # 2) 126a sampled -------------------------------------------------------
    s126a = SC.run_protocol('mir', '126a', items, MIR / '126a', scorer, '0/1')
    summary['126a'] = s126a
    log(f'126a: {s126a}')

    # 3) §4.4 ambiguous units -> full 5,131 clips (same rule as the GPU jobs)
    full = SC.full_rerun('mir', '126a', MIR / '126a', scorer, C.SAMPLE_N, job=176,
                         require_full=C.require_full_126a())
    summary['126a_full'] = full
    log(f'126a full rerun: {full}')

    # 4) 134 all items (key/chroma missing; pulse/ibi same pass) -------------
    items134 = C.protocol_items('134', 'mir')
    s134 = SC.run_protocol('mir', '134', items134, MIR / '134', scorer, '0/1')
    ref_keys = {it['key'] for it in items134 if it.get('kind') == 'ref'}
    scores134 = {}
    for f in C.final_tsv_files(MIR / '134', '134', 'mir'):
        scores134.update(C.parse_score_tsv(f, COLS))
    S_134_extra = {}
    for c in COLS:
        v = np.array([scores134[k][c] for k in ref_keys if k in scores134], dtype=float)
        v = v[np.isfinite(v)]
        S_134_extra[c] = float(np.std(v)) if v.size > 1 else None
    summary['134'] = {**s134, 'S_134_extra': S_134_extra}
    log(f'134: {s134} S_134_extra={S_134_extra}')

    doc = {'document_kind': 'metric_coverage_s_values_mir_v1', 'job': 176,
           'written_at': C.written_at(), 'S_126a': S_126a,
           'S_126a_definition': 'SD of all clean clips (5 systems x 5,131), sign kept positive; '
                                'SD is scale so the raw-reading convention in the driver does '
                                'not change it (§8.2)',
           'S_134_extra': S_134_extra,
           'S_134_extra_definition': 'SD of the 134 clean (kind=ref) clips, same pass',
           'S_by_protocol': {'126a': S_126a, '134': {**S_134_extra,
                                                     **(C.read_json(C.S_134_SUMMARY, {})
                                                        .get('protocol', {}).get('S', {}))}},
           'runs': summary, 'elapsed_s': round(time.time() - t0, 1)}
    C.write_json_atomic(S_MIR, doc)
    C.write_json_atomic(C.OUT_ROOT / 'jobs' / 'job_done_176.json', summary)
    log(f'S -> {S_MIR}')
    return 0


def validate():
    errs = []
    if not S_MIR.is_file():
        errs.append(f'missing {S_MIR}')
    else:
        d = C.read_json(S_MIR, {})
        for k in ('S_126a', 'S_134_extra'):
            if not d.get(k):
                errs.append(f'{S_MIR}: no {k}')
    if not C.final_tsv_files(MIR / '126a', '126a', 'mir'):
        errs.append('126a MIR shards missing')
    if not C.final_tsv_files(MIR / '134', '134', 'mir'):
        errs.append('134 MIR shards missing')
    return errs
