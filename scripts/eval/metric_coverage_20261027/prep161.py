#!/usr/bin/env python3
"""Job 161 (CPU): sample manifest, sample-vs-full agreement, 126a audio cache,
rec/floor analysis backfill for SongEval / MERT / MuQ.

Design: docs/experiments/metric_coverage_completion_20261027.md §4.4, §6, §7.3, §8.1.

Every output is resume-safe: the cache is per file, the agreement JSON is written
atomically after the simulation, and the backfill files are pure functions of the
existing 126b per-clip TSVs.  Nothing here touches a GPU or a model.
"""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import common as C  # noqa: E402

EXPECTED = {
    'clips_126a': 5131,
    # measured 2026-10-27 after the seagate offload: only lvl30/audio exists.
    # The plan's 70,603 counted the pre-offload layout (all levels); the 126a
    # protocol only ever reads these five lvl30 trees.
    'r115_flac': 37867,
    'units_126a': 265,
    '126b': 87653,
    '126c': 36366,
    '134': 49587,
    'musiccaps_rows': 5521,
    'cache_files_512': 5 * 53 * 512,
}
REC_SOURCES = ('pro', 'poor', 'poor_sing', 'csd_real')
AGREEMENT = C.OUT_ROOT / 'sample_agreement_20261027.json'
CACHE = C.OUT_ROOT / 'audio_cache'
CACHE_MANIFEST = C.OUT_ROOT / 'audio_cache_manifest.json'
BACKFILL = C.OUT_ROOT / 'backfill'
PREP = C.OUT_ROOT / 'prep_161.json'


def log(*a):
    C.log('[161]', *a)


# ── preflight counts (§3.5 / §7.2) ───────────────────────────────────────────
def musiccaps_rows(path=C.MUSICCAPS_TSV):
    with open(path, newline='') as f:
        return sum(1 for _ in csv.DictReader(f, delimiter='\t'))


def check_inputs():
    errs = []
    p = C.sample_manifest_path()
    if p.is_file():
        man = C.read_json(p)
        if C.sha256_json(man) != C.sha256_json(C.build_sample_manifest()):
            errs.append('committed sample manifest does not regenerate (seed 20261027)')
    else:
        errs.append(f'sample manifest missing: {p}')
    sel = C.R115 / 'selection.json'
    if not sel.is_file():
        errs.append(f'R115 selection.json missing: {sel}')
    clips = {cid for cid, _ in C.A.items()}
    for s, d in C.SYSTEM_DIRS.items():
        p = C.R115 / d / 'lvl30' / 'audio'
        have = {q.stem for q in p.glob('*.flac')} if p.is_dir() else set()
        miss = clips - have
        if miss:
            errs.append(f'R115 {s}: {len(miss)} protocol clips missing (first: '
                        f'{sorted(miss)[:3]})')
        # generated systems carry the 5,521 MusicCaps rows; real only the 5,131
        # clips that also have a reference.  Both satisfy the 126a pool.
        if len(have) not in (EXPECTED['clips_126a'], 5521):
            errs.append(f'R115 {s}: {len(have)} flac, expected 5,131 or 5,521')
    total = sum(1 for _ in C.R115.rglob('*.flac'))
    if total != EXPECTED['r115_flac']:
        errs.append(f'R115 total {total} flac, expected {EXPECTED["r115_flac"]}')
    if Path(C.MUSICCAPS_TSV).is_file():
        n = musiccaps_rows()
        if n != EXPECTED['musiccaps_rows']:
            errs.append(f'MusicCaps TSV has {n} records, expected {EXPECTED["musiccaps_rows"]}')
    else:
        errs.append(f'MusicCaps TSV missing: {C.MUSICCAPS_TSV}')
    for proto, want in (('126b', EXPECTED['126b']), ('126c', EXPECTED['126c']),
                        ('134', EXPECTED['134'])):
        items = C.load_items_file(proto)
        if len(items) != want:
            errs.append(f'{proto}: {len(items)} items, expected {want}')
        miss = sum(1 for it in items[:200] if not Path(it['path']).is_file())
        if miss:
            errs.append(f'{proto}: {miss}/200 sampled stimuli missing on disk')
    units = C.PROTOCOL_126A / 'units'
    n_units = sum(1 for v in units.glob('*') if v.is_dir()
                  for s in v.glob('*') if (s / 'scores.tsv').is_file())
    if n_units != EXPECTED['units_126a']:
        errs.append(f'126a units: {n_units} score TSVs, expected {EXPECTED["units_126a"]}')
    free = __import__('shutil').disk_usage('/home/kojiek').free
    if free < 100 * 2 ** 30:
        # §7.3 stop condition; the cache falls back to on-the-fly rendering above 100 GB
        errs.append(f'disk: {free / 2**30:.0f} GB free < 100 GB stop condition')
    return errs


# ── 126a scores -> per-cell boolean pass arrays ──────────────────────────────
def _read_scores(path, cols):
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            try:
                out[r['id']] = {c: float(r[c]) for c in cols}
            except (KeyError, ValueError):
                continue
    return out


def agreement_matrices():
    """(pool ids, {(metric, kind): bool array [n_cells x n_clips]}).

    Cells are the 52 non-ref variants x 5 systems; a True entry means that pair
    passes at m (|d| <= m for tie, d < -m for lose).  Built in one pass over the
    existing 265 score TSVs, so memory stays at 6 x 2 x 260 x 5,131 bytes."""
    metrics = list(C.A.METRICS)
    pool = [cid for cid, _ in C.A.items()]
    idx = {cid: i for i, cid in enumerate(pool)}
    cases = [v for v in C.VARIANTS if v['kind'] in ('tie', 'lose')]
    cells = [(v['name'], s) for v in cases for s in C.SYSTEMS]
    M = {(m, k): np.zeros((len(cells), len(pool)), dtype=bool)
         for m in metrics for k in ('tie', 'lose')}
    S = C.read_json(C.S_126A_SUMMARY, {}).get('protocol', {}).get('S', {})
    thresholds = {m: C.M_FRAC * S.get(m, 0) for m in metrics}
    for s in C.SYSTEMS:
        clean = _read_scores(C.PROTOCOL_126A / 'units' / 'clean' / s / 'scores.tsv', metrics)
        for ci, (vname, s2) in enumerate(cells):
            if s2 != s:
                continue
            case = _read_scores(C.PROTOCOL_126A / 'units' / vname / s / 'scores.tsv', metrics)
            for cid, cvals in case.items():
                i = idx.get(cid)
                b = clean.get(cid)
                if i is None or not b:
                    continue
                for m in metrics:
                    if m not in cvals or m not in b:
                        continue
                    d = cvals[m] - b[m]
                    mm = thresholds[m]
                    M[(m, 'tie')][ci, i] = abs(d) <= mm
                    M[(m, 'lose')][ci, i] = d < -mm
    return pool, M, cells


def simulate_agreement(pool, M, n_values=(256, 512, 1024), B=C.BOOTSTRAP_B):
    rng = np.random.default_rng(C.SAMPLE_SEED)
    names = sorted(M)
    label = {key: f'{key[0]}/{key[1]}' for key in names}
    acc = {n: {label[key]: {'agreement': [], 'ambiguous': [], 'oob': []}
               for key in names} for n in n_values}
    rep0 = None
    for b in range(B):
        perm = rng.permutation(len(pool))
        if rep0 is None:
            rep0 = perm
        for n in n_values:
            ix = perm[:n]
            for key in names:
                P = M[key]
                full = P.mean(axis=1)
                rates = P[:, ix].mean(axis=1)
                sampled = rates >= C.PASS
                full_pass = full >= C.PASS
                acc[n][label[key]]['agreement'].append(float(np.mean(sampled == full_pass)))
                acc[n][label[key]]['ambiguous'].append(float(np.sum(np.abs(rates - C.PASS) <= C.AMBIG_BAND)))
                oob = np.abs(rates - full)[np.abs(full - C.PASS) > C.AMBIG_BAND]
                acc[n][label[key]]['oob'].append(float(np.max(oob)) if oob.size else 0.0)
        if (b + 1) % 25 == 0:
            log(f'agreement bootstrap {b + 1}/{B}')
    out = {}
    for n in n_values:
        out[str(n)] = {}
        for key in names:
            a = acc[n][label[key]]
            out[str(n)][label[key]] = {
                'agreement': float(np.mean(a['agreement'])),
                'ambiguous_mean': float(np.mean(a['ambiguous'])),
                'ambiguous_max': float(np.max(a['ambiguous'])),
                'out_of_band_error_max': float(np.max(a['oob'])),
            }
    return out, rep0


def min_agreement(by_n, n):
    return min(v['agreement'] for v in by_n[str(n)].values())


def max_ambiguous(by_n, n):
    return max(v['ambiguous_max'] for v in by_n[str(n)].values())


# ── 126a audio cache (§7.3) ──────────────────────────────────────────────────
def cache_targets():
    clips = C.clips_for_n(C.SAMPLE_N)
    for system in C.SYSTEMS:
        for v in C.VARIANTS:
            for cid in clips:
                yield system, v['name'], cid


def build_cache(dry_run=False):
    import soundfile as sf
    targets = list(cache_targets())
    todo = [t for t in targets if not (CACHE / t[0] / t[1] / f'{t[2]}.flac').is_file()]
    log(f'cache: {len(targets) - len(todo)}/{len(targets)} exist, {len(todo)} to render')
    if dry_run:
        return {'n_targets': len(targets), 'n_missing': len(todo), 'dry_run': True}
    t0 = time.time()
    for i, (system, vname, cid) in enumerate(todo):
        var = C.A.VARIANTS[vname]
        src = C.R115 / C.SYSTEM_DIRS[system] / 'lvl30' / 'audio' / f'{cid}.flac'
        dst = CACHE / system / vname / f'{cid}.flac'
        x, sr = sf.read(src, dtype='float64')
        if getattr(x, 'ndim', 1) > 1:
            x = x.mean(axis=1)
        y, _ = C.A.transform(x, var, C.A.rng_for(system, vname, cid), int(sr))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + f'.tmp{os.getpid()}')
        sf.write(str(tmp), y, int(sr), format='FLAC', subtype=var['subtype'])
        os.replace(tmp, dst)
        if (i + 1) % 500 == 0:
            r = (i + 1) / max(time.time() - t0, 1e-9)
            log(f'cache {i + 1}/{len(todo)} {r:.1f}/s eta {(len(todo) - i - 1) / max(r, 1e-9) / 60:.1f} min')
            C.write_json_atomic(C.OUT_ROOT / 'cache_heartbeat.json',
                                {'document_kind': 'metric_coverage_cache_heartbeat_v1',
                                 'written_at': C.written_at(), 'n_done': i + 1,
                                 'n_total': len(todo), 'files_per_s': round(r, 2)})
    per_unit, total_bytes = {}, 0
    for system, vname, cid in targets:
        p = CACHE / system / vname / f'{cid}.flac'
        if p.is_file():
            per_unit[f'{vname}__{system}'] = per_unit.get(f'{vname}__{system}', 0) + 1
            total_bytes += p.stat().st_size
    man = {'document_kind': 'metric_coverage_audio_cache_v1',
           'written_at': C.written_at(), 'root': str(CACHE),
           'n_files': sum(per_unit.values()), 'n_expected': len(targets),
           'bytes': total_bytes, 'per_unit': per_unit,
           'clip_list': 'sample_manifest clip_ids (512); 1024 escalation renders on the fly',
           'elapsed_s': round(time.time() - t0, 1)}
    C.write_json_atomic(CACHE_MANIFEST, man)
    log(f'cache manifest -> {CACHE_MANIFEST} ({man["n_files"]} files, {total_bytes / 2**30:.1f} GiB)')
    return man


# ── rec / floor backfill for the already-scored SongEval / MERT / MuQ rows ───
def _parse_scores(path, cols):
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            try:
                out[r['key']] = {c: float(r[c]) for c in cols}
            except (KeyError, ValueError):
                continue
    return out


def _winrate(xs, ys, m):
    x, y = np.asarray(xs, dtype=float)[:, None], np.asarray(ys, dtype=float)[None, :]
    d = (x - y).ravel()
    d = d[np.isfinite(d)]
    if not d.size:
        return None
    return {'better': float(np.mean(d > m)), 'tie': float(np.mean(np.abs(d) <= m)),
            'worse': float(np.mean(d < -m))}


def backfill_one(model):
    cols = C.MODELS[model]['cols']
    src = (C.PROTOCOL_126D / '126b_songeval.tsv' if model == 'songeval'
           else C.PROTOCOL_133 / f'126b_{model}_cosim.tsv')
    scores = _parse_scores(src, cols)
    items = C.load_items_file('126b')
    by_key = {it['key']: it for it in items}
    S = C.s_values(model, '126a')
    rec = {'sources': {}, 'pairs_by_source': {}}
    for source in REC_SOURCES:
        rows = [it for it in items if it.get('source') == source and it.get('base')
                and it['base'] in by_key]
        rec['pairs_by_source'][source] = len(rows)
        rec['sources'][source] = {}
        for col in cols:
            ms = C.M_FRAC * S.get(col, 0) if S.get(col) else None
            if not ms:
                rec['sources'][source][col] = {'status': 'inconclusive', 'reason': 'S missing'}
                continue
            for kind in ('tie', 'lose'):
                pairs = [it for it in rows if it.get('kind') == kind
                         and it['key'] in scores and it['base'] in scores]
                d = [scores[it['key']][col] - scores[it['base']][col] for it in pairs]
                rec['sources'][source].setdefault(col, {})[kind] = C.judge_deltas(
                    d, ms, kind, True)
    floor = None
    if model == 'songeval':
        music = [it['key'] for it in items
                 if it.get('kind') == 'ref' and it.get('block') in ('midi', 'rec')
                 and it['key'] in scores]
        fl = [it for it in items if it.get('block') == 'floor']
        floor = {}
        for case in sorted({it['case'] for it in fl}):
            ks = [it['key'] for it in fl if it['case'] == case and it['key'] in scores]
            if not ks:
                continue
            lvl = case.rsplit('_', 1)[-1]
            white = [it['key'] for it in fl
                     if it['case'] == f'white_{lvl}' and it['key'] in scores]
            floor[case] = {}
            for col in cols:
                ms = C.M_FRAC * S.get(col, 0) if S.get(col) else None
                v = np.array([scores[k][col] for k in ks], dtype=float)
                mv = np.array([scores[k][col] for k in music], dtype=float)
                if not np.isfinite(v).any():
                    continue
                fm = float(np.nanmean(v))
                floor[case][col] = {
                    'mean': fm,
                    'frac_music_below_floor': float(np.nanmean(mv < fm)),
                    'vs_white': _winrate(v, [scores[k][col] for k in white], ms)
                    if white and ms else None,
                }
    out = {'document_kind': 'metric_coverage_rec_floor_backfill_v1',
           'written_at': C.written_at(), 'model': model, 'protocol': '126b',
           'source_tsv': str(src), 'source_sha256': C.sha256_file(src),
           'S': {c: S.get(c) for c in cols},
           'S_source': '126d' if model == 'songeval' else '133',
           'rec': rec,
           'floor': floor,
           'floor_status': ('n/a (reference-based metrics have no floor pairing, §1.4)'
                            if model != 'songeval' else 'descriptive percentiles only')}
    p = BACKFILL / f'rec_floor_{model}.json'
    C.write_json_atomic(p, out)
    log(f'backfill {model} -> {p}')
    return out


# ── job body ────────────────────────────────────────────────────────────────
def run(dry_run=False):
    t0 = time.time()
    C.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    C.S_FILE.mkdir(parents=True, exist_ok=True)
    BACKFILL.mkdir(parents=True, exist_ok=True)
    errs = check_inputs()
    if errs:
        for e in errs:
            print('PRECHECK FAILED: ' + e, file=sys.stderr)
        return 2
    man = C.build_sample_manifest()
    committed = C.read_json(C.sample_manifest_path())
    if C.sha256_json(committed) != C.sha256_json(man):
        print('committed sample manifest differs from the regenerated one', file=sys.stderr)
        return 2
    C.write_json_atomic(C.OUT_ROOT / 'sample_manifest.json', man)
    if dry_run:
        log('dry-run: would build agreement, cache and backfill')
        return 0
    pool, M, cells = agreement_matrices()
    log(f'agreement matrices: {len(cells)} cells x {len(pool)} clips x {len(M)} metric/kinds')
    by_n, rep0 = simulate_agreement(pool, M)
    rep0_ids = [pool[int(i)] for i in rep0[:1024]]
    rep0_ok = rep0_ids == man['clip_ids_1024']
    agg = {'document_kind': 'metric_coverage_sample_agreement_v1',
           'written_at': C.written_at(), 'seed': C.SAMPLE_SEED, 'B': C.BOOTSTRAP_B,
           'n_pool': len(pool), 'pass': C.PASS, 'band': C.AMBIG_BAND,
           'metric_kinds': [f'{m}/{k}' for m, k in sorted(M)],
           'by_n': by_n,
           'min_agreement_at_512': min_agreement(by_n, 512),
           'min_agreement_at_1024': min_agreement(by_n, 1024),
           'max_ambiguous_at_512': max_ambiguous(by_n, 512),
           'rep0_matches_committed_manifest': bool(rep0_ok)}
    esc = {'to_1024': bool(agg['min_agreement_at_512'] < 0.9634
                           or agg['max_ambiguous_at_512'] > 24),
           'rule': 'doc §4.4/§8.1: min_agreement_at_512 < 0.9634 or any metric ambiguous > 24'}
    esc['require_full_126a'] = bool(esc['to_1024']
                                    and agg['min_agreement_at_1024'] < 0.9716)
    esc['note'] = ('rule + thresholds pre-registered in the doc; never tuned after seeing data'
                   if not esc['to_1024'] else
                   'escalated to N=1024 (PAM 512); 512 rows are kept, only the new clips run')
    agg['escalation'] = esc
    C.write_json_atomic(AGREEMENT, agg)
    log(f'agreement -> {AGREEMENT} min@512={agg["min_agreement_at_512"]:.4f} '
        f'to_1024={esc["to_1024"]} full={esc["require_full_126a"]}')
    cache = build_cache()
    backfill = {m: backfill_one(m) for m in ('songeval', 'mert', 'muq')}
    prep = {'document_kind': 'metric_coverage_prep_v1', 'job': 161,
            'written_at': C.written_at(),
            'sample_manifest_sha256': C.sha256_file(C.sample_manifest_path()),
            'sample_manifest_json_sha256': C.sha256_json(man),
            'agreement_sha256': C.sha256_file(AGREEMENT),
            'cache': {k: v for k, v in cache.items() if k != 'per_unit'},
            'cache_manifest_sha256': (C.sha256_file(CACHE_MANIFEST)
                                      if C.CACHE_MANIFEST.is_file() else None),
            'backfill': {m: str(BACKFILL / f'rec_floor_{m}.json') for m in backfill},
            'expected': EXPECTED, 'elapsed_s': round(time.time() - t0, 1)}
    C.write_json_atomic(PREP, prep)
    log(f'prep -> {PREP} ({prep["elapsed_s"] / 60:.1f} min)')
    return 0


def validate():
    errs = []
    for p in (PREP, AGREEMENT, CACHE_MANIFEST, C.sample_manifest_path()):
        if not Path(p).is_file():
            errs.append(f'missing {p}')
    if AGREEMENT.is_file():
        d = C.read_json(AGREEMENT, {})
        if not d.get('escalation'):
            errs.append('agreement JSON has no escalation block')
        if d.get('B') != C.BOOTSTRAP_B:
            errs.append('agreement B != pre-registered bootstrap count')
    for m in ('songeval', 'mert', 'muq'):
        if not (BACKFILL / f'rec_floor_{m}.json').is_file():
            errs.append(f'backfill missing for {m}')
    return errs
