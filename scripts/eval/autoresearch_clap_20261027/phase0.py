#!/usr/bin/env python
"""Phase 0: the five measurements the design doc requires before rung 1 (10.4).

    phase0.py all   [--scratch SCRATCH] [--force]     # resumable per step
    phase0.py verify|baselines|copy|fad-parity|timing|report [--scratch SCRATCH] [--force]

Steps (each writes <scratch>/phase0/<step>.json and is skipped when it exists):
  verify      source TSV / split sha vs prereg, checkpoint sha, prereg sha, disk
  baselines   six subset baselines per split (no generation; FAD via cached refs)
  copy        C0 twice on dev256 -> G0 (vs canonical 256/256) and G1 (run A == B),
              plus C1/C2 seed-floor trials (reused by rung 1 through config_sha)
  fad-parity  canonical full cfg3 cell recomputed through fad_cache -> 5.0836 +-1e-3
  timing      one sequential full-5521 lvl30 cell on the canonical cfg0 raw audio
  report      p0_summary.json with the gates and the measured budget inputs
"""
import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                   # noqa: E402
import baselines as baselines_mod               # noqa: E402
import fad_cache                                # noqa: E402
import trial as trial_mod                       # noqa: E402

STEPS = ('verify', 'baselines', 'copy', 'fad-parity', 'timing', 'report')
SPLITS = ('dev256', 'dev1024', 'heldout1024')
EXPECTED_FAD = 5.0836
FAD_TOL = 1e-3


def p0_dir(scratch):
    d = Path(scratch) / 'phase0'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _done(path, force):
    rec = common.read_json(path)
    if rec and not force:
        print(f'[phase0] skip {path.name}', flush=True)
        return rec
    return None


# ── verify ───────────────────────────────────────────────
def step_verify(scratch, force=False):
    out = p0_dir(scratch) / 'verify.json'
    prev = common.read_json(out)
    if prev and not force:
        if prev.get('prereg_sha256') == common.sha256(common.PREREG):
            print(f'[phase0] skip {out.name}', flush=True)
            return prev
        print('[phase0] prereg sha changed -> re-verify', flush=True)
    prereg = common.load_prereg()
    sub = Path(scratch) / 'tsv'
    import subprocess
    subprocess.run([common.PY, str(Path(__file__).parent / 'split_subsets.py'), '--out', str(sub)],
                   check=True, capture_output=True)
    manifest = common.read_json(sub / 'manifest.json')
    problems = []
    if manifest['source_sha256'] != prereg['splits']['source_sha256']:
        problems.append('source TSV sha differs from prereg')
    if manifest['n_records'] != prereg['splits']['n_records']:
        problems.append('source record count differs from prereg')
    split_check = {}
    for name in SPLITS:
        want = prereg['splits'].get(name, {})
        got = manifest['splits'][name]
        ok_sha = want.get('sha256') is None or got['sha256'] == want['sha256']
        split_check[name] = {'sha256': got['sha256'], 'ids_sha256': got['ids_sha256'],
                             'n': got['n'], 'prereg_sha256': want.get('sha256'),
                             'matches_prereg': ok_sha}
        if not ok_sha:
            problems.append(f'{name} sha differs from prereg')
    if manifest['splits']['dev256']['ids_sha256'] != hashlib_ids_prefix(
            sub / 'dev1024.tsv', 256):
        problems.append('dev256 is not the first 256 ids of dev1024')
    ckpt = {}
    for s in common.SEEDS:
        p = common.CHECKPOINTS[s]
        got = common.sha256(p) if Path(p).is_file() else None
        ckpt[s] = {'path': str(p), 'sha256': got, 'expected': common.CHECKPOINT_SHA256[s],
                   'ok': got == common.CHECKPOINT_SHA256[s]}
        if not ckpt[s]['ok']:
            problems.append(f'checkpoint {s} sha mismatch')
    rec = {'prereg_sha256': common.sha256(common.PREREG), 'source': manifest['source_sha256'],
           'splits': split_check, 'checkpoints': ckpt,
           'disk_free_bytes': common.free_bytes(scratch), 'problems': problems,
           'ok': not problems, 'written_at': common.now_iso()}
    common.atomic_json(out, rec)
    return rec


def hashlib_ids_prefix(tsv, n):
    import hashlib
    ids = [i for i, _ in common.rows_for_tsv(tsv)][:n]
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


# ── baselines ────────────────────────────────────────────
def step_baselines(scratch, force=False):
    out = p0_dir(scratch) / 'baselines.json'
    if (rec := _done(out, force)):
        return rec
    rows_by_split = {}
    cells = {}
    for split in SPLITS:
        rows = common.split_rows(scratch, split)
        rows_by_split[split] = rows
        ids = [i for i, _ in rows]
        for ckpt in common.SEEDS:
            for cfgkey in common.CFG_KEYS:
                dst = common.baseline_path(scratch, split, ckpt, cfgkey)
                if not dst.exists() or force:
                    rec = baselines_mod.build_one(scratch, split, ckpt, cfgkey, ids, rows)
                    rec['seconds'] = None
                    common.atomic_json(dst, rec)
                b = common.read_json(dst)
                cells[f'{split}/{ckpt}/{cfgkey}'] = {
                    'lvl30_clap': b['lvl30']['clap'], 'lvl30_PQ': b['lvl30']['PQ'],
                    'fad': (b.get('fad') or {}).get('value'), 'n': b['n']}
    fad_vals = [v['fad'] for v in cells.values() if common.finite(v['fad'])]
    rec = {'cells': cells, 'fad_min': min(fad_vals) if fad_vals else None,
           'fad_max': max(fad_vals) if fad_vals else None,
           'fad_spread': (max(fad_vals) - min(fad_vals)) if fad_vals else None,
           'written_at': common.now_iso()}
    common.atomic_json(out, rec)
    return rec


# ── copy / seed floor ────────────────────────────────────
def step_copy(scratch, force=False):
    out = p0_dir(scratch) / 'copy.json'
    if (rec := _done(out, force)):
        return rec
    scratch = Path(scratch)
    r1 = p0_dir(scratch) / 'r1_phase0_plan.json'
    plan = {'round': 'r1', 'split': 'dev256', 'checkpoints': [common.SEEDS[0]], 'seeds': [42],
            'configs': common.load_prereg()['search_space_rung1']['configs']}
    common.atomic_json(r1, plan)
    by_id = {c['id']: c for c in plan['configs']}
    runs = {}
    for cid in ('C0_dup_seed42', 'C1_null_seed7', 'C2_null_seed161803'):
        cfg = by_id[cid]
        seed = int(cfg.get('seed', 42))
        t0 = time.time()
        res = trial_mod.run_trial(r1, cfg, common.SEEDS[0], seed, scratch, 'dev256')
        runs[cid] = {'summary': res, 'wall_s': round(time.time() - t0, 1),
                     'dir': str(common.trial_dir(scratch, plan, cfg, common.SEEDS[0], seed))}
    b_plan = p0_dir(scratch) / 'p0b_plan.json'
    b_cfg = dict(by_id['C0_dup_seed42'], id='C0_p0b')
    common.atomic_json(b_plan, {'round': 'p0b', 'split': 'dev256',
                                'checkpoints': [common.SEEDS[0]], 'seeds': [42],
                                'configs': [b_cfg]})
    t0 = time.time()
    b_res = trial_mod.run_trial(b_plan, b_cfg, common.SEEDS[0], 42, scratch, 'dev256')
    runs['C0_p0b'] = {'summary': b_res, 'wall_s': round(time.time() - t0, 1)}

    def sha_map(path):
        recs = {}
        for line in Path(path).read_text().splitlines():
            i, h = line.split('\t')
            recs[i] = h
        return recs

    a = sha_map(Path(runs['C0_dup_seed42']['dir']) / 'audio_sha256.tsv')
    b = sha_map(Path(common.trial_dir(scratch, {'round': 'p0b', 'checkpoints': ['x'], 'seeds': [42]},
                                      b_cfg, common.SEEDS[0], 42)) / 'audio_sha256.tsv')
    canon = sha_map(common.baseline_path(scratch, 'dev256', common.SEEDS[0], 'cfg3_neg')
                    .with_name('audio_sha256.tsv'))
    g0 = sum(1 for i in a if canon.get(i) == a[i])
    g1 = sum(1 for i in a if b.get(i) == a[i])
    rec = {'runs': {k: {kk: vv for kk, vv in v['summary'].items()
                        if kk in ('status', 'runtime_s', 'delta_vs_baseline', 'gates')}
                    | {'wall_s': v['wall_s'], 'dir': v.get('dir')} for k, v in runs.items()},
           'G0_matches': g0, 'G0_total': len(canon), 'G0_pass': g0 == len(canon) == 256,
           'G1_matches': g1, 'G1_total': len(a), 'G1_pass': g1 == len(a) == 256,
           'C1_delta': (runs['C1_null_seed7']['summary'].get('delta_vs_baseline') or {}).get('lvl30_clap'),
           'C2_delta': (runs['C2_null_seed161803']['summary'].get('delta_vs_baseline') or {}).get('lvl30_clap'),
           'dev256_trial_seconds_median': round(
               statistics.median(v['wall_s'] for v in runs.values()), 1),
           'written_at': common.now_iso()}
    common.atomic_json(out, rec)
    return rec


# ── FAD parity ───────────────────────────────────────────
def step_fad_parity(scratch, force=False):
    out = p0_dir(scratch) / 'fad_parity.json'
    if (rec := _done(out, force)):
        return rec
    rows = common.split_rows(scratch, 'full5521')
    cell = common.canonical_cell(common.SEEDS[0], 'cfg3_neg', 'raw')
    t0 = time.time()
    fad = fad_cache.score(cell / 'audio', rows, cache_dir=Path(scratch) / 'fad_cache')
    rec = {'value': fad['value'], 'expected': EXPECTED_FAD, 'abs_delta': abs(fad['value'] - EXPECTED_FAD),
           'tolerance': FAD_TOL, 'pass': abs(fad['value'] - EXPECTED_FAD) <= FAD_TOL,
           'pairs': fad['pairs'], 'seconds': round(time.time() - t0, 1),
           'seconds_per_pair': round((time.time() - t0) / fad['pairs'], 4),
           'cell': str(cell), 'written_at': common.now_iso()}
    common.atomic_json(out, rec)
    return rec


# ── sequential lvl30 cost ────────────────────────────────
def step_timing(scratch, force=False):
    out = p0_dir(scratch) / 'timing.json'
    if (rec := _done(out, force)):
        return rec
    probe = p0_dir(scratch) / 'lvl30_probe'
    if probe.exists():
        shutil.rmtree(probe)
    src = common.canonical_cell(common.SEEDS[0], 'cfg0', 'raw')
    tsv = common.split_tsv(scratch, 'full5521')
    out_cell = probe / f'{src.name}_lvl30'
    t0 = time.time()
    trial_mod._run([common.PY, common.ROOT / 'scripts/eval/level_match_rescore.py',
                    '--cell_dir', src,
                    '--tsv', tsv, '--target', '-30', '--peak_cap', '0.999', '--workers', '8',
                    '--out_root', probe], p0_dir(scratch) / 'timing_lvl30.log', timeout=7200)
    wall = time.time() - t0
    n = len(list((out_cell / 'audio').glob('*.flac')))
    metrics = common.read_json(out_cell / out_cell.name / 'metrics.json')
    shutil.rmtree(probe, ignore_errors=True)
    rec = {'n': n, 'seconds': round(wall, 1), 'seconds_per_clip': round(wall / max(n, 1), 4),
           'source_audio_seconds_per_clip': None,
           'lvl30_clap': ((metrics or {}).get('metrics') or {}).get('clap_score'),
           'deleted_probe_audio': True, 'written_at': common.now_iso()}
    common.atomic_json(out, rec)
    return rec


# ── report ───────────────────────────────────────────────
def step_report(scratch, force=False):
    out = p0_dir(scratch) / 'p0_summary.json'
    if (rec := _done(out, force)):
        return rec
    v = common.read_json(p0_dir(scratch) / 'verify.json') or {}
    b = common.read_json(p0_dir(scratch) / 'baselines.json') or {}
    c = common.read_json(p0_dir(scratch) / 'copy.json') or {}
    f = common.read_json(p0_dir(scratch) / 'fad_parity.json') or {}
    t = common.read_json(p0_dir(scratch) / 'timing.json') or {}
    c1 = abs(c.get('C1_delta') or 0.0)
    c2 = abs(c.get('C2_delta') or 0.0)
    floor = common.sigma_floor(256)
    gates = {
        'verify': bool(v.get('ok')),
        'G0_copy': bool(c.get('G0_pass')),
        'G1_determinism': bool(c.get('G1_pass')),
        'fad_parity': bool(f.get('pass')),
        'baselines_complete': bool(b.get('cells') and all(
            common.finite(x.get('fad')) and common.finite(x.get('lvl30_clap'))
            for x in b['cells'].values())),
        'timing_measured': bool(t.get('n')),
    }
    rec = {'status': 'pass' if all(gates.values()) else 'fail', 'gates': gates,
           'sigma_seed_256_prereg': floor,
           'sigma_measured_max_null': max(c1, c2, floor),
           'sigma_used_256': max(c1, c2, floor),
           'measured': {
               'dev256_trial_seconds': c.get('dev256_trial_seconds_median'),
               'fad_seconds_per_pair': f.get('seconds_per_pair'),
               'fad_full5521_pairs': f.get('pairs'),
               'lvl30_full5521_seconds': t.get('seconds'),
               'lvl30_seconds_per_clip': t.get('seconds_per_clip'),
               'fad_subset_spread': b.get('fad_spread'),
           },
           'baselines': b.get('cells'), 'written_at': common.now_iso()}
    common.atomic_json(out, rec)
    return rec


STEPS_FN = {'verify': step_verify, 'baselines': step_baselines, 'copy': step_copy,
            'fad-parity': step_fad_parity, 'timing': step_timing, 'report': step_report}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('all',) + STEPS)
    p.add_argument('--scratch', default=str(common.SCRATCH))
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    todo = STEPS if a.action == 'all' else (a.action,)
    results = {}
    for name in todo:
        print(f'[phase0] start {name}', flush=True)
        results[name] = STEPS_FN[name](a.scratch, a.force)
        print(f'[phase0] done {name}: '
              f'{ {k: results[name].get(k) for k in ("ok", "G0_pass", "G1_pass", "pass", "status")} }',
              flush=True)
    summary = common.read_json(p0_dir(a.scratch) / 'p0_summary.json') if a.action == 'all' else None
    if a.action == 'all':
        if not summary:
            summary = step_report(a.scratch, force=True)
        print(json.dumps({'status': summary['status'], 'gates': summary['gates'],
                          'measured': summary['measured']}, indent=1), flush=True)
        return 0 if summary['status'] == 'pass' else 4
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
