#!/usr/bin/env python
"""One trial: render -> raw level -> lvl30 rescore -> FAD -> gates -> summary.json.

Every stage is a separate subprocess so VRAM comes back between them, and every
heavyweight script is the canonical one (eval.py through render.py,
eval_metrics.py, level_match_rescore.py); this file only sequences them and
applies the prereg gates.

    trial.py --configs rounds/r1.json --config-id F2_short --split dev256 \
             --checkpoint s14159265 --scratch SCRATCH [--seed 42] [--force]
    trial.py --score-only ...          # recompute gates from existing artifacts

Resume rule (prereg loop.resume): a trial dir whose summary.json carries the same
config_sha is skipped. Audio is deleted after scoring (dispatch restriction);
audio_sha256.tsv is kept so the copy gate survives the deletion.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                   # noqa: E402
import fad_cache                                # noqa: E402

PY = common.PY


def _run(cmd, log_path, timeout=7200):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f'\n[{common.now_iso()}] $ {" ".join(str(c) for c in cmd)}\n')
        log.flush()
        t0 = time.time()
        proc = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(common.ROOT), timeout=timeout)
        log.write(f'[{common.now_iso()}] exit={proc.returncode} {time.time() - t0:.1f}s\n')
    if proc.returncode != 0:
        raise RuntimeError(f'exit {proc.returncode}: {cmd[1]} (log {log_path})')


def raw_cell(t, cfg_id):
    return t / 'cells' / f'{cfg_id}_raw'


def lvl30_cell(t, cfg_id):
    return t / 'lvl30' / f'{cfg_id}_raw_lvl30'


def write_sha_map(audio_dir, ids, path):
    lines = []
    for i in ids:
        p = Path(audio_dir) / f'{i}.flac'
        if not p.is_file():
            raise RuntimeError(f'missing rendered clip {p}')
        lines.append(f'{i}\t{common.sha256(p)}')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _ids_of(audio_dir):
    return sorted(p.stem for p in Path(audio_dir).glob('*.flac'))


def run_trial(plan_path, cfg, ckpt, seed, scratch, split, do_fad=True, force=False,
              keep_audio=False):
    scratch = Path(scratch)
    plan = common.load_plan(plan_path)
    t = common.trial_dir(scratch, plan, cfg, ckpt, seed)
    cfg_sha = common.config_sha(cfg, ckpt, seed, split)
    summary_path = t / 'summary.json'
    prev = common.read_json(summary_path)
    if prev and prev.get('config_sha') == cfg_sha and not force:
        return dict(prev, skipped=True)
    if t.exists():
        # stale config or a partial (killed) run: eval.py skips existing flacs and
        # does not draw their noise, so a top-up is not the same audio -> restart
        shutil.rmtree(t)
    t.mkdir(parents=True, exist_ok=True)
    raw, lvl = raw_cell(t, cfg['id']), lvl30_cell(t, cfg['id'])
    tsv = common.split_tsv(scratch, split)
    rows = common.rows_for_tsv(tsv)
    ids = [i for i, _ in rows]
    started = time.time()
    common.atomic_json(t / 'config.json', {
        'config': cfg, 'checkpoint': ckpt, 'checkpoint_sha256': common.CHECKPOINT_SHA256[ckpt],
        'seed': int(seed), 'split': split, 'n': len(ids), 'config_sha': cfg_sha,
        'plan': str(plan_path), 'plan_round': plan['round'], 'tsv': str(tsv),
        'tsv_sha256': common.sha256(tsv), 'started_at': common.now_iso(),
    })
    try:
        _run([PY, common.HERE / 'render.py', '--out', raw / 'audio', '--tsv', tsv,
              '--checkpoint', common.CHECKPOINTS[ckpt], '--configs', plan_path,
              '--config-id', cfg['id'], '--seed', seed], raw / 'gen.log')
        got = _ids_of(raw / 'audio')
        if got != sorted(ids):
            raise RuntimeError(f'render produced {len(got)} clips, expected {len(ids)} '
                               f'(missing {sorted(set(ids) - set(got))[:3]})')
        write_sha_map(raw / 'audio', ids, t / 'audio_sha256.tsv')
        _run([PY, common.ROOT / 'scripts/eval/eval_metrics.py', '--gen_dir', raw / 'audio',
              '--tsv', tsv, '--exp_name', f"{cfg['id']}_raw", '--out_dir', raw,
              '--skip_clap', '--skip_aes'], raw / 'level.log')
        _run([PY, common.ROOT / 'scripts/eval/level_match_rescore.py', '--cell_dir', raw,
              '--tsv', tsv, '--target', '-30', '--peak_cap', '0.999', '--workers', '8',
              '--out_root', t / 'lvl30'], t / 'lvl30.log')
        if do_fad:
            _run([PY, common.HERE / 'fad_cache.py', '--gen-dir', raw / 'audio', '--tsv', tsv,
                  '--split', split, '--out', t / 'fad.json',
                  '--cache-dir', scratch / 'fad_cache'], t / 'fad.log')
        out = score_only(t, cfg, ckpt, seed, scratch, split, ids)
        out['config_sha'] = cfg_sha
        out['runtime_s'] = round(time.time() - started, 1)
        out['finished_at'] = common.now_iso()
        if not keep_audio:
            shutil.rmtree(raw / 'audio', ignore_errors=True)
            shutil.rmtree(lvl / 'audio', ignore_errors=True)
            out['audio_deleted'] = True
        common.atomic_json(summary_path, out)
        return out
    except Exception as exc:                     # noqa: BLE001 - recorded, not swallowed
        rec = {'config_id': cfg['id'], 'config_sha': cfg_sha, 'split': split,
               'checkpoint': {'name': ckpt, 'sha256': common.CHECKPOINT_SHA256[ckpt]},
               'n': len(ids), 'status': 'error', 'error': str(exc)[:800],
               'runtime_s': round(time.time() - started, 1), 'finished_at': common.now_iso()}
        common.atomic_json(summary_path, rec)
        return rec


def score_only(t, cfg, ckpt, seed, scratch, split, ids=None):
    t = Path(t)
    scratch = Path(scratch)
    raw, lvl = raw_cell(t, cfg['id']), lvl30_cell(t, cfg['id'])
    if ids is None:
        ids = common.split_ids(scratch, split)
    n = len(ids)
    base = common.load_baseline(scratch, split, ckpt, common.baseline_cfgkey(cfg))
    problems = []

    def load_metrics(cell, name):
        rec = common.read_json(Path(cell) / name / 'metrics.json')
        if not rec:
            problems.append(f'missing metrics {cell / name / "metrics.json"}')
        return (rec or {}).get('metrics', {})

    lvl_m = load_metrics(lvl, f"{cfg['id']}_raw_lvl30")
    raw_m = load_metrics(raw, f"{cfg['id']}_raw")
    fad = common.read_json(t / 'fad.json')
    gain = lvl / 'gain_status.tsv'
    gain_rows = common.read_gain_status(gain) if gain.is_file() else []
    if not gain_rows:
        problems.append('missing gain_status.tsv')

    raw_audio_ids = _ids_of(raw / 'audio') if (raw / 'audio').is_dir() else []
    lvl_audio_ids = _ids_of(lvl / 'audio') if (lvl / 'audio').is_dir() else []
    audio_deleted = not raw_audio_ids and (common.read_json(t / 'summary.json') or {}).get(
        'audio_deleted')
    if not audio_deleted:
        if raw_audio_ids != sorted(ids):
            problems.append(f'raw audio ids differ from split ({len(raw_audio_ids)} vs {n})')
        if lvl_audio_ids != sorted(ids):
            problems.append(f'lvl30 audio ids differ from split ({len(lvl_audio_ids)} vs {n})')

    lvl30_m = {
        'clap': lvl_m.get('clap_score'), 'pq': lvl_m.get('aes_PQ'),
        'ce': lvl_m.get('aes_CE'), 'cu': lvl_m.get('aes_CU'), 'pc': lvl_m.get('aes_PC'),
        'lufs': lvl_m.get('level_lufs_mean'), 'silent_n': lvl_m.get('level_silent_n'),
        'clipped_n': lvl_m.get('level_clipped_n'),
        'peak_capped_n': sum(r.get('status') == 'peak_capped' for r in gain_rows),
        'lufs_off_0p05_n': None, 'n_scored': lvl_m.get('clap_n'),
    }
    per_clip = lvl / f"{cfg['id']}_raw_lvl30" / 'per_clip.tsv'
    if per_clip.is_file():
        recs = common.read_per_clip(per_clip)
        lvl30_m['lufs_off_0p05_n'] = sum(
            1 for r in recs if r['lufs'] is not None and abs(r['lufs'] + 30.0) > 0.05)
    raw_lvl = {
        'clap': raw_m.get('clap_score'), 'silent_n': raw_m.get('level_silent_n'),
        'clipped_n': raw_m.get('level_clipped_n'), 'crest_mean': raw_m.get('level_crest_mean'),
        'lufs': raw_m.get('level_lufs_mean'),
    }

    for key in ('clap', 'pq', 'lufs', 'peak_capped_n', 'lufs_off_0p05_n'):
        if not common.finite(lvl30_m.get(key)):
            problems.append(f'lvl30.{key} not finite ({lvl30_m.get(key)!r})')
    for key in ('silent_n', 'clipped_n', 'crest_mean', 'lufs'):
        if not common.finite(raw_lvl.get(key)):
            problems.append(f'raw.{key} not finite ({raw_lvl.get(key)!r})')

    gates = {}
    dl = {}
    dl['lvl30_clap'] = (lvl30_m['clap'] - base['lvl30']['clap']
                        if common.finite(lvl30_m.get('clap')) and common.finite(base['lvl30']['clap'])
                        else None)
    dl['lvl30_pq'] = (lvl30_m['pq'] - base['lvl30']['PQ']
                      if common.finite(lvl30_m.get('pq')) and common.finite(base['lvl30']['PQ'])
                      else None)
    dl['fad'] = (fad['value'] - base['fad']['value']
                 if fad and 'fad' in base and common.finite(fad.get('value'))
                 and common.finite(base['fad'].get('value')) else None)
    gates, status = compute_gates(lvl30_m, raw_lvl, base, dl, n, problems)
    return {
        'config_id': cfg['id'], 'family': cfg.get('family'), 'split': split, 'n': n,
        'checkpoint': {'name': ckpt, 'sha256': common.CHECKPOINT_SHA256[ckpt]},
        'seed': int(seed), 'cfg': cfg.get('cfg'), 'interval': cfg.get('interval'),
        'schedule': cfg.get('schedule'), 'geometry': cfg.get('geometry'),
        'steps': cfg.get('steps'), 'grid': cfg.get('grid'), 'prompt': cfg.get('prompt'),
        'lvl30': lvl30_m, 'raw': raw_lvl,
        'fad': ({'value': fad.get('value'), 'pairs': fad.get('pairs'),
                 'ref_cache_hit': fad.get('ref_cache_hit')} if fad else None),
        'baseline': {'cfg': common.baseline_cfgkey(cfg),
                     'lvl30_clap': base['lvl30']['clap'], 'lvl30_pq': base['lvl30']['PQ'],
                     'raw_silent_n': base['raw']['silent_n'],
                     'raw_clipped_n': base['raw']['clipped_n'],
                     'raw_crest_mean': base['raw']['crest_mean'],
                     'peak_capped_n': base['lvl30']['peak_capped_n'],
                     'fad': (base.get('fad') or {}).get('value')},
        'delta_vs_baseline': dl, 'gates': gates, 'status': status,
        'problems': problems[:8],
    }


def compute_gates(lvl30_m, raw_lvl, base, dl, n, problems):
    """The prereg gates, as a pure function so the unit tests can drive every branch."""
    gates = {'G0': None, 'G1': None, 'G8': True}
    gates['G2'] = bool(dl.get('fad') is not None and dl['fad'] <= common.THRESHOLDS['fad'])
    gates['G3'] = bool(dl.get('lvl30_pq') is not None and dl['lvl30_pq'] >= common.THRESHOLDS['pq'])
    base_silent = base['raw']['silent_n']
    gates['G4'] = bool(common.finite(raw_lvl.get('silent_n')) and
                       raw_lvl['silent_n'] - base_silent <= max(3, 2 * base_silent))
    gates['G5'] = bool(common.finite(raw_lvl.get('crest_mean'))
                       and common.finite(raw_lvl.get('clipped_n'))
                       and raw_lvl['clipped_n'] <= base['raw']['clipped_n']
                       and raw_lvl['crest_mean'] >= base['raw']['crest_mean'] - 0.5)
    cap_limit = max(base['lvl30']['peak_capped_n'] + 3, 0.005 * n)
    lufs_limit = max(1, int(0.01 * n))
    gates['G6'] = bool(common.finite(lvl30_m.get('peak_capped_n'))
                       and common.finite(lvl30_m.get('lufs_off_0p05_n'))
                       and lvl30_m['peak_capped_n'] <= cap_limit
                       and lvl30_m['lufs_off_0p05_n'] <= lufs_limit)
    gates['G7'] = not problems
    hard = [g for g in ('G3', 'G4', 'G5', 'G6', 'G7') if gates[g] is False]
    if hard:
        status = 'reject'
    elif gates['G2'] is False:
        status = 'fad_pending'
    else:
        status = 'pass'
    return gates, status


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--configs', required=True)
    p.add_argument('--config-id', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--scratch', default=str(common.SCRATCH))
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--score-only', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--no-fad', action='store_true')
    p.add_argument('--keep-audio', action='store_true')
    a = p.parse_args(argv)
    plan = common.load_plan(a.configs)
    cfg = next((c for c in plan['configs'] if c['id'] == a.config_id), None)
    if cfg is None:
        raise SystemExit(f'[FAIL] config {a.config_id} not in {a.configs}')
    seed = a.seed if a.seed is not None else common.effective_seed(plan, cfg)
    if a.score_only:
        t = common.trial_dir(Path(a.scratch), plan, cfg, a.checkpoint, seed)
        out = score_only(t, cfg, a.checkpoint, seed, a.scratch, a.split)
        out['config_sha'] = common.config_sha(cfg, a.checkpoint, seed, a.split)
        common.atomic_json(t / 'summary.json', out)
    else:
        out = run_trial(a.configs, cfg, a.checkpoint, seed, a.scratch, a.split,
                        do_fad=not a.no_fad, force=a.force, keep_audio=a.keep_audio)
    print(json.dumps({k: out.get(k) for k in ('config_id', 'split', 'status', 'runtime_s',
                                              'delta_vs_baseline', 'gates', 'skipped')},
                     ensure_ascii=False), flush=True)
    return 0 if out.get('status') != 'error' else 4


if __name__ == '__main__':
    raise SystemExit(main())
