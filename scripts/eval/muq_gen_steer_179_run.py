#!/usr/bin/env python
"""179 orchestrator: prompt subset -> generation (6 arms) -> scoring/analysis.

  python scripts/eval/muq_gen_steer_179_run.py --preflight
  python scripts/eval/muq_gen_steer_179_run.py            # full run
  python scripts/eval/muq_gen_steer_179_run.py --smoke 4  # GPU smoke, writes under OUT/_smoke
  python scripts/eval/muq_gen_steer_179_run.py --validate-only

Generation skips an arm whose audio directory is already complete (a partial arm is
regenerated from scratch: eval.py draws one RNG stream per run). Scoring caches one
JSON per arm and the lowpass reference, so a restart only redoes missing stages.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
import muq_gen_steer_179_common as C  # noqa: E402

PY = '/home/kojiek/venvs/dac/bin/python'
MIN_FREE = 8 * 2 ** 30


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sh(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log, 'a', encoding='utf-8') as f:
        f.write(f'\n=== {" ".join(map(str, cmd))} @ {now()} ===\n')
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT).returncode
    return {'cmd': [str(c) for c in cmd], 'rc': rc, 'seconds': round(time.time() - t, 1)}


def preflight():
    if not C.CKPT.is_file():
        print(f'[FAIL] checkpoint missing: {C.CKPT}')
        return 2
    if not C.MC_TSV.is_file():
        print(f'[FAIL] MusicCaps TSV missing: {C.MC_TSV}')
        return 2
    free = shutil.disk_usage('/home/kojiek').free
    if free < MIN_FREE:
        print(f'[FAIL] NVMe free {free} B < {MIN_FREE} B')
        return 3
    C.OUT.mkdir(parents=True, exist_ok=True)
    print(f'[OK] ckpt {C.CKPT.stat().st_size} B; free {free} B')
    return 0


def plan(mode, root):
    """(arms_root, tsv, arms, score args)"""
    if mode == 'smoke':
        n = 4
        tsv, _ = C.write_subset_tsv(root / 'inputs' / f'mc{n}.tsv', n)
        return root, tsv, C.SMOKE_ARMS, ['--no-docs', '--no-lowpass']
    tsv, n = C.write_subset_tsv(root / 'inputs' / f'mc{C.N_PROMPTS}.tsv', C.N_PROMPTS)
    if n != C.N_PROMPTS:
        raise SystemExit(f'[FAIL] TSV has {n} records, expected {C.N_PROMPTS}')
    return root, tsv, C.FULL_ARMS, []


def run(mode):
    root = C.OUT / '_smoke' if mode == 'smoke' else C.OUT
    root.mkdir(parents=True, exist_ok=True)
    arms_root, tsv, arms, score_args = plan(mode, root)
    stages = []
    gen = sh([PY, '-I', HERE / 'muq_gen_steer_179_gen.py', '--out', arms_root,
              '--tsv', tsv, '--arms', ','.join(arms)], root / 'logs' / 'gen.log')
    stages.append({'stage': 'gen', **gen})
    if gen['rc']:
        report(root, mode, stages, status='failed')
        return 3
    score = sh([PY, '-I', HERE / 'muq_gen_steer_179_score.py', '--root', arms_root,
                '--tsv', tsv, '--arms', ','.join(arms), *score_args], root / 'logs' / 'score.log')
    stages.append({'stage': 'score', **score})
    report(root, mode, stages, status='completed' if score['rc'] == 0 else 'failed')
    return 0 if score['rc'] == 0 else 4


def report(root, mode, stages, status):
    out = {'document_kind': 'muq_gen_steer_179_run_v1', 'status': status, 'mode': mode,
           'written_at': now(), 'stages': stages}
    for name in ('summary.json',):
        p = Path(root) / name
        if p.is_file():
            out['summary_sha256'] = C.sha256(p)
    C.write_json(Path(root) / 'run.json', out)


def validate_only():
    root = C.OUT
    problems = []
    for arm in C.FULL_ARMS:
        d = root / arm / 'audio'
        n = len(list(d.glob('*.flac'))) if d.is_dir() else 0
        if n != C.N_PROMPTS:
            problems.append(f'{arm}: {n}/{C.N_PROMPTS} clips')
        sc = root / 'scores' / f'{arm}.json'
        rec = C.read_json(sc)
        if not rec or len(rec) != C.N_PROMPTS:
            problems.append(f'{arm}: score cache missing/short')
    lp = C.read_json(root / 'scores' / 'lowpass_ref.json')
    if not lp or len(lp) != C.N_PROMPTS * 3:
        problems.append('lowpass_ref: missing/short')
    for p in (root / 'summary.json', root / 'run.json',
              C.RESULTS_DIR / 'muq_gen_steer_179_summary.json',
              C.RESULTS_DIR / 'muq_gen_steer_179_results.md'):
        if not p.is_file() or p.stat().st_size <= 0:
            problems.append(f'missing/empty: {p}')
    S = C.read_json(root / 'summary.json')
    if not S or not S.get('verdict'):
        problems.append('summary has no verdict')
    if problems:
        print('[FAIL] validate: ' + '; '.join(problems))
        return 2
    print(f'[OK] validate: 6 arms x {C.N_PROMPTS} clips, lowpass ref, summary, results docs')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--smoke', type=int, default=0, help='smoke run with this many prompts')
    a = ap.parse_args()
    if a.preflight:
        return preflight()
    if a.validate_only:
        return validate_only()
    return run('smoke' if a.smoke else 'full')


if __name__ == '__main__':
    raise SystemExit(main())
