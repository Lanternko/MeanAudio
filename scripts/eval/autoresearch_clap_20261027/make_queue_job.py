#!/usr/bin/env python3
"""Generate the p2 contract + launcher for one autoresearch-clap job.

    make_queue_job.py --job 153 --mode phase0
    make_queue_job.py --job 154 --mode round --round r1 --plan rounds/r1.json

Writes, in this order:
  docs/experiments/autoresearch_<job>_<tag>_contract.json
  scripts/queue_candidates/<job>_autoresearch_<tag>.sh   (mode 755)

The launcher's sha is what the contract binds, so the launcher is rendered first
(its text only depends on paths, not on the contract's own hash). Copy the
launcher into ~/gpu_queue/p2/pending/ only after the contract is committed:
`pending/` must not contain dotfiles, and the contract has to exist first.
"""
import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import common                                   # noqa: E402

PY = common.PY
DOCS = ROOT / 'docs' / 'experiments'
CANDIDATES = ROOT / 'scripts' / 'queue_candidates'
PENDING = Path('/home/kojiek/gpu_queue/p2/pending')
HARN = ROOT / 'scripts' / 'experiment_harness' / 'autoresearch_clap_20261027_guest.py'
ACTION = HERE / 'run_round.py'
PREREG = common.PREREG
MIN_FREE = 20_000_000_000

# every module the action can import, plus the canonical eval scripts it shells out to
HARNESS = ('common.py', 'patch_meanflow.py', 'render.py', 'trial.py', 'fad_cache.py',
           'baselines.py', 'phase0.py', 'driver.py', 'analyze.py', 'make_round_configs.py',
           'split_subsets.py', 'run_round.py')
CANONICAL = ('eval.py', 'scripts/eval/eval_metrics.py',
             'scripts/eval/level_match_rescore.py', 'scripts/eval/mc_mf25_eval.sh',
             'meanaudio/model/networks.py', 'meanaudio/model/mean_flow.py')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def names(job, mode, rnd):
    tag = 'p0' if mode == 'phase0' else rnd
    return (f'{job}_autoresearch_{tag}.sh', f'autoresearch_{job}_{tag}_contract.json')


def launcher_text(contract_path):
    return f'''#!/bin/bash
# GPU_QUEUE_CONTRACT={contract_path}
set -eo pipefail
free=$(df -B1 --output=avail /home/kojiek | tail -1 | tr -d ' ')
if [ "${{free:-0}}" -lt {MIN_FREE} ]; then
  echo "[DISK] NVMe avail ${{free:-0}} B < {MIN_FREE} B, refusing to start" >&2
  exit 3
fi
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT={contract_path}
export PYTHONUNBUFFERED=1; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
cd {ROOT} || exit 1
exec {PY} -I {HARN}
'''


def code_bindings():
    out = {}
    for name in HARNESS:
        p = HERE / name
        out[f'autoresearch_clap_20261027/{name}'] = sha(p)
    for rel in CANONICAL:
        p = ROOT / rel
        out[rel] = sha(p)
    out['scripts/experiment_harness/autoresearch_clap_20261027_guest.py'] = sha(HARN)
    out['docs/experiments/autoresearch_clap_20261027_prereg.json'] = sha(PREREG)
    return out


def build(job, mode, rnd=None, plan=None, scratch=None, dry=False):
    scratch = Path(scratch or common.SCRATCH)
    launcher_name, contract_name = names(job, mode, rnd)
    launcher = CANDIDATES / launcher_name
    contract_path = DOCS / contract_name
    plan_abs = (scratch / plan) if plan else None
    if mode == 'round':
        if not rnd or not plan:
            raise SystemExit('[FAIL] round mode needs --round and --plan')
        plan_rec = common.load_plan(plan_abs)
        if plan_rec.get('round') != rnd:
            raise SystemExit(f'[FAIL] plan round {plan_rec.get("round")} != {rnd}')
        n_trials = len(plan_rec['configs']) * len(plan_rec['checkpoints']) * len(plan_rec['seeds'])
        split = plan_rec['split']
        question = (f'Rung {rnd} of the inference-time CLAP search: {n_trials} planned trial(s) '
                    f'on {split} (plan {plan_abs}); thresholds, splits and gates are frozen by '
                    f'the prereg.')
        report = scratch / 'rounds_out' / f'{rnd}_summary.json'
        deps = [f'phase 0 passed ({scratch}/phase0/p0_summary.json)',
                f'plan registered ({scratch}/rounds/{rnd}.sha256)']
    else:
        question = ('Phase 0 of the inference-time CLAP search: split/checkpoint sha verify, '
                    'six subset baselines per split, copy gate + determinism on dev256, FAD '
                    'parity 5.0836, and one sequential full-5521 lvl30 timing cell; '
                    'p0_summary.json gates rung 1.')
        report = scratch / 'phase0' / 'p0_summary.json'
        deps = ['canonical cfg0/cfg3_neg full cells', 'three checkpoints']

    text = launcher_text(contract_path)
    if not dry:
        CANDIDATES.mkdir(parents=True, exist_ok=True)
        launcher.write_text(text, encoding='utf-8')
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    contract = {
        'schema_version': 1,
        'document_kind': 'eval_only_metrics_contract',
        'experiment_id': f'autoresearch-clap-20261027-{job}',
        'job': str(job),
        'status': 'authorized_p2_pending',
        'priority': 'P2',
        'question': question,
        'design_document': str(ROOT / 'docs/experiments/autoresearch_inference_clap_20261027.md'),
        'operator_instruction': {
            'recorded_at': '2026-10-27',
            'text': ('Run the preregistered inference-time CLAP search; the 38 rung-1 configs, '
                     'gates, splits and thresholds are not changed. Phase 0 gates rung 1.'),
        },
        'approval_required': True,
        'launch_authorization': {
            'gpu_launch_allowed': True,
            'queue_position': 'p2',
            'scope': ('Eval-only inference of the three frozen checkpoints; no training, no '
                      f'checkpoint writes. Output {scratch}.'),
            'valid': True,
        },
        'bindings': {
            'launcher': str(PENDING / launcher_name),
            'launcher_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'harn': str(HARN),
            'harn_sha256': sha(HARN),
            'action': str(ACTION),
            'action_sha256': sha(ACTION),
            'prereg': str(PREREG),
            'prereg_sha256': sha(PREREG),
        },
        'code_bindings': code_bindings(),
        'commands': {
            'preflight': [PY, '-I', str(ACTION), '--preflight'],
            'run': [PY, '-I', str(ACTION)],
            'postflight': [PY, '-I', str(ACTION), '--validate-only'],
        },
        'completion_evidence': {
            'eval_only': True,
            'method': ('guest writes terminal with sha256 of the summary json after '
                       '--validate-only passes (phase 0: p0_summary status pass; round: every '
                       'planned trial summary carries the planned config_sha)'),
        },
        'fixed_protocol': {
            'mode': mode,
            'scratch': str(scratch),
            'round': (rnd if mode == 'round' else 'p0'),
            'plan': (str(plan_abs) if plan_abs else None),
            'prereg_sha256': sha(PREREG),
            'min_free_bytes': MIN_FREE,
        },
        'storage': {'path': str(scratch), 'min_free_bytes': MIN_FREE},
        'resume': {
            'kind': 'from_scratch_with_autoresume',
            'note': ('trial-level resume: a trial whose summary.json carries the planned '
                     'config_sha is skipped; audio is deleted after scoring and only '
                     'audio_sha256.tsv is kept'),
            'pause_progress': str(scratch / f'pause_progress_{job}.json'),
            'autoresume': str(scratch / f'pause_progress_{job}.json'),
            'checkpoint': None,
            'iteration': 0,
        },
        'dependencies': deps,
        'reports': [{'path': str(report)}],
        'queue_name': launcher_name,
        'queue_role': 'p2',
        'watcher': {'stall_seconds': 3600},
        'launch_allowed': True,
        'revision_history': [{
            'rev': 'r1', 'date': '2026-10-27',
            'note': ('generated by make_queue_job.py from prereg r5; phase 0 is job 153, '
                     'rungs 1-5 are 154-158'),
        }],
    }
    payload = json.dumps(contract, indent=1, ensure_ascii=False) + '\n'
    if not dry:
        contract_path.write_text(payload, encoding='utf-8')
    return {
        'launcher': str(launcher), 'launcher_sha256': contract['bindings']['launcher_sha256'],
        'contract': str(contract_path),
        'contract_sha256': hashlib.sha256(payload.encode()).hexdigest(),
        'mode': mode, 'round': contract['fixed_protocol']['round'],
        'pending_path': str(PENDING / launcher_name), 'dry_run': dry,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--mode', required=True, choices=('phase0', 'round'))
    p.add_argument('--round', choices=('r1', 'r2', 'r3', 'r4', 'r5'))
    p.add_argument('--plan', help='plan path relative to --scratch (round mode)')
    p.add_argument('--scratch', default=str(common.SCRATCH))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    rec = build(a.job, a.mode, a.round, a.plan, a.scratch, a.dry_run)
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
