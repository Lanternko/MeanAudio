#!/usr/bin/env python3
"""Write the launcher candidate + contract for one round-2 queue job (contract first, wrapper last).

  python make_queue_job.py <queue_no> <label> <plan_job> <ds=model,model>[ <ds=model,...>] [--question TEXT]
Writes scripts/queue_candidates/<no>_bench_r2_<label>.sh and docs/experiments/bench_r2_<label>_contract.json
(shas of the committed action/guest/scorer files).  Copy the launcher into ~/gpu_queue/p2/pending/ afterwards.
"""
import argparse
import hashlib
import json
from pathlib import Path

M = Path('/home/kojiek/MeanAudio')
S = M / 'scripts'
PY = '/home/kojiek/venvs/dac/bin/python'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('no')
    ap.add_argument('label')
    ap.add_argument('plan_job')
    ap.add_argument('spec', nargs='+')
    ap.add_argument('--question', default='')
    ap.add_argument('--hours', type=float, default=0)
    a = ap.parse_args()
    models = {x.split('=')[0]: x.split('=')[1].split(',') for x in a.spec}
    name = f'{a.no}_bench_r2_{a.label}'
    contract = M / f'docs/experiments/bench_r2_{a.label}_contract.json'
    launcher = S / f'queue_candidates/{name}.sh'
    b = {'launcher': f'/home/kojiek/gpu_queue/p2/pending/{name}.sh', 'launcher_sha256': None}
    launcher.write_text(f'''#!/bin/bash
# GPU_QUEUE_CONTRACT={contract}
set -eo pipefail
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT={contract}
export PYTHONUNBUFFERED=1; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
cd /home/kojiek/MeanAudio || exit 1
exec {PY} /home/kojiek/MeanAudio/scripts/experiment_harness/bench_r2_guest.py
''')
    launcher.chmod(0o755)
    b['launcher_sha256'] = sha(launcher)
    for k, p in [('harn', 'experiment_harness/bench_r2_guest.py'), ('action', 'eval/bench_r2/run_job.py'),
                 ('score_r2', 'eval/bench_r2/score_r2.py'), ('decode_cache', 'eval/bench_r2/decode_cache.py'),
                 ('bench_datasets', 'eval/bench_r2/bench_datasets.py'), ('eval_metrics', 'eval/eval_metrics.py')]:
        b[k] = str(S / p)
        b[k + '_sha256'] = sha(S / p)
    act = str(S / 'eval/bench_r2/run_job.py')
    rep = str(M / f'docs/experiments/results/bench_r2_{a.label}_summary.json')
    c = {
        'schema_version': 1, 'document_kind': 'eval_only_metrics_contract',
        'experiment_id': f'bench-r2-{a.label}-20261025', 'job': a.no, 'status': 'authorized_p2_pending', 'priority': 'P2',
        'question': a.question or f'Round-2 benchmark screening {a.label} (preregistered §9.3): scores only; analysis is a later step.',
        'design_document': str(M / 'docs/experiments/benchmark_screening_r2_20261025.md'),
        'operator_instruction': {'recorded_at': '2026-10-25', 'text': 'Run the preregistered round-2 screening scoring jobs; thresholds, seeds and variants are not changed.'},
        'approval_required': True,
        'launch_authorization': {'gpu_launch_allowed': True, 'queue_position': 'p2',
                                 'scope': f'Eval-only scoring, plan key {a.plan_job}, estimate {a.hours or "?"} h. Output /mnt/seagate/benchmarks_r2/{{derived,scores}}.', 'valid': True},
        'bindings': b,
        'commands': {'preflight': [PY, '-I', act, '--preflight'], 'run': [PY, '-I', act], 'postflight': [PY, '-I', act, '--validate-only']},
        'completion_evidence': {'eval_only': True, 'method': 'guest writes terminal with sha256 of the summary json after --validate-only passes (every planned unit scored or in .err.tsv)'},
        'fixed_protocol': {'job': a.plan_job, 'datasets': list(models), 'models': models,
                           'min_free_bytes': 21474836480, 'seed': 20261025},
        'storage': {'path': '/mnt/seagate/benchmarks_r2', 'min_free_bytes': 21474836480},
        'resume': {'kind': 'from_scratch_with_autoresume', 'note': 'row-level resume on unit_id in scores/*.tsv; decode is idempotent; preflight waits (rc 75) while a decode_cache.py for the same datasets runs',
                   'pause_progress': f'/mnt/seagate/benchmarks_r2/pause_progress_{a.label}.json',
                   'autoresume': f'/mnt/seagate/benchmarks_r2/pause_progress_{a.label}.json', 'checkpoint': None, 'iteration': 0},
        'dependencies': ['downloads and manifests (done)'],
        'reports': [{'path': rep}],
        'queue_name': f'{name}.sh', 'queue_role': 'p2', 'watcher': {'stall_seconds': 28800}, 'launch_allowed': True,
        'revision_history': [{'rev': 'r1', 'date': '2026-10-25', 'note': 'dispatch task 20261025-044112-c1b491; score_r2 also imports 126d/133/meva_runtime (not sha-bound).'}],
    }
    contract.write_text(json.dumps(c, indent=1))
    print(contract, launcher)


if __name__ == '__main__':
    main()
