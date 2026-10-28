#!/usr/bin/env python3
"""Generate the 17 p2 launchers and contracts for jobs 161-177 (§6).

Format follows docs/experiments/autoresearch_154_r1_contract.json: bindings carry the
launcher / guest / driver / prereg shas, code_bindings carry every module that can
change a score, and the guest (`metric_coverage_20261027_guest.py`) runs
`commands.preflight` at startup so a missing prerequisite exits nonzero immediately.

Run this last: the shas are computed from the files on disk.  Editing a bound module
afterwards invalidates its contract (score_coverage --preflight reports the mismatch).
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import common as C  # noqa: E402

REPO = C.REPO
PENDING = Path('/home/kojiek/gpu_queue/p2/pending')
PY = '/home/kojiek/venvs/dac/bin/python'
DRIVER = HERE / 'score_coverage.py'
GUEST = REPO / 'scripts/experiment_harness/metric_coverage_20261027_guest.py'
PLAN = C.DOCS / 'metric_coverage_completion_20261027.md'

NAMES = {
    161: 'prep', 162: 'clean_s',
    163: 'muqeval_a1_rest', 164: 'muqeval_a1_126a',
    165: 'muqeval_a3a_rest', 166: 'muqeval_a3a_126a',
    167: 'cmirm_rest', 168: 'cmirm_126a',
    169: 'pam_rest', 170: 'pam_126a_ours', 171: 'pam_126a_baselines',
    172: 'singmos_rest', 173: 'singmos_126a',
    174: 'muq_126a_134', 175: 'songeval_mert_134',
    176: 'mir', 177: 'judge',
}

QUESTIONS = {
    161: 'CPU prep: regenerate the sample manifest (must match the committed bytes), '
         'sample-vs-full agreement, the 126a render cache, and SongEval/MERT/MuQ rec+floor backfill.',
    162: 'Clean S for six new models on the 5 TTM clean systems (126a definition; MuQ-embed '
         'uses the 133 centroid definition as preregistered).',
    163: 'MuQ-Eval A1 on 126b/126c/134 (full stimuli).',
    164: 'MuQ-Eval A1 on the 126a sample (512 clips, 53 variants x 5 systems).',
    165: 'MuQ-Eval A3a on 126b/126c/134 (full stimuli).',
    166: 'MuQ-Eval A3a on the 126a sample (512 clips, 53 variants x 5 systems).',
    167: 'CMI-RM on 126b/126c/134 (full stimuli).',
    168: 'CMI-RM on the 126a sample (512 clips, 53 variants x 5 systems).',
    169: 'PAM on 126b/126c/134 (full stimuli).',
    170: 'PAM on the 126a sample, TTM systems real/ours_cfg3neg/ours_cfg0 (256 clips).',
    171: 'PAM on the 126a sample, TTM systems musicgen/musicldm (256 clips); strictly after 170.',
    172: 'SingMOS on 126b/126c/134 (full stimuli).',
    173: 'SingMOS on the 126a sample (512 clips, 53 variants x 5 systems).',
    174: 'MuQ-embed cosine on the 126a sample and 134 (reference-based).',
    175: 'SongEval and MERT-ref cosines on 134 (reference-based / aesthetics).',
    176: 'MIR four readings on the 126a sample and 134, plus the clean-S passes.',
    177: 'Judge every matrix cell from the job 163-176 outputs and emit the coverage summary, '
         'matrix and deck data.',
}

# 126a jobs read the 58 GB cache; 161 is the job that writes it (100 GB stop condition,
# §7.3).  Everything else follows §4.5: 60 GB for 126a, 10 GB otherwise.
JOBS_126A = {161, 162, 164, 166, 168, 170, 171, 173, 174, 176}
MIN_FREE = {161: 100 * 2 ** 30}
for _j in NAMES:
    MIN_FREE.setdefault(_j, 60 * 2 ** 30 if _j in JOBS_126A else 10 * 2 ** 30)

CODE_FILES = [
    'scripts/eval/metric_coverage_20261027/common.py',
    'scripts/eval/metric_coverage_20261027/adapters.py',
    'scripts/eval/metric_coverage_20261027/score_coverage.py',
    'scripts/eval/metric_coverage_20261027/prep161.py',
    'scripts/eval/metric_coverage_20261027/prep162.py',
    'scripts/eval/metric_coverage_20261027/mir_prep.py',
    'scripts/eval/metric_coverage_20261027/judge_coverage.py',
    'scripts/eval/metric_coverage_20261027/make_artifacts.py',
    'scripts/experiment_harness/metric_coverage_20261027_guest.py',
    'scripts/eval/metric_win_tie_126d.py',
    'scripts/eval/metric_win_tie_133.py',
    'scripts/eval/mir_metrics.py',
    'scripts/eval/newmetrics_common.py',
    'scripts/eval/newmetrics_muqeval.py',
    'scripts/eval/newmetrics_cmirm.py',
    'scripts/eval/newmetrics_pam.py',
    'docs/experiments/metric_coverage_completion_20261027.md',
    'docs/experiments/metric_coverage_sample_manifest_20261027.json',
    'docs/experiments/metric_coverage_model_pins_20261027.json',
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def launcher(job):
    name = f'{job}_coverage_{NAMES[job]}.sh'
    contract = C.DOCS / f'metric_coverage_{job}_contract.json'
    min_free = MIN_FREE[job]
    return name, f'''#!/bin/bash
# GPU_QUEUE_CONTRACT={contract}
set -eo pipefail
free=$(df -B1 --output=avail /home/kojiek | tail -1 | tr -d ' ')
if [ "${{free:-0}}" -lt {min_free} ]; then
  echo "[DISK] NVMe avail ${{free:-0}} B < {min_free} B, refusing to start" >&2
  exit 3
fi
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT={contract}
export PYTHONUNBUFFERED=1; export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export METRIC_COVERAGE_PIN_FULL_HASH={'1' if job == 161 else '0'}
cd /home/kojiek/MeanAudio || exit 1
exec {PY} -I {GUEST}
'''


def progress_spec(job):
    out = C.OUT_ROOT
    spec = C.JOBS[job]
    kind = spec['kind']
    roots = [str(out)]
    files = []
    if kind == 'prep':
        roots = [str(out / 'audio_cache'), str(out / 'backfill')]
        files = [str(out / 'cache_heartbeat.json')]
    elif kind == 'clean_s':
        roots = [str(out / 'clean_s')]
    elif kind == 'mir':
        roots = [str(out / 'mir')]
    elif kind == 'judge':
        files = [str(out / 'judge_heartbeat.json')]
    elif kind == 'multi':
        roots = [str(out / s['model']) for s in spec['specs']]
    elif spec.get('model'):
        roots = [str(out / spec['model'])]
    evidence = {
        'prep': [str(out / 'prep_161.json')],
        'clean_s': [str(out / 'jobs' / 'job_done_162.json')],
        'mir': [str(out / 'jobs' / 'job_done_176.json')],
        'judge': [str(out / 'jobs' / 'job_done_177.json')],
    }.get(kind, [str(out / 'jobs' / f'job_done_{job}.json')])
    return roots, files, evidence


def dependencies(job):
    out = C.OUT_ROOT
    spec = C.JOBS[job]
    kind = spec['kind']
    need = []
    if job == 161:
        return [f'5 TTM systems lvl30 audio under {C.R115}',
                '126b/126c/134 items.json', 'MusicCaps TSV', '126a per-unit scores.tsv']
    if kind == 'clean_s':
        return [str(out / 'prep_161.json'), str(C.R115 / 'selection.json')]
    if kind == 'mir':
        return [str(out / 'prep_161.json'), str(out / 'audio_cache_manifest.json')]
    if kind == 'judge':
        return [f'{out}/jobs/job_done_163..176.json', str(out / 'prep_161.json'),
                str(C.S_FILE / 's_values_mir.json')]
    if job == 171:
        need.append(str(out / 'jobs' / 'job_done_170.json'))
    for model, protocols, _n, _s in _job_specs(job):
        if model and model not in ('mert', 'muq', 'songeval'):
            need.append(str(C.S_FILE / f's_values_{model}.json'))
        if '126a' in protocols:
            need.append(str(C.sample_manifest_path()))
    need += [str(out / 'prep_161.json'), str(out / 'audio_cache_manifest.json')]
    return need


def _job_specs(job):
    spec = C.JOBS[job]
    if spec['kind'] == 'multi':
        return [(s['model'], s['protocols'], spec.get('sample_n'), spec.get('systems'))
                for s in spec['specs']]
    return [(spec.get('model'), spec.get('protocols', []), spec.get('sample_n'),
             spec.get('systems'))]


def contract(job, lname, lcontent):
    out = C.OUT_ROOT
    name = f'{job}_coverage_{NAMES[job]}.sh'
    lp = PENDING / name
    roots, files, evidence = progress_spec(job)
    spec = C.JOBS[job]
    protocols = [p for _m, ps, _n, _s in _job_specs(job) for p in ps]
    models = [m for m, _ps, _n, _s in _job_specs(job) if m]
    d = {
        'schema_version': 1,
        'document_kind': 'eval_only_metrics_contract',
        'experiment_id': f'metric-coverage-20261027-{job}',
        'job': str(job),
        'status': 'authorized_p2_pending',
        'priority': 'P2',
        'question': QUESTIONS[job],
        'design_document': str(PLAN),
        'operator_instruction': {
            'recorded_at': '2026-10-27',
            'text': 'Run the preregistered metric coverage job; judging rules '
                    '(m = 0.10*S, 80% tie/lose), sample sizes (512, PAM 256) and the '
                    'OOD exclusion are frozen by the plan.'},
        'approval_required': True,
        'launch_authorization': {
            'gpu_launch_allowed': True,
            'queue_position': 'p2',
            'scope': f'{"GPU inference" if models else "CPU-only"} job {job} of the metric '
                     f'coverage line; no training, no checkpoint writes. Output {out}.',
            'valid': True},
        'bindings': {
            'launcher': str(lp), 'launcher_sha256': hashlib.sha256(lcontent.encode()).hexdigest(),
            'harn': str(GUEST), 'harn_sha256': sha(GUEST),
            'action': str(DRIVER), 'action_sha256': sha(DRIVER),
            'prereg': str(PLAN), 'prereg_sha256': sha(PLAN),
            'sample_manifest': str(C.sample_manifest_path()),
            'sample_manifest_sha256': sha(C.sample_manifest_path()),
        },
        'code_bindings': {},
        'commands': {
            # The queue's accept_guest checks that commands.run[-1] is the bound action
            # file, so run must end with the driver path; the driver reads the job number
            # from GPU_QUEUE_CONTRACT (same convention as bench_r2 / autoresearch).
            'preflight': [PY, '-I', str(DRIVER), '--preflight'],
            'run': [PY, '-I', str(DRIVER)],
            'postflight': [PY, '-I', str(DRIVER), '--validate-only'],
        },
        'completion_evidence': {
            'eval_only': True,
            'method': 'guest writes terminal with sha256 of the first existing evidence file '
                      'after --validate-only passes; evidence is '
                      + ', '.join(evidence),
        },
        'fixed_protocol': {
            'mode': spec['kind'],
            'scratch': str(out),
            'models': models,
            'protocols': protocols,
            'sample_n': spec.get('sample_n'),
            'systems': spec.get('systems'),
            'progress_roots': roots,
            'progress_files': files,
            'evidence': evidence,
            'min_free_bytes': MIN_FREE[job],
        },
        'storage': {'path': str(out), 'min_free_bytes': MIN_FREE[job]},
        'resume': {
            'kind': 'from_scratch_with_autoresume',
            'note': 'every job skips keys that already carry a finite score row; 126a shards '
                    'are append-only and compacted only after a completed run; a missing '
                    'REPORT/run json forces a rerun',
            'pause_progress': str(out / f'pause_progress_{job}.json'),
            'autoresume': str(out / f'pause_progress_{job}.json'),
            'checkpoint': None,
            'iteration': 0,
        },
        'dependencies': dependencies(job),
        'reports': [{'path': evidence[0]}],
        'queue_name': name,
        'queue_role': 'p2',
        'watcher': {'stall_seconds': 1800},
        'launch_allowed': True,
        'revision_history': [{
            'rev': 'r1', 'date': '2026-10-27',
            'note': 'generated by make_contracts.py from '
                    'docs/experiments/metric_coverage_completion_20261027.md'}],
    }
    for rel in CODE_FILES:
        p = REPO / rel
        if not p.is_file():
            raise SystemExit(f'missing code binding: {rel}')
        d['code_bindings'][rel] = sha(p)
    return d


def main(dry_run=False):
    made = []
    for job in sorted(NAMES):
        if not (C.DOCS / f'metric_coverage_{job}_contract.json').parent.is_dir():
            raise SystemExit('docs/experiments missing')
        lname, lcontent = launcher(job)
        lpath = REPO / 'scripts/queue_candidates' / lname
        if not dry_run:
            lpath.write_text(lcontent, encoding='utf-8')
            lpath.chmod(0o755)
        c = contract(job, lname, lcontent)
        cpath = C.DOCS / f'metric_coverage_{job}_contract.json'
        if not dry_run:
            C.write_json_atomic(cpath, c)
        made.append((job, lpath, cpath))
        print(f'{job}: {lpath.name} {len(lcontent)} B, contract {cpath.name}')
    print(f'{len(made)} launchers + contracts')
    return 0


if __name__ == '__main__':
    sys.exit(main(dry_run='--dry-run' in sys.argv))
