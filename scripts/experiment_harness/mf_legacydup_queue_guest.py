#!/usr/bin/env python3
"""Registered fail-closed gate for the approved legacy-caption ablation.

The candidate is deliberately NOT prepared for GPU launch while its structural
gate fails. This entry records the exact hold and returns resource ownership to
the host. It never treats an old CFG0 artifact as canonical completion.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/preprocess'))
sys.path.insert(0, str(ROOT / 'scripts/experiment_harness'))
from build_mf_legacydup_inputs import sha, write_json
from notification_receipts import deliver_required

BUNDLE = ROOT / 'docs/experiments/harn/mf_legacydup_quarter_20260908'
RUNTIME = Path('/home/kojiek/mf_legacydup_quarter_runtime')


def check():
    science = json.loads((BUNDLE / 'science.json').read_text())
    gate = json.loads((RUNTIME / 'inputs/gate.json').read_text())
    for source in science['sources'].values():
        if sha(source['path']) != source['sha256']:
            return 'invalid', 'source bytes drifted: ' + source['path']
    if gate['science_sha256'] != sha(BUNDLE / 'science.json'):
        return 'invalid', 'science binding drift'
    if gate['corpus_sha256'] != sha(RUNTIME / 'inputs/mf_legacydup_train.tsv'):
        return 'invalid', 'corpus bytes drifted'
    if gate['defects_sha256'] != sha(RUNTIME / 'inputs/defects.json'):
        return 'invalid', 'defect evidence drift'
    if gate['status'] != 'passed':
        return 'fail', f"{gate['defect_rows']} structural-defect rows; exact text retained; operator disposition required"
    return 'invalid', 'GPU launcher unprepared: stop provenance, cache binding, canonical evaluation and runtime acceptance must pass before registration is promoted'


def main():
    try:
        verdict, reason = check()
    except (OSError, ValueError, KeyError) as exc:
        verdict, reason = 'invalid', str(exc)
    state = {'experiment_id': 'mf-legacydup-noq-quarter',
             'run_id': 'run-20260908-mf-legacydup-noq-quarter',
             'state': 'held_for_operator', 'prepared': False, 'gpu_launched': False,
             'gate_verdict': verdict, 'reason': reason,
             'next_action': 'operator resolves exact-text corpus exceptions; then prepare and validate GPU launcher before requeue',
             'gate_report': str(RUNTIME / 'inputs/gate.json'),
             'gate_sha256': sha(RUNTIME / 'inputs/gate.json')}
    write_json(RUNTIME / 'state.json', state)
    if '--check' in sys.argv:
        print(json.dumps(state, indent=2))
        return 2
    script = Path(os.environ['GPU_QUEUE_JOB_SCRIPT'])
    contract = Path(os.environ['GPU_QUEUE_CONTRACT'])
    receipt = deliver_required(contract_path=contract, launcher_path=script,
        event='corpus-gate-hold', status='held',
        summary='043 legacy-duplicate quarter withheld: ' + reason + '; current MF dedup full is unaffected.',
        idempotency_key='run-20260908-mf-legacydup-noq-quarter:corpus-gate-hold',
        notifier=ROOT / 'scripts/notify_experiment_webhook.py')
    write_json(script.with_name(script.stem + '.terminal.json'),
               {'status': 'held', 'reason': reason,
                'notification_receipt': {'path': str(receipt)}})
    return 2


if __name__ == '__main__':
    sys.exit(main())
