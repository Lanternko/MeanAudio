#!/usr/bin/env python3
"""Contract-bound recovery for 054; no training or protocol changes."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/kojiek/MeanAudio')
CONTRACT = ROOT / 'docs/experiments/qwen_bucket_quarter_eval_backfill_recovery_contract.json'
PYTHON = '/home/kojiek/venvs/dac/bin/python'
STATE = Path('/home/kojiek/logs/qwen_bucket_backfill_recovery_harn')
NEG = 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
sys.path[:0] = [str(Path(__file__).parent), str(ROOT / 'scripts/experiment_harness')]
from validate_report import validate
from notification_receipts import atomic_secure_json, deliver_required, validate_delivered_receipt


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def config():
    return json.loads(CONTRACT.read_text())


def notify(c, event, summary):
    return deliver_required(contract_path=CONTRACT,
        launcher_path=Path(os.environ.get('GPU_QUEUE_JOB_SCRIPT', c['candidate_launcher'])),
        event=event, status='start', summary=summary,
        idempotency_key=f"{c['experiment_id']}:{c['run_id']}:{event}",
        notifier=Path(c['notification_receipts']['notifier']), python=Path(PYTHON),
        root=Path(c['notification_receipts']['root']))


def storage(c):
    measurements = []
    for raw in c['resource_budget']['writable_filesystems']:
        fs = os.statvfs(raw); free = fs.f_bavail * fs.f_frsize
        measurements.append({'path': raw, 'free_bytes': free})
        if free < c['storage']['hard_stop_free_bytes']:
            print(json.dumps({'status': 'resource_wait', 'storage': measurements}), flush=True)
            raise SystemExit(75)
    return measurements


def validate_protocol(c):
    p = c['fixed_protocol']
    if (p['rows'],p['steps'],p['seed'],p['solver'],p['mask'],p['precision']) != (5521,25,42,'MeanFlow','NoMask','full'):
        raise ValueError('fixed protocol drift')
    old = json.loads(Path(c['original_contract']).read_text())
    if c['cells'][:9] != old['cells']:
        raise ValueError('original CFG0 scientific cells changed')
    if c['decision_rule'] != old['decision_rule'] or c['ordering_dependencies'] != old['ordering_dependencies']:
        raise ValueError('scientific decision or queue order changed')
    if len(c['cells']) != 14: raise ValueError('14 cells required')
    for cell in c['cells'][9:]:
        if (cell['cfg_strength'],cell['negative_prompt'],cell['negative_prompt_id']) != (3,NEG,'fidelity8'):
            raise ValueError('canonical CFG3 fidelity8 drift')
    with Path(p['tsv']).open() as f: rows = list(csv.DictReader(f, delimiter='\t'))
    if len(rows) != 5521 or len({r['id'] for r in rows}) != 5521: raise ValueError('MusicCaps IDs invalid')
    if 'q_level' in rows[0]: raise ValueError('TSV q_level would override explicit conditioning')


def preflight(c, allow_unregistered=False):
    validate_protocol(c)
    if not c['launch_allowed']: raise ValueError('launch not authorized')
    for item in c['inputs']:
        if digest(item['path']) != item['sha256']: raise ValueError('input hash drift: '+item['path'])
    for item in c.get('ordering_evidence', []):
        if digest(item['path']) != item['sha256']: raise ValueError('ordering evidence changed')
        if json.loads(Path(item['path']).read_text()).get('status') not in ('completed','failed','interrupted'): raise ValueError('ordering predecessor not terminal')
    for raw in c['runtime_roots']:
        p = Path(raw)
        if p.is_symlink() or p.stat().st_uid != os.geteuid() or p.stat().st_mode & 0o777 != 0o700:
            raise ValueError('insecure runtime root: '+raw)
    measured = storage(c)
    sys.path.insert(0, str(ROOT/'scripts/eval'))
    from validate_cfg0_output_path import validate_root_target
    for cell in c['cells']:
        out, metrics = cell_paths(c, cell)
        for base, target in ((out.parent,out),(metrics.parent,metrics),(Path(cell['report']).parent,Path(cell['report']))):
            validate_root_target(base, target)
    b = Path(c['harn_bundle'])
    if (b/'contract.json').exists():
        argv = ['/usr/bin/python3', str(ROOT/'scripts/validate_experiment_harness_documents.py')]
        for kind in ('contract','preflight','ledger','queue'): argv += ['--'+kind, str(b/(kind+'.json'))]
        subprocess.run(argv, check=True)
        sources = {x['path']:x['sha256'] for x in json.loads((b/'contract.json').read_text())['corpus']['source_artifacts']}
        if sources.get(str(CONTRACT)) != digest(CONTRACT): raise ValueError('bundle contract hash mismatch')
    elif not allow_unregistered: raise ValueError('missing HARN bundle')
    if not allow_unregistered:
        rec = json.loads((b/'queue_registration.json').read_text())
        ok, reason = validate_delivered_receipt(Path(rec['receipt']), contract_path=CONTRACT,
            launcher_path=Path(c['candidate_launcher']), event='queue_registration', status='start')
        if not ok: raise ValueError('registration receipt: '+reason)
    print(json.dumps({'status':'passed','gpu_launched':False,'storage':measured}), flush=True)


def cell_paths(c, cell):
    root = Path(c['runtime_roots'][1 if cell.get('cfg_strength') == 3 else 0])
    return root/'output'/cell['label'], root/'metrics'/cell['label']


def prepare(c, cell):
    storage(c)
    if digest(cell['checkpoint']) != cell['checkpoint_sha256'] or digest(c['fixed_protocol']['tsv']) != c['fixed_protocol']['tsv_sha256']:
        raise ValueError('phase input hash drift')
    if Path(cell['report']).exists():
        validate(CONTRACT, cell['cell_id'], Path(cell['report']))
        return
    # eval.py skips existing clips without advancing its RNG. Preserve partial
    # files in a unique quarantine, then replay seed42 from row zero.
    for path in cell_paths(c, cell):
        if path.exists():
            if path.is_symlink() or path.stat().st_uid != os.geteuid(): raise ValueError('unsafe partial path')
            if any(path.iterdir()):
                target = path.with_name(path.name+'.partial-'+str(time.time_ns()))
                path.rename(target)
                atomic_secure_json(STATE/('quarantine-'+str(time.time_ns())+'.json'), {'source':str(path),'retained_at':str(target),'cell':cell['cell_id']})


def run(c):
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    for cell in c['cells']:
        prepare(c, cell)
        event = 'cell_'+cell['cell_id']
        notify(c, event+'_preflight', f"054 {cell['cell_id']}: checkpoint/TSV hashes and storage passed; run registered evaluation.")
        atomic_secure_json(STATE/'phase.json', {'cell':cell['cell_id'],'phase':'evaluation','written_at':time.time()})
        helper = ROOT/'scripts/eval/qwen_bucket_backfill'/('cfg3_fidelity8.sh' if cell.get('cfg_strength')==3 else 'cfg0.sh')
        cond = ['--no_q'] if cell['conditioning']=='no_q' else ['--quality_level',cell['conditioning'][1:]]
        env = dict(os.environ, CFG0_CONTRACT=str(CONTRACT), CFG0_ARM=cell['cell_id'])
        subprocess.run(['/bin/bash',str(helper),cell['label'],cell['checkpoint'],*cond], env=env, check=True)
        validate(CONTRACT, cell['cell_id'], Path(cell['report']))
        notify(c, event+'_passed', f"054 {cell['cell_id']}: exact MusicCaps 5521 report and metrics validated; next registered cell eligible.")
        if cell['cell_id'] == c['cells'][0]['cell_id']:
            notify(c, 'recovery_validated', '054 recovery verified: first formerly blocked NoQ cell now has a valid full MusicCaps 5521 report and five finite metrics. Continuing the preserved remaining cells.')
            atomic_secure_json(STATE/'recovery.json', {'status':'recovered','evidence':cell['report'],'sha256':digest(cell['report'])})
        atomic_secure_json(STATE/'phase.json', {'cell':cell['cell_id'],'phase':'passed','report_sha256':digest(cell['report']),'written_at':time.time()})
    postflight(c)


def postflight(c):
    evidence = []
    for cell in c['cells']:
        validate(CONTRACT, cell['cell_id'], Path(cell['report']))
        evidence.append({'cell':cell['cell_id'],'report':cell['report'],'sha256':digest(cell['report'])})
    atomic_secure_json(Path(c['summary']), {'status':'passed','experiment_id':c['experiment_id'],'reports':evidence})
    print('ALL_14_REPORTS_VALIDATED', flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    c = config()
    if '--preflight' in sys.argv: preflight(c, '--allow-unregistered' in sys.argv)
    elif '--postflight' in sys.argv: postflight(c)
    else: run(c)
