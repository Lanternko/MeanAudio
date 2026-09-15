#!/usr/bin/env python3
"""No-GPU immutable, scientific, storage, dependency and harness preflight."""
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from loudness_aes_cfg3_20260911 import CONTRACT, ROOT, NEGATIVE, digest

def validate_protocol(c):
    p=c['protocol']
    if not c['launch_allowed'] or not c['launch_authorization']['gpu_launch_allowed']: raise ValueError('not approved')
    if (p['rows'],p['steps'],p['cfg_strength'],p['generation_seed'],p['mask'],p['precision'],p['solver'],p['negative_key'],p['negative_prompt']) != (5521,25,3,42,'NoMask','full','MeanFlow','fidelity8',NEGATIVE): raise ValueError('canonical protocol drift')
    if c['analysis']['gains_db']!=[0,-3,-6,-9]: raise ValueError('gain protocol drift')
    rows=list(csv.DictReader(Path(c['tsv']).open(),delimiter='\t'))
    source=list(csv.DictReader(Path(c['source_tsv']).open(),delimiter='\t'))
    if len(rows)!=5521 or len({r['id'] for r in rows})!=5521: raise ValueError('not 5521 unique IDs')
    if [(r['id'],r['caption']) for r in rows]!=[(r['id'],r['caption']) for r in source]: raise ValueError('TSV order/caption drift')
    if any(r['negative_prompt']!=NEGATIVE for r in rows): raise ValueError('negative mismatch')
    argv=c['commands_generation']['fidelity8']
    for flag,val in {'--cfg_strength':'3','--num_steps':'25','--seed':'42','--tsv':c['tsv'],'--model_path':c['checkpoint'],'--negative_prompt_column':'negative_prompt'}.items():
        if argv.count(flag)!=1 or argv[argv.index(flag)+1]!=val: raise ValueError('argv drift '+flag)
    for flag in ('--use_meanflow','--full_precision','--no_text_attention_mask','--no_q'):
        if argv.count(flag)!=1: raise ValueError('missing flag '+flag)
    if argv[argv.index('--output')+1]!=str(Path(c['storage']['path'])/'_audio'/'baseline'): raise ValueError('output drift')
    if Path(c['approval_record']).read_text().strip()!='設計實驗並加入 queue': raise ValueError('operator record drift')

def main():
    c=json.loads(CONTRACT.read_text());validate_protocol(c)
    for i in c['inputs']:
        f=Path(i['path'])
        if not f.is_file() or f.stat().st_size!=i['bytes'] or digest(f)!=i['sha256']: raise ValueError('input drift: '+str(f))
    for package,version in c['package_versions'].items():
        if importlib.metadata.version(package)!=version: raise ValueError('package version drift: '+package)
    root=Path(c['storage']['path'])
    if root.is_symlink() or root.stat().st_uid!=os.geteuid() or root.stat().st_mode&0o777!=0o700: raise ValueError('insecure runtime root')
    measured=[]
    for path in c['resource_budget']['writable_filesystems']:
        fs=os.statvfs(path);free=fs.f_bavail*fs.f_frsize
        measured.append({'path':path,'free_bytes':free})
        if free<c['storage']['hard_stop_free_bytes']:
            print(json.dumps({'status':'resource_wait','storage':measured}));raise SystemExit(75)
    # Package imports are CPU-only. Pin source files and weights separately in inputs.
    import numpy, scipy, soundfile, pyloudnorm, torch, audiobox_aesthetics
    b=Path(c['harn_bundle']); argv=['/usr/bin/python3',str(ROOT/'scripts/validate_experiment_harness_documents.py')]
    for kind in ('contract','preflight','ledger','queue'): argv+=['--'+kind,str(b/(kind+'.json'))]
    subprocess.run(argv,check=True)
    bundle=json.loads((b/'contract.json').read_text())
    sources={x['path']:x['sha256'] for x in bundle['corpus']['source_artifacts']}
    if sources.get(str(CONTRACT))!=digest(CONTRACT): raise ValueError('schema bundle does not bind current science contract')
    if '--allow-unregistered' not in sys.argv:
        sys.path.insert(0,str(ROOT/'scripts/experiment_harness'))
        from notification_receipts import validate_delivered_receipt
        registration=json.loads((b/'queue_registration.json').read_text())
        receipt=Path(registration['receipt'])
        if receipt.parent.parent.resolve()!=Path(c['notification_receipts']['root']).resolve(): raise ValueError('wrong receipt root')
        ok,reason=validate_delivered_receipt(receipt,contract_path=CONTRACT,launcher_path=ROOT/'scripts/queue_candidates'/c['queue_name'],event='queue_registration',status='start')
        if not ok: raise ValueError('registration notification: '+reason)
    print(json.dumps({'status':'passed','storage':measured,'rows':5521,'gpu_launched':False}))
if __name__=='__main__': main()
