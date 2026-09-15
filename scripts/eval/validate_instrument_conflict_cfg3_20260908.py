#!/usr/bin/env python3
"""No-GPU exact protocol, row-negative pairing, provenance and capacity gate."""
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
ROOT=Path('/home/kojiek/MeanAudio')
CONTRACT=ROOT/'docs/experiments/instrument_conflict_cfg3_20260908_contract.json'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def validate_rows(c):
    root=Path(c['storage']['path']);assignment=json.loads((root/'assignments.json').read_text())
    data=assignment['assignments'];ids=[a['id'] for a in data]
    assert len(ids)==len(set(ids))==3207
    fidelity='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
    for key,argv in c['commands_generation'].items():
        rows=list(csv.DictReader(Path(argv[argv.index('--tsv')+1]).open(),delimiter='\t'))
        assert [r['id'] for r in rows]==ids
        assert argv[argv.index('--negative_prompt_column')+1]=='negative_prompt'
        assert argv[argv.index('--cfg_strength')+1]=='3' and argv[argv.index('--seed')+1]=='42'
        for a,r in zip(data,rows):
            assert r['caption']==a['caption'] and a['target'] in a['mentioned'] and a['unmentioned'] not in a['mentioned']
            term={'fidelity8':'','fidelity8_conflict':a['target'],'fidelity8_unmentioned':a['unmentioned']}[key]
            assert r['negative_prompt']==fidelity+(', '+term if term else '')
def main():
    c=json.loads(CONTRACT.read_text());p=c['protocol']
    assert c['launch_allowed'] and c['launch_authorization']['gpu_launch_allowed']
    assert (p['rows'],p['steps'],p['cfg_strength'],p['generation_seed'],p['mask'],p['precision'])==(3207,25,3,42,'NoMask','full')
    for i in c['inputs']:
        f=Path(i['path']);assert f.is_file() and f.stat().st_size==i['bytes'] and sha(f)==i['sha256'],f'input drift: {f}'
    validate_rows(c)
    root=Path(c['storage']['path']);assert root.stat().st_uid==os.geteuid() and root.stat().st_mode&0o777==0o700
    fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize
    if free<c['storage']['hard_stop_free_bytes']:
        print(json.dumps({'status':'resource_wait','free_bytes':free}));raise SystemExit(75)
    bundle=CONTRACT.parent/'harn'/'instrument_conflict_cfg3_20260908'
    argv=['/usr/bin/python3',str(ROOT/'scripts/validate_experiment_harness_documents.py')]
    for kind in ('contract','preflight','ledger','queue'):argv+=['--'+kind,str(bundle/(kind+'.json'))]
    subprocess.run(argv,check=True)
    print(json.dumps({'status':'passed','free_bytes':free,'rows':3207}))
if __name__=='__main__':main()
