#!/usr/bin/env python3
import hashlib, json, os, sys
from pathlib import Path
CONTRACT=Path('/home/kojiek/MeanAudio/docs/experiments/vocal_negative_cfg3_20260908_contract.json')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def main():
    c=json.loads(CONTRACT.read_text())
    assert c['launch_allowed'] and c['launch_authorization']['gpu_launch_allowed']
    p=c['protocol']
    assert (p['rows'],p['steps'],p['cfg_strength'],p['generation_seed'],p['mask'],p['precision']) == (5521,25,3,42,'NoMask','full')
    for i in c['inputs']:
        f=Path(i['path'])
        assert f.is_file() and f.stat().st_size==i['bytes'] and sha(f)==i['sha256'], f'input drift: {f}'
    root=Path(c['storage']['path'])
    assert root.stat().st_uid==os.geteuid() and root.stat().st_mode & 0o777 == 0o700
    fs=os.statvfs(root); free=fs.f_bavail*fs.f_frsize
    assert free>=c['storage']['hard_stop_free_bytes'], f'storage: {free}'
    import subprocess
    bundle=CONTRACT.parent/'harn'/'vocal_negative_cfg3_20260908'
    argv=['/usr/bin/python3',str(CONTRACT.parents[2]/'scripts/validate_experiment_harness_documents.py')]
    for kind in ('contract','preflight','ledger','queue'): argv += ['--'+kind,str(bundle/(kind+'.json'))]
    subprocess.run(argv,check=True)
    print(json.dumps({'status':'passed','free_bytes':free}))
if __name__=='__main__': main()
