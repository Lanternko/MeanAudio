#!/usr/bin/env python3
"""One-time operator-authorized publication of reviewed 054 recovery."""
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
R=Path('/home/kojiek/MeanAudio')
sys.path.insert(0,str(R/'scripts/eval/qwen_bucket_backfill'))
import run as m
from notification_receipts import atomic_secure_json
B=Path(__file__).resolve().parent
c=m.config();q=Path('/home/kojiek/gpu_queue/p2');name=c['queue_name']
review=json.loads((B/'review.json').read_text())
assert review['verdict']=='CONFIRMED'
assert review['contract_sha256']==m.digest(m.CONTRACT)
for item in c['inputs']: assert m.digest(item['path'])==item['sha256'], item['path']
subprocess.run([m.PYTHON,'/home/kojiek/gpu_queue/lib_scheduler.py','accept',c['candidate_launcher']],check=True)
m.preflight(c,allow_unregistered=True)
receipt=m.notify(c,'queue_registration','054 recovery registered: checkpoint hashes verified; private path/NoQ arm fix; same 9 CFG0 and 5 CFG3 fidelity8 cells and order. Returning failed 054 to empty P2 tail; host owns launch.')
atomic_secure_json(B/'queue_registration.json',{'event':'queue_registration','status':'delivered','receipt':str(receipt)})
m.preflight(c)
with (q/'maintenance'/'054_recovery.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    assert not list((q/'pending').glob('*.sh')), 'pending queue changed; preserve its order'
    assert not list((q/'running').glob('*.sh')), 'P2 is already active'
    assert not (q/'done'/name).exists(), '054 already completed'
    source=q/'failed'/name;assert source.is_file(), 'original failed launcher missing'
    terminal=source.with_suffix('.terminal.json');assert json.loads(terminal.read_text())['status']=='failed'
    archive=q/'maintenance'/'054_recovery_20260914';archive.mkdir(mode=0o700,exist_ok=False)
    transaction={'state':'applying','operator_instruction':'fix it then continue the experiment','contract':str(m.CONTRACT),'contract_sha256':m.digest(m.CONTRACT),'candidate_sha256':m.digest(c['candidate_launcher']),'before_pending':[],'after_pending':[name],'archive':str(archive),'registration_receipt':str(receipt)}
    atomic_secure_json(B/'queue_mutation.json',transaction)
    source.rename(archive/name);terminal.rename(archive/terminal.name)
    target=q/'pending'/name;tmp=target.with_suffix('.recovery-tmp')
    with tmp.open('xb') as f:
        f.write(Path(c['candidate_launcher']).read_bytes());f.flush();os.fsync(f.fileno())
    tmp.chmod(0o755);os.replace(tmp,target)
    transaction['state']='queued';atomic_secure_json(B/'queue_mutation.json',transaction)
    atomic_secure_json(m.STATE/'recovery.json',{'status':'queued_for_validation','incident':'fail-054_qwen_bucket_quarter_eval_backfill','contract_sha256':m.digest(m.CONTRACT),'next_evidence':c['cells'][0]['report']})
print('054_RECOVERY_QUEUED')
