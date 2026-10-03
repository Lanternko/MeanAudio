"""No-GPU, no-network production guest and signal fixtures + shared harness tests."""
import sys,json,os,tempfile,subprocess,importlib.util,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=Path('/home/kojiek/MeanAudio')
sys.path.insert(0,str(ROOT));import guest as h
from common import dump,sha,load_records

def main():
 checks={}
 with tempfile.TemporaryDirectory() as td:
  td=Path(td);bad=td/'scores.jsonl';bad.write_bytes(b'{"key":"one"}\n{"bad":');assert load_records(bad)=={'one':{'key':'one'}};assert load_records(bad)=={'one':{'key':'one'}};assert len(list(td.glob('*.partial-tail-*')))==1;checks['truncated_tail_resume']='pass: only own partial last record quarantined, valid records retained'
 import source_study as study
 rng=np.random.default_rng(43);other=rng.standard_normal(159744).astype(np.float32)*.02;residual=rng.standard_normal(159744).astype(np.float32)*.01;y=other+residual
 items=[('stem_recompose',y,'clean',None),('stem_other_off',residual,'stem_recompose',None),('stem_other_m6',(residual+other*10**(-6/20)).astype(np.float32),'stem_recompose',None)]
 cases=list(study.variants(y,items,123));assert [r[0] for r in cases]==study.CASES and len(cases)==20;assert all(z.dtype==np.float32 and np.isfinite(z).all() and len(z)==159744 for _,z,_ in cases)
 randomized=dict((case,z) for case,z,_ in cases)['other_phase_random']-residual;assert np.isclose(np.sum(randomized.astype(float)**2),np.sum(other.astype(float)**2),rtol=2e-6);checks['source_interventions']='pass:20 cases, identity/control, energy/spectrum phase fixture, float32/coverage'
 class Ctl:
  instances=[]
  def __init__(self,script,contract,state):self.contract=json.loads(contract.read_text());self.events=[];self.terminal_args=None;Ctl.instances.append(self)
  def _load(self):return {'events':[]}
  def notify(self,*a,**k):self.events.append(a)
  def pre_child_notifications(self,*a):self.events.append(('start',))
  def record(self,*a):pass
  def terminal(self,*a,**k):self.terminal_args=a
 class Child:
  pid=987654
  def __init__(self,rc):self.rc=rc
  def poll(self):return self.rc
  def wait(self,timeout=None):return self.rc
 for case in ['success','failure','exit137','interrupt','pause','invalid_preflight','invalid_postflight','storage_recover','notification_failure']:
  with tempfile.TemporaryDirectory() as td:
   td=Path(td);launcher=td/'fixture.sh';launcher.write_text('fixture');report=td/'report.json';report.write_text('{}');contract=td/'contract.json'
   c=dict(runtime_root=str(td),state_root=str(td),reports=[dict(path=str(report))],summary=str(report),storage=dict(path=str(td),hard_stop_free_bytes=80000000000,warning_free_bytes=120000000000),resume=dict(autoresume=str(td/'resume.json')),commands=dict(preflight=['preflight'],run=['run'],postflight=['postflight']),watcher=dict(stall_seconds=1800));contract.write_text(json.dumps(c));calls=[];child=Child(2 if case=='failure' else 137 if case=='exit137' else 0)
   def read(path):
    if path.name=='p2.running.json':return dict(pid=os.getpid(),start_time=h.pid_start_time(os.getpid()),job_id=launcher.stem,run_id='fixture')
    if case=='pause':return dict(run_id='fixture',job_id=launcher.stem,request_id='request1')
    return None
   def spawn(*a,**k):calls.append('spawn');return child
   frees=iter([0,10**15] if case=='storage_recover' else [10**15]);fs=lambda _:SimpleNamespace(f_bavail=next(frees,10**15),f_frsize=1)
   def run(argv,*a,**k):return SimpleNamespace(returncode=1 if case=='invalid_preflight' or (case=='invalid_postflight' and argv==['postflight']) else 0)
   with patch.dict(os.environ,dict(GPU_QUEUE_JOB_SCRIPT=str(launcher),GPU_QUEUE_CONTRACT=str(contract),P2_RUN_ID='fixture',P2_CONTROL_DIR=str(td))),patch.object(h,'rearm_queue_idle'),patch.object(h,'Controller',Ctl),patch.object(h,'accept_guest',return_value=(True,'ok')),patch.object(h,'read_json',side_effect=read),patch.object(h,'progress',return_value=(0,0)),patch.object(h.os,'statvfs',side_effect=fs),patch.object(h.subprocess,'run',side_effect=run),patch.object(h.subprocess,'Popen',side_effect=spawn),patch.object(h.time,'sleep'),patch.object(h,'INTERRUPTED',case=='interrupt'):
    if case=='notification_failure':
     with patch.object(Ctl,'pre_child_notifications',side_effect=RuntimeError('notification_pending')):assert h.main()==2
     assert not calls
    else:
     rc=h.main();terminal=Ctl.instances[-1].terminal_args[0];expect={'success':(0,'completed'),'failure':(2,'failed'),'exit137':(137,'interrupted'),'interrupt':(130,'interrupted'),'pause':(h.PAUSE_EXIT,'paused'),'invalid_preflight':(2,'held'),'invalid_postflight':(2,'held'),'storage_recover':(0,'completed')}[case];assert (rc,terminal)==expect,(case,rc,terminal)
     if case in ['interrupt','pause','invalid_preflight']:assert not calls
   checks['production_guest_'+case]='pass'
 paths=[REPO/'scripts/tests/selftest_secondary_queue_notifications.py',REPO/'scripts/tests/selftest_experiment_harness_schemas.py',Path('/home/kojiek/gpu_queue/tests/test_notification_receipts.py'),Path('/home/kojiek/gpu_queue/tests/test_lib_scheduler.py')]
 for path in paths:
  log=ROOT/(path.stem+'.log');r=subprocess.run(['/usr/bin/python3' if path.stem=='selftest_experiment_harness_schemas' else sys.executable,str(path)],stdout=open(log,'w'),stderr=subprocess.STDOUT);assert r.returncode==0,str(path);checks[path.stem]='pass'
 dump(ROOT/'acceptance_partial.json',dict(status='passed',gpu_used=False,network_used=False,checks=checks,code_hashes={str(p):sha(p) for p in [ROOT/'guest.py',ROOT/'resume.py',ROOT/'source_study.py',ROOT/'common.py',Path(__file__)]+paths}));print('PASS production guest, intervention and shared schema/notification fixtures',flush=True)
if __name__=='__main__':main()
