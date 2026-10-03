"""Durable private queue: verified recovery, then full population confirmation.
Local ledger only; no shared queue mutation or external notification sends.
"""
import os,sys,json,time,fcntl,subprocess,signal,shutil,hashlib
from pathlib import Path
from datetime import datetime,timedelta,timezone
from generate import ROOT,sha,dump
PY='/home/kojiek/venvs/dac/bin/python';STOP=False

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text())
def event(key,kind,**fields):
 p=ROOT/'events.jsonl';keys={json.loads(s)['key'] for s in p.read_text().splitlines()} if p.exists() else set()
 if key not in keys:
  with open(p,'a',buffering=1) as f:f.write(json.dumps(dict(key=key,kind=kind,time=now(),**fields))+'\n')
def prefix_sha(p,n):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while n:
   b=f.read(min(n,8<<20));assert b;n-=len(b);h.update(b)
 return h.hexdigest()
def pids():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 return set(map(int,s.split()))
def verify(entry,c):
 out=ROOT/'cells'/entry['cell'];p=out/(entry['stage']+'_complete.json')
 if not p.exists():return False
 r=read(p);n=512 if entry['stage']=='recovery' else 5521;cases=28 if entry['stage']=='recovery' else 32
 assert r['source_contract_sha256']==sha(ROOT/'contract.json') and r['n_prompts_requested']==n and r['n_cases']==cases
 assert r['gain_control_max_all_axes']<.01 and r['n_scores']==r['n_prompts_eligible']*cases*2
 assert prefix_sha(out/'probe.jsonl',r['cache_bytes'])==r['cache_sha256']
 assert r['exclusions_sha256']==sha(out/(entry['stage']+'_source_exclusions.json'))
 assert r['variant_exclusions_sha256']==sha(out/(entry['stage']+'_variant_exclusions.json'))
 return True

def terminate(child):
 if child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def main():
 global STOP
 signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True));signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('STOP',True))
 own=open(ROOT/'controller.lock','a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 c=read(ROOT/'contract.json');statefile=ROOT/'STATE.json';state=read(statefile) if statefile.exists() else dict(status='preflight',steps={},started_at=now(),horizon_end=(datetime.now(timezone(timedelta(hours=8)))+timedelta(hours=12)).isoformat(timespec='seconds'),contract_sha256=sha(ROOT/'contract.json'))
 assert state['contract_sha256']==sha(ROOT/'contract.json')
 state.update(controller_pid=os.getpid(),heartbeat=now());dump(statefile,state)
 assert sha(Path(c['parent_root'])/'contract.json')==c['parent_contract_sha256']
 old=read(Path(c['parent_root'])/'contract.json')
 for p,h in old['code_hashes'].items():assert sha(Path(c['parent_root'])/p)==h
 for p,h in c['code_hashes'].items():assert sha(ROOT/p)==h
 for group in ['source_hashes','weight_hashes']:
  for p,h in c[group].items():assert sha(p)==h
 assert sha(c['tsv'])==c['tsv_sha256'] and len(set(c['probe_ids']))==5521
 assert len(c['discovery_ids'])==512 and c['discovery_ids']==old['probe_ids']
 assert len(set(c['probe_ids'])-set(c['discovery_ids']))==5009
 assert read(ROOT/'smoke_pass.json')['returncode']==0
 while not (ROOT/'raw_audit.json').exists():
  assert not STOP;state.update(status='waiting_for_audio_hash_audit',heartbeat=now());dump(statefile,state);time.sleep(5)
 assert read(ROOT/'raw_audit.json')['audio_files_verified']==88336
 assert shutil.disk_usage(ROOT).free>c['hard_stop_free_bytes']+c['estimated_peak_additional_bytes']
 event('preflight_pass','gate_pass',contract_sha256=state['contract_sha256'])
 resource=open(c['shared_resource_lock'],'a')
 while not STOP:
  try:fcntl.flock(resource,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:state.update(status='waiting_for_resource',heartbeat=now());dump(statefile,state);time.sleep(10)
 if STOP:return 130
 env=os.environ.copy();env.update(PYTHONPATH=c['isolated_packages'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
 event('start','start',pid=os.getpid(),queue=c['queue_order'],horizon_end=state['horizon_end'])
 completed_times=[]
 for qi,entry in enumerate(c['queue_order']):
  key=entry['cell']+'__'+entry['stage'];out=ROOT/'cells'/entry['cell']
  if verify(entry,c):state['steps'][key]=dict(status='completed',verified_at=now(),recovered=True);dump(statefile,state);continue
  for attempt in range(1,4):
   while pids()-set(c['allowed_existing_gpu_pids']):
    state.update(status='waiting_for_unregistered_compute',heartbeat=now());dump(statefile,state)
    if STOP:return 130
    time.sleep(10)
   assert shutil.disk_usage(ROOT).free>c['hard_stop_free_bytes']
   argv=[PY,'-u',str(ROOT/'probe.py'),entry['cell'],'--stage',entry['stage']]
   state['steps'][key]=dict(status='running',attempt=attempt,started_at=now(),argv=argv)
   event(key+'__start_'+str(attempt),'stage_start',**entry,attempt=attempt)
   logfile=out/(entry['stage']+'.log')
   with open(logfile,'a',buffering=1) as log:
    child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
    started=time.monotonic();lastchange=started;fingerprint=None;idle_since=None
    while child.poll() is None:
     progress=out/(entry['stage']+'_progress.json');fp=(logfile.stat().st_size,progress.stat().st_mtime_ns if progress.exists() else None)
     if fp!=fingerprint:lastchange=time.monotonic();fingerprint=fp
     gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0])
     idle_since=(idle_since or time.monotonic()) if gpu==0 else None
     if idle_since and time.monotonic()-idle_since>180:event(key+'__idle_'+str(attempt),'unexpected_gpu_idle',child_alive=True,last_progress_age=time.monotonic()-lastchange)
     state.update(status='running',current_cell=entry['cell'],current_stage=entry['stage'],queue_index=qi,total_entries=len(c['queue_order']),child_pid=child.pid,heartbeat=now(),free_bytes=shutil.disk_usage(ROOT).free,gpu_utilization_percent=gpu,stage_elapsed_seconds=time.monotonic()-started)
     if progress.exists():state['progress']=read(progress)
     # Estimate all prepared successors using measured prompt rate; no artificial repetitions.
     rate=state.get('progress',{}).get('elapsed_seconds',0)/max(1,state.get('progress',{}).get('completed_prompts',0))
     if rate>0:
      remaining=state['progress']['total_prompts']-state['progress']['completed_prompts']+sum(512 if e['stage']=='recovery' else 5521 for e in c['queue_order'][qi+1:])
      state['estimated_queue_remaining_hours']=remaining*rate/3600
     dump(statefile,state)
     if STOP or state['free_bytes']<c['hard_stop_free_bytes']:
      terminate(child);state['steps'][key].update(status='interrupted',ended_at=now());state.update(status='interrupted' if STOP else 'disk_hold');dump(statefile,state);event(key+'__hold_'+str(attempt),state['status']);return 130
     if time.monotonic()-lastchange>1800:terminate(child);event(key+'__stall_'+str(attempt),'stall');break
     time.sleep(5)
   valid=child.returncode==0 and verify(entry,c)
   state['steps'][key].update(status='completed' if valid else 'failed',returncode=child.returncode,ended_at=now(),wall_seconds=time.monotonic()-started);dump(statefile,state)
   event(key+'__terminal_'+str(attempt),'stage_complete' if valid else 'stage_failure',returncode=child.returncode)
   if valid:break
   tail=logfile.read_text(errors='replace').splitlines()[-8:];state['last_failure']=tail;dump(statefile,state)
   if attempt<3:state.update(status='retrying_same_entry');dump(statefile,state);time.sleep(5)
   else:
    # A shared implementation failure blocks its successors; do not report completion.
    state.update(status='implementation_hold',heartbeat=now(),hold_reason='Same phase failed three attempts; see last_failure; successors preserved');dump(statefile,state);event(key+'__implementation_hold','hold',reason=state['hold_reason']);return 2
  event(key+'__handoff','handoff',next_entry=c['queue_order'][qi+1] if qi+1<len(c['queue_order']) else None)
  if qi==15:
   with open(ROOT/'recovery_analysis.log','a') as log:
    r=subprocess.run([PY,str(ROOT/'analyze.py'),'--stage','recovery'],env=env,stdout=log,stderr=subprocess.STDOUT)
   event('recovery_analysis','analysis_complete' if r.returncode==0 else 'analysis_failure',returncode=r.returncode)
 with open(ROOT/'confirmatory_analysis.log','a') as log:r=subprocess.run([PY,str(ROOT/'analyze.py'),'--stage','confirmatory'],env=env,stdout=log,stderr=subprocess.STDOUT)
 state.update(status='completed' if r.returncode==0 else 'analysis_failure',ended_at=now(),heartbeat=now());dump(statefile,state);event('terminal','terminal',status=state['status']);return r.returncode
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:
  dump(ROOT/'controller_failure.json',dict(error=repr(exc),time=now()))
  p=ROOT/'STATE.json'
  if p.exists():s=read(p);s.update(status='controller_failure',error=repr(exc),heartbeat=now());dump(p,s)
  raise
