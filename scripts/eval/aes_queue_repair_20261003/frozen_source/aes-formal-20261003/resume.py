"""Missing-only replay of the unchanged scientific contract under registered P2 HARN."""
import os,sys,json,time,subprocess,fcntl
from pathlib import Path
from common import ROOT,SRC,PY,sha,dump,spec,notify,load_records
sys.path.insert(0,str(SRC))
import controller as original

def main():
 c=spec();science=json.loads((SRC/'contract.json').read_text());state=Path(c['runtime_root']);state.mkdir(exist_ok=True)
 lock=open(SRC/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(PYTHONPATH=science['isolated_packages'],CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 verified=[]
 for index,entry in enumerate(science['queue_order']):
  key=entry['cell']+'_'+entry['stage'];out=SRC/'cells'/entry['cell'];receipt=out/(entry['stage']+'_complete.json');adopted=original.verify(entry,science)
  if not adopted:
   load_records(out/'probe.jsonl') # registered last-line-only recovery with forensic copy
   notify(key+'_start','start',f"Starting missing-only {entry['cell']} / {entry['stage']}; original source/recipe hashes unchanged. GPU0 owned by formal P2 seat.")
   argv=[PY,'-u',str(SRC/'probe.py'),entry['cell'],'--stage',entry['stage']];logfile=state/(key+'.log')
   with open(logfile,'a') as log:
    child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=SRC) # same owned process group
    last=None
    try:
     while child.poll() is None:
      progress=out/(entry['stage']+'_progress.json');fp=(logfile.stat().st_size,progress.stat().st_mtime_ns if progress.exists() else None)
      if fp!=last:
       last=fp;dump(state/'progress.json',dict(phase=key,index=index,total_phases=32,child_pid=child.pid,progress=json.loads(progress.read_text()) if progress.exists() else None))
      time.sleep(2)
    finally:
     if child.poll() is None:child.terminate();child.wait(timeout=25)
   if child.returncode!=0:
    notify(key+'_fail','failure',f'{key}: child rc={child.returncode}; phase failed; artifacts retained; no later phase promoted in this recovery entry.');raise RuntimeError(f'{key} child exit {child.returncode}')
   if not original.verify(entry,science):
    notify(key+'_invalid','held',f'{key}: coverage/hash/gain-control gate invalid; no promotion.');raise RuntimeError(key+' invalid artifacts')
  r=json.loads(receipt.read_text());notify(key+'_pass','start',f"Gate PASS {key}: {r['n_prompts_eligible']}/{r['n_prompts_requested']} eligible prompts, {r['n_scores']} records, gain-control max={r['gain_control_max_all_axes']:.6f} <0.01; receipt SHA={sha(receipt)}. {'Adopted pre-existing verified completion; no repeat compute.' if adopted else 'New completion verified.'} Next registered phase eligible.")
  verified.append(dict(**entry,receipt=str(receipt),sha256=sha(receipt),adopted=adopted));dump(state/'resume.json',dict(verified=verified,index=index));dump(state/'progress.json',dict(phase='gate_pass',index=index,total_phases=32))
  if index==15:
   subprocess.run([PY,str(SRC/'analyze.py'),'--stage','recovery'],env=env,check=True)
   notify('discovery_analysis_pass','start',f"Discovery analysis passed: {SRC/'recovery_summary.csv'} SHA={sha(SRC/'recovery_summary.csv')}; exploratory512, not holdout.")
 subprocess.run([PY,str(SRC/'analyze.py'),'--stage','confirmatory'],env=env,check=True)
 analysis=SRC/'confirmatory_analysis_complete.json';r=json.loads(analysis.read_text());assert r['n_cells']==16 and r['n_prompts']==5009
 dump(state/'summary.json',dict(status='completed',source_contract_sha256=sha(SRC/'contract.json'),verified_phases=verified,analysis=str(analysis),analysis_sha256=sha(analysis),report=str(SRC/'confirmatory_REPORT.md'),report_sha256=sha(SRC/'confirmatory_REPORT.md')))
 notify('confirmatory_analysis_pass','start',f"Confirmatory analysis PASS:16 cells,5009 held-out prompts, {analysis} SHA={sha(analysis)}; next queued independent source-study may start after terminal notification.")
if __name__=='__main__':main()
