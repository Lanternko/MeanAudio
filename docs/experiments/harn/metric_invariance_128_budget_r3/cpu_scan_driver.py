import os,sys,json,subprocess,time,signal,fcntl,hashlib
from pathlib import Path
R=Path('/home/kojiek/MeanAudio');B=R/'docs/experiments/harn/metric_invariance_128_budget_r3';S=B/'lufs_scans';PY='/home/kojiek/venvs/dac/bin/python'
sys.path.insert(0,str(R/'scripts/eval'));import metric_invariance_128 as M
lock=(B/'scans.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
child=None

def stop(sig,frame):
 if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
 raise SystemExit(128+sig)
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)

def notice(status,text):
 p=subprocess.run([PY,str(R/'scripts/notify_experiment_webhook.py'),'--status',status,'--experiment','128-budget-r3-cpu-lufs-scans','--summary',text],capture_output=True,text=True)
 M.write(B/f'scan_notification_{status}.json',{'status':status,'returncode':p.returncode,'at':time.time()})
 if p.returncode:raise RuntimeError('required scan notification failed')

commands=[]
for part in 'abc':
 cp=R/f'docs/experiments/metric_invariance_128{part}_20261017_quiet_r2_contract.json'
 assert not (S/f'{part}.json').exists(),'refuse replacing scan report'
 commands.append({'part':part,'contract':str(cp),'contract_sha256':M.sha(cp),'argv':[PY,str(R/'scripts/eval/metric_invariance_128.py'),'--scan-lufs',str(S/f'{part}.json'),'--scan-workers','8']})
M.write(B/'cpu_scan_contract.json',{'run_id':'128-budget-r3-cpu-lufs-scans','authorization':'operator_console: user approved 7h to 9h feasibility amendment and a fresh run; regenerate all three source-bound CPU scans','action_sha256':M.sha(R/'scripts/eval/metric_invariance_128.py'),'driver_sha256':M.sha(__file__),'manifest_sha256':M.sha('/mnt/seagate/kojiek_data/metric_invariance_128_20261017/inputs.json'),'cpu_workers':8,'gpu':False,'hard_floor_bytes':50*2**30,'commands':commands,'repair':False,'resume':'No GPU launch; retain all partial/failed evidence. New output directory required for rerun.'})
notice('start','128 budget_r3 CPU-only full LUFS scans starting (8 workers); GPU launch blocked until a/b/c all pass.')
results={}
for entry in commands:
 part=entry['part'];env=dict(os.environ,GPU_QUEUE_CONTRACT=entry['contract'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 with (S/f'{part}.log').open('w') as log:
  child=subprocess.Popen(entry['argv'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while child.poll() is None:
   free={str(p):os.statvfs(p).f_bavail*os.statvfs(p).f_frsize for p in (R,Path('/mnt/seagate'))}
   M.write(B/'scan_monitor.json',{'at':time.time(),'part':part,'child_pid':child.pid,'free_bytes':free})
   if min(free.values())<50*2**30:
    os.killpg(child.pid,signal.SIGTERM);child.wait();notice('held','128 CPU LUFS scan storage hard stop; no GPU work authorized.');raise SystemExit(75)
   time.sleep(5)
  results[part]=child.returncode
  M.write(B/'scan_results.json',{'parts':results,'status':'scanning' if part!='c' else ('passed' if not any(results.values()) else 'failed')})
notice('success' if not any(results.values()) else 'failure',f'128 CPU LUFS scans complete: exit codes {results}. GPU remains blocked unless all pass and new contracts are registered.')
raise SystemExit(1 if any(results.values()) else 0)
