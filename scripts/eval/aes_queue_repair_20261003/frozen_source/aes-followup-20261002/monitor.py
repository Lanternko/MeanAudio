"""Deterministic local forecast and controller liveness monitor. No model calls."""
import os,json,time,fcntl
from pathlib import Path
from datetime import datetime,timezone,timedelta
from statistics import median
from generate import ROOT,dump

def main():
 lock=open(ROOT/'monitor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 c=json.loads((ROOT/'contract.json').read_text());initial={}
 for cell in c['cells']:
  p=ROOT/'cells'/cell['id']/'probe.jsonl';initial[cell['id']]=sum(1 for _ in open(p)) if p.exists() else 0
 while True:
  s=json.loads((ROOT/'STATE.json').read_text());rates=[];counts=dict(initial)
  for cell in c['cells']:
   for stage in ['recovery','confirmatory']:
    p=ROOT/'cells'/cell['id']/(stage+'_complete.json')
    if p.exists():
     r=json.loads(p.read_text());counts[cell['id']]=max(counts[cell['id']],r['n_cached_total']);rates.append(r['elapsed_seconds']/r['n_scores'])
  progress=s.get('progress',{})
  if progress.get('cell')==s.get('current_cell'):counts[s['current_cell']]=max(counts[s['current_cell']],progress.get('scored_now',0)+progress.get('reused_scores',0))
  remaining=max(0,5521*16*32*2-sum(counts.values()));horizon=datetime.fromisoformat(s['horizon_end']);now=datetime.now(timezone(timedelta(hours=8)));rate=median(rates) if rates else None
  pid=s['controller_pid'];cmd=Path(f'/proc/{pid}/cmdline');alive=False
  if cmd.exists():
   argv=cmd.read_bytes().decode(errors='replace').split('\0');cwd=Path(f'/proc/{pid}/cwd').resolve()
   alive=any((Path(a) if a.startswith('/') else cwd/a).resolve()==ROOT/'controller.py' for a in argv if a)
  forecast=dict(time=now.isoformat(timespec='seconds'),controller_alive=alive,controller_pid=pid,queue_status=s['status'],observed_complete_phase_rates_seconds_per_score=rates,remaining_unique_AES_score_upper_count=remaining,nominal_remaining_hours=remaining*rate/3600 if rate else None,lower_fast_remaining_hours=remaining*rate*.75/3600 if rate else None,upper_slow_remaining_hours=remaining*rate*1.6/3600 if rate else None,horizon_hours_remaining=max(0,(horizon-now).total_seconds()/3600),method='Median fully completed phase wall seconds per score; target32cases incl raw+normalized. Count existing valid cache conservatively. Includes CLAP/Demucs/preprocessing wall time, not CUDA active time. Warm cache prefixes excluded from rate update until phase complete. Future unseen stems/extra conditions may alter throughput.')
  forecast['needs_more_design_before_horizon']=rate is not None and forecast['lower_fast_remaining_hours']<forecast['horizon_hours_remaining'];dump(ROOT/'FORECAST.json',forecast)
  terminal=s['status'] in ['completed','analysis_failure','implementation_hold','interrupted','disk_hold','controller_failure']
  if not alive and not terminal:
   dump(ROOT/'MONITOR_INCIDENT.json',dict(reason='controller absent while state expects live execution',forecast=forecast));return 2
  if terminal:return 0
  time.sleep(60)
if __name__=='__main__':raise SystemExit(main())
