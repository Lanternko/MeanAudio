#!/usr/bin/env python3
"""Read-only monitor for the queued Phase-8 Q-safe fine-tuning sequence."""

from __future__ import annotations

import argparse, json, math, os, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/home/kojiek/MeanAudio'); LOG=Path('/home/kojiek/logs')
STATE=LOG/'phase8_qsafe_ft_monitor'; STATUS=STATE/'status.json'; ALERT=STATE/'ALERT.json'
TMUX='p8_qsafe_ft'
ARMS=[('real','phase8_qsafe_realq_ft100k'),('shuffled','phase8_qsafe_shuffledq_ft100k')]
ITER=re.compile(r'\bit\s+(\d+):'); LOSS=re.compile(r'loss:([+\-]?(?:nan|inf|\d+(?:\.\d+)?(?:e[+\-]?\d+)?))',re.I); GRAD=re.compile(r'grad_norm:([+\-]?(?:nan|inf|\d+(?:\.\d+)?(?:e[+\-]?\d+)?))',re.I)

def run(cmd):
    try:return subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
    except Exception:return ''
def tail(path,limit=4*1024*1024):
    if not path.is_file():return ''
    with path.open('rb') as f:f.seek(0,2);n=f.tell();f.seek(max(0,n-limit));return f.read().decode(errors='replace')
def number(match):
    try:return float(match.group(1)) if match else None
    except ValueError:return None
def read_metrics(path):
    out={}
    if path.is_file():
        for line in path.read_text().splitlines():
            if ':' in line:
                k,v=line.split(':',1)
                if k.strip() in {'clap_score','aes_CE','aes_CU','aes_PC','aes_PQ'}:out[k.strip()]=float(v)
    return out

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--once',action='store_true'); parser.parse_args()
    STATE.mkdir(parents=True,exist_ok=True); now=datetime.now(timezone.utc)
    processes=run(['pgrep','-af','phase8_qsafe_(realq|shuffledq)_ft100k'])
    tmux=run(['tmux','list-sessions','-F','#{session_name}']).splitlines()
    arms={};active_mode=None;active_prefix=None;active_log=None
    for mode,prefix in ARMS:
        exp=f'{prefix}_stage2_ft100000'; audit=STATE/f'{prefix}_FINAL_AUDIT.json'
        audited=json.loads(audit.read_text()) if audit.is_file() else None
        arms[mode]={'prefix':prefix,'complete':bool(audited and audited.get('status')=='passed'),'final_audit':audited,
                    'metrics':{f'q{q}':read_metrics(ROOT/'eval_output/metrics'/f'{exp}_musiccaps_q{q}'/'metrics.txt') for q in (9,6)}}
        if prefix in processes:active_mode,active_prefix,active_log=mode,prefix,LOG/f'{exp}.log'
    final=STATE/'FINAL_COMPARISON.json'; old_final=LOG/'phase8_s2_q_ablation_monitor/FINAL_COMPARISON.json'
    if final.is_file():phase='complete'
    elif active_mode:
        exp=f'{active_prefix}_stage2_ft100000';q9=LOG/f'{exp}_musiccaps_q9_eval.log';q6=LOG/f'{exp}_musiccaps_q6_eval.log'
        if q6.is_file() and not arms[active_mode]['metrics']['q6']:phase=f'{active_mode}_eval_q6';active_log=q6
        elif q9.is_file() and not arms[active_mode]['metrics']['q9']:phase=f'{active_mode}_eval_q9';active_log=q9
        else:phase=f'{active_mode}_training'
    elif all(arms[m]['complete'] for m,_ in ARMS):phase='paired_bootstrap_or_finalizing'
    elif arms['real']['complete']:phase='between_arms'
    elif not old_final.is_file():phase='queued_waiting_for_predecessor'
    else:phase='queued_or_starting'

    latest={};issues=[];review=[];log_age=None;grad_health={'nonfinite_trailing':0,'nonfinite_recent_20':0,'nonfinite_recent_100':0}
    if active_log and active_log.is_file():
        text=re.sub(r'\x1b\[[0-9;]*m','',tail(active_log));log_age=max(0,now.timestamp()-active_log.stat().st_mtime);records=[]
        for line in text.replace('\r','\n').splitlines():
            m=ITER.search(line)
            if m:records.append({'iteration':int(m.group(1)),'loss':number(LOSS.search(line)),'grad_norm':number(GRAD.search(line))})
        if records:
            latest=records[-1];recent=records[-100:];bad=[not math.isfinite(x['grad_norm']) for x in recent if x['grad_norm'] is not None];trailing=0
            for value in reversed(bad):
                if not value:break
                trailing+=1
            grad_health={'nonfinite_trailing':trailing,'nonfinite_recent_20':sum(bad[-20:]),'nonfinite_recent_100':sum(bad)}
            if trailing>=2 or sum(bad[-20:])>=3 or sum(bad)>=10:issues.append(f'persistent/dense nonfinite grad: {grad_health}')
            elif any(bad):review.append(f'isolated recovered AMP overflow: {grad_health}')
            if latest.get('loss') is not None and not math.isfinite(latest['loss']):issues.append('nonfinite latest loss')
        found=[p for p in (r'CUDA out of memory',r'ChildFailedError',r'Traceback \(most recent call last\)',r'segmentation fault') if re.search(p,text,re.I)]
        if found:issues.append(f'hard runtime signatures: {found}')
        if processes and log_age is not None and log_age>1200:issues.append(f'active process log stale {log_age:.0f}s')
    root_free=shutil.disk_usage('/').free/1024**3;hdd_free=shutil.disk_usage('/mnt/HDD').free/1024**3
    if root_free<50:issues.append(f'root free below 50 GiB: {root_free:.1f}')
    if hdd_free<80:issues.append(f'HDD free below 80 GiB: {hdd_free:.1f}')
    if phase!='complete' and TMUX not in tmux and not processes:issues.append('incomplete but queue/training tmux absent')
    gpu={};raw=run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'])
    if raw:
        try:u,used,total,temp=[float(x.strip()) for x in raw.splitlines()[0].split(',')];gpu={'util_pct':u,'mem_used_mib':used,'mem_total_mib':total,'temp_c':temp}
        except ValueError:pass
    comparison=json.loads(final.read_text()) if final.is_file() else None
    status='incident' if issues else ('review' if review else 'healthy')
    progress=round(100*(latest['iteration']-600000)/100000,3) if latest.get('iteration') is not None else None
    payload={'updated_at':now.isoformat(),'experiment':'phase8_qsafe_ft_sequence','status':status,'phase':phase,'active_mode':active_mode,'active_prefix':active_prefix,
             'latest':latest,'fine_tune_progress_pct':progress,'target_iteration':700000 if active_mode else None,'issues':issues,'review':review,'grad_health':grad_health,
             'log_age_sec':log_age,'gpu':gpu,'root_free_gb':round(root_free,1),'hdd_free_gb':round(hdd_free,1),'tmux':tmux,'processes':processes.splitlines() if processes else [],
             'arms':arms,'final_comparison':comparison,
             'targets':{'baseline_noq':0.1888,'restoration':'Real-Q q9 >= 0.1900','q_information':'paired bootstrap CI95 Real-Q minus Shuffled-Q > 0','net_q_gain':'paired CI95 Real-Q minus NoQ > 0'}}
    tmp=STATUS.with_suffix('.json.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');tmp.replace(STATUS)
    if issues:
        tmp=ALERT.with_suffix('.json.tmp');tmp.write_text(json.dumps({'created_at':now.isoformat(),'phase':phase,'issues':issues,'stop_authorized':False},indent=2)+'\n');tmp.replace(ALERT)
    elif ALERT.exists():ALERT.unlink()
    print(f"status={status} phase={phase} mode={active_mode} it={latest.get('iteration')}/700000 progress={progress}% loss={latest.get('loss')} grad={latest.get('grad_norm')} gpu={gpu.get('util_pct')}% root={root_free:.1f}G hdd={hdd_free:.1f}G issues={len(issues)}")
    raise SystemExit(1 if issues else 0)

if __name__=='__main__':main()
