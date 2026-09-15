#!/usr/bin/env python3
"""P2-owned, restart-safe three-arm instrument-conflict evaluation supervisor."""
import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path[:0] = ['/home/kojiek/gpu_queue', '/home/kojiek/MeanAudio/scripts/experiment_harness']
from lib_scheduler import PAUSE_EXIT, accept_guest, atomic_json, now, pid_start_time, read_json
from secondary_queue_controller import Controller
STATE = Path('/home/kojiek/logs/instrument_conflict_20260908_harn')
INTERRUPTED = False

def handler(_sig, _frame):
    global INTERRUPTED
    INTERRUPTED = True

def stop(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()

def progress(c):
    root=Path(c['storage']['transient_root'])
    files=list(root.glob('*/*.flac'))
    return (sum(Path(r['path']).is_file() for r in c['reports']),len(files),max((p.stat().st_mtime_ns for p in files if p.exists()),default=0))

def storage_state(c, free):
    if free < c['storage']['hard_stop_free_bytes']: return 'hard_stop'
    if free < c['storage']['warning_free_bytes']: return 'warning'
    return 'ok'

def main():
    for sig in (signal.SIGHUP,signal.SIGINT,signal.SIGTERM): signal.signal(sig,handler)
    script=Path(os.environ.get('GPU_QUEUE_JOB_SCRIPT') or sys.argv[0]).resolve()
    accepted,reason=accept_guest(script)
    if not accepted: raise SystemExit('HOLD: '+reason)
    for _ in range(50):
        seat=read_json(Path('/home/kojiek/gpu_queue/p2.running.json')) or {}
        if seat.get('pid')==os.getpid():break
        time.sleep(.1)
    if (seat.get('pid'),seat.get('start_time'),seat.get('job_id'),seat.get('run_id')) != (os.getpid(),pid_start_time(os.getpid()),script.stem,os.environ.get('P2_RUN_ID')):
        raise SystemExit('HOLD: exact P2 process ownership required')
    ctl=Controller(script,Path(os.environ['GPU_QUEUE_CONTRACT']),STATE);c=ctl.contract
    delivered={event['event'] for event in ctl._load()['events']}
    def incident(event,status,summary):
        marker='notification_'+event+'_delivered'
        if marker not in delivered:
            ctl.notify(event,status,summary);delivered.add(marker)
    control=Path(os.environ['P2_CONTROL_DIR']);child=None
    last=None;changed=time.monotonic();held=False
    try:
        while True:
            request=read_json(control/'pause.request.json')
            if request:
                stop(child)
                resume=Path(c['resume']['autoresume'])
                atomic_json(resume,{'document_kind':'instrument_conflict_resume_v1','written_at':now(),'progress':progress(c)})
                atomic_json(control/'pause.ack.json',{'run_id':request['run_id'],'job_id':request['job_id'],'request_id':request['request_id'],'checkpoint':str(resume),'checkpoint_bytes':resume.stat().st_size,'iteration':progress(c)[0],'pid':os.getpid(),'start_time':pid_start_time(os.getpid())})
                ctl.terminal('paused','interrupted','interrupted','045 paused by P1; verified complete reports and blind audio retained.')
                return PAUSE_EXIT
            if INTERRUPTED:
                stop(child);ctl.terminal('interrupted','interrupted','interrupted','045 interrupted; resume requires the registered P2 seat.');return 130
            fs=os.statvfs(c['storage']['path']);free=fs.f_bavail*fs.f_frsize
            state=storage_state(c,free)
            atomic_json(STATE/'monitor.json',{'written_at':now(),'free_bytes':free,'storage':state,'phase':'evaluation' if child else 'preflight_or_storage_wait','progress':progress(c),'child_pid':child.pid if child else None})
            if state=='hard_stop':
                stop(child);child=None;held=True
                incident('disk_hard_stop','held','045 storage hard stop; no evaluation child running. Poll capacity and resume the same registered cell; see monitor.json.')
                time.sleep(2);continue
            if state=='warning':
                incident('disk_warning','start','045 storage advisory: warning floor crossed; above hard floor. Evaluation remains eligible; see monitor.json.')
            if held:
                incident('storage_recovered','start','045 storage hard floor cleared; repeat immutable preflight before resuming.');held=False
            if child is None:
                rc=subprocess.run(c['commands']['preflight']).returncode
                if rc==75: time.sleep(2);continue
                if rc:
                    ctl.terminal('held','held','held',f'045 preflight invalid rc={rc}; no GPU child launched.',reason='preflight');return 2
                ctl.notify('preflight_pass','start','045 protocol, row assignments, provenance, storage and HARN bundle checks passed.')
                ctl.pre_child_notifications('045 CFG3 fidelity8 instrument-conflict ablation')
                child=subprocess.Popen(c['commands']['run'],start_new_session=True)
                ctl.record('evaluation_child_started',{'pid':child.pid,'start_time':pid_start_time(child.pid)})
                last=progress(c);changed=time.monotonic()
            rc=child.poll()
            if rc is not None:
                if rc==75: child=None;time.sleep(2);continue
                if rc:
                    ctl.terminal('failed','failure','failure',f'045 evaluation/gate failed rc={rc}; no later arm promoted.',rc=rc);return rc
                if subprocess.run(c['commands']['postflight']).returncode:
                    ctl.terminal('held','held','held','045 final reports or blind artifacts invalid.',reason='postflight');return 2
                evidence=[{'path':r['path'],'sha256':hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()} for r in c['reports']]
                evidence.append({'path':c['summary'],'sha256':hashlib.sha256(Path(c['summary']).read_bytes()).hexdigest()})
                ctl.terminal('completed','success','success','045 compute complete: three paired arms and blind-listening pack ready. Audible suppression remains unconfirmed until listening ratings.',evidence={'ema':c['completion_evidence']['ema'],'cfg0_report':c['completion_evidence']['cfg0_report'],'secondary_reports':evidence})
                return 0
            current=progress(c)
            if current!=last:last=current;changed=time.monotonic()
            elif time.monotonic()-changed>c['watcher']['stall_seconds']:
                stop(child);ctl.terminal('held','held','held','045 stalled beyond registered 7200s threshold; child stopped, no automatic repair.',reason='stall');return 2
            time.sleep(2)
    finally:
        stop(child)
if __name__=='__main__':raise SystemExit(main())
