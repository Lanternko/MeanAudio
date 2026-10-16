#!/usr/bin/env python3
"""P2 supervisor for 128a/129b/130c; only this owned child may be signalled."""
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path[:0] = ['/home/kojiek/gpu_queue', '/home/kojiek/MeanAudio/scripts/experiment_harness']
from lib_scheduler import PAUSE_EXIT, accept_guest, atomic_json, now, pid_start_time, read_json
from secondary_queue_controller import Controller

INTERRUPTED = False
DISK_HOLD = '128 storage hard stop: owned evaluation child stopped. Polling capacity; no cleanup or automatic scientific changes.'


def handler(sig, frame):
    global INTERRUPTED
    INTERRUPTED = True


def stop(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()


def storage_state(c):
    measurements = []
    for fs in c['filesystems']:
        s = os.statvfs(fs['path']); free = s.f_bavail * s.f_frsize
        measurements.append({'path': fs['path'], 'free_bytes': free,
            'state': 'hard_stop' if free < fs['hard_floor_bytes'] else
                     'warning' if free < fs['warning_floor_bytes'] else 'ok'})
    return measurements


def progress(c):
    root = Path(c['storage']['path'])
    paths = [root/'progress.json', root/'smoke.json', root/'summary.json']
    return [p.stat().st_mtime_ns if p.exists() else 0 for p in paths]


def main():
    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM): signal.signal(sig, handler)
    script = Path(os.environ['GPU_QUEUE_JOB_SCRIPT']).resolve()
    accepted, reason = accept_guest(script)
    if not accepted: raise RuntimeError('HOLD: '+reason)
    for _ in range(50):
        seat = read_json(Path('/home/kojiek/gpu_queue/p2.running.json')) or {}
        if seat.get('pid') == os.getpid(): break
        time.sleep(.1)
    if (seat.get('pid'), seat.get('start_time'), seat.get('job_id'), seat.get('run_id')) != (
            os.getpid(), pid_start_time(os.getpid()), script.stem, os.environ.get('P2_RUN_ID')):
        raise RuntimeError('exact P2 ownership required')
    cp = Path(os.environ['GPU_QUEUE_CONTRACT']); c = json.loads(cp.read_text())
    state = Path(c['storage']['path'])/'supervisor'
    state.mkdir(parents=True, exist_ok=True)
    lock = (state/'owner.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    ctl = Controller(script, cp, state)
    (Path('/home/kojiek/gpu_queue/watcher_state')/'queue_idle_notified').unlink(missing_ok=True)
    child = None; previous = None; changed = time.monotonic(); disk_state = None
    try:
        while True:
            control = Path(os.environ['P2_CONTROL_DIR'])
            request = read_json(control/'pause.request.json')
            if request or INTERRUPTED:
                stop(child)
                if request:
                    resume = Path(c['resume']['autoresume'])
                    atomic_json(resume, {'at':now(), 'progress':progress(c)})
                    atomic_json(control/'pause.ack.json', {'run_id':request['run_id'],'job_id':request['job_id'],
                        'request_id':request['request_id'],'checkpoint':str(resume),'checkpoint_bytes':resume.stat().st_size,
                        'iteration':sum(v>0 for v in progress(c)), 'pid':os.getpid(),'start_time':pid_start_time(os.getpid())})
                    ctl.terminal('paused','interrupted','interrupted','128 paused for P1; only owned child stopped; verified score units retained.')
                    return PAUSE_EXIT
                ctl.terminal('interrupted','interrupted','interrupted','128 interrupted; verified score units retained.')
                return 130
            measured = storage_state(c)
            current_disk = 'hard_stop' if any(r['state']=='hard_stop' for r in measured) else 'warning' if any(r['state']=='warning' for r in measured) else 'ok'
            atomic_json(state/'monitor.json', {'at':now(),'filesystems':measured,'progress':progress(c),
                'child_pid':child.pid if child else None,'child_start_time':pid_start_time(child.pid) if child else None})
            if current_disk == 'hard_stop':
                stop(child); child = None
                if disk_state != current_disk:
                    ctl.notify('disk_hard_stop','held',DISK_HOLD)
                disk_state=current_disk; time.sleep(5); continue
            if disk_state=='hard_stop':
                ctl.notify('storage_recovered','start','128 capacity recovered; repeating preflight before resuming existing units.')
            elif current_disk=='warning' and disk_state!=current_disk:
                ctl.notify('disk_warning','start','128 storage advisory: below warning threshold, above hard floor. Evaluation remains eligible.')
            disk_state=current_disk
            if child is None:
                rc = subprocess.run(c['commands']['preflight']).returncode
                if rc==75:
                    if disk_state!='hard_stop':ctl.notify('disk_hard_stop','held',DISK_HOLD)
                    disk_state='hard_stop';time.sleep(5);continue
                if rc:
                    ctl.terminal('held','held','held',f'128 preflight invalid rc={rc}; no scorer launched.',reason='preflight_invalid')
                    return 2
                ctl.notify('preflight_pass','start','128 immutable inputs, schemas, dependency reports and storage gates passed.')
                ctl.pre_child_notifications(c['experiment_id'])
                child=subprocess.Popen(c['commands']['run'],start_new_session=True)
                ctl.record('child_started',{'pid':child.pid,'start_time':pid_start_time(child.pid)})
                previous=progress(c); changed=time.monotonic()
            rc=child.poll()
            if rc is not None:
                if rc==75:
                    child=None
                    if disk_state!='hard_stop':ctl.notify('disk_hard_stop','held',DISK_HOLD)
                    disk_state='hard_stop';time.sleep(5);continue
                if rc:
                    status='interrupted' if rc in (-1,-2,-9,-15,129,130,137,143) else 'failed'
                    ctl.terminal(status,'interrupted' if status=='interrupted' else 'failure','interrupted' if status=='interrupted' else 'failure',
                                 f'128 action exited rc={rc}; successor must pass its scientific dependency gate.',rc=rc)
                    return min(abs(rc),255)
                if subprocess.run(c['commands']['postflight']).returncode:
                    ctl.terminal('held','held','held','128 output validation failed; no completion evidence granted.',reason='postflight_invalid')
                    return 2
                reports=[{'path':r['path'],'sha256':hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()} for r in c['reports']]
                deferred=json.loads(Path(c['reports'][0]['path']).read_text()).get('status')=='deferred'
                message='128c deferred under the registered 17h rule; no NegMF re-adjudication was performed.' if deferred else '128 registered result reports validated; release resources and continue approved queue.'
                ctl.terminal('completed','success','success',message,
                             evidence={'eval_only':True,'reports':reports})
                return 0
            current=progress(c)
            if current!=previous: previous=current; changed=time.monotonic()
            elif time.monotonic()-changed>c['watcher']['stall_seconds']:
                stop(child);ctl.terminal('held','held','held','128 no completed score unit within registered stall interval; owned child stopped.',reason='stall')
                return 2
            time.sleep(5)
    except Exception as exc:
        stop(child);ctl.record('incident',{'type':type(exc).__name__,'reason':str(exc)[:500]})
        try:ctl.terminal('held','held','held','128 controller/notification incident; no owned child remains. Inspect supervisor ledger.',reason='controller_incident')
        except Exception:pass
        return 2
    finally:
        stop(child);lock.close()


if __name__=='__main__':raise SystemExit(main())
