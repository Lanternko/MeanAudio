#!/usr/bin/env python
"""194 P2 guest: contract-driven supervisor for the CMI-RM weakness probe (text, 30 s window).

Copy of muq_attractor_probe_188_guest.py (sha-bound by 188, left untouched); only names differ. 188 was a copy of
muq_train_exploit_183_guest.py (sha-bound by 183-186, left untouched) with one
functional change: terminal evidence carries eval_only=True, so the queue host validates the
contract's reports instead of looking for a training ema / cfg0 report (the 183 guest omits
the flag and its completed jobs are filed under held/). Everything job-specific lives in the
contract:
  watcher.progress_globs  newest mtime among matches = liveness signal
  watcher.done_globs      number of matches = coarse progress counter
  watcher.stall_seconds   no change in (done, mtime) for this long -> held
The child is resumable (per training run via DONE, per scoring cell), so a pause just stops
the child and records progress; the next seat re-runs the same command.
Terminal evidence is the contract's reports (sha256), checked after postflight.
"""
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path[:0] = ['/home/kojiek/gpu_queue']
from lib_scheduler import PAUSE_EXIT, accept_guest, atomic_json, now, pid_start_time, read_json

INTERRUPTED = False


def handler(_sig, _frame):
    global INTERRUPTED
    INTERRUPTED = True


def stop(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def terminal(script, status, **extra):
    atomic_json(script.with_name(script.stem + '.terminal.json'),
                {'status': status, 'written_at': now(), **extra})


def progress(c):
    w = c['watcher']
    done = sum(len(glob.glob(g, recursive=True)) for g in w.get('done_globs', []))
    m = 0
    for g in w.get('progress_globs', []):
        for p in glob.glob(g, recursive=True):
            try:
                m = max(m, os.stat(p).st_mtime_ns)
            except FileNotFoundError:
                pass
    return (done, m)


def main():
    for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)
    script = Path(os.environ.get('GPU_QUEUE_JOB_SCRIPT') or sys.argv[0]).resolve()
    accepted, reason = accept_guest(script)
    if not accepted:
        terminal(script, 'held', reason=reason)
        raise SystemExit('HOLD: ' + reason)
    c = json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())

    seat = {}
    for _ in range(50):
        seat = read_json(Path('/home/kojiek/gpu_queue/p2.running.json')) or {}
        if seat.get('pid') == os.getpid():
            break
        time.sleep(.1)
    if (seat.get('pid'), seat.get('start_time'), seat.get('job_id'), seat.get('run_id')) != (
            os.getpid(), pid_start_time(os.getpid()), script.stem, os.environ.get('P2_RUN_ID')):
        terminal(script, 'held', reason='exact P2 process ownership required')
        raise SystemExit('HOLD: exact P2 process ownership required')

    control = Path(os.environ.get('P2_CONTROL_DIR') or '')
    child = None
    last, changed = None, time.monotonic()
    wait_note = 0.0
    try:
        while True:
            request = read_json(control / 'pause.request.json') if control.is_dir() else None
            if request:
                stop(child)
                resume = Path(c['resume']['pause_progress'])
                resume.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(resume, {'document_kind': 'cmirm_weakness_194_resume_v1',
                                     'job': script.stem, 'written_at': now(), 'progress': progress(c)})
                atomic_json(control / 'pause.ack.json', {
                    'run_id': request['run_id'], 'job_id': request['job_id'],
                    'request_id': request['request_id'], 'checkpoint': str(resume),
                    'checkpoint_bytes': resume.stat().st_size, 'iteration': progress(c)[0],
                    'pid': os.getpid(), 'start_time': pid_start_time(os.getpid())})
                terminal(script, 'paused', progress=progress(c))
                return PAUSE_EXIT
            if INTERRUPTED:
                stop(child)
                terminal(script, 'interrupted', progress=progress(c))
                return 130
            if child is None:
                rc = subprocess.run(c['commands']['preflight']).returncode
                if rc == 75:
                    if time.monotonic() - wait_note > 600:
                        print(f'[{now()}] preflight wait (rc 75)', flush=True)
                        wait_note = time.monotonic()
                    time.sleep(30)
                    continue
                if rc:
                    terminal(script, 'held', reason=f'preflight invalid rc={rc}')
                    return 2
                child = subprocess.Popen(c['commands']['run'], start_new_session=True)
                last, changed = progress(c), time.monotonic()
            rc = child.poll()
            if rc is not None:
                if rc == 75:
                    child = None
                    time.sleep(30)
                    continue
                if rc:
                    status = 'interrupted' if rc in (-1, -2, -9, -15, 129, 130, 137, 143) else 'failed'
                    terminal(script, status, rc=rc, progress=progress(c))
                    return rc
                if subprocess.run(c['commands']['postflight']).returncode:
                    terminal(script, 'held', reason='postflight invalid')
                    return 2
                evidence = [{'path': r['path'],
                             'sha256': hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()}
                            for r in c['reports']]
                terminal(script, 'completed', evidence={'eval_only': True, 'reports': evidence, 'progress': progress(c)})
                return 0
            current = progress(c)
            if current != last:
                last, changed = current, time.monotonic()
            elif time.monotonic() - changed > c['watcher']['stall_seconds']:
                stop(child)
                terminal(script, 'held', reason='stalled beyond registered threshold')
                return 2
            time.sleep(5)
    except Exception as exc:
        stop(child)
        terminal(script, 'held', reason=f'{type(exc).__name__}: {str(exc)[:300]}')
        return 2
    finally:
        stop(child)


if __name__ == '__main__':
    raise SystemExit(main())
