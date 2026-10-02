#!/usr/bin/env python3
"""095 P2 guest: resumable, eval-only MEva shadow scoring supervisor.

The generic queue guest gates completion on a training EMA plus a CFG0 report. This
job trains nothing and produces neither, so it owns its terminal evidence instead of
being held by a gate it can never satisfy.
"""
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
from preflight_capture import run_preflight

INTERRUPTED = False


def handler(_sig, _frame):
    global INTERRUPTED
    INTERRUPTED = True


def stop(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def terminal(script, status, **extra):
    atomic_json(script.with_name(script.stem + '.terminal.json'),
                {'status': status, 'written_at': now(), **extra})


def progress(c):
    marker = Path(c['storage']['path']) / 'progress.json'
    value = read_json(marker) or {}
    return (int(value.get('index', 0)), marker.stat().st_mtime_ns if marker.exists() else 0)


def notify_gate(c, script, event, status, summary):
    from notification_receipts import deliver_required
    cfg = c['notification_receipts']
    return deliver_required(contract_path=Path(os.environ['GPU_QUEUE_CONTRACT']),
        launcher_path=script, event=event, status=status, summary=summary,
        idempotency_key=c['experiment_id']+':'+event, notifier=Path(cfg['notifier']),
        root=Path(cfg['root']))


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
    try:
        while True:
            request = read_json(control / 'pause.request.json') if control.is_dir() else None
            if request:
                stop(child)
                resume = Path(c['resume']['pause_progress'])
                atomic_json(resume, {'document_kind': 'meva_095_resume_v1',
                                     'written_at': now(), 'progress': progress(c)})
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
                rc = run_preflight(c['commands']['preflight'], Path(c['storage']['path']))
                if rc == 75:
                    time.sleep(2)
                    continue
                if rc:
                    terminal(script, 'held', reason=f'preflight invalid rc={rc}')
                    return 2
                notify_gate(c, script, 'preflight-pass', 'success',
                            'MEva095 preflight PASS: bound inputs/runtime and storage. Next: shadow scoring.')
                child = subprocess.Popen(c['commands']['run'], start_new_session=True)
                last, changed = progress(c), time.monotonic()
            rc = child.poll()
            if rc is not None:
                if rc == 75:
                    child = None
                    time.sleep(2)
                    continue
                if rc:
                    status = 'interrupted' if rc in (-1, -2, -9, -15, 129, 130, 137, 143) else 'failed'
                    terminal(script, status, rc=rc, progress=progress(c))
                    return rc
                if subprocess.run(c['commands']['postflight']).returncode:
                    terminal(script, 'held', reason='postflight invalid')
                    return 2
                notify_gate(c, script, 'postflight-pass', 'success',
                            'MEva095 full coverage postflight PASS. Results are shadow scores; AES is retained.')
                evidence = [{'path': r['path'],
                             'sha256': hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()}
                            for r in c['reports']]
                terminal(script, 'completed', evidence={'eval_only': True, 'reports': evidence,
                                                        'progress': progress(c)})
                return 0
            fs = os.statvfs(c['storage']['path'])
            if fs.f_bavail * fs.f_frsize < c['storage']['hard_stop_free_bytes']:
                stop(child)
                terminal(script, 'held', reason='storage hard stop', progress=progress(c))
                return 2
            if fs.f_bavail * fs.f_frsize < c['storage']['warning_free_bytes']:
                notify_gate(c, script, 'storage-warning', 'start',
                            'MEva095 storage advisory: free bytes crossed registered warning floor; current queue entry remains running.')
            current = progress(c)
            if current != last:
                last, changed = current, time.monotonic()
            elif time.monotonic() - changed > c['watcher']['stall_seconds']:
                stop(child)
                terminal(script, 'held', reason='stalled beyond registered threshold')
                return 2
            time.sleep(2)
    except Exception as exc:
        stop(child)
        terminal(script, 'held', reason=f'{type(exc).__name__}: {str(exc)[:300]}')
        return 2
    finally:
        stop(child)


if __name__ == '__main__':
    raise SystemExit(main())
