#!/usr/bin/env python3
"""P2 guest for the autoresearch inference-CLAP line: supervisor around run_round.py.

Same protocol as scripts/experiment_harness/bench_r2_guest.py (accept_guest, exact
P2 process ownership, preflight rc 75 = retry, pause ack, stall watchdog, terminal
evidence); progress is trial summaries + phase-0 step files, so a long render is
covered by the registered stall window.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True

sys.path[:0] = ['/home/kojiek/gpu_queue', str(HERE)]
from lib_scheduler import PAUSE_EXIT, accept_guest, atomic_json, now, pid_start_time, read_json  # noqa: E402

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
    """(finished units, newest mtime_ns) over trial summaries, audio dirs, phase-0 steps.

    Audio directories are included so a long full-5521 trial still moves the counter
    every few seconds while eval.py writes flacs; summaries alone would leave a
    40-minute gap between two trials.
    """
    root = Path(c['fixed_protocol']['scratch'])
    n, m = 0, 0
    for pattern in ('trials/*/*/summary.json', 'trials/*/*/*/summary.json',
                    'trials/*/*/cells/*/audio', 'trials/*/*/lvl30/*/audio',
                    'trials/*/*/*/cells/*/audio', 'trials/*/*/*/lvl30/*/audio',
                    'phase0/*.json', 'rounds_out/*_run.json'):
        for p in root.glob(pattern):
            try:
                st = p.stat()
            except OSError:
                continue
            n += 1
            m = max(m, st.st_mtime_ns)
    return (n, m)


def evidence(c):
    root = Path(c['fixed_protocol']['scratch'])
    mode = c['fixed_protocol']['mode']
    rel = (Path('phase0') / 'p0_summary.json' if mode == 'phase0'
           else Path('rounds_out') / f"{c['fixed_protocol']['round']}_summary.json")
    p = root / rel
    if not p.is_file():
        return {'path': str(p), 'sha256': None, 'present': False}
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'present': True}


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
                atomic_json(resume, {'document_kind': 'autoresearch_clap_resume_v1',
                                     'written_at': now(), 'progress': progress(c),
                                     'round': c['fixed_protocol'].get('round'),
                                     'mode': c['fixed_protocol']['mode']})
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
                    time.sleep(5)
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
                    time.sleep(5)
                    continue
                if rc:
                    status = 'interrupted' if rc in (-1, -2, -9, -15, 129, 130, 137, 143) else 'failed'
                    terminal(script, status, rc=rc, progress=progress(c), evidence=evidence(c))
                    return rc
                if subprocess.run(c['commands']['postflight']).returncode:
                    terminal(script, 'held', reason='postflight invalid', progress=progress(c))
                    return 2
                terminal(script, 'completed', evidence=evidence(c), progress=progress(c))
                return 0
            current = progress(c)
            if current != last:
                last, changed = current, time.monotonic()
            elif time.monotonic() - changed > c['watcher']['stall_seconds']:
                stop(child)
                terminal(script, 'held', reason='stalled beyond registered threshold',
                         progress=progress(c))
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
