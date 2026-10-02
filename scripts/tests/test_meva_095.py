"""No-GPU acceptance fixtures for MEva scoring bindings and queue branches."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/eval'),str(ROOT/'scripts/experiment_harness'),'/home/kojiek/gpu_queue']
import meva_promptcc as batch
from lib_scheduler import pid_start_time

def guest_fixture(status, preflight=0, postflight=0, pause=False, disk=False, interrupt=False, notify_fail=False):
    spec=importlib.util.spec_from_file_location('guest',ROOT/'scripts/experiment_harness/meva_095_guest.py')
    guest=importlib.util.module_from_spec(spec);spec.loader.exec_module(guest)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);script=root/'095_test.sh';script.write_text('#!/bin/bash\n');script.chmod(0o700)
        report=root/'summary.json';report.write_text('{}')
        control=root/'control';control.mkdir()
        contract=root/'contract.json'
        c={'storage':{'path':str(root),'hard_stop_free_bytes':50<<30,'warning_free_bytes':80<<30},
           'commands':{'run':['dummy'],'preflight':['pre'],'postflight':['post']},
           'watcher':{'stall_seconds':3600},'resume':{'pause_progress':str(root/'resume.json')},
           'reports':[{'path':str(report)}]}
        contract.write_text(json.dumps(c))
        seat={'pid':os.getpid(),'start_time':pid_start_time(os.getpid()),'job_id':script.stem,'run_id':'run-test'}
        calls=[]
        def read(path):
            if path.name=='p2.running.json':return seat
            if path.name=='pause.request.json':return {'run_id':'run-test','job_id':script.stem,'request_id':'request-test'} if pause else None
            return json.loads(path.read_text()) if path.exists() else None
        class Child:
            pid=12345
            def poll(self):return None if disk else status
        def spawn(*a,**k):calls.append('spawn');return Child()
        def notify(*a,**k):
            calls.append('notify')
            if notify_fail:raise RuntimeError('notifier unavailable')
        class FS: f_bavail=1 if disk else 100*(1<<30);f_frsize=1
        env={'GPU_QUEUE_JOB_SCRIPT':str(script),'GPU_QUEUE_CONTRACT':str(contract),
             'P2_CONTROL_DIR':str(control),'P2_RUN_ID':'run-test'}
        with patch.dict(os.environ,env),patch.object(guest.signal,'signal'),patch.object(guest,'read_json',read),\
             patch.object(guest,'accept_guest',return_value=(True,'ok')),patch.object(guest,'notify_gate',notify),\
             patch.object(guest,'run_preflight',return_value=preflight),patch.object(guest.subprocess,'Popen',spawn),\
             patch.object(guest.subprocess,'run',return_value=type('RC',(),{'returncode':postflight})()),\
             patch.object(guest,'stop'),patch.object(guest.os,'statvfs',return_value=FS()):
            guest.INTERRUPTED=interrupt
            rc=guest.main()
        terminal=json.loads((root/'095_test.terminal.json').read_text())
        return rc,terminal,calls

results={}
for label,kwargs,expected in [
    ('success',{'status':0},'completed'),('failure',{'status':1},'failed'),
    ('exit137',{'status':137},'interrupted'),('preflight_invalid',{'status':0,'preflight':2},'held'),
    ('postflight_invalid',{'status':0,'postflight':2},'held'),
    ('pause',{'status':0,'pause':True},'paused'),('signal',{'status':0,'interrupt':True},'interrupted'),
    ('disk_hard_stop',{'status':0,'disk':True},'held'),
    ('notification_failure',{'status':0,'notify_fail':True},'held')]:
    rc,terminal,calls=guest_fixture(**kwargs)
    assert terminal['status']==expected,(label,terminal)
    if label in ['preflight_invalid','notification_failure']:assert 'spawn' not in calls
    results[label]='pass'
with tempfile.TemporaryDirectory() as temp:
    old=batch.OUT;batch.OUT=Path(temp)
    row={'set':'a','key':'b','audio_sha256':'abc'};p=batch.result_path(row);p.parent.mkdir()
    assert batch.existing(row,'binding') is None
    record={'input_sha256':'abc','model_binding':'binding','status':'scored','meva_raw':3.0}
    p.write_text(json.dumps(record));assert batch.existing(row,'binding')['meva_raw']==3.0
    try:batch.existing(row,'changed')
    except ValueError:pass
    else:raise AssertionError('stale model accepted')
    record['meva_raw']=float('nan');p.write_text(json.dumps(record))
    try:batch.existing(row,'binding')
    except ValueError:pass
    else:raise AssertionError('NaN accepted')
    batch.OUT=old
results.update(resume_binding='pass',stale_model_rejected='pass',nonfinite_rejected='pass')
print(json.dumps({'status':'passed','fixtures':results},indent=2))
