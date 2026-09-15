#!/usr/bin/env python3
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'scripts/eval/qwen_bucket_backfill'))
import run as m
import validate_report as v
spec=importlib.util.spec_from_file_location('guest',R/'scripts/experiment_harness/qwen_bucket_backfill_guest.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
c=m.config();m.validate_protocol(c)
# Protocol rejection precedes any GPU invocation.
for field,value in [('seed',0),('rows',5400),('steps',1)]:
    bad=json.loads(json.dumps(c));bad['fixed_protocol'][field]=value
    try:m.validate_protocol(bad)
    except ValueError:pass
    else:raise AssertionError('protocol drift accepted')
with tempfile.TemporaryDirectory() as raw:
    d=Path(raw);ck=d/'ck';ck.write_bytes(b'checkpoint');tsv=d/'tsv';tsv.write_bytes(b'tsv');metrics=d/'metrics'/'test'/'metrics.txt';metrics.parent.mkdir(parents=True);metrics.write_text('metrics')
    cell={'cell_id':'a','label':'test','conditioning':'q9','checkpoint':str(ck),'checkpoint_sha256':m.digest(ck),'report':str(d/'report.json'),'cfg_strength':3,'protocol':'cfg3','negative_prompt':m.NEG,'metrics_path':str(metrics)}
    cc={'cells':[cell],'fixed_protocol':{'tsv':str(tsv),'tsv_sha256':m.digest(tsv)},'runtime_storage':{'metrics_root':str(d/'metrics')}}
    cp=d/'contract.json';cp.write_text(json.dumps(cc))
    report={'status':'passed','label':'test','protocol':'cfg3','cfg_strength':3,'num_steps':25,'seed':42,'conditioning':'q9','checkpoint_sha256':m.digest(ck),'tsv_sha256':m.digest(tsv),'checkpoint':str(ck),'tsv':str(tsv),'metrics_path':str(metrics),'metrics_sha256':m.digest(metrics),'negative_prompt':m.NEG,'negative_prompt_id':'fidelity8','audio_validation':{'rows':5521,'unique_ids':5521,'sample_rate':16000,'channels':1},'metrics':{key:1.0 for key in v.METRIC_KEYS}}
    rp=d/'report.json';rp.write_text(json.dumps(report));v.validate(cp,'a',rp)
    for field,value in [('cfg_strength',0),('negative_prompt','silence'),('conditioning','q0'),('audio_validation',{'rows':5400})]:
        rp.write_text(json.dumps(report|{field:value}))
        try:v.validate(cp,'a',rp)
        except ValueError:pass
        else:raise AssertionError('report drift accepted: '+field)
    rp.unlink()
    cc.update(resource_budget={'writable_filesystems':[str(d)]},storage={'hard_stop_free_bytes':0},runtime_roots=[str(d/'zero'),str(d/'three')])
    out,met=m.cell_paths(cc,cell);(out/'audio').mkdir(parents=True);(out/'audio'/'partial.flac').write_bytes(b'partial');met.mkdir(parents=True);(met/'metrics.txt').write_text('unbound')
    with patch.object(m,'STATE',d):m.prepare(cc,cell)
    assert not out.exists() and not met.exists()
    assert len(list(out.parent.glob('test.partial-*/audio/partial.flac')))==1
    # Run all 14 actual registered argv through fake CPU child; NoQ arm matches contract.
    calls=[];events=[]
    with patch.object(m,'STATE',d),patch.object(m,'prepare'),patch.object(m,'notify',side_effect=lambda c,e,s:events.append(e)),patch.object(m,'validate'),patch.object(m,'digest',return_value='fixture'),patch.object(m,'postflight'),patch.object(m.subprocess,'run',side_effect=lambda argv,**kw:calls.append((argv,kw))):m.run(c)
    assert len(calls)==14 and calls[0][1]['env']['CFG0_ARM']=='noq_cfg0_noq'
    assert [x[1]['env']['CFG0_ARM'] for x in calls]==[x['cell_id'] for x in c['cells']]
    assert calls[0][0][-1]=='--no_q' and calls[1][0][-2:]==['--quality_level','9']
    assert all('cfg3_fidelity8.sh' in x[0][1] for x in calls[9:])
    assert len(events)==29
    calls=[]
    with patch.object(m,'STATE',d),patch.object(m,'prepare'),patch.object(m,'notify',side_effect=RuntimeError('notification_pending')),patch.object(m.subprocess,'run',side_effect=lambda *a,**kw:calls.append(a)):
        try:m.run(c)
        except RuntimeError:pass
    assert not calls
print('PASS protocol/report rejection, 14 ordered argv, exact NoQ arm, partial replay, notification hold')

# Host-side capacity and process/notification branches use the actual supervisor main.
assert h.storage_state(c,0)=='hard_stop'
assert h.storage_state(c,c['storage']['hard_stop_free_bytes'])=='warning'
assert h.storage_state(c,c['storage']['warning_free_bytes'])=='ok'
class FakeController:
    instances=[]
    def __init__(self,script,contract,state):
        self.contract=json.loads(contract.read_text());self.events=[];self.terminal_args=None;FakeController.instances.append(self)
    def _load(self):return {'events':[]}
    def notify(self,*args,**kwargs):self.events.append(args)
    def pre_child_notifications(self,*args):self.events.append(('start',))
    def record(self,*args):pass
    def terminal(self,*args,**kwargs):self.terminal_args=args
class FakeChild:
    pid=987654
    def __init__(self,rc):self.rc=rc
    def poll(self):return self.rc
    def wait(self,timeout=None):return self.rc
for case in ('success','failure','exit137','interrupt','pause','invalid_preflight','storage_recover','notification_failure'):
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);script=d/'051_fixture.sh';script.write_text('fixture');contract=d/'contract.json'
        report=d/'report.json';report.write_text('{}');summary=d/'summary.json';summary.write_text('{}')
        spec={**c,'reports':[{'path':str(report)}],'summary':str(summary),'resume':{'autoresume':str(d/'resume.json')}};contract.write_text(json.dumps(spec))
        calls=[];child=FakeChild(2 if case=='failure' else 137 if case=='exit137' else 0)
        def read(path):
            if path.name=='p2.running.json':return {'pid':os.getpid(),'start_time':h.pid_start_time(os.getpid()),'job_id':script.stem,'run_id':'fixture'}
            if case=='pause':return {'run_id':'fixture','job_id':script.stem,'request_id':'request1'}
            return None
        def spawn(*args,**kwargs):calls.append('spawn');return child
        frees=iter([0,10**15] if case=='storage_recover' else [10**15])
        def fs(_):return SimpleNamespace(f_bavail=next(frees,10**15),f_frsize=1)
        run=lambda *a,**kw:SimpleNamespace(returncode=1 if case=='invalid_preflight' else 0)
        with patch.dict(os.environ,{'GPU_QUEUE_JOB_SCRIPT':str(script),'GPU_QUEUE_CONTRACT':str(contract),'P2_RUN_ID':'fixture','P2_CONTROL_DIR':str(d)}),patch.object(h,'rearm_queue_idle'),patch.object(h,'STATE',d),patch.object(h,'Controller',FakeController),patch.object(h,'accept_guest',return_value=(True,'ok')),patch.object(h,'read_json',side_effect=read),patch.object(h,'progress',return_value=(0,0,0)),patch.object(h.os,'statvfs',side_effect=fs),patch.object(h.subprocess,'run',side_effect=run),patch('preflight_capture.run_preflight',side_effect=lambda *a,**kw:1 if case=='invalid_preflight' else 0),patch.object(h.subprocess,'Popen',side_effect=spawn),patch.object(h.time,'sleep'),patch.object(h,'INTERRUPTED',case=='interrupt'):
            if case=='notification_failure':
                with patch.object(FakeController,'pre_child_notifications',side_effect=RuntimeError('notification_pending')):
                    assert h.main()==2
                assert not calls
            else:
                rc=h.main();terminal=FakeController.instances[-1].terminal_args[0]
                expected={'success':(0,'completed'),'failure':(2,'failed'),'exit137':(137,'interrupted'),'interrupt':(130,'interrupted'),'pause':(h.PAUSE_EXIT,'paused'),'invalid_preflight':(2,'held'),'storage_recover':(0,'completed')}[case]
                assert (rc,terminal)==expected,(case,rc,terminal)
                if case in ('interrupt','pause','invalid_preflight'):assert not calls
                if case=='storage_recover':
                    events=[e[0] for e in FakeController.instances[-1].events]
                    assert events.count('disk_hard_stop')==1 and events.count('storage_recovered')==1 and len(calls)==1
print('PASS: actual supervisor success/failure/137/interruption/pause/preflight-invalid/storage-recovery/notification-pending branches; no GPU processes')
