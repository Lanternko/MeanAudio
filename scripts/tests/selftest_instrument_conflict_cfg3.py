#!/usr/bin/env python3
"""No-GPU execution/analysis/resume fixtures for the instrument-conflict run."""
import ast
import csv
import importlib.util
import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
R=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('instrument',R/'scripts/eval/instrument_conflict_cfg3_20260908.py')
v=load('instrument_preflight',R/'scripts/eval/validate_instrument_conflict_cfg3_20260908.py')
b=load('builder',R/'scripts/eval/build_instrument_conflict_inputs.py')
h=load('guest',R/'scripts/experiment_harness/instrument_conflict_cfg3_20260908_guest.py')
c=json.loads(m.CONTRACT.read_text());v.validate_rows(c)
assert b.mentions('a violin and piano')==['piano','violin']
assert b.mentions('bass, brass and strings')==[]
fixture=[{'id':'a','caption':'violin with piano'},{'id':'b','caption':'guitar solo'},{'id':'c','caption':'no drums or guitar'}]
frozen,matched=b.freeze(fixture);assert len(frozen)==2 and frozen==b.freeze(fixture)[0]
assert all(a['target'] in a['mentioned'] and a['unmentioned'] not in a['mentioned'] for a in frozen)
# Execute the real generator's TSV reader and generation loop with a fake model only.
tree=ast.parse((R/'scripts/eval/instrument_conflict_cfg3_20260908_generate.py').read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
reader=next(n for n in main.body if isinstance(n,ast.With))
loop=next(n for n in main.body if isinstance(n,ast.For))
class Audio:
    def float(self):return self
    def cpu(self):return self
    def __getitem__(self,k):return self
    def squeeze(self,k):return self
    def numpy(self):return np.ones(100)
class Flag:
    def any(self):return False
with tempfile.TemporaryDirectory() as d:
    d=Path(d); tsv=d/'input.tsv'
    tsv.write_text('id\tcaption\tnegative_prompt\na\tviolin solo\tfidelity, violin\nb\tpiano solo\tfidelity, flute\n')
    calls=[]
    def gen(prompts,**kw):calls.append((prompts,kw['negative_text'],kw['rng']));return Audio()
    rng=object()
    env={'args':SimpleNamespace(negative_prompt_column='negative_prompt',prompt_suffix='',no_q=True,use_meanflow=True,no_text_attention_mask=True),
         'audio_ids':[],'text_prompts':[],'q_levels':[],'row_negatives':[],'eval_file':str(tsv),'csv':csv,
         'output_dir':d,'tqdm':lambda x:x,'generate_mf':gen,'feature_utils':None,'net':None,'mf':None,'rng':rng,'cfg_strength':3,
         'log':logging.getLogger('test'),'torch':SimpleNamespace(isnan=lambda a:Flag(),isinf=lambda a:Flag(),cuda=SimpleNamespace(max_memory_allocated=lambda:0)),
         'sf':SimpleNamespace(write=lambda *a:None),'seq_cfg':SimpleNamespace(sampling_rate=16000)}
    exec(compile(ast.Module(body=[reader,loop],type_ignores=[]),'generator_fixture','exec'),env)
    assert calls==[(['violin solo'],['fidelity, violin'],rng),(['piano solo'],['fidelity, flute'],rng)]
# Complete synthetic report validates; wrong negative identity and missing audio hashes do not.
ids={r['id'] for r in m.rows()}
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'report.json';per={i:{k:1. for k in m.METRICS} for i in ids}
    payload={'label':m.label('fidelity8'),'exp_id':m.EXP_ID,'provenance':m.provenance(),'cfg_strength':3,'negative_key':'fidelity8','negative_prompt':m.NEGATIVES['fidelity8'],'protocol_id':'musiccaps3207_mf25_cfg3_fidelity8_instrument_conflict_seed42_nomask_fp32','per_clip':per,'generated_audio_sha256':{i:'a'*64 for i in ids},'aggregates':{'full':{'n':3207,**{k:1. for k in m.METRICS}}}}
    p.write_text(json.dumps(payload));assert m.valid_report(p,'fidelity8',ids)
    payload['negative_key']='wrong';p.write_text(json.dumps(payload));assert not m.valid_report(p,'fidelity8',ids)
    payload['negative_key']='fidelity8';payload['generated_audio_sha256']={};p.write_text(json.dumps(payload));assert not m.valid_report(p,'fidelity8',ids)
# All storage branches are explicit; a preflight hard stop is retryable, not terminal success.
assert h.storage_state(c,0)=='hard_stop'
assert h.storage_state(c,c['storage']['hard_stop_free_bytes'])=='warning'
assert h.storage_state(c,c['storage']['warning_free_bytes'])=='ok'
with patch.object(v.os,'statvfs',return_value=SimpleNamespace(f_bavail=0,f_frsize=4096)):
    try:v.main()
    except SystemExit as e:assert e.code==75
    else:raise AssertionError('storage did not hold')
# Summary contrasts and blind retention, without heavy models.
with tempfile.TemporaryDirectory() as d,patch.object(m,'OUT',Path(d)),patch.object(m,'AUDIO_ROOT',Path(d)/'_audio'),patch.object(m,'provenance',return_value={'fixture':True}):
    d=Path(d);(d/'listening').mkdir()
    assignments={'assignments':[{'id':str(i),'target':'violin' if i<5 else 'piano'} for i in range(10)],'control_frequency_exact_match':True,'target_counts':{'violin':5,'piano':5},'unmentioned_counts':{'violin':5,'piano':5},'listening_sample':[{'id':'0','target':'violin','slots':dict(zip('ABC',m.ORDER))}]}
    (d/'assignments.json').write_text(json.dumps(assignments))
    for key in m.ORDER:
        per={str(i):{k:1.+(-.1 if key=='fidelity8_conflict' else 0) for k in m.METRICS} for i in range(10)}
        directory=m.audio_dir(key);directory.mkdir(parents=True);clip=directory/'0.flac';clip.write_bytes(b'fixture audio')
        m.report(key).write_text(json.dumps({'per_clip':per,'generated_audio_sha256':{'0':m.digest(clip)}}))
        m.retain_and_cleanup(key);assert not directory.exists()
        m.retain_and_cleanup(key) # restart is safe after source cleanup
    m.write_summary();summary=json.loads((d/'summary.json').read_text())
    assert summary['decision']['semantic_interference_proxy']=='supported'
    assert summary['decision']['audible_instrument_suppression']=='pending_blind_listening'
    assert summary['comparisons']['conflict_vs_unmentioned']['all']['clap']['mean_delta']<0
    (d/'listening/ratings.csv').write_text('human ratings\n');m.write_summary();assert (d/'listening/ratings.csv').read_text()=='human ratings\n'
print('PASS: fixed assignments, actual per-row generator loop, immutable report resume, storage pass/warn/retryable hold, paired contrast, blind-retention resume, human rating preservation')
# Drive the actual supervisor through terminal, hold and interruption branches.
import os
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
        d=Path(d);script=d/'045_fixture.sh';script.write_text('fixture');contract=d/'contract.json'
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
        with patch.dict(os.environ,{'GPU_QUEUE_JOB_SCRIPT':str(script),'GPU_QUEUE_CONTRACT':str(contract),'P2_RUN_ID':'fixture','P2_CONTROL_DIR':str(d)}),patch.object(h,'STATE',d),patch.object(h,'Controller',FakeController),patch.object(h,'accept_guest',return_value=(True,'ok')),patch.object(h,'read_json',side_effect=read),patch.object(h,'progress',return_value=(0,0,0)),patch.object(h.os,'statvfs',side_effect=fs),patch.object(h.subprocess,'run',side_effect=run),patch.object(h.subprocess,'Popen',side_effect=spawn),patch.object(h.time,'sleep'),patch.object(h,'INTERRUPTED',case=='interrupt'):
            if case=='notification_failure':
                with patch.object(FakeController,'pre_child_notifications',side_effect=RuntimeError('notification_pending')):
                    try:h.main()
                    except RuntimeError:pass
                    else:raise AssertionError('notification failure did not block')
                assert not calls
            else:
                rc=h.main();terminal=FakeController.instances[-1].terminal_args[0]
                expected={'success':(0,'completed'),'failure':(2,'failed'),'exit137':(137,'failed'),'interrupt':(130,'interrupted'),'pause':(h.PAUSE_EXIT,'paused'),'invalid_preflight':(2,'held'),'storage_recover':(0,'completed')}[case]
                assert (rc,terminal)==expected,(case,rc,terminal)
                if case in ('interrupt','pause','invalid_preflight'):assert not calls
                if case=='storage_recover':
                    events=[e[0] for e in FakeController.instances[-1].events]
                    assert events.count('disk_hard_stop')==1 and events.count('storage_recovered')==1 and len(calls)==1
print('PASS: actual supervisor success/failure/137/interruption/pause/preflight-invalid/storage-recovery/notification-pending branches; no GPU processes')
