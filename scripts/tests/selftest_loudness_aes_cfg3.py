#!/usr/bin/env python3
"""CPU fixtures: waveform gain, statistical effects, artifact integrity and real supervisor branches."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'scripts/eval'))
import loudness_aes_cfg3_20260911 as m
import score_musiccaps_per_item as scorer
import validate_loudness_aes_cfg3_20260911 as v
spec=importlib.util.spec_from_file_location('guest',R/'scripts/experiment_harness/loudness_aes_cfg3_20260911_guest.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
c=json.loads(m.CONTRACT.read_text());v.validate_protocol(c)
for change in ({'cfg_strength':0},{'negative_prompt':'silence'},{'rows':1024}):
    try:v.validate_protocol(c|{'protocol':c['protocol']|change})
    except ValueError:pass
    else:raise AssertionError('protocol drift accepted')
import soundfile as sf
import torch
class FakeAES:
    calls=0
    model=SimpleNamespace(wavlm_model=SimpleNamespace(cfg=SimpleNamespace(normalize=False)))
    def _load_audio(self,p):return torch.from_numpy(m.read_audio(p)).unsqueeze(0)
    def forward(self,batch):
        FakeAES.calls+=len(batch)
        return [{a:float(np.sqrt(np.mean(m.read_audio(x['path']).astype(float)**2)))*10 for a in m.AXES} for x in batch]
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp);root=tmp/'run';root.mkdir();directory=root/'_audio'/'baseline';directory.mkdir(parents=True)
    fixture=c|{'storage':c['storage']|{'path':str(root)},'analysis':c['analysis']|{'bootstrap_replicates':50},'protocol':c['protocol']|{'rows':20},'resource_budget':{'writable_filesystems':[str(root)]}}
    rec=[SimpleNamespace(id=f'id{i:02d}',caption='music') for i in range(20)]
    for i,r in enumerate(rec):sf.write(directory/(r.id+'.flac'),(.01+i*.01)*np.sin(2*np.pi*440*np.arange(16000)/16000),16000)
    manifest={'contract_sha256':'fixture','audio_sha256':{r.id:m.digest(directory/(r.id+'.flac')) for r in rec}}
    m.atomic(root/'audio_manifest.json',manifest)
    with patch.object(m,'binding',return_value='fixture'),patch.object(m,'records',return_value=rec),patch.object(m,'notify'),patch.object(m,'capacity'),patch.object(scorer,'load_aes_predictor',return_value=FakeAES()):
        for g in m.GAINS:m.score_gain(fixture,g)
        assert FakeAES.calls==80
        for g in m.GAINS:m.score_gain(fixture,g)
        assert FakeAES.calls==80,'resume duplicated inference'
        x=m.read_audio(directory/'id10.flac');z=m.gain_copy(directory/'id10.flac',tmp/'test.wav',-6)
        a=m.signal_stats(x);b=m.signal_stats(z)
        assert abs(b['rms_dbfs']-a['rms_dbfs']+6)<1e-5
        assert abs(b['crest_db']-a['crest_db'])<1e-5
        assert abs(b['lufs']-a['lufs']+6)<1e-4
        zero=m.signal_stats(np.zeros(16000));assert zero['lufs'] is None and zero['rms_dbfs'] is None and zero['crest_db'] is None
        assert np.array_equal(x,m.gain_copy(directory/'id10.flac',tmp/'identity.wav',0))
        cuts,idx=m.group_indices([1]*20,5);assert len(idx[0])==20 and all(len(z)==0 for z in idx[1:])
        items=m.load_items(fixture);analysis=m.analyze_data(items,123,50)
        assert analysis['primary']['association_PQ_Q5_minus_Q1']['mean']>0
        assert analysis['primary']['gain_PQ_minus6_minus0']['ci95'][1]<0
        assert analysis==m.analyze_data(items,123,50),'bootstrap not reproducible'
        with patch.object(scorer,'load_clap_model',return_value=object()),patch.object(scorer,'_clap_batch',side_effect=lambda model,paths,captions:[.3]*len(paths)):
            m.clap(fixture)
        m.report(fixture);m.validate_all(fixture)
        path=root/'items'/'-6'/'id00.json';value=json.loads(path.read_text());value['source_sha256']='wrong';m.atomic(path,value)
        try:m.score_gain(fixture,-6)
        except ValueError:pass
        else:raise AssertionError('stale source accepted')
print('PASS: real score/report pipeline with CPU fake models; 80 paired records; 0dB identity; -6dB RMS/LUFS; crest invariance; silence/ties; signed effects; reproducible bootstrap; resume/no duplicate; stale source rejection')
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
        with patch.dict(os.environ,{'GPU_QUEUE_JOB_SCRIPT':str(script),'GPU_QUEUE_CONTRACT':str(contract),'P2_RUN_ID':'fixture','P2_CONTROL_DIR':str(d)}),patch.object(h,'rearm_queue_idle'),patch.object(h,'STATE',d),patch.object(h,'Controller',FakeController),patch.object(h,'accept_guest',return_value=(True,'ok')),patch.object(h,'read_json',side_effect=read),patch.object(h,'progress',return_value=(0,0,0)),patch.object(h.os,'statvfs',side_effect=fs),patch.object(h.subprocess,'run',side_effect=run),patch.object(h.subprocess,'Popen',side_effect=spawn),patch.object(h.time,'sleep'),patch.object(h,'INTERRUPTED',case=='interrupt'):
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
