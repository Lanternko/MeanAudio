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
import meva_expansion_20261004 as batch
from lib_scheduler import pid_start_time

def guest_fixture(status, preflight=0, postflight=0, pause=False, disk=False, interrupt=False, notify_fail=False):
    spec=importlib.util.spec_from_file_location('guest',ROOT/'scripts/experiment_harness/meva_expansion_20261004_guest.py')
    guest=importlib.util.module_from_spec(spec);spec.loader.exec_module(guest)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);script=root/'095_test.sh';script.write_text('#!/bin/bash\n');script.chmod(0o700)
        report=root/'summary.json';report.write_text('{}')
        control=root/'control';control.mkdir()
        contract=root/'contract.json'
        c={'storage':{'path':str(root),'hard_stop_free_bytes':50<<30,'warning_free_bytes':80<<30},
           'commands':{'run':['dummy'],'preflight':['pre'],'postflight':['post']},
           'watcher':{'stall_seconds':3600},'resume':{'pause_progress':str(root/'resume.json')},
           'reports':[{'path':str(report)}], 'harn_bundle':str(root)}
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
            return {'path':'mock','event':'mock','status':'mock'}
        class FS: f_bavail=1 if disk else 100*(1<<30);f_frsize=1
        env={'GPU_QUEUE_JOB_SCRIPT':str(script),'GPU_QUEUE_CONTRACT':str(contract),
             'P2_CONTROL_DIR':str(control),'P2_RUN_ID':'run-test'}
        with patch.dict(os.environ,env),patch.object(guest.signal,'signal'),patch.object(guest,'read_json',read),\
             patch.object(guest,'accept_guest',return_value=(True,'ok')),patch.object(guest,'notify_gate',notify),\
             patch.object(guest,'run_preflight',return_value=preflight),patch.object(guest.subprocess,'Popen',spawn),\
             patch.object(guest.subprocess,'run',return_value=type('RC',(),{'returncode':postflight})()),\
             patch.object(guest,'notify',return_value={'path':'mock','event':'mock','status':'mock'}),\
             patch.object(guest,'append'),patch.object(guest,'rearm_queue_idle'),\
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
    old=batch.RUN;batch.RUN=Path(temp)
    row={'group':'probe','key':'a','audio_sha256':'audio'}
    p=batch.RUN/'scores/probe/a.json';p.parent.mkdir(parents=True)
    v={'status':'scored','binding':'model','input_sha256':'audio','meva_raw':3.}
    p.write_text(json.dumps(v));assert batch.existing(row,'model')['meva_raw']==3.
    for label,changed in [('stale_model',{'binding':'changed'}),('stale_audio',{'input_sha256':'changed'}),('nonfinite',{'meva_raw':float('nan')})]:
        p.write_text(json.dumps({**v,**changed}))
        try:batch.existing(row,'model')
        except AssertionError:results[label]='pass'
        else:raise AssertionError(label+' accepted')
    checkpoint=batch.RUN/'checkpoint';checkpoint.write_text('fixture')
    exp='probe';out=batch.RUN/'generated/probe_mc_mf25_cfg3_neg';out.mkdir(parents=True)
    report={'status':'passed','cfg_strength':3,'num_steps':25,'negative_prompt':'registered',
            'checkpoint_sha256':batch.sha(checkpoint),'seed':42,'conditioning':'no_q','text_attention_mask':False,
            'gen_tsv_sha256':'tsv','score_tsv_sha256':'tsv'}
    c={'generation':[{'experiment':exp,'checkpoint':str(checkpoint),'checkpoint_sha256':batch.sha(checkpoint),'command':['mock']}],
       'protocol':{'negative_prompt':'registered'},'musiccaps_sha256':'tsv'}
    class Complete:
        returncode=0
        def poll(self):return 0
    for label,changed in [('wrong_CFG',{'cfg_strength':0}),('wrong_seed',{'seed':7}),('wrong_mask',{'text_attention_mask':True}),('wrong_negative',{'negative_prompt':'other'}),('wrong_TSV',{'gen_tsv_sha256':'other'})]:
        (out/'probe_mc_mf25_cfg3_neg_REPORT.json').write_text(json.dumps({**report,**changed}))
        with patch.object(batch.subprocess,'Popen',return_value=Complete()),patch.object(batch,'phase_gate'):
            try:batch.generate(c)
            except AssertionError:results[label]='pass'
            else:raise AssertionError(label+' accepted')
    bound=batch.RUN/'bound';bound.write_text('changed')
    with patch.object(batch,'notify') as notifier:
        try:batch.phase_gate({'raw_bindings':[{'path':str(bound),'sha256':'wrong'}]},'fixture')
        except AssertionError:results['phase_input_drift']='pass'
        else:raise AssertionError('phase input drift accepted')
        assert not notifier.called
    class EmptyFS:f_bavail=1;f_frsize=1
    with patch.object(batch,'notify') as notifier,patch.object(batch.os,'statvfs',return_value=EmptyFS()):
        try:batch.phase_gate({'raw_bindings':[]},'fixture')
        except AssertionError:results['phase_storage_stop']='pass'
        else:raise AssertionError('phase storage stop accepted')
        assert not notifier.called
    contract=batch.RUN/'contract.json';contract.write_text(json.dumps(c))
    (batch.RUN/'manifest.json').write_text('[]')
    (batch.RUN/'probe_generated_manifest.json').write_text('[{"audio_sha256":"before"}]')
    with patch.object(batch,'C',contract),patch.object(batch,'preflight',return_value=0),\
         patch.object(batch,'generated_rows',return_value=[{'audio_sha256':'after'}]),\
         patch.object(sys,'argv',['test','--validate-only']),patch.object(batch,'summarize') as summary:
        try:batch.main()
        except AssertionError:results['postflight_generated_drift']='pass'
        else:raise AssertionError('changed generated audio accepted by postflight')
        assert not summary.called
    (batch.RUN/'coverage.json').write_text('{}')
    paired=[]
    for group,scores in [('left',[1.,1.]),('right',[3.,3.])]:
        for key,value in zip(['a','b'],scores):
            row={'group':group,'key':key,'audio_sha256':'paired','protocol':'fixture'};paired.append(row)
            path=batch.RUN/'scores'/group/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'status':'scored','binding':'model','input_sha256':'paired','meva_raw':value}))
    stats={'model_binding':'model','comparisons':[{'left':'left','right':'right','label':'fixture','n':2,'expectation':'left better'}],'limitations':[]}
    batch.summarize(paired,stats,partial=True)
    pair=json.loads((batch.RUN/'interim_existing.json').read_text())['comparisons'][0]
    assert pair['mean_delta']==-2 and pair['left_win_fraction']==0 and pair['direction']=='right higher'
    results['expected_winner_can_lose']='pass'
    for row in paired:
        path=batch.RUN/'scores'/row['group']/(row['key']+'.json');v=json.loads(path.read_text());v['meva_raw']=2.;path.write_text(json.dumps(v))
    batch.summarize(paired,stats,partial=True)
    pair=json.loads((batch.RUN/'interim_existing.json').read_text())['comparisons'][0]
    assert pair['mean_delta']==0 and pair['left_win_fraction']==.5 and pair['direction']=='inconclusive'
    results['paired_ties_half']='pass'
    batch.RUN=old
print(json.dumps({'status':'passed','fixtures':results},indent=2))
