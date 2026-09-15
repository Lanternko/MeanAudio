#!/usr/bin/env python3
"""No-GPU regression checks for the preregistered vocal-negative ablation."""
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[2]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m=load('vocal',R/'scripts/eval/vocal_negative_cfg3_20260908.py')
c=json.loads(m.CONTRACT.read_text()); ids={r['id'] for r in m.rows()}
assert len(ids)==5521 and len(m.NEGATIVES)==5
s=json.loads((m.OUT/'subsets.json').read_text())['groups']
assert set(s['caption_no_vocal']).isdisjoint(s['caption_vocal'])
assert set(s['caption_no_vocal'])|set(s['caption_vocal'])==ids
for key, text in m.NEGATIVES.items():
    argv=c['commands_generation'][key]
    assert argv[argv.index('--negative_prompt')+1]==text
    assert argv[argv.index('--cfg_strength')+1]=='3'
    assert argv[argv.index('--seed')+1]=='42'
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'report.json'
    assert not m.valid_report(p,'fidelity8',ids)
    per={i:{k:1. for k in m.METRICS} for i in ids}
    payload={'label':m.label('fidelity8'),'exp_id':m.EXP_ID,'provenance':m.provenance(),'cfg_strength':3.,'negative_key':'fidelity8','negative_prompt':m.NEGATIVES['fidelity8'],'protocol_id':'musiccaps5521_mf25_cfg3_fidelity8_vocal_ablation_seed42_nomask_fp32','per_clip':per,'aggregates':{'full':{'n':5521,**{k:1. for k in m.METRICS}}}}
    p.write_text(json.dumps(payload)); assert m.valid_report(p,'fidelity8',ids)
    payload['provenance']={};p.write_text(json.dumps(payload));assert not m.valid_report(p,'fidelity8',ids)
    payload['provenance']=m.provenance();payload['per_clip'].pop(next(iter(ids)));p.write_text(json.dumps(payload));assert not m.valid_report(p,'fidelity8',ids)
    bad=Path(d)/'bad';bad.mkdir();(bad/'unrelated.txt').write_text('preserve')
    try: m.clear_partial(bad)
    except SystemExit: pass
    else: raise AssertionError('unsafe cleanup accepted')
    assert (bad/'unrelated.txt').exists()
print('PASS: fixed partition, five exact commands, complete report resume, missing/stale report rejection, safe cleanup')
v=load('vocal_preflight',R/'scripts/eval/validate_vocal_negative_cfg3_20260908.py')
class EmptyFS: f_bavail=0; f_frsize=4096
with patch.object(v.os,'statvfs',return_value=EmptyFS()):
    try: v.main()
    except AssertionError as exc: assert 'storage' in str(exc)
    else: raise AssertionError('disk hard stop failed open')
with tempfile.TemporaryDirectory() as d, patch.object(m,'OUT',Path(d)), patch.object(m,'provenance',return_value={'fixture':True}):
    small=[str(i) for i in range(20)]
    (Path(d)/'subsets.json').write_text(json.dumps({'groups':{'full':small,'caption_no_vocal':small[:10],'caption_vocal':small[10:]}}))
    for key in m.ORDER:
        p={i:{metric:1.+(.1 if key!='fidelity8' else 0) for metric in m.METRICS} for i in small}
        m.report(key).write_text(json.dumps({'per_clip':p}))
    m.write_summary()
    result=json.loads((Path(d)/'summary.json').read_text())
    assert set(result['decision'].values())=={'metric_improvement'}
    for key in m.ORDER[1:]:
        p={i:{metric:(0.5 if metric=='clap' else 1.1) for metric in m.METRICS} for i in small}
        m.report(key).write_text(json.dumps({'per_clip':p}))
    m.write_summary()
    assert set(json.loads((Path(d)/'summary.json').read_text())['decision'].values())=={'not_demonstrated'}
print('PASS: disk hard-stop fail-closed; paired bootstrap improvement and CLAP-regression rejection')
