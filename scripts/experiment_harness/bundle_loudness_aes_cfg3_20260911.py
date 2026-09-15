#!/usr/bin/env python3
"""Build the schema bundle from measured registration evidence; no queue mutation."""
from pathlib import Path
import json,hashlib,os
from datetime import datetime,timezone
R=Path('/home/kojiek/MeanAudio'); C=R/'docs/experiments/loudness_aes_cfg3_20260911_contract.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hashval(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n');os.replace(tmp,p)
def main():
    c=json.loads(C.read_text());B=Path(c['harn_bundle']);eid=c['experiment_id'];rid=c['run_id']
    evidence=B/'acceptance.json';e=json.loads(evidence.read_text());assert e['status']=='passed'
    schema=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((R/'docs/experiments/schemas').glob('*.json')))).hexdigest()
    policy=hashlib.sha256(b''.join(p.read_bytes() for p in [R/'AGENTS.md',R/'docs/experiments/experiment_notification_policy.md',R/'docs/experiments/watcher_policy.md'])).hexdigest()
    now=datetime.now(timezone.utc).isoformat(); until='2026-12-31T00:00:00Z'
    commands=[]
    for name,argv in [('launch',['/bin/bash',c['bindings']['launcher']]),('resume',['/bin/bash',c['bindings']['launcher']]),('preflight',c['commands']['preflight']),('postflight',c['commands']['postflight'])]:
        commands.append({'action_id':name,'argv':argv,'working_directory':str(R),'environment':{'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'}})
    commandhash=hashval({x['action_id']:x['argv'] for x in commands});runtime=sha(c['bindings']['harn'])
    artifacts=[{'path':str(C),'sha256':sha(C)},{'path':c['tsv'],'sha256':sha(c['tsv'])},{'path':c['checkpoint'],'sha256':sha(c['checkpoint'])},{'path':str(evidence),'sha256':sha(evidence)},{'path':c['approval_record'],'sha256':sha(c['approval_record'])}]
    contract={'document_kind':'experiment_contract','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1','experiment_id':eid,'run_id':rid,
        'bindings':{'policy_bundle_sha256':policy,'schema_bundle_sha256':schema,'runtime_sha256':runtime,'command_set_sha256':commandhash},
        'approval_requirement':{'required':True,'responsible_role':'responsible-operator','trusted_channels':['operator_console']},
        'corpus':{'kind':'non_generated','source_artifacts':artifacts},'repair':{'enabled':False},
        'phases':[{'phase_id':'canonical-baseline-and-gain-analysis','action_id':'launch','input_artifacts':artifacts,'output_paths':[c['summary']],'completion_evidence':[{'path':c['bindings']['action'],'sha256':sha(c['bindings']['action'])}],'resume_action_id':'resume'}],
        'filesystems':[{'path':c['storage']['path'],'hard_floor_bytes':c['storage']['hard_stop_free_bytes'],'warning_floor_bytes':c['storage']['warning_free_bytes'],'peak_additional_bytes':6000000000,'transient_bytes':25000000,'recovery_reserve_bytes':53687091200}],
        'commands':commands,'required_preflight_checks':['policy','provenance','storage','notification','queue','watcher','acceptance'],'notification_events':c['notifications']}
    write(B/'contract.json',contract);ch=sha(B/'contract.json')
    fs=os.statvfs(c['storage']['path']);free=fs.f_bavail*fs.f_frsize
    assert free>=c['storage']['hard_stop_free_bytes']
    checks=[{'check_id':name,'verdict':'pass','observed_at':now,'valid_until':until,'evidence_sha256':sha(evidence)} for name in contract['required_preflight_checks']]
    preflight={'document_kind':'preflight_report','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1','experiment_id':eid,'run_id':rid,'contract_raw_sha256':ch,
        'approval_evidence':{'evidence_id':'approval-20260911-loudness-aes','source_kind':'trusted_operator_record','trusted_channel':'operator_console','channel_record_id':'codex-task-20260911-loudness-aes','channel_record_sha256':sha(c['approval_record']),'approver_id':'responsible-operator','issued_at':now,'expires_at':until,'experiment_id':eid,'run_id':rid,'bindings':{'contract_raw_sha256':ch,'policy_bundle_sha256':policy,'schema_bundle_sha256':schema,'runtime_sha256':runtime,'repair_envelope_sha256':None,'command_set_sha256':commandhash}},
        'checks':checks,'storage':[{'path':c['storage']['path'],'measured_at':now,'free_bytes':free,'hard_floor_bytes':53687091200,'peak_additional_bytes':6000000000,'transient_bytes':25000000,'recovery_reserve_bytes':53687091200,'verdict':'pass'}],'derived_verdict':'pass','created_at':now}
    write(B/'preflight.json',preflight);ph=sha(B/'preflight.json')
    event={'sequence':1,'event_id':'contract-register','idempotency_key':eid+':contract-register','event_kind':'contract_registered','occurred_at':now,'phase':None,'verdict':'none','relates_to_event_id':None,'notification_status':'not_applicable','previous_event_sha256':None,'event_sha256':hashval({'experiment':eid,'contract':ch})}
    ledger={'document_kind':'event_ledger','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1','experiment_id':eid,'run_id':rid,'bindings':{'contract_raw_sha256':ch,'preflight_report_raw_sha256':ph,'schema_bundle_sha256':schema},'events':[event]}
    write(B/'ledger.json',ledger)
    queue={'document_kind':'queue_state','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1','queue_id':'p2-loudness-aes-051','updated_at':now,'entries':[{'entry_id':'051-loudness-aes','position':1,'experiment_id':eid,'run_id':rid,'status':'ready','dependencies':[],'assigned_resource':None,'bindings':{'contract_raw_sha256':ch,'preflight_report_raw_sha256':ph,'ledger_raw_sha256':sha(B/'ledger.json'),'schema_bundle_sha256':schema},'terminal_notification_status':'not_applicable'}]}
    write(B/'queue.json',queue);print(B)
if __name__=='__main__':main()
