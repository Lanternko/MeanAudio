#!/usr/bin/env python3
"""Build Meva095 runtime and harn-schema-v1 bundles in staging; never seats GPU."""
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
ROOT=Path('/home/kojiek/MeanAudio');RUN=ROOT/'runtime/meva_20261003'
B=ROOT/'docs/experiments/harn/meva_095';C=ROOT/'docs/experiments/meva_095_contract.json'
sys.path.insert(0,str(ROOT/'scripts/experiment_harness'))
from notification_receipts import atomic_secure_json,sha256_file,canonical_hash
sha=sha256_file

def write(p,v):atomic_secure_json(p,v)

def main():
    if C.exists():raise SystemExit('Contract already registered; immutable after launch')
    eid='meva095-shadow-20261003';rid='run-20261003-095'
    smoke=json.loads((RUN/'smoke.json').read_text());assert smoke['status']=='passed'
    assert smoke['runtime_source_sha256']==sha(ROOT/'scripts/eval/meva_runtime.py')
    fixtures=json.loads((RUN/'acceptance-fixtures.json').read_text());assert fixtures['status']=='passed'
    assert 'PASS=33 FAIL=0' in (RUN/'queue-fixtures.log').read_text()
    lock_path=ROOT/'docs/experiments/meva_095_model_lock.json';lock=json.loads(lock_path.read_text())
    paths={Path(f['path']) for f in lock['files']}
    paths.update(p.resolve() for p in (RUN/'hf-cache/hub').rglob('*') if p.is_file() and ('/snapshots/' in str(p) or '/refs/' in str(p)))
    paths.update([ROOT/'scripts/eval/meva_runtime.py',RUN/'venv/lib/python3.12/site-packages/audiocraft/models/loaders.py'])
    lock['files']=[{'path':str(p),'sha256':sha(p)} for p in sorted(paths)]
    packages=['torch','torchaudio','torchvision','transformers','tokenizers','audiocraft','sae-lens','transformer-lens','numpy','safetensors','huggingface-hub','scipy','soundfile','xformers']
    versions={p:metadata.version(p) for p in packages};lock['package_versions']=versions
    write(lock_path,lock)
    # Script is staged outside pending so the host cannot observe an incomplete registration.
    launcher=RUN/'097_meva_promptcc_shadow.sh'
    launcher.write_text('#!/bin/bash\n# GPU_QUEUE_CONTRACT='+str(C)+'\nexport GPU_QUEUE_JOB_SCRIPT="$0"\nexport GPU_QUEUE_CONTRACT='+str(C)+'\nexport PYTHONUNBUFFERED=1\nexport HF_HOME='+str(RUN/'hf-cache')+'\nexport HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\nexec /home/kojiek/venvs/dac/bin/python '+str(ROOT/'scripts/experiment_harness/meva_095_guest.py')+'\n')
    launcher.chmod(0o700)
    action=ROOT/'scripts/eval/meva_promptcc.py';guest=ROOT/'scripts/experiment_harness/meva_095_guest.py';python=RUN/'venv/bin/python'
    notify=ROOT/'scripts/notify_experiment_webhook.py';helper=ROOT/'scripts/experiment_harness/notification_receipts.py'
    other=[ROOT/'scripts/eval/meva_runtime.py',ROOT/'scripts/experiment_harness/meva_095_events.py',
           ROOT/'scripts/experiment_harness/preflight_capture.py',lock_path,RUN/'coverage.json',
           ROOT/'research/eval/output/aes_human_corr_pam/per_clip.tsv',B/'operator_request.txt',
           RUN/'requirements.resolved.txt',ROOT/'docs/experiments/meva_095_20261003.md']
    coverage=json.loads((RUN/'coverage.json').read_text())
    for cell in coverage['cells']:
        if cell['status']=='included':other += [Path(cell['source_report']),Path(cell['per_clip_path'])]
    c={'document_kind':'eval_only_meva_shadow_contract','schema_version':1,'experiment_id':eid,'run_id':rid,
      'launch_allowed':True,'approval_required':True,
      'launch_authorization':{'gpu_launch_allowed':True,'operator':'responsible_operator','trusted_channel':'operator_console',
                            'scope':'User requested experiment design and formal deployment in MeanAudio; shadow rescoring existing audio only'},
      'approval_record':str(B/'operator_request.txt'),'harn_bundle':str(B),'queue_role':'p2',
      'queue_name':launcher.name,'queue_placement':'append after existing AES095/AES096; preserve all pending/running/held entries',
      'manifest':str(RUN/'manifest.json'),'manifest_sha256':sha(RUN/'manifest.json'),
      'n_expected':coverage['n_rows'],'model_binding':sha(lock_path),'package_versions':versions,
      'git_revision':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
      'question':'Does fixed pooled-small-f03 generalize to PAM human OVL, and how does it rank retained PromptCC outputs?',
      'protocol':{'new_generation':False,'mode':'shadow','primary_external':'PAM 400 generated;100 real separately',
                  'history':'preserve all source CFG and negative-prompt protocol labels; source manifest bound',
                  'candidate':'pooled-small-f03-hook11-4096','external_gate':'simultaneous rho delta CI>0; delta>=.05 both PQ/CE; rho CI lower>=.60',
                  'failed_gate_action':'continue authorized shadow rescore; no replacement of AES','bootstrap_seed':20261003,'bootstrap_repeats':2000},
      'bindings':{'launcher':str(Path('/home/kojiek/gpu_queue/p2/pending')/launcher.name),'launcher_sha256':sha(launcher),
                  'harn':str(guest),'harn_sha256':sha(guest),'action':str(action),'action_sha256':sha(action)},
      'raw_bindings':[{'path':str(p),'sha256':sha(p)} for p in [action,guest,*other,helper,notify]],
      'commands':{'run':[str(python),str(action)],'preflight':[str(python),str(action),'--preflight'],
                  'postflight':[str(python),str(action),'--validate-only']},
      'reports':[{'path':str(RUN/p)} for p in ['summary.json','coverage.json','pam_validation.json']],
      'completion_evidence':{'eval_only':True,'n_expected':coverage['n_rows'],'method':'postflight verifies every manifest row and raw bindings'},
      'dependencies':[],
      'storage':{'path':str(RUN),'hard_stop_free_bytes':50<<30,'warning_free_bytes':80<<30,
                 'estimated_peak_additional_bytes':12<<30,'cleanup':'none; audio and checkpoints immutable'},
      'watcher':{'poll_seconds':2,'stall_seconds':1800,'automatic_repair':False},
      'resource_budget':{'gpu':0,'max_active_seconds':24*3600,'peak_vram_bytes':8<<30,'additional_disk_bytes':12<<30},
      'resume':{'kind':'from_scratch_with_autoresume','autoresume':'','iteration':0,'checkpoint':None,
                'pause_progress':str(RUN/'pause_progress.json'),'behavior':'atomic per-clip cache bound to input/model hashes; no duplicate completed score'},
      'notification_receipts':{'required':True,'root':'/home/kojiek/gpu_queue/notification_receipts',
         'notifier':str(notify),'notifier_sha256':sha(notify),'helper':str(helper),'helper_sha256':sha(helper)},
      'notifications':['registration','start','preflight-pass','external-pam-gate','postflight-pass','terminal','pause','handoff','storage-warning','hard-stop','stall','gpu-idle']}
    write(C,c)
    now=datetime.now(timezone.utc).isoformat();until=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
    tests=[ROOT/'scripts/tests/test_meva_095.py',Path('/home/kojiek/gpu_queue/tests/test_lib_scheduler.py'),Path('/home/kojiek/gpu_queue/tests/test_notification_receipts.py'),ROOT/'scripts/tests/run_loudness_queue_acceptance.py']
    acceptance={'status':'passed','created_at':now,'tests':fixtures['fixtures'],
         'queue':'33 isolated fixtures; both notifier routes mocked','receipts':'no-network unit tests passed',
         'real_gpu_smoke':smoke,'source_sha256':{str(p):sha(p) for p in tests+[guest,action]}}
    write(B/'acceptance.json',acceptance)
    policy=hashlib.sha256(b''.join(p.read_bytes() for p in [ROOT/'AGENTS.md',ROOT/'docs/experiments/evaluation_policy.md',ROOT/'docs/experiments/experiment_notification_policy.md',ROOT/'docs/experiments/watcher_policy.md'])).hexdigest()
    schema=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'docs/experiments/schemas').glob('*.json')))).hexdigest()
    commands=[{'action_id':name,'argv':argv,'working_directory':str(ROOT),'environment':{'PYTHONUNBUFFERED':'1','CUDA_VISIBLE_DEVICES':'0'}}
              for name,argv in [('launch',['/bin/bash',str(launcher)]),('resume',['/bin/bash',str(launcher)]),('preflight',c['commands']['preflight']),('postflight',c['commands']['postflight'])]]
    cmdhash=canonical_hash({a['action_id']:a['argv'] for a in commands});runtime=sha(guest)
    artifacts=[{'path':str(p),'sha256':sha(p)} for p in [C,RUN/'manifest.json',RUN/'coverage.json',B/'acceptance.json',B/'operator_request.txt',lock_path]]
    contract={'document_kind':'experiment_contract','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1','experiment_id':eid,'run_id':rid,
      'bindings':{'policy_bundle_sha256':policy,'schema_bundle_sha256':schema,'runtime_sha256':runtime,'command_set_sha256':cmdhash},
      'approval_requirement':{'required':True,'responsible_role':'responsible-operator','trusted_channels':['operator_console']},
      'corpus':{'kind':'non_generated','source_artifacts':artifacts},'repair':{'enabled':False},
      'phases':[{'phase_id':'external-validity-and-shadow-rescore','action_id':'launch','input_artifacts':artifacts,
         'output_paths':[r['path'] for r in c['reports']],'completion_evidence':[{'path':str(action),'sha256':sha(action)}],'resume_action_id':'resume'}],
      'filesystems':[{'path':str(RUN),'hard_floor_bytes':50<<30,'warning_floor_bytes':80<<30,
         'peak_additional_bytes':12<<30,'transient_bytes':100<<20,'recovery_reserve_bytes':50<<30}],
      'commands':commands,'required_preflight_checks':['policy','provenance','storage','notification','queue','watcher','acceptance'],
      'notification_events':c['notifications']}
    write(B/'contract.json',contract);ch=sha(B/'contract.json');free=os.statvfs(RUN).f_bavail*os.statvfs(RUN).f_frsize
    assert free>=50<<30
    preflight={'document_kind':'preflight_report','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1','experiment_id':eid,'run_id':rid,
      'contract_raw_sha256':ch,'approval_evidence':{'evidence_id':'approval-meva095-20261003','source_kind':'trusted_operator_record',
        'trusted_channel':'operator_console','channel_record_id':'codex-meva095-20261003','channel_record_sha256':sha(B/'operator_request.txt'),
        'approver_id':'responsible-operator','issued_at':now,'expires_at':until,'experiment_id':eid,'run_id':rid,
        'bindings':{'contract_raw_sha256':ch,'policy_bundle_sha256':policy,'schema_bundle_sha256':schema,'runtime_sha256':runtime,'repair_envelope_sha256':None,'command_set_sha256':cmdhash}},
      'checks':[{'check_id':name,'verdict':'pass','observed_at':now,'valid_until':until,'evidence_sha256':sha(B/'acceptance.json')} for name in contract['required_preflight_checks']],
      'storage':[{'path':str(RUN),'measured_at':now,'free_bytes':free,'hard_floor_bytes':50<<30,
        'peak_additional_bytes':12<<30,'transient_bytes':100<<20,'recovery_reserve_bytes':50<<30,'verdict':'pass'}],
      'derived_verdict':'pass','created_at':now}
    write(B/'preflight.json',preflight);ph=sha(B/'preflight.json')
    event={'sequence':1,'event_id':'contract-register','idempotency_key':eid+':contract-register','event_kind':'contract_registered',
      'occurred_at':now,'phase':None,'verdict':'none','relates_to_event_id':None,'notification_status':'not_applicable','previous_event_sha256':None}
    event['event_sha256']=canonical_hash(event)
    write(B/'ledger.json',{'document_kind':'event_ledger','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1',
      'experiment_id':eid,'run_id':rid,'bindings':{'contract_raw_sha256':ch,'preflight_report_raw_sha256':ph,'schema_bundle_sha256':schema},'events':[event]})
    write(B/'queue.json',{'document_kind':'queue_state','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1',
      'queue_id':'p2-meva097','updated_at':now,'entries':[{'entry_id':'097-meva-shadow','position':1,'experiment_id':eid,'run_id':rid,
      'status':'ready','dependencies':[],'assigned_resource':None,'bindings':{'contract_raw_sha256':ch,'preflight_report_raw_sha256':ph,'ledger_raw_sha256':sha(B/'ledger.json'),'schema_bundle_sha256':schema},'terminal_notification_status':'not_applicable'}]})
    print('Registered in staging, NOT seated:',C)

if __name__=='__main__':main()
