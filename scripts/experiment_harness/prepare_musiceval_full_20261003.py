"""Prepare a new full MusicEval contract and schema bundle; never seats the GPU."""
import copy
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import soundfile as sf

ROOT=Path('/home/kojiek/MeanAudio')
RUN=ROOT/'runtime/musiceval_full_20261003'
B=ROOT/'docs/experiments/harn/musiceval_full_20261003'
C=ROOT/'docs/experiments/musiceval_full_20261003_contract.json'
sys.path[:0]=[str(ROOT/'scripts/eval'),str(ROOT/'scripts/experiment_harness')]
from meva_runtime import sha
from notification_receipts import atomic_secure_json as write, canonical_hash


def main():
    if C.exists():
        raise SystemExit('Contract exists; immutable after registration')
    RUN.mkdir(parents=True,exist_ok=True);B.mkdir(parents=True,exist_ok=True)
    (B/'operator_request.txt').write_text('2026-10-03 responsible operator via Codex conversation:\n測試完整的 PQCECU MEVA 在 musiceval 的分數\nAuthorized scope: full MusicEval existing-audio evaluation, fixed pretrained models, no training or adoption.\n')
    aes_path=ROOT/'research/eval/output/musiceval_mir/musiceval_aes.tsv'
    aes={r['key']:r for r in csv.DictReader(aes_path.open(),delimiter='\t')}
    label=ROOT/'research/eval/musiceval/sets/total_mos_list.txt'
    rows=[]
    for name,mi,ta in csv.reader(label.open()):
        key=Path(name).stem;p=ROOT/'research/eval/musiceval/wav'/name
        system,prompt=re.search(r'-(S\d+)_(P\d+)$',key).groups()
        rows.append(dict(key=key,system=system,prompt=prompt,MI=float(mi),TA=float(ta),
            PQ=float(aes[key]['PQ']),CE=float(aes[key]['CE']),CU=float(aes[key]['CU']),
            duration_sec=sf.info(p).duration,audio_path=str(p),audio_sha256=sha(p)))
    assert len(rows)==len(aes)==len({r['key'] for r in rows})==2748
    write(RUN/'manifest.json',rows)
    eid='musiceval-full-20261003';rid='run-20261003-musiceval100'
    guest=ROOT/'scripts/experiment_harness/musiceval_full_20261003_guest.py'
    action=ROOT/'scripts/eval/musiceval_full_20261003.py'
    launcher=RUN/'100_musiceval_full_20261003.sh'
    launcher.write_text('#!/bin/bash\n# GPU_QUEUE_CONTRACT='+str(C)+'\nexport GPU_QUEUE_JOB_SCRIPT="$0"\nexport GPU_QUEUE_CONTRACT='+str(C)+'\nexport PYTHONUNBUFFERED=1\nexport HF_HOME='+str(ROOT/'runtime/meva_20261003/hf-cache')+'\nexport HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\nexport OMP_NUM_THREADS=4\nexport MKL_NUM_THREADS=4\nexec /home/kojiek/venvs/dac/bin/python '+str(guest)+'\n')
    launcher.chmod(0o700)
    result=subprocess.run(['/home/kojiek/venvs/dac/bin/python',str(ROOT/'scripts/tests/test_musiceval_full_20261003.py')],capture_output=True,text=True,check=True)
    acceptance=json.loads(result.stdout)
    acceptance.update(source_sha256={str(p):sha(p) for p in [guest,action,ROOT/'scripts/tests/test_musiceval_full_20261003.py']},
        inherited_queue_acceptance=str(ROOT/'runtime/meva_20261003/acceptance-fixtures.json'),
        inherited_gpu_smoke=str(ROOT/'runtime/meva_20261003/smoke.json'),
        note='Unchanged queue host/events and inference adapter; new supervisor fixtures rerun with injected failures.')
    write(B/'acceptance.json',acceptance)
    old=json.loads((ROOT/'docs/experiments/meva_095_contract.json').read_text())
    c=copy.deepcopy(old)
    c.update(document_kind='eval_only_musiceval_full_contract',experiment_id=eid,run_id=rid,
        approval_record=str(B/'operator_request.txt'),harn_bundle=str(B),queue_name=launcher.name,
        queue_placement='append_tail; preserve pending/running/held entries',
        manifest=str(RUN/'manifest.json'),manifest_sha256=sha(RUN/'manifest.json'),n_expected=2748,
        question='Full MusicEval human MI correlations and same-prompt comparisons for AES PQ/CE/CU and fixed pooled-small-f03 MEva',
        protocol=dict(new_generation=False,mode='in-domain descriptive',candidate='pooled-small-f03-hook11-4096',
            labels='original five-rater mean musical impression; alignment secondary',
            aes='reuse full frozen raw predictions, S013_P013 first90s exception',
            meva='all native clips, fixed checkpoint; no training or calibration',
            pair_subset='shared2500,25systems,100prompts; human ties excluded, model ties half',
            sensitivity='exclude S013_P013 from both models',bootstrap_seed=20261003,bootstrap_repeats=2000,
            gate='coverage/hash/finite checks only; no model adoption or accuracy threshold'),
        bindings=dict(launcher=str(Path('/home/kojiek/gpu_queue/p2/pending')/launcher.name),launcher_sha256=sha(launcher),
            harn=str(guest),harn_sha256=sha(guest),action=str(action),action_sha256=sha(action)),
        commands={name:[str(ROOT/'runtime/meva_20261003/venv/bin/python'),str(action)]+extra
                  for name,extra in [('run',[]),('preflight',['--preflight']),('postflight',['--validate-only'])]},
        reports=[{'path':str(p)} for p in [ROOT/'docs/experiments/results/musiceval_full_20261003.json',
            ROOT/'docs/experiments/results/musiceval_full_20261003.md',RUN/'per_clip.tsv',RUN/'per_system.tsv']],
        completion_evidence=dict(eval_only=True,n_expected=2748,method='all unique input-bound finite scores plus deterministic full report'),
        git_revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
    c['launch_authorization']['scope']='User explicitly requested full PQ/CE/CU/MEva MusicEval scoring; existing audio only'
    c['storage'].update(path=str(RUN),estimated_peak_additional_bytes=1<<30)
    c['resource_budget'].update(additional_disk_bytes=1<<30,max_active_seconds=12*3600)
    c['resume']['pause_progress']=str(RUN/'pause_progress.json')
    c['notifications']=[e for e in c['notifications'] if e!='external-pam-gate']
    paths=[action,guest,ROOT/'scripts/eval/meva_runtime.py',ROOT/'scripts/experiment_harness/meva_095_events.py',
        ROOT/'scripts/experiment_harness/preflight_capture.py',ROOT/'docs/experiments/meva_095_model_lock.json',
        B/'operator_request.txt',label,aes_path,RUN/'manifest.json',ROOT/'research/eval/musiceval_mir_incremental.py',
        Path(c['notification_receipts']['helper']),Path(c['notification_receipts']['notifier'])]
    c['raw_bindings']=[{'path':str(p),'sha256':sha(p)} for p in paths]
    write(C,c)
    now=datetime.now(timezone.utc).isoformat();until=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
    oldb=ROOT/'docs/experiments/harn/meva_095'
    sc=copy.deepcopy(json.loads((oldb/'contract.json').read_text()))
    sc.update(experiment_id=eid,run_id=rid)
    commands=[dict(action_id=name,argv=argv,working_directory=str(ROOT),environment={'PYTHONUNBUFFERED':'1','CUDA_VISIBLE_DEVICES':'0'})
        for name,argv in [('launch',['/bin/bash',str(launcher)]),('resume',['/bin/bash',str(launcher)]),
                          ('preflight',c['commands']['preflight']),('postflight',c['commands']['postflight'])]]
    sc['commands']=commands
    sc['bindings']['runtime_sha256']=sha(guest)
    sc['bindings']['command_set_sha256']=canonical_hash({a['action_id']:a['argv'] for a in commands})
    artifacts=[{'path':str(p),'sha256':sha(p)} for p in [C,RUN/'manifest.json',B/'acceptance.json',B/'operator_request.txt',aes_path,label]]
    sc['corpus']['source_artifacts']=artifacts
    sc['phases']=[dict(phase_id='full-musiceval-comparison',action_id='launch',input_artifacts=artifacts,
        output_paths=[r['path'] for r in c['reports']],completion_evidence=[{'path':str(action),'sha256':sha(action)}],resume_action_id='resume')]
    sc['filesystems']=[dict(path=str(RUN),hard_floor_bytes=50<<30,warning_floor_bytes=80<<30,
        peak_additional_bytes=1<<30,transient_bytes=100<<20,recovery_reserve_bytes=50<<30)]
    sc['notification_events']=c['notifications'];write(B/'contract.json',sc)
    ch=sha(B/'contract.json');free=os.statvfs(RUN).f_bavail*os.statvfs(RUN).f_frsize;assert free>=50<<30
    pf=copy.deepcopy(json.loads((oldb/'preflight.json').read_text()))
    pf.update(experiment_id=eid,run_id=rid,contract_raw_sha256=ch,created_at=now)
    ap=pf['approval_evidence'];ap.update(evidence_id='approval-musiceval100-20261003',channel_record_id='codex-musiceval100-20261003',
        channel_record_sha256=sha(B/'operator_request.txt'),issued_at=now,expires_at=until,experiment_id=eid,run_id=rid)
    ap['bindings'].update(contract_raw_sha256=ch,runtime_sha256=sha(guest),command_set_sha256=sc['bindings']['command_set_sha256'])
    for check in pf['checks']:
        check.update(observed_at=now,valid_until=until,evidence_sha256=sha(B/'acceptance.json'))
    pf['storage']=[dict(path=str(RUN),measured_at=now,free_bytes=free,hard_floor_bytes=50<<30,
        peak_additional_bytes=1<<30,transient_bytes=100<<20,recovery_reserve_bytes=50<<30,verdict='pass')]
    write(B/'preflight.json',pf)
    event=dict(sequence=1,event_id='contract-register',idempotency_key=eid+':contract-register',event_kind='contract_registered',
        occurred_at=now,phase=None,verdict='none',relates_to_event_id=None,notification_status='not_applicable',previous_event_sha256=None)
    event['event_sha256']=canonical_hash(event)
    write(B/'ledger.json',dict(document_kind='event_ledger',schema_version='1.0.0',schema_bundle_id='harn-schema-v1',
        experiment_id=eid,run_id=rid,bindings=dict(contract_raw_sha256=ch,preflight_report_raw_sha256=sha(B/'preflight.json'),
        schema_bundle_sha256=sc['bindings']['schema_bundle_sha256']),events=[event]))
    write(B/'queue.json',dict(document_kind='queue_state',schema_version='1.0.0',schema_bundle_id='harn-schema-v1',
        queue_id='p2-musiceval100',updated_at=now,entries=[dict(entry_id='100-musiceval-full',position=1,experiment_id=eid,run_id=rid,
        status='ready',dependencies=[],assigned_resource=None,bindings=dict(contract_raw_sha256=ch,
        preflight_report_raw_sha256=sha(B/'preflight.json'),ledger_raw_sha256=sha(B/'ledger.json'),
        schema_bundle_sha256=sc['bindings']['schema_bundle_sha256']),terminal_notification_status='not_applicable')]))
    print('Prepared, not seated:',C)


if __name__=='__main__':main()
