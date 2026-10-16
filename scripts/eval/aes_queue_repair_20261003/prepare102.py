"""Append a separately registered diagnosis using the approved P2 HARN."""
import os,json,hashlib,copy
from pathlib import Path
from datetime import datetime,timezone,timedelta
R=Path('/home/kojiek/MeanAudio');scripts=Path(__file__).parent
F=Path('/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs/aes-formal-20261003')
S=F.parent/'aes-followup-20261002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    oldpath=R/'docs/experiments/aes-negmf-other-source-envfix-20261003_contract.json';old=json.loads(oldpath.read_text());oldbundle=Path(old['harn_bundle'])
    name='aes-negmf-corrected-source-20261003';bundle=R/'docs/experiments/harn/aes_diagnostic_102';bundle.mkdir(parents=True,exist_ok=True)
    path=R/'docs/experiments'/f'{name}_contract.json';assert not path.exists(), 'immutable registration already exists'
    now=datetime.now(timezone.utc).isoformat();until=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
    request=bundle/'operator_request.txt';request.write_text('檢測清楚，失敗的話就重跑；成功的話 觀看結果，並設計新的實驗 依照正式流程上架\n')
    audit=json.loads((F/'CONTROL_FAILURE_AUDIT.json').read_text());sources=[]
    for entry,cell in zip(audit,['N100_s27182818_mf25_cfg0_historical_null','control_s14159265_mf25_cfg3_fidelity8']):
        for identifier in dict.fromkeys(v[1] for v in entry['bad'][:16]):
            p=S/'cells'/cell/'audio'/(identifier+'.flac');sources.append(dict(cell=cell,id=identifier,sha256=sha(p)))
    c=copy.deepcopy(old);c.update(experiment_id=name,run_id='run-20261003-102',job='diagnostic',queue_name='102_'+name+'.sh',
        operator_instruction=request.read_text().strip(),approval_record=str(request),harn_bundle=str(bundle),
        state_root='/home/kojiek/logs/aes_negmf_diagnostic_102_harn',runtime_root=str(R/'runtime/aes_queue_repair_20261003/diagnostic102'),
        original_contract=str(oldpath),original_contract_sha256=sha(oldpath),ordering_dependencies=[],diagnostic_sources=sources,
        scientific_scope='Explicit v2 scientific normalization: fixed RMS before -23 BS1770, strict FP32 with TF32 disabled; all thresholds retained including gain control <0.01. Fresh versioned scores; old failed results preserved. Same audio/seeds/conditions/primary holdout and inferential plan. CFG0 remains historical. Source study may reuse only v2 scores with identical waveform and source SHA.')
    c['job']='source_study'; c['corrected_science_root']='/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs/aes-corrected-20261003'; c['normalization_revision']='fixed_rms_seed_then_bs1770_minus23_v2'; c['inference_revision']=dict(matmul_tf32=False,cudnn_tf32=False,aes_dtype='float32',early_gain_gate=True); c['original_preflight']=str(F/'preflight.py'); c['runtime_environment']=dict(PYTHONPATH=str(F.parents[1]/'work/structure-packages')); c['commands']['preflight']=[c['commands']['preflight'][0],str(scripts/'preflight101.py')]
    c['launch_authorization'].update(authorized_at=now,scope='Existing operator instruction to detect failures, rerun, design and formally register new experiments authorizes this explicit v2 repair contract. Fixed-RMS normalization and strict FP32 are declared revisions, never silent edits to original science. Original gain-control threshold unchanged.')
    root=Path(c['runtime_root']);root.mkdir(parents=True,exist_ok=True);c['summary']=str(root/'summary.json');c['reports']=[dict(path=c['summary'])]
    c['commands']['run']=[c['commands']['run'][0],str(scripts/'source102.py')];c['commands']['postflight']=[c['commands']['postflight'][0],str(scripts/'postflight101.py')]
    c['storage'].update(path=str(R),transient_root=str(root),estimated_peak_additional_bytes=7000000000)
    c['resource_budget']['estimated_gpu_hours_range']=[5,10]
    manifest=bundle/'resume_manifest.json';dump(manifest,dict(kind='verified_per_item_cache',science_root=c['corrected_science_root'],normalization_revision=c['normalization_revision'],cache_root=str(root),original_audio_reused=True,old_scores_not_adopted=True))
    c['resume'].update(checkpoint=str(manifest),checkpoint_sha256=sha(manifest),checkpoint_bytes=manifest.stat().st_size,autoresume=str(root/'host_resume.json'),behavior='Resume only v2 keys with source/waveform/hash/control verification. Failed v1 scores are retained in their original roots and not adopted. Official P2 seat and bound scientific revision required.')
    launcher=R/'scripts/queue_candidates'/c['queue_name'];launcher.write_text(Path(old['candidate_launcher']).read_text().replace(str(oldpath),str(path)));launcher.chmod(0o700)
    launcher.write_text(launcher.read_text().replace('export PYTHONUNBUFFERED=1','export PYTHONPATH='+c['runtime_environment']['PYTHONPATH']+'\nexport PYTHONUNBUFFERED=1')); c['candidate_launcher']=str(launcher);c['bindings'].update(launcher=str(Path('/home/kojiek/gpu_queue/p2/pending')/launcher.name),launcher_sha256=sha(launcher),action=str(scripts/'source102.py'),action_sha256=sha(scripts/'source102.py'))
    new_science_path=Path(c['corrected_science_root'])/'contract.json'
    additional=[new_science_path,scripts/'probe101.py',scripts/'analyze101.py',scripts/'resume101.py',scripts/'source102.py',scripts/'preflight101.py',Path('/home/kojiek/MeanAudio/runtime/aes_queue_repair_20261003/diagnostic100/summary.json'),oldpath,request,launcher,manifest,scripts/'source102.py',scripts/'postflight101.py',scripts/'normalize_candidate.py',F/'CONTROL_FAILURE_AUDIT.json',F/'NORMALIZATION_DIAGNOSIS.json']+[S/'cells'/x['cell']/'audio'/(x['id']+'.flac') for x in sources]
    c['inputs'] += [dict(kind='immutable_input',path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in additional];dump(path,c)
    b=json.loads((oldbundle/'contract.json').read_text());b.update(experiment_id=name,run_id=c['run_id'])
    for x in b['commands']:
        if x['action_id'] in ['launch','resume']:x['argv']=['/bin/bash',str(launcher)]
        elif x['action_id'] in c['commands']:x['argv']=c['commands'][x['action_id']]
    b['bindings']['command_set_sha256']=hashlib.sha256(json.dumps({x['action_id']:x['argv'] for x in b['commands']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for group in [b['corpus']['source_artifacts'],b['phases'][0]['input_artifacts']]:
        for a in group:
            if a['path']==str(oldpath):a.update(path=str(path),sha256=sha(path))
            elif a['path']==old['approval_record']:a.update(path=str(request),sha256=sha(request))
        group.extend(dict(path=str(p),sha256=sha(p)) for p in additional)
    b['phases'][0].update(phase_id='verified-corrected-evaluation',output_paths=[c['summary']],completion_evidence=[dict(path=str(scripts/'postflight101.py'),sha256=sha(scripts/'postflight101.py'))])
    for f in b['filesystems']:f.update(peak_additional_bytes=7000000000)
    dump(bundle/'contract.json',b);ch=sha(bundle/'contract.json')
    p=json.loads((oldbundle/'preflight.json').read_text());p.update(experiment_id=name,run_id=c['run_id'],contract_raw_sha256=ch,created_at=now)
    a=p['approval_evidence'];a.update(evidence_id='operator-diagnosis-99',channel_record_id='codex-user-diagnose-failed-rerun',channel_record_sha256=sha(request),issued_at=now,expires_at=until,experiment_id=name,run_id=c['run_id']);a['bindings'].update(contract_raw_sha256=ch,command_set_sha256=b['bindings']['command_set_sha256'])
    for check in p['checks']:check.update(observed_at=now,valid_until=until)
    for f in p['storage']:
        fs=os.statvfs(f['path']);f.update(measured_at=now,free_bytes=fs.f_bavail*fs.f_frsize,peak_additional_bytes=7000000000)
    dump(bundle/'preflight.json',p)
    l=json.loads((oldbundle/'ledger.json').read_text());l.update(experiment_id=name,run_id=c['run_id']);l['bindings'].update(contract_raw_sha256=ch,preflight_report_raw_sha256=sha(bundle/'preflight.json'));l['events']=l['events'][:1];l['events'][0].update(idempotency_key=name+':contract-register',occurred_at=now,event_sha256=hashlib.sha256(ch.encode()).hexdigest());dump(bundle/'ledger.json',l)
    q=json.loads((oldbundle/'queue.json').read_text());q.update(queue_id='p2-aes-diagnostic-102',updated_at=now);e=q['entries'][0];e.update(entry_id='102-aes-corrected',experiment_id=name,run_id=c['run_id']);e['bindings'].update(contract_raw_sha256=ch,preflight_report_raw_sha256=sha(bundle/'preflight.json'),ledger_raw_sha256=sha(bundle/'ledger.json'));dump(bundle/'queue.json',q)
    print('PREPARED',path,len(sources),'diagnostic sources')
if __name__=='__main__':main()
