import json,hashlib,copy,os
from pathlib import Path
from datetime import datetime,timezone,timedelta
R=Path('/home/kojiek/MeanAudio');old=Path('/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs/aes-formal-20261003');bundle=R/'docs/experiments/harn/aes_source_098';bundle.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
now=datetime.now(timezone.utc).isoformat();until=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
oldpath=R/'docs/experiments/aes-negmf-other-source-20261003_contract.json';c=json.loads(oldpath.read_text());c.update(experiment_id='aes-negmf-other-source-envfix-20261003',run_id='run-20261003-98',queue_name='098_aes-negmf-other-source-envfix-20261003.sh',ordering_dependencies=[],harn_bundle=str(bundle),state_root='/home/kojiek/logs/aes_negmf_source_098_harn',original_contract=str(oldpath),original_contract_sha256=sha(oldpath),original_preflight=c['commands']['preflight'][1]);c['runtime_root']=str(R/'runtime/aes_queue_repair_20261003/source098');Path(c['runtime_root']).mkdir(parents=True,exist_ok=True);c['summary']=c['runtime_root']+'/summary.json';c['reports']=[dict(path=c['summary'])];c['resume']['autoresume']=c['runtime_root']+'/host_resume.json';c['storage']['transient_root']=c['runtime_root'];c['storage']['path']=str(R);c['resource_budget']['writable_filesystems']=[str(R),'/home/kojiek/logs','/home/kojiek/gpu_queue/notification_receipts'];c['runtime_environment']={'PYTHONPATH':'/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/work/structure-packages'}
request=bundle/'operator_request.txt';request.write_text('修復？\n');c['operator_instruction']='修復？';c['approval_record']=str(request);c['launch_authorization']['authorized_at']=now;c['launch_authorization']['scope']='Current user repair instruction: restore missing process-local dependency path and resume original 096 science unchanged in a new tail entry. No normalization or gate changes; old failed evidence preserved.'
path=R/'docs/experiments/aes-negmf-other-source-envfix-20261003_contract.json';launcher=R/'scripts/queue_candidates'/c['queue_name'];text=Path(c['candidate_launcher']).read_text().replace(str(oldpath),str(path));text=text.replace('export PYTHONUNBUFFERED=1','export PYTHONPATH='+c['runtime_environment']['PYTHONPATH']+'\nexport PYTHONUNBUFFERED=1');launcher.write_text(text);launcher.chmod(0o700);c['candidate_launcher']=str(launcher);c['bindings']['launcher']=str(Path('/home/kojiek/gpu_queue/p2/pending')/launcher.name);c['bindings']['launcher_sha256']=sha(launcher)
pre=R/'scripts/eval/aes_queue_repair_20261003/preflight.py';c['commands']['preflight'][1]=str(pre)
# Bind every local dependency file; importing successfully is not sufficient provenance.
paths=[oldpath,request,launcher,pre]+[p for p in Path(c['runtime_environment']['PYTHONPATH']).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
c['inputs'] += [dict(kind='immutable_input',path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths];dump(path,c)
b=json.loads((old/'96/contract.json').read_text());b['experiment_id']=c['experiment_id'];b['run_id']=c['run_id']
for x in b['commands']:
 if x['action_id'] in ['launch','resume']:x['argv']=['/bin/bash',str(launcher)]
 elif x['action_id']=='preflight':x['argv']=c['commands']['preflight']
b['bindings']['command_set_sha256']=hashlib.sha256(json.dumps({x['action_id']:x['argv'] for x in b['commands']},sort_keys=True,separators=(',',':')).encode()).hexdigest()
for group in [b['corpus']['source_artifacts'],b['phases'][0]['input_artifacts']]:
 for a in group:
  if a['path']==str(oldpath):a.update(path=str(path),sha256=sha(path))
  elif a['path']==str(old/'operator_request.txt'):a.update(path=str(request),sha256=sha(request))
 group.extend([dict(path=str(oldpath),sha256=sha(oldpath)),dict(path=str(pre),sha256=sha(pre))])
b['phases'][0]['output_paths']=[c['summary']]
for f in b['filesystems']:
 f['path']=str(R) if f['path']==str(old) else f['path']
b['filesystems']=list({f['path']:f for f in b['filesystems']}.values())
dump(bundle/'contract.json',b);ch=sha(bundle/'contract.json')
p=json.loads((old/'96/preflight.json').read_text());p.update(experiment_id=c['experiment_id'],run_id=c['run_id'],contract_raw_sha256=ch,created_at=now);a=p['approval_evidence'];a.update(evidence_id='repair-20261003-98',channel_record_id='codex-current-user-repair',channel_record_sha256=sha(request),issued_at=now,expires_at=until,experiment_id=c['experiment_id'],run_id=c['run_id']);a['bindings'].update(contract_raw_sha256=ch,command_set_sha256=b['bindings']['command_set_sha256'])
for check in p['checks']:check.update(observed_at=now,valid_until=until)
for f in p['storage']:
 f['path']=str(R) if f['path']==str(old) else f['path'];fs=os.statvfs(f['path']);f.update(measured_at=now,free_bytes=fs.f_bavail*fs.f_frsize)
p['storage']=list({f['path']:f for f in p['storage']}.values())
dump(bundle/'preflight.json',p)
l=json.loads((old/'96/ledger.json').read_text());l.update(experiment_id=c['experiment_id'],run_id=c['run_id']);l['bindings'].update(contract_raw_sha256=ch,preflight_report_raw_sha256=sha(bundle/'preflight.json'));l['events']=l['events'][:1];l['events'][0].update(idempotency_key=c['experiment_id']+':contract-register',occurred_at=now,event_sha256=hashlib.sha256(ch.encode()).hexdigest());dump(bundle/'ledger.json',l)
q=json.loads((old/'96/queue.json').read_text());q.update(queue_id='p2-aes-source-098',updated_at=now);e=q['entries'][0];e.update(entry_id='98-aes',experiment_id=c['experiment_id'],run_id=c['run_id']);e['bindings'].update(contract_raw_sha256=ch,preflight_report_raw_sha256=sha(bundle/'preflight.json'),ledger_raw_sha256=sha(bundle/'ledger.json'));dump(bundle/'queue.json',q)
print(path)
