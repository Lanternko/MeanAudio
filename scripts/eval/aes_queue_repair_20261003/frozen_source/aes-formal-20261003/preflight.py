"""No-GPU, fail-closed admission for two formally registered eval-only jobs."""
import sys,os,json,csv,stat,subprocess,hashlib
from pathlib import Path
from common import ROOT,SRC,REPO,sha,spec,dump

def main():
 c=spec();source=json.loads((SRC/'contract.json').read_text());assert c['launch_allowed'] and c['launch_authorization']['gpu_launch_allowed']
 assert Path(c['approval_record']).read_text()==c['operator_instruction']+'\n'
 assert source['generation_seed']==42 and source['steps']==25 and source['mask']==False
 assert source['negative_prompt']=='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
 assert len(source['probe_ids'])==5521 and len(source['discovery_ids'])==512
 assert sha(source['tsv'])==source['tsv_sha256'];rows=list(csv.DictReader(open(source['tsv']),delimiter='\t'));assert len(rows)==len({r['id'] for r in rows})==5521
 for p,h in source['code_hashes'].items():assert sha(SRC/p)==h
 for i in c['inputs']:
  p=Path(i['path']);assert p.is_file() and p.stat().st_size==i['bytes'] and sha(p)==i['sha256'],str(p)
 audit=json.loads((ROOT/'AUDIT.json').read_text());assert audit['n_completed']==sum(r['status']=='verified_complete' for r in audit['cell_audit']) and audit['n_completed']>=10
 assert json.loads((SRC/'raw_audit.json').read_text())['audio_files_verified']==88336
 assert json.loads((ROOT/'acceptance.json').read_text())['status']=='passed'
 for pid in json.loads((SRC/'STATE.json').read_text()).get('controller_pid'),json.loads((SRC/'STATE.json').read_text()).get('child_pid'):
  p=Path(f'/proc/{pid}/cmdline')
  if p.exists() and b'aes-followup-20261002' in p.read_bytes():raise ValueError('old private compute still active')
 if c['job']=='source_study':
  expected=[x for x in source['cells'] if x['cfg_strength']==3 and x['family'] in ['N100','control']]
  assert len(c['cells'])==6 and c['cells']==expected and len(c['analysis_ids'])==5009
  assert not set(c['analysis_ids'])&set(source['discovery_ids']) and c['n_cases']==20
 for path in c['resource_budget']['writable_filesystems']:
  fs=os.statvfs(path);free=fs.f_bavail*fs.f_frsize
  if free<c['storage']['hard_stop_free_bytes']+c['storage']['estimated_peak_additional_bytes']:return 75
 secret=Path('/home/kojiek/.config/meanaudio/discord_webhook_url').stat();assert stat.S_IMODE(secret.st_mode)==0o600 and secret.st_uid==os.getuid()
 b=Path(c['harn_bundle']);args=['/usr/bin/python3',str(REPO/'scripts/validate_experiment_harness_documents.py')]
 for name in ['contract','preflight','ledger','queue']:args +=['--'+name,str(b/(name+'.json'))]
 subprocess.run(args,check=True)
 bsrc={r['path']:r['sha256'] for r in json.loads((b/'contract.json').read_text())['corpus']['source_artifacts']};assert bsrc[os.environ['GPU_QUEUE_CONTRACT']]==sha(os.environ['GPU_QUEUE_CONTRACT'])
 if '--allow-unregistered' not in sys.argv:
  sys.path.insert(0,str(REPO/'scripts/experiment_harness'));from notification_receipts import validate_delivered_receipt
  reg=json.loads((b/'queue_registration.json').read_text());ok,reason=validate_delivered_receipt(Path(reg['receipt']),contract_path=Path(os.environ['GPU_QUEUE_CONTRACT']),launcher_path=Path(c['candidate_launcher']),event='queue_registration',status='start');assert ok,reason
 print('PREFLIGHT PASS',c['experiment_id'],'no GPU launched',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
