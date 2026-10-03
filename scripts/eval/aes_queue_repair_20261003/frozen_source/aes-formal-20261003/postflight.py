import json
from pathlib import Path
from common import ROOT,SRC,spec,sha

def main():
 c=spec();p=Path(c['summary']);s=json.loads(p.read_text());assert s['status']=='completed'
 if c['job']=='recovery':
  assert len(s['verified_phases'])==32 and s['source_contract_sha256']==sha(SRC/'contract.json')
  for item in s['verified_phases']:assert sha(item['receipt'])==item['sha256']
  assert sha(s['analysis'])==s['analysis_sha256'] and sha(s['report'])==s['report_sha256'];a=json.loads(Path(s['analysis']).read_text());assert a['n_cells']==16 and a['n_prompts']==5009
 else:
  assert s['n_cells']==6 and s['n_prompts_requested']==5009 and s['n_cases']==20
  for r in s['cells']:
   assert sha(r['cache'])==r['cache_sha256'] and r['n_records']==r['n_eligible']*20*2 and r['gain_control_max']<.01
 assert all(Path(r['path']).is_file() and Path(r['path']).stat().st_size for r in c['reports'])
 print('POSTFLIGHT PASS',c['experiment_id']);return 0
if __name__=='__main__':raise SystemExit(main())
