import os,json,csv,hashlib
from pathlib import Path
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent;SRC=ROOT.parent/'aes-followup-20261002'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((SRC/'contract.json').read_text());results=[];probes={};ids=c['discovery_ids'];byid={i:j for j,i in enumerate(ids)};cases=['clean','white40','pink40','quiet_m6','quiet_m12','dark_m6','stem_recompose','stem_other_off','stem_vocals_off','stem_drums_off'];lookup={(x['family'],x['cfg_strength'],x['training_seed']):x['id'] for x in c['cells']}
 for cell in c['cells']:
  out=SRC/'cells'/cell['id'];receipt=out/'recovery_complete.json';cache=out/'probe.jsonl';status='incomplete';nvalid=0
  if receipt.exists():
   r=json.loads(receipt.read_text());raw=cache.read_bytes();assert sha(SRC/'contract.json')==r['source_contract_sha256'];assert hashlib.sha256(raw[:r['cache_bytes']]).hexdigest()==r['cache_sha256'];assert r['n_scores']==r['n_prompts_eligible']*28*2;assert r['gain_control_max_all_axes']<.01;status='verified_complete';nvalid=r['n_scores']
   scores={v['key']:v for v in map(json.loads,raw.decode().splitlines())};expected={f'{i}__{case}__{mode}' for i in ids for case in cases for mode in ['raw','lufs23']};assert expected<=scores.keys();probes[cell['id']]=scores
  else:
   # Reject incomplete JSONL writes; retain forensic tail and remove only own partial line.
   raw=cache.read_bytes() if cache.exists() else b''
   if raw and not raw.endswith(b'\n'):
    try:json.loads(raw.splitlines()[-1]);raw+=b'\n'
    except json.JSONDecodeError:
     split=raw.rfind(b'\n')+1;(ROOT/(cell['id']+'_truncated_tail.bin')).write_bytes(raw[split:]);raw=raw[:split]
    cache.write_bytes(raw)
   for line in raw.splitlines():
    r=json.loads(line);source=out/'audio'/(r['id']+'.flac');assert sha(source)==r['source_sha256'];nvalid+=1
  results.append(dict(cell=cell['id'],status=status,n_records=nvalid,cache_sha256=sha(cache)))
 groups=[]
 for cfg in [3,0]:
  pairs=[(lookup['N100',cfg,s],lookup['control',cfg,s]) for s in [14159265,16180339,27182818]]
  if not all(a in probes and b in probes for a,b in pairs):continue
  for case in cases:
   for mode in ['raw','lufs23']:
    for axis in ['PQ','CE','CU','PC','clap']:
     vals=[];baseline='stem_recompose' if case.startswith('stem_') and case!='stem_recompose' else 'clean'
     for i in ids:
      differences=[];gaps=[]
      for a,b in pairs:
       aa=probes[a][f'{i}__{case}__{mode}'];bb=probes[b][f'{i}__{case}__{mode}'];ab=probes[a][f'{i}__{baseline}__{mode}'];cb=probes[b][f'{i}__{baseline}__{mode}']
       if any(r[axis] is None for r in [aa,bb,ab,cb]):break
       gaps.append(aa[axis]-bb[axis]);differences.append(aa[axis]-ab[axis]-bb[axis]+cb[axis])
      if len(gaps)==3:vals.append((np.mean(gaps),np.mean(differences)))
     for j,kind in enumerate(['model_gap','gap_change']):
      v=np.array([x[j] for x in vals]);n=len(v)
      if n<2:continue
      half=float(stats.t.ppf(.975,n-1)*stats.sem(v));groups.append(dict(cfg=cfg,case=case,mode=mode,axis=axis,kind=kind,n_prompts=n,n_seeds=3,mean=float(v.mean()),ci_low=float(v.mean()-half),ci_high=float(v.mean()+half)))
 (ROOT/'AUDIT.json').write_text(json.dumps(dict(source_contract_sha256=sha(SRC/'contract.json'),cell_audit=results,results=groups,n_completed=sum(r['status']=='verified_complete' for r in results)),indent=2)+'\n')
 lines=['# Formal recovery audit and available results','','10 complete guided512-prompt cells independently checked; incomplete records are not called completed. Guided N100-control effects average all3 matched training seeds, then prompts. Intervals below are exploratory t intervals on the original discovery prompts; no confirmatory holdout claim.','', '| Kind | CFG | Case | Mode | Axis | Mean | 95% exploratory CI |','|---|---:|---|---|---|---:|---|']
 for r in groups:
  if r['mode']=='lufs23' and r['axis'] in ['PQ','clap'] and (r['kind']=='gap_change' or r['case']=='clean'):lines.append(f'| {r["kind"]} | {r["cfg"]} | {r["case"]} | {r["mode"]} | {r["axis"]} | {r["mean"]:+.5f} | [{r["ci_low"]:+.5f}, {r["ci_high"]:+.5f}] |')
 (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
