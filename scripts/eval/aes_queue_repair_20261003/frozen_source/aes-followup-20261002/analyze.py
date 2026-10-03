"""Complete paired prompt analysis; no survivor-only completion claims."""
import json,csv,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
from generate import ROOT,dump,sha
AXES=['PQ','CE','CU','PC','clap']
def ci(v,bootstrap=False):
 v=np.asarray(v,float);v=v[np.isfinite(v)];n=len(v)
 if n<2:return dict(n_prompts=n,mean=float(v.mean()) if n else None,ci_low=None,ci_high=None,pvalue=None,ci_method='insufficient pairs')
 t=stats.ttest_1samp(v,0);mean=float(v.mean());half=float(stats.t.ppf(.975,n-1)*stats.sem(v));lo,hi=mean-half,mean+half
 if bootstrap:
  rng=np.random.default_rng(20261002);bs=[]
  for _ in range(100):bs.extend(v[rng.integers(0,n,(100,n))].mean(1))
  lo,hi=map(float,np.quantile(bs,[.025,.975]))
 return dict(n_prompts=n,mean=mean,ci_low=lo,ci_high=hi,pvalue=float(t.pvalue) if np.isfinite(t.pvalue) else (1. if mean==0 else 0.),ci_method='prompt bootstrap10000' if bootstrap else 'paired-prompt t interval; exploratory')
def main(stage):
 c=json.loads((ROOT/'contract.json').read_text());ids=c['discovery_ids'] if stage=='recovery' else [i for i in c['probe_ids'] if i not in set(c['discovery_ids'])];idsmap={s:i for i,s in enumerate(ids)}
 cases=['clean','white50','white40','white30','white20','pink50','pink40','pink30','pink20','quiet_m6','quiet_m12','dark_m6','local_shuffle1','local_shuffle0.25','gain_m6','stem_recompose']+[f'stem_{n}{suffix}' for n in ['drums','bass','other','vocals','guitar','piano'] for suffix in ['_m6','_off']]
 if stage=='confirmatory':cases += [f'{color}_absolute_m{level}' for color in ['white','pink'] for level in [55,45]]
 cmap={s:i for i,s in enumerate(cases)};mmap={'raw':0,'lufs23':1};scores={};coverage=[]
 for cell in c['cells']:
  out=ROOT/'cells'/cell['id'];receipt=out/(stage+'_complete.json');assert receipt.exists(),str(receipt)
  arr=np.full((len(ids),len(cases),2,5),np.nan)
  for line in open(out/'probe.jsonl'):
   r=json.loads(line)
   if r['id'] not in idsmap or r['case'] not in cmap:continue
   arr[idsmap[r['id']],cmap[r['case']],mmap[r['mode']]]=[r.get(a) if r.get(a) is not None else np.nan for a in AXES]
  scores[cell['id']]=arr;coverage.append(dict(cell=cell['id'],stage=stage,n_requested=len(ids),n_finite_PQ_clean_norm=int(np.isfinite(arr[:,0,1,0]).sum()),receipt_sha256=sha(receipt)))
 groups=defaultdict(list)
 for cell in c['cells']:groups[cell['family'],cell['cfg_strength']].append(cell)
 lookup={(r['family'],r['cfg_strength'],r['training_seed']):r['id'] for r in c['cells']};rows=[]
 for (family,cfg),cells in groups.items():
  effects=[];gaps=[];changes=[]
  for cell in cells:
   arr=scores[cell['id']];baselines=np.stack([arr[:,cmap['stem_recompose'] if case.startswith('stem_') and case!='stem_recompose' else 0] for case in cases],axis=1)
   effects.append(arr-baselines)
   if family!='control':
    control=scores[lookup['control',cfg,cell['training_seed']]];cb=np.stack([control[:,cmap['stem_recompose'] if case.startswith('stem_') and case!='stem_recompose' else 0] for case in cases],axis=1)
    gaps.append(arr-control);changes.append((arr-baselines)-(control-cb))
  for kind,vectors in [('effect',effects),('gap',gaps),('gap_change',changes)]:
   if not vectors:continue
   # A NaN in any training seed excludes that paired prompt from that contrast.
   v=np.stack(vectors).mean(0)
   for case in cases:
    for mode in mmap:
     for axis in AXES:
      primary=(stage=='confirmatory' and family=='N100' and kind=='gap_change' and mode=='lufs23' and case in c['analysis']['primary_conditions'] and axis in c['analysis']['primary_axes'])
      row=dict(kind=kind,family=family,cfg=cfg,case=case,mode=mode,axis=axis,n_training_seeds=len(cells),primary=primary,holm_pvalue=None,**ci(v[:,cmap[case],mmap[mode],AXES.index(axis)],bootstrap=primary));rows.append(row)
 for cfg in [0,3]:
  selected=[r for r in rows if r['primary'] and r['cfg']==cfg and r['pvalue'] is not None];selected.sort(key=lambda r:r['pvalue']);previous=0.
  for j,r in enumerate(selected):previous=max(previous,min(1.,r['pvalue']*(len(selected)-j)));r['holm_pvalue']=previous
 with open(ROOT/(stage+'_summary.csv'),'w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# '+stage+' intervention results','',f'Analyzed {len(ids)} hash-selected prompts in each of 16 complete cells. Confirmatory inference excludes the original512 prompts. All three matched training seeds must be finite within a contrast. Nhi is one-seed exploratory.','', 'PQ/CLAP changes are intervention minus its registered baseline; stem removals use neural recomposition. Gap change is (intervention-baseline in the trained family) minus (the same in control). A negative gap change means the model advantage shrinks under that intervention. This alone does not prove human quality or a causal training mediator.','', 'Primary holdout gap changes: 10,000 paired-prompt bootstrap draws; two-sided paired-prompt t-test with Holm adjustment within CFG across all12 primary conditions ×2 axes. Other intervals are exploratory t intervals. CIs describe this prompt population and fixed checkpoints, not the distribution of training seeds.','', '| Kind | Family | CFG | Condition | Mode | Axis | Mean | 95% CI | N | Holm p |','|---|---|---:|---|---|---|---:|---|---:|---:|']
 for r in rows:
  if r['axis'] not in ['PQ','clap'] or r['family']!='N100' or r['kind'] not in ['gap','gap_change'] or r['case'] not in ['clean']+c['analysis']['primary_conditions'] or r['mode']!='lufs23':continue
  if r['mean'] is None:continue
  lines.append(f'| {r["kind"]} | {r["family"]} | {r["cfg"]} | {r["case"]} | {r["mode"]} | {r["axis"]} | {r["mean"]:+.5f} | [{r["ci_low"]:+.5f}, {r["ci_high"]:+.5f}] | {r["n_prompts"]} | {r["holm_pvalue"]} |')
 (ROOT/(stage+'_REPORT.md')).write_text('\n'.join(lines)+'\n');dump(ROOT/(stage+'_analysis_complete.json'),dict(stage=stage,n_cells=16,n_prompts=len(ids),coverage=coverage,summary_sha256=sha(ROOT/(stage+'_summary.csv')),contract_sha256=sha(ROOT/'contract.json')))
 print('ANALYSIS COMPLETE',stage,len(rows),flush=True)
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('--stage',required=True);main(p.parse_args().stage)
