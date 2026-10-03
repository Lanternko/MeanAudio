import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'aes-longrun-20261002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 c=json.loads((OLD/'contract.json').read_text());ids=[r['id'] for r in csv.DictReader(open(c['tsv']),delimiter='\t')];data={};audit=[]
 for cell in c['cells']:
  out=OLD/'cells'/cell['id'];m=out/'metrics'/cell['id'];rec=json.loads((m/'metrics.json').read_text());rr={r['id']:r for r in csv.DictReader(open(m/'per_clip.tsv'),delimiter='\t')}
  assert len(rr)==5521 and set(rr)==set(ids) and rec['n_present']==5521 and rec['n_rows']==5521 and not rec['missing_ids']
  assert rec['tsv_sha256']==c['tsv_sha256'];data[cell['id']]=np.array([[float(rr[i][a]) for a in ['PQ','CE','CU','PC','clap','rms_dbfs']] for i in ids]);assert np.isfinite(data[cell['id']]).all()
  n=0
  for i in ids:
   r=json.loads((out/'receipts'/(i+'.json')).read_text());assert r['sha256']==sha(out/'audio'/(i+'.flac'));n+=1
  audit.append(dict(cell=cell['id'],n_audio_hash_verified=n,n_metric_rows=len(rr),metrics_sha256=sha(m/'per_clip.tsv')))
 lookup={(x['family'],x['cfg_strength'],x['training_seed']):x['id'] for x in c['cells']};groups=defaultdict(list);seeds=[]
 for cell in c['cells']:
  if cell['family']=='control':continue
  base=lookup['control',cell['cfg_strength'],cell['training_seed']];diff=data[cell['id']]-data[base];groups[cell['family'],cell['cfg_strength']].append(diff)
  seeds.append(dict(family=cell['family'],cfg=cell['cfg_strength'],training_seed=cell['training_seed'],**dict(zip(['PQ','CE','CU','PC','clap','rms_dbfs'],map(float,diff.mean(0))))))
 sums=[]
 for (family,cfg),values in groups.items():
  v=np.mean(values,axis=0);rng=np.random.default_rng(20261002);boots=[]
  for _ in range(100):boots.append(v[rng.integers(0,len(v),(100,len(v)))].mean(1))
  lo,hi=np.quantile(np.concatenate(boots),[.025,.975],axis=0)
  for j,a in enumerate(['PQ','CE','CU','PC','clap','rms_dbfs']):sums.append(dict(family=family,cfg=cfg,axis=a,n_prompts=5521,n_training_seeds=len(values),mean=float(v[:,j].mean()),ci_low=float(lo[j]),ci_high=float(hi[j])))
 (ROOT/'raw_audit.json').write_text(json.dumps(dict(audio_files_verified=sum(r['n_audio_hash_verified'] for r in audit),audit=audit,per_seed=seeds,summary=sums,unit='paired prompt; average seeds within prompt; exploratory bootstrap10000; not population CI over training seeds',old_contract_sha256=sha(OLD/'contract.json')),indent=2)+'\n')
 lines=['# Verified raw results','', 'All 88,336 audio file hashes and 16 × 5,521 raw metric rows verified. Failed intervention phases do not invalidate these raw results. This replay did not compute FAD or human listening.','', '| Model vs matched control | CFG | Training seeds | Axis | Delta | Prompt-bootstrap 95% CI |','|---|---:|---:|---|---:|---|']
 for r in sums:lines.append(f'| {r["family"]} | {r["cfg"]} | {r["n_training_seeds"]} | {r["axis"]} | {r["mean"]:+.5f} | [{r["ci_low"]:+.5f}, {r["ci_high"]:+.5f}] |')
 lines+=['','CFG0 is an explicitly historical secondary replay. CFG3 fidelity8 is the canonical guided protocol. Nhi has one seed and cannot establish cross-seed reproducibility. CIs average training replicates within each of 5,521 prompts; uncertainty across all possible training runs is not estimated.','', 'Mechanistic interpretation remains unconfirmed until interventions complete. CLAP measures caption alignment, not human audio quality. The previous report incorrectly gated raw results on probe completion; that report and failed logs are retained unchanged.']
 (ROOT/'RAW_RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(sums,indent=2))
if __name__=='__main__':main()
