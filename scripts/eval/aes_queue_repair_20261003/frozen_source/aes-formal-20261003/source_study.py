"""Preregistered other-stem dose, band and structure diagnostic; no new generation."""
import os,sys,json,time,hashlib,csv
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal,stats
from common import ROOT,SRC,PY,sha,dump,spec,notify,load_records
sys.path.insert(0,str(SRC))
import probe as core
import metrics
AXES=['PQ','CE','CU','PC','clap']
CASES=['clean','stem_recompose','other_m18','other_m12','stem_other_m6','stem_other_off','other_p6']+[f'other_band{k}_{suffix}' for k in range(4) for suffix in ['m12','off']]+['other_only','other_phase_random','other_shift250ms','other_shift1000ms','gain_m6']

def variants(y,items,seed):
 bycase={name:z for name,z,_,_ in items};recon=bycase['stem_recompose'];off=bycase['stem_other_off'];other=(recon.astype(float)-off.astype(float)).astype(np.float32)
 yield 'clean',y,'clean'
 yield 'stem_recompose',recon,'clean'
 for db in [-18,-12,-6,-99,6]:
  case={-6:'stem_other_m6',-99:'stem_other_off',6:'other_p6'}.get(db,f'other_m{-db}')
  z=bycase[case] if case in bycase else (recon+((0 if db==-99 else 10**(db/20))-1)*other).astype(np.float32)
  yield case,z,'stem_recompose'
 freq=np.fft.rfftfreq(len(other),1/16000);spectrum=np.fft.rfft(other);edges=[0,300,1200,4000,8001]
 for k,(low,high) in enumerate(zip(edges,edges[1:])):
  band=np.fft.irfft(spectrum*((freq>=low)&(freq<high)),n=len(other)).astype(np.float32)
  for suffix,g in [('m12',10**(-12/20)),('off',0)]:yield f'other_band{k}_{suffix}',(recon+(g-1)*band).astype(np.float32),'stem_recompose'
 yield 'other_only',other,'stem_recompose'
 phase=np.exp(1j*np.random.default_rng(seed).uniform(-np.pi,np.pi,len(spectrum)));phase[0]=np.exp(1j*np.angle(spectrum[0]));phase[-1]=np.exp(1j*np.angle(spectrum[-1]))
 randomized=np.fft.irfft(np.abs(spectrum)*phase,n=len(other)).astype(np.float32)
 assert np.isclose(np.sum(randomized.astype(float)**2),np.sum(other.astype(float)**2),rtol=2e-6,atol=1e-10)
 yield 'other_phase_random',(off+randomized).astype(np.float32),'stem_recompose'
 for ms in [250,1000]:yield f'other_shift{ms}ms',(off+np.roll(other,ms*16)).astype(np.float32),'stem_recompose'
 yield 'gain_m6',(y*10**(-6/20)).astype(np.float32),'clean'

def analyze(c,root):
 ids=c['analysis_ids'];imap={i:j for j,i in enumerate(ids)};cmap={n:j for j,n in enumerate(CASES)};arrays={}
 for cell in c['cells']:
  arr=np.full((len(ids),20,2,5),np.nan)
  for line in open(root/'cells'/cell['id']/'scores.jsonl'):
   r=json.loads(line);arr[imap[r['id']],cmap[r['case']],0 if r['mode']=='raw' else 1]=[r[a] if r[a] is not None else np.nan for a in AXES]
  arrays[cell['family'],cell['training_seed']]=arr
 gaps=[];effects=[]
 for seed in [14159265,16180339,27182818]:
  a=arrays['N100',seed];b=arrays['control',seed];baseline=[cmap['clean'] if name in ['clean','stem_recompose','gain_m6'] else cmap['stem_recompose'] for name in CASES]
  gaps.append(a-b);effects.append((a-a[:,baseline])-(b-b[:,baseline]))
 rows=[]
 from analyze import ci
 for kind,stack in [('model_gap',gaps),('gap_change',effects)]:
  values=np.stack(stack).mean(0)
  for j,name in enumerate(CASES):
   for mode,mi in [('raw',0),('lufs23',1)]:
    for ai,axis in enumerate(AXES):
     primary=(kind=='gap_change' and mode=='lufs23' and axis in ['PQ','clap'] and name in c['primary_cases'])
     rows.append(dict(kind=kind,case=name,mode=mode,axis=axis,primary=primary,holm_pvalue=None,**ci(values[:,j,mi,ai],bootstrap=primary)))
 tests=sorted([r for r in rows if r['primary'] and r['pvalue'] is not None],key=lambda r:r['pvalue']);p=0.
 for j,r in enumerate(tests):p=max(p,min(1.,r['pvalue']*(len(tests)-j)));r['holm_pvalue']=p
 with open(root/'summary.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Other-source diagnosis: primary CFG3 fidelity8 only','','5,009 original held-out prompts; all3 matched training-seed values required per contrast. Primary noise/dose/band/phase contrasts use10,000 prompt bootstrap and Holm-adjusted paired-prompt tests. Other intervals exploratory. `other` is a neural source class that includes musical content, not verified hiss or background noise. Band removal may introduce ringing, circular shifts alter sync and wrap boundaries, phase randomization changes temporal structure and crest while preserving global energy/spectrum. No human-quality or causal-training-mediator claim.','', '| Kind | Case | Axis | Mean | N | 95% CI | Holm p |','|---|---|---|---:|---:|---|---:|']
 for r in rows:
  if r['mode']=='lufs23' and r['axis'] in ['PQ','clap'] and r['mean'] is not None:lines.append(f'| {r["kind"]} | {r["case"]} | {r["axis"]} | {r["mean"]:+.5f} | {r["n_prompts"]} | [{r["ci_low"]:+.5f},{r["ci_high"]:+.5f}] | {r["holm_pvalue"]} |')
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 import warnings;warnings.filterwarnings('ignore')
 import torch
 from audiobox_aesthetics.infer import initialize_predictor
 from demucs_infer.pretrained import get_model
 c=spec();root=Path(c['runtime_root']);root.mkdir(exist_ok=True);captions={r['id']:r['caption'] for r in csv.DictReader(open(c['tsv']),delimiter='\t')};torch.set_num_threads(4)
 predictor=initialize_predictor();separator=get_model('htdemucs_6s',repo=Path(c['demucs_model_repo'])).to('cuda').eval();clap=metrics.load_clap();receipts=[]
 for cell in c['cells']:
  out=root/'cells'/cell['id'];out.mkdir(parents=True,exist_ok=True);cache=out/'scores.jsonl';done={};existing=SRC/'cells'/cell['id']/'probe.jsonl'
  if cache.exists():done=load_records(cache)
  prior={r['key']:r for r in map(json.loads,existing.read_text().splitlines())} if existing.exists() else {}
  excluded=[];variant_exclusions=[];eligible=[];start=time.monotonic();new_scores=0;reused=0
  with open(cache,'a',buffering=1) as f:
   for index,identifier in enumerate(c['analysis_ids']):
    source=SRC/'cells'/cell['id']/'audio'/(identifier+'.flac');y,sr=sf.read(source,dtype='float32');assert sr==16000 and len(y)==159744;source_hash=sha(source)
    if not np.isfinite(core.loudness(y)):excluded.append(identifier);continue
    eligible.append(identifier);seed=int(hashlib.sha256(('20261002'+identifier).encode()).hexdigest()[:8],16)
    items=list(core.stem_items(y,identifier,SRC/'cells'/cell['id'],separator));vs=list(variants(y,items,seed));assert [v[0] for v in vs]==CASES
    with torch.inference_mode():te=clap.get_text_embedding([captions[identifier]],use_tensor=True)
    prepared=[]
    for case,z,baseline in vs:
     for mode in ['raw','lufs23']:
      key=f'{identifier}__{case}__{mode}'
      if key in done:assert done[key]['source_sha256']==source_hash;continue
      meta=dict(key=key,id=identifier,case=case,baseline=baseline,mode=mode,source_sha256=source_hash,source_features=core.features(z),clap=None)
      try:zz,g,lu=(z,1.,core.loudness(z)) if mode=='raw' else core.normalize(z);zz=np.asarray(zz,dtype=np.float32)
      except ValueError as exc:
       variant_exclusions.append(dict(key=key,reason=str(exc)));record=dict(**meta,status='excluded',**{a:None for a in AXES if a!='clap'});f.write(json.dumps(record,allow_nan=False)+'\n');done[key]=record;continue
      meta.update(gain=float(g),lufs_after=float(lu) if np.isfinite(lu) else None,scored_waveform_sha256=core.waveform_hash(zz))
      old=prior.get(key)
      if old and old['source_sha256']==source_hash and old.get('scored_waveform_sha256')==meta['scored_waveform_sha256']:
       record=dict(**meta,**{a:old[a] for a in AXES if a!='clap'});record['clap']=old['clap'];record['reused_from']=str(existing);f.write(json.dumps(record,allow_nan=False)+'\n');done[key]=record;reused+=1;continue
      prepared.append((zz,meta))
    for offset in range(0,len(prepared),16):
     batch=prepared[offset:offset+16];scores=predictor.forward([dict(path=torch.from_numpy(z[None]),sample_rate=16000) for z,_ in batch])
     for (z,meta),score in zip(batch,scores):
      assert all(np.isfinite(score[a]) for a in AXES if a!='clap');record=dict(**meta,**score)
      if meta['mode']=='lufs23':
       np.random.seed(seed);torch.manual_seed(seed)
       with torch.inference_mode():ae=clap.get_audio_embedding_from_data(x=torch.from_numpy(signal.resample_poly(z,3,1).astype(np.float32))[None],use_tensor=True);record['clap']=float(torch.nn.functional.cosine_similarity(ae,te,dim=-1).item())
      f.write(json.dumps(record,allow_nan=False)+'\n');done[record['key']]=record;new_scores+=1
    if index%8==0 or index==len(c['analysis_ids'])-1:
     dump(root/'progress.json',dict(cell=cell['id'],completed=index+1,total=len(c['analysis_ids']),new_scores=new_scores,reused=reused,elapsed_seconds=time.monotonic()-start));print('SOURCE',cell['id'],index+1,flush=True)
   expected={f'{i}__{case}__{mode}' for i in eligible for case in CASES for mode in ['raw','lufs23']};assert expected==set(done)
   ctrl=max([abs(done[f'{i}__clean__lufs23'][a]-done[f'{i}__gain_m6__lufs23'][a]) for i in eligible for a in AXES[:4] if done[f'{i}__clean__lufs23'][a] is not None and done[f'{i}__gain_m6__lufs23'][a] is not None] or [0]);assert ctrl<.01
   rec=dict(cell=cell['id'],n_eligible=len(eligible),n_records=len(done),cache=str(cache),cache_sha256=sha(cache),gain_control_max=ctrl,excluded_sources=excluded,excluded_variants=variant_exclusions,new_scores=new_scores,reused=reused);dump(out/'complete.json',rec);receipts.append(rec)
   notify(cell['id']+'_source_pass','start',f"Other-source gate PASS {cell['id']}: {len(eligible)}/5009 eligible, {len(done)} records, control max={ctrl:.6f}<.01; SHA={sha(out/'complete.json')}; next cell eligible.")
 analyze(c,root);dump(root/'summary.json',dict(status='completed',n_cells=6,n_prompts_requested=5009,n_cases=20,cells=receipts,summary_sha256=sha(root/'summary.csv'),contract_sha256=sha(os.environ['GPU_QUEUE_CONTRACT'])))
 notify('analysis_pass','start',f"Other-source analysis PASS:6 CFG3 cells,5009 prompts,20 conditions; {root/'summary.csv'} SHA={sha(root/'summary.csv')}; terminal report next.")
if __name__=='__main__':main()
