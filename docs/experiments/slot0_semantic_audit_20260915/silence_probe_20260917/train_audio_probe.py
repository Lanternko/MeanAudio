import csv,re,json,numpy as np,librosa,warnings,sys
from multiprocessing import Pool
warnings.filterwarnings('ignore'); csv.field_size_limit(10**9)
def load(p): return {r['id']:r['caption'] for r in csv.DictReader(open(p),delimiter='\t')}
SRC=load('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv')
CLN=load('/home/kojiek/exps_nvme/slot0clean/arm_inputs/phase8_caption2p0_slot0clean_train.tsv')
A=load('/home/kojiek/exps_nvme/slot0_rowmatched/arm_inputs/phase8_caption2p0_slot0_rowmatched_train.tsv')
B=load('/home/kojiek/exps_nvme/slot0nm/arm_inputs/phase8_caption2p0_slot0nm_train.tsv')
ids=sorted(A); rng=np.random.default_rng(0)
regen=[i for i in ids if SRC[i]!=CLN[i]]
lost20=[i for i in ids if len(A[i].split())-len(B[i].split())>20]
gained=[i for i in ids if A[i]!=B[i] and len(B[i].split())>len(A[i].split())]
changed=[i for i in ids if A[i]!=B[i]]
unchanged=[i for i in ids if A[i]==B[i]]
G={'regen_1499':regen,'lost>20w':lost20,'nm_gained_words':gained,
   'changed_sample2000':list(rng.choice(changed,2000,replace=False)),'control_unchanged_3000':list(rng.choice(unchanged,3000,replace=False))}
print({k:len(v) for k,v in G.items()}, 'regen∩lost20',len(set(regen)&set(lost20)),flush=True)
def path(cid):
  cid=re.sub(r'_\d+$','',cid); p=cid.split('_'); s=p.index('segment')
  return f"/mnt/HDD/hsiehyian/segments_no_vocals/{'_'.join(p[:s-1])}/{p[s-1]}/segment_{p[s+1]}.mp3"
def feat(cid):
  try:
    y,sr=librosa.load(path(cid),sr=16000,mono=True,duration=10.0)
    if len(y)<sr: return None
    w=y[:len(y)//1600*1600].reshape(-1,1600); r=np.sqrt((w**2).mean(1))
    return (float(20*np.log10(np.sqrt((y**2).mean())+1e-9)),float((r<0.01).mean()),float(np.abs(y).max()))
  except Exception as e: return None
allids=sorted(set().union(*G.values()))
with Pool(24) as pool: res=dict(zip(allids,pool.map(feat,allids,chunksize=16)))
json.dump({'groups':G,'feat':res},open(sys.argv[1],'w'))
print('%-24s %5s %6s %8s %8s %8s %8s %9s'%('group','n','fail','med dB','p10 dB','<-40dB%','<-45dB%','silent>50%'))
for k,v in G.items():
  f=np.array([res[i] for i in v if res[i] is not None]); fail=sum(res[i] is None for i in v)
  print('%-24s %5d %6d %8.1f %8.1f %8.2f %8.2f %9.2f'%(k,len(v),fail,np.median(f[:,0]),np.percentile(f[:,0],10),100*(f[:,0]<-40).mean(),100*(f[:,0]<-45).mean(),100*(f[:,1]>0.5).mean()))
