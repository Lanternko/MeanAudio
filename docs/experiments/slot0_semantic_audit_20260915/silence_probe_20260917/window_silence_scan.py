# 訓練 latent = 整個 30s 檔先 peak-normalize 到 0.95，再取前 10s（clips.tsv 全是 _0、start=0；extract_audio_latents.py normalize_audio=True）。
# 這裡照同樣順序處理，對 [0,10) [10,20) [20,30) 三窗各算：RMS dBFS、100ms 幀 RMS<0.01 的比例。
import csv,re,sys,numpy as np,soundfile as sf
from multiprocessing import Pool
csv.field_size_limit(10**9)
TSV='/home/kojiek/exps_nvme/slot0nm/arm_inputs/phase8_caption2p0_slot0nm_train.tsv'
ids=sorted(r['id'] for r in csv.DictReader(open(TSV),delimiter='\t'))
N=int(sys.argv[2]); rng=np.random.default_rng(20260917)
sel=sorted(rng.choice(ids,N,replace=False)) if N<len(ids) else ids
def path(cid):
  p=re.sub(r'_\d+$','',cid).split('_'); s=p.index('segment')
  return f"/mnt/HDD/hsiehyian/segments_no_vocals/{'_'.join(p[:s-1])}/{p[s-1]}/segment_{p[s+1]}.mp3"
def feat(cid):
  try:
    y,sr=sf.read(path(cid),dtype='float32'); y=y if y.ndim==1 else y.mean(1)
    pk=float(np.abs(y).max()); raw_rms=float(np.sqrt((y[:10*sr]**2).mean()))
    if pk<1e-6: return [cid,len(y)/sr,pk,raw_rms]+[-120.0,1.0]*3
    y=y/pk*0.95; out=[cid,len(y)/sr,pk,raw_rms]; h=sr//10
    for k in range(3):
      w=y[k*10*sr:(k+1)*10*sr]
      if len(w)<sr: out+=[np.nan,np.nan]; continue
      fr=w[:len(w)//h*h].reshape(-1,h); r=np.sqrt((fr**2).mean(1))
      out+=[float(20*np.log10(np.sqrt((w**2).mean())+1e-9)),float((r<0.01).mean())]
    return out
  except Exception as e: return [cid]+[np.nan]*9
with Pool(16) as pool, open(sys.argv[1],'w') as f:
  f.write('id\tdur\tpeak\traw_rms10\tdb0\tsil0\tdb1\tsil1\tdb2\tsil2\n')
  for i,r in enumerate(pool.imap(feat,sel,chunksize=32)):
    f.write('\t'.join(str(x) for x in r)+'\n')
    if i%2000==0: print(i,flush=True)
print('done',len(sel))
