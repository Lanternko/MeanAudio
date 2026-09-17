import numpy as np,soundfile as sf,os,json,sys
from multiprocessing import Pool
N='/home/kojiek/eval_output_nvme/'; C='/home/kojiek/cfg0_eval_runtime/output/'
D={'cfg3_A_slot0':N+'c2p0_slot0_quarter_mc_mf25_cfg3_neg/audio','cfg3_A_nm':N+'phase8_qwen_caption2p0_slot0nm_noq_quarter_mc_mf25_cfg3_neg/audio',
   'cfg3_B_slot0':N+'phase8_qwen_caption2p0_slot0_noq_quarter_s27182818_mc_mf25_cfg3_neg/audio','cfg3_B_nm':N+'phase8_qwen_caption2p0_slot0nm_noq_quarter_s27182818_mc_mf25_cfg3_neg/audio',
   'cfg0_A_nm':C+'phase8_qwen_caption2p0_slot0nm_noq_quarter_musiccaps_mf25_cfg0_noq/audio',
   'cfg0_B_slot0':C+'phase8_qwen_caption2p0_slot0_noq_quarter_s27182818_musiccaps_mf25_cfg0_noq/audio','cfg0_B_nm':C+'phase8_qwen_caption2p0_slot0nm_noq_quarter_s27182818_musiccaps_mf25_cfg0_noq/audio'}
def f(p):
  y,sr=sf.read(p,dtype='float32'); y=y if y.ndim==1 else y.mean(1)
  fr=len(y)//(sr//10)*(sr//10); w=y[:fr].reshape(-1,sr//10); r=np.sqrt((w**2).mean(1))
  return float(20*np.log10(np.sqrt((y**2).mean())+1e-9)), float((r<0.01).mean())
out={}
with Pool(16) as pool:
  for k,d in D.items():
    fs=sorted(x for x in os.listdir(d) if x.endswith(('.flac','.wav')))
    res=pool.map(f,[os.path.join(d,x) for x in fs],chunksize=64)
    out[k]={x.rsplit('.',1)[0]:r for x,r in zip(fs,res)}; print(k,len(fs),flush=True)
json.dump(out,open(sys.argv[1],'w'))
