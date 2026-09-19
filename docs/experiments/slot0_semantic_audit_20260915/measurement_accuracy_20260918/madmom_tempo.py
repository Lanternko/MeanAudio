import pickle,re,sys,numpy as np
from multiprocessing import Pool
D='/tmp/claude-1005/-home-kojiek-MeanAudio/770eef45-4b45-4b28-8976-cf3e6027f940/scratchpad/'
B,K,bk,kk,R=pickle.load(open(D+'verify.pkl','rb'))
ids=[k for k in bk if R.get(k)]
def path(cid):
  cid=re.sub(r'_\d+$','',cid); p=cid.split('_'); s=p.index('segment')
  return f"/mnt/HDD/hsiehyian/segments_no_vocals/{'_'.join(p[:s-1])}/{p[s-1]}/segment_{p[s+1]}.mp3"
def f(cid):
    try:
        from madmom.features.beats import RNNBeatProcessor
        from madmom.features.tempo import TempoEstimationProcessor
        from madmom.io.audio import load_audio_file
        y,_=load_audio_file(path(cid),sample_rate=44100,num_channels=1,stop=10.0)
        y=y.astype(np.float32)/32768.0 if y.dtype==np.int16 else y.astype(np.float32)
        act=RNNBeatProcessor()(y.astype(np.float32))
        t=TempoEstimationProcessor(fps=100)(act)
        return cid,float(t[0][0])
    except Exception as e: return cid,repr(e)
with Pool(16) as p: M=dict(p.map(f,ids))
pickle.dump(M,open('/tmp/claude-1005/-home-kojiek-MeanAudio/e94eb9e3-54c8-4c79-b492-3ab94e4bab76/scratchpad/madmom.pkl','wb'))
bad=[v for v in M.values() if isinstance(v,str)]
print('ok',len(M)-len(bad),'bad',len(bad),bad[:2])
