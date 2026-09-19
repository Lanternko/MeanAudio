import pickle,re,random,numpy as np,librosa,json
from multiprocessing import Pool
src,clean,s4,nm=pickle.load(open('corp.pkl','rb'))
def path(cid):
  cid=re.sub(r'_\d+$','',cid); p=cid.split('_'); s=p.index('segment')
  return f"/mnt/HDD/hsiehyian/segments_no_vocals/{'_'.join(p[:s-1])}/{p[s-1]}/segment_{p[s+1]}.mp3"
B={};K={}
for k,c in src.items():
    m=re.search(r'(\d{2,3}(?:\.\d+)?)\s*(?:bpm|beats per minute)',c,re.I)
    if m and 40<=float(m.group(1))<=240: B[k]=float(m.group(1))
    m=re.search(r'\b([A-G])(#|b| sharp| flat)? (major|minor)\b',c)
    if m: K[k]=(m.group(1),(m.group(2) or '').strip(),m.group(3))
random.seed(0)
bk=random.sample(sorted(B),600); kk=random.sample(sorted(K),600)
MAJ=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]);MIN=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
def feat(cid):
    try:
        y,sr=librosa.load(path(cid),sr=22050,mono=True,duration=10.0)
        if len(y)<sr*3: return cid,None
        t=float(np.atleast_1d(librosa.beat.beat_track(y=y,sr=sr)[0])[0])
        ch=librosa.feature.chroma_cqt(y=y,sr=sr).mean(1)
        sc=[]
        for mode,prof in (('major',MAJ),('minor',MIN)):
            for r in range(12): sc.append((np.corrcoef(ch,np.roll(prof,r))[0,1],r,mode))
        sc.sort(reverse=True)
        return cid,(t,sc[0][1],sc[0][2])
    except Exception as e: return cid,None
with Pool(16) as p: R=dict(p.map(feat,sorted(set(bk+kk))))
pickle.dump((B,K,bk,kk,R),open('verify.pkl','wb'))
print('done',sum(v is not None for v in R.values()))
