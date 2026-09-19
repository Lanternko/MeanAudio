import pickle,re,random,numpy as np
from scipy.stats import spearmanr
D='/tmp/claude-1005/-home-kojiek-MeanAudio/770eef45-4b45-4b28-8976-cf3e6027f940/scratchpad/'
src,clean,s4,nm=pickle.load(open(D+'corp.pkl','rb'))
B,K,bk,kk,R=pickle.load(open(D+'verify.pkl','rb'))
ids=[k for k in bk if R.get(k)]
c=np.array([B[k] for k in ids]); l=np.array([R[k][0] for k in ids]); n=len(ids)
print('n',n)
print('claim  pct 10/25/50/75/90:',np.percentile(c,[10,25,50,75,90]).round(1),' in[110,130]:',round(np.mean((c>=110)&(c<=130)),3),' ==120:',round(np.mean(c==120),3))
print('librosa pct 10/25/50/75/90:',np.percentile(l,[10,25,50,75,90]).round(1),' in[110,130]:',round(np.mean((l>=110)&(l<=130)),3))
def hit(a,b,tol,octv):
    r=np.abs(a/b-1)<=tol
    if octv: r|=np.abs(2*a/b-1)<=tol; r|=np.abs(a/(2*b)-1)<=tol
    return r.mean()
GEN=['rock','pop','jazz','electronic','hip hop','hip-hop','classical','folk','ambient','metal','funk','reggae','blues','country','techno','house','soul','r&b','dance','lounge','soundtrack','cinematic','orchestral','punk','latin','world','indie','edm','trance','disco']
def genre(k):
    s=src[k].lower()
    for g in GEN:
        if g in s: return g
    return 'other'
G=np.array([genre(k) for k in ids])
rng=np.random.default_rng(0)
def shuf(within):
    if not within: return c[rng.permutation(n)]
    o=c.copy()
    for g in set(G):
        ix=np.where(G==g)[0]; o[ix]=c[ix[rng.permutation(len(ix))]]
    return o
def report(mask,label):
    cc,ll=c[mask],l[mask]
    print(f'\n== {label}  n={mask.sum()}')
    for tol in (0.02,0.04,0.08):
        for octv in (False,True):
            obs=hit(cc,ll,tol,octv)
            glob=[];wg=[]
            for _ in range(1000):
                s=shuf(False)[mask]; glob.append(hit(s,ll,tol,octv))
                s=shuf(True)[mask]; wg.append(hit(s,ll,tol,octv))
            const=max(hit(np.full_like(ll,v),ll,tol,octv) for v in range(60,200))
            print(f' ±{int(tol*100)}% oct={int(octv)}: obs {obs:.3f} | 全體打亂 {np.mean(glob):.3f} (p95 {np.percentile(glob,95):.3f}) | 同曲風內打亂 {np.mean(wg):.3f} (p95 {np.percentile(wg,95):.3f}) | 最佳常數猜測 {const:.3f}')
    rho=spearmanr(cc,ll).correlation
    nul=[spearmanr(shuf(False)[mask],ll).correlation for _ in range(500)]
    nulg=[spearmanr(shuf(True)[mask],ll).correlation for _ in range(500)]
    print(f' Spearman {rho:.3f} | 全體打亂 p95 {np.percentile(nul,95):.3f} | 同曲風內打亂 mean {np.mean(nulg):.3f} p95 {np.percentile(nulg,95):.3f}')
    ae=np.median(np.abs(cc-ll)/ll); print(f' 相對誤差中位數 {ae:.3f}')
report(np.ones(n,bool),'全部')
report(~((c>=110)&(c<=130)),'排除 claim 110–130')
report(~((l>=110)&(l<=130))&~((c>=110)&(c<=130)),'claim 與 librosa 都不在 110–130')
print('\ngenre counts',{g:int((G==g).sum()) for g in set(G) if (G==g).sum()>=15})
