import pickle,numpy as np
from scipy.stats import spearmanr
D='/tmp/claude-1005/-home-kojiek-MeanAudio/770eef45-4b45-4b28-8976-cf3e6027f940/scratchpad/'
S='/tmp/claude-1005/-home-kojiek-MeanAudio/e94eb9e3-54c8-4c79-b492-3ab94e4bab76/scratchpad/'
B,K,bk,kk,R=pickle.load(open(D+'verify.pkl','rb')); M=pickle.load(open(S+'madmom.pkl','rb'))
ids=[k for k in bk if R.get(k)]
c=np.array([B[k] for k in ids]); l=np.array([R[k][0] for k in ids]); m=np.array([M[k] for k in ids]); n=len(c)
rng=np.random.default_rng(0)
def hit(a,b,tol,octv=False):
    r=np.abs(a/b-1)<=tol
    if octv: r|=np.abs(2*a/b-1)<=tol; r|=np.abs(a/(2*b)-1)<=tol
    return r
print('madmom pct',np.percentile(m,[10,25,50,75,90]).round(1))
print('librosa vs madmom ±4% 同意率',hit(l,m,.04).mean().round(3),' 含倍頻',hit(l,m,.04,True).mean().round(3))
for ref,name in ((m,'madmom'),(l,'librosa')):
    for tol in (.02,.04):
        obs=hit(c,ref,tol).mean(); nul=[hit(c[rng.permutation(n)],ref,tol).mean() for _ in range(2000)]
        print(f'claim vs {name} ±{int(tol*100)}% 不含倍頻: {obs:.3f}  打亂 {np.mean(nul):.3f} (p95 {np.percentile(nul,95):.3f})')
# 兩個估計器都同意的子集 = 較可信的「真值」
agree=hit(l,m,.04)
t=(l+m)/2
print(f'\n兩估計器同意(±4%)子集 n={agree.sum()}')
for tol in (.02,.04,.08):
    obs=hit(c[agree],t[agree],tol).mean(); nul=[hit(c[rng.permutation(n)][agree],t[agree],tol).mean() for _ in range(2000)]
    print(f' claim ±{int(tol*100)}%: {obs:.3f}  打亂 {np.mean(nul):.3f} (p95 {np.percentile(nul,95):.3f})')
print(' Spearman',round(spearmanr(c[agree],t[agree]).correlation,3))
# 誤差分布（對 madmom、以倍頻折算後的最小相對誤差）
e=np.min(np.abs(np.stack([c/m,2*c/m,c/(2*m)])-1),axis=0)
print('\n對 madmom 的相對誤差分布（取倍頻最佳）: ≤2%',np.mean(e<=.02).round(3),' 2–8%',np.mean((e>.02)&(e<=.08)).round(3),' 8–20%',np.mean((e>.08)&(e<=.2)).round(3),' >20%',np.mean(e>.2).round(3))
er=np.abs(c/m-1)
print('不含倍頻: ≤2%',np.mean(er<=.02).round(3),' >20%',np.mean(er>.2).round(3))
# 小數點 BPM（如 117.45）比例：看是否像工具輸出
dec=np.array([abs(x-round(x))>1e-6 for x in c]); print('\n帶小數的 claim 比例',dec.mean().round(3))
for sub,name in ((dec,'帶小數'),(~dec,'整數')):
    print(f' {name} n={sub.sum()}: ±2% vs madmom {hit(c[sub],m[sub],.02).mean():.3f}')
