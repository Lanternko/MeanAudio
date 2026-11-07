import json, numpy as np, pandas as pd
from common import S_of
D='/home/kojiek/eval_output_nvme/volume_recovery_20261020/'
sc=pd.read_csv(D+'scores.tsv',sep='\t'); wf=pd.read_csv(D+'waveform.tsv',sep='\t')
S=S_of('126a'); m0={k.lower() if k=='meva' else k:0.1*v for k,v in S.items()} if isinstance(S,dict) else None
M=['PQ','CE','CU','PC','meva','clap']
def m0of(k):
    for kk in (k,k.upper(),k.lower()):
        if kk in S: return 0.1*S[kk]
    return np.nan
sc['src']=sc.id.str.split('__').str[0]
wf['src']=wf.id.str.split('__').str[0]
rng=np.random.default_rng(0)
out=['## 波形誤差（中位數，dB／絕對值）','','| variant | src | SNR dB 中位 | max_err 中位 |','|---|---|---|---|']
for (v,s),g in wf.groupby(['variant','src'],sort=False):
    if v.endswith('_rec'): out.append(f'| {v} | {s} | {g.snr_db.median():.1f} | {g.max_err.median():.2e} |')
out+=['','## 配對差（變體−clean）：mean [95% CI]（逐段 bootstrap 1000），tie% = |Δ|≤m₀ 比例','']
base=sc[sc.variant=='clean'].set_index('id')
rows=[]
for v,g in sc[sc.variant!='clean'].groupby('variant',sort=False):
    for s,gg in g.groupby('src'):
        gg=gg.set_index('id'); b=base.loc[gg.index]
        r={'variant':v,'src':s,'n':len(gg)}
        for k in M:
            d=(gg[k]-b[k]).dropna().values
            if len(d)==0: continue
            bs=rng.choice(d,(1000,len(d))).mean(1)
            lo,hi=np.percentile(bs,[2.5,97.5]); t=np.mean(np.abs(d)<=m0of(k))*100
            r[k]=f'{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] t{t:.0f}'
        rows.append(r)
df=pd.DataFrame(rows).fillna('-')
out.append('m₀: '+', '.join(f'{k} {m0of(k):.4f}' for k in M)); out.append('')
out.append(df.to_markdown(index=False))
open('B_tables.md','w').write('\n'.join(out))
