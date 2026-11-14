import gzip,json,collections
import pandas as pd
r=pd.DataFrame(json.load(gzip.open('evidence_rows.json.gz','rt')))
pd.set_option('display.width',250,'display.max_rows',500)
f=r[(r.cat=='明確問題')&(r.why.str.contains('上升'))&(r.metric!='pulse_clarity')]
print(f.sort_values('mean',ascending=False).head(22)[['ds','cell','metric','n','ncl','mean','lo','hi','m0','dz']].round(3).to_string())
print(r[(r.cat=='明確問題')&(r.why.str.contains('<'))][['ds','cell','metric','n','mean','m0','worse']].round(3).to_string())
# pitch error cases
p=r[(r.ds=='126b')&r.cell.str.contains('wrong|randpitch|pitch|jitter',regex=True)&r.metric.isin(['PQ','CE','CU','meva','clap','key_cnn_conf'])]
print(p.groupby(['cell','metric']).agg(n=('n','sum'),mean=('mean','mean'),worse=('worse','mean')).round(3).head(0))
print(sorted(set(p.cell.str.split('|').str[2])))
