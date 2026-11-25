import gzip,json,collections
import pandas as pd
r=pd.DataFrame(json.load(gzip.open('evidence_rows.json.gz','rt')))
pd.set_option('display.width',250,'display.max_rows',500,'display.max_colwidth',60)
print(r.groupby(['ds','cat']).size().unstack(fill_value=0))
print(r.groupby(['ds','metric','cat']).size().unstack(fill_value=0).to_string())
# flags
f1=r[(r.kind=='lose')&(r.lo>0)&(~r.metric.isin(['pulse_clarity']))]
print('LOSE with CI>0 count',len(f1)); print(f1.groupby(['ds','metric']).size())
print(f1.sort_values('mean',ascending=False).head(25)[['ds','cell','metric','n','ncl','mean','lo','hi','m0','worse','sign']].to_string())
f2=r[(r.kind=='lose')&(r.cat=='爭議')&(r.why.str.contains('小於'))]
print('CI excl zero below m0',len(f2)); print(f2.groupby(['ds','metric']).size())
print(r[r.cat=='明確問題'].groupby(['ds','metric','why']).size().to_string())
