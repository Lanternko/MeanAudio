import csv,numpy as np,re
from sklearn.feature_extraction.text import TfidfVectorizer
csv.field_size_limit(10**9)
def load(p): return {r['id']:r['caption'] for r in csv.DictReader(open(p),delimiter='\t')}
A=load('/home/kojiek/exps_nvme/slot0_rowmatched/arm_inputs/phase8_caption2p0_slot0_rowmatched_train.tsv')
B=load('/home/kojiek/exps_nvme/slot0nm/arm_inputs/phase8_caption2p0_slot0nm_train.tsv')
ids=sorted(A); chg=np.array([A[i]!=B[i] for i in ids])
cap={r['id']:r['caption'] for r in csv.DictReader(open('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv'),delimiter='\t')}
bad=['0PMFAO4TIU4_30','G13NEVAm6-o_30','HVA9-fjtv6U_60','RI71ebbU0PQ_0','VCrnnx9jTqs_50','j8z9a9A8LV4_230','lUp9WnLYetg_30','tOb0M2k3deo_30','yTq3Kr3jkvs_50']
rng=np.random.default_rng(1); others=[k for k in cap if k not in bad]; ctrl=list(rng.choice(others,200,replace=False))
vec=TfidfVectorizer(stop_words='english',sublinear_tf=True,min_df=3).fit([A[i] for i in ids])
M=vec.transform([A[i] for i in ids])
def stat(q,k=200):
  s=(M@vec.transform([cap[q]]).T).toarray().ravel(); top=np.argsort(-s)[:k]
  return chg[top].mean(), np.mean([len(A[ids[j]].split())-len(B[ids[j]].split()) for j in top])
print('baseline changed rate %.3f'%chg.mean())
bs=[stat(q) for q in bad]
for q,(c,w) in zip(bad,bs): print(f'{q:18s} top200 neighbors changed {c:.3f}  words lost/row {w:.2f}')
cs=np.array([stat(q) for q in ctrl])
print('9 bad mean changed %.3f words %.2f | 200 random MC prompts mean changed %.3f (p10 %.3f p90 %.3f) words %.2f'%(np.mean([b[0] for b in bs]),np.mean([b[1] for b in bs]),cs[:,0].mean(),*np.percentile(cs[:,0],[10,90]),cs[:,1].mean()))
broken=sum(bool(re.search(r'\b(is|are|with|of|at|in|a|the)\s*[,.]|\bis the \w+ is\b',B[i])) and not re.search(r'\b(is|are|with|of|at|in|a|the)\s*[,.]',A[i]) for i in ids if chg[ids.index(i)] ) if False else None
