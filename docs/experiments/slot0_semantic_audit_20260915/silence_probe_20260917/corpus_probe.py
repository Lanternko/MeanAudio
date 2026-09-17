import csv,re,json,numpy as np,collections,sys
csv.field_size_limit(10**9)
def load(p): return {r['id']:r['caption'] for r in csv.DictReader(open(p),delimiter='\t')}
A=load('/home/kojiek/exps_nvme/slot0_rowmatched/arm_inputs/phase8_caption2p0_slot0_rowmatched_train.tsv')
B=load('/home/kojiek/exps_nvme/slot0nm/arm_inputs/phase8_caption2p0_slot0nm_train.tsv')
ids=sorted(A); assert set(ids)==set(B)
chg=np.array([A[i]!=B[i] for i in ids]); print('rows',len(ids),'changed',chg.sum(),'%.1f%%'%(100*chg.mean()))
low={i:A[i].lower() for i in ids}
MEL=r'guitar|piano|synth|keyboard|vocal|sing|voice|string|violin|bass|organ|horn|brass|sax|flute|trumpet|melod'
def drumonly(t): return bool(re.search(r'drum|percussion|beat',t)) and not re.search(MEL,t)
groups={'all':np.ones(len(ids),bool),
 'mentions drum/percussion':np.array([bool(re.search(r'drum|percussion',low[i])) for i in ids]),
 'drum-only (no melodic words)':np.array([drumonly(low[i]) for i in ids]),
 'mentions count/counting':np.array(['count' in low[i] for i in ids]),
 'mentions speech/spoken/talk':np.array([bool(re.search(r'speech|spoken|speak|talk|narrat',low[i])) for i in ids]),
 'didgeridoo':np.array(['didgeridoo' in low[i] for i in ids]),
}
TERMS={'bpm/數字速度':r'\bbpm\b|beats per minute|\d+\s*bpm','拍號':r'\b\d/\d\b|time signature|meter\b','數字':r'\d',
 'tempo 字':r'\btempo\b','fast/slow/moderate':r'\bfast|\bslow|moderate|mid-tempo|upbeat','rhythm/groove':r'rhythm|groove|syncopat|pattern',
 'eighth/sixteenth/quarter':r'eighth|sixteenth|quarter[- ]note|triplet|16th|8th','key/調性':r'\bin the key\b|\bmajor\b|\bminor\b|\bkey of\b'}
rows=[]
for g,m in groups.items():
  idx=[i for i,k in zip(ids,m) if k]; mm=m
  la=np.mean([len(A[i].split()) for i in idx]); lb=np.mean([len(B[i].split()) for i in idx])
  r={'group':g,'n':len(idx),'changed%':100*chg[mm].mean(),'words_before':la,'words_after':lb}
  for t,pat in TERMS.items():
    r[t]=(100*np.mean([bool(re.search(pat,A[i].lower())) for i in idx]),100*np.mean([bool(re.search(pat,B[i].lower())) for i in idx]))
  rows.append(r)
for r in rows:
  print(f"\n## {r['group']}  n={r['n']}  changed {r['changed%']:.1f}%  words {r['words_before']:.1f}->{r['words_after']:.1f}")
  for t in TERMS: print(f"   {t:24s} {r[t][0]:5.1f}% -> {r[t][1]:5.1f}%")
# among changed drum-only rows: word loss and examples
do=[i for i,k in zip(ids,groups['drum-only (no melodic words)']) if k and A[i]!=B[i]]
loss=[len(A[i].split())-len(B[i].split()) for i in do]
print('\ndrum-only changed rows',len(do),'mean words lost %.1f  median %.1f'%(np.mean(loss),np.median(loss)))
allc=[i for i in ids if A[i]!=B[i]]
print('all changed rows mean words lost %.1f'%np.mean([len(A[i].split())-len(B[i].split()) for i in allc]))
# sentences dropped entirely: count rhythm-info words remaining
rng=np.random.default_rng(0)
print('\n=== drum-only changed examples ===')
for i in rng.choice(do,8,replace=False): print('-',i,'\n  BEFORE:',A[i],'\n  AFTER :',B[i])
cnt=[i for i,k in zip(ids,groups['mentions count/counting']) if k and A[i]!=B[i]]
print('\n=== count changed examples ===')
for i in cnt[:4]: print('-',i,'\n  BEFORE:',A[i],'\n  AFTER :',B[i])
json.dump({'drumonly_changed':do},open(sys.argv[1],'w'))
