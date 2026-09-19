import csv, re, sys, json, collections, random
csv.field_size_limit(10**9)
def load(p):
    d={}
    with open(p) as f:
        r=csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        for row in r: d[row['id']]=row['caption']
    return d
src=load('/mnt/HDD/kojiek/phase4_jamendo_data/phase8_qwen_caption10s_multisent_train.tsv')
clean=load('/home/kojiek/exps_nvme/slot0clean/arm_inputs/phase8_caption2p0_slot0clean_train.tsv')
s4=load('/home/kojiek/exps_nvme/slot4v2/arm_inputs/phase8_caption2p0_slot4v2_train.tsv')
nm=load('/home/kojiek/exps_nvme/slot0nm/arm_inputs/phase8_caption2p0_slot0nm_train.tsv')
print(len(src),len(clean),len(s4),len(nm))
regen={k for k in src if clean[k]!=src[k]}
dig={k for k in src if s4.get(k)!=src[k]}
changed={k for k in nm if nm[k]!=src[k]}
print('changed',len(changed),'regen',len(regen&changed))
rest=changed-regen
d_only=[k for k in rest if s4[k]!=src[k] and nm[k]==s4[k]]
nm_only=[k for k in rest if s4[k]==src[k]]
both=[k for k in rest if s4[k]!=src[k] and nm[k]!=s4[k]]
print('digits-stage only',len(d_only),'nm-stage only',len(nm_only),'both',len(both))
tok=lambda s: re.findall(r"[A-Za-z0-9/'\-]+",s.lower())
tot=sum(len(tok(src[k])) for k in nm); 
ch=0
for k in changed:
    a=collections.Counter(tok(src[k])); b=collections.Counter(tok(nm[k]))
    ch+=sum((a-b).values())+sum((b-a).values())
print('token churn / corpus tokens = %.4f'%(ch/tot))
json.dump({'regen':sorted(regen&changed),'d_only':d_only,'nm_only':nm_only,'both':both},open('groups.json','w'))
import pickle; pickle.dump((src,clean,s4,nm),open('corp.pkl','wb'))
