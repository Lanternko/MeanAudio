import pickle, json, re, collections, random, difflib
src,clean,s4,nm=pickle.load(open('corp.pkl','rb')); g=json.load(open('groups.json'))
tok=lambda s: re.findall(r"[a-z0-9#/'\-]+",s.lower())
STOP=set("a an the and or of in on at to with is are was be been being it its this that which as by for from while there plays played playing set has have into throughout overall all very also but than more most some slightly quite".split())
NUMW=set("one two three four five six seven eight nine ten eleven twelve sixteen twenty thirty forty fifty sixty seventy eighty ninety hundred half quarter triple duple compound quadruple".split())
MEAS=set("""bpm beats beat per minute minutes hz khz db decibels decibel key keys major minor a b c d e f g sharp flat # time signature signatures common waltz meter metre mode modal dorian mixolydian lydian phrygian aeolian ionian locrian tonality tonal tonic scale around approximately about roughly consistent steady tempo speed pace measure measures bar bars count rate note notes chord chords""".split())
def sents(s): return [x.strip() for x in re.split(r'(?<=[.!?])\s+',s) if x.strip()]
def classify(a,b):
    ta=collections.Counter(tok(a)); tb=collections.Counter(tok(b))
    rem=[w for w in (ta-tb).elements()]; add=[w for w in (tb-ta).elements()]
    bad_rem=[w for w in rem if w not in STOP and w not in MEAS and w not in NUMW and not re.search(r'\d',w)]
    bad_add=[w for w in add if w not in STOP]
    return rem,add,bad_rem,bad_add
def pairs(A,B):
    sa,sb=sents(A),sents(B)
    sm=difflib.SequenceMatcher(a=sa,b=sb,autojunk=False)
    out=[]
    for op,i1,i2,j1,j2 in sm.get_opcodes():
        if op=='equal': continue
        out.append((' '.join(sa[i1:i2]),' '.join(sb[j1:j2])))
    return out
res={}
for stage,keys,A,B in [('digits(slot0->slot4v2)',g['d_only']+g['both'],src,s4),('nm(slot4v2->slot0nm)',g['nm_only']+g['both'],s4,nm)]:
    n=0; rows_bad_rem=0; rows_bad_add=0; brc=collections.Counter(); bac=collections.Counter(); ex_rem=[]; ex_add=[]
    trig=collections.Counter()
    for k in keys:
        rb=ra=False
        for a,b in pairs(A[k],B[k]):
            rem,add,br,ba=classify(a,b)
            for w in rem:
                if re.search(r'\d',w): trig['digit']+=1
            if br: rb=True; brc.update(br); ex_rem.append((k,a,b,br))
            if ba: ra=True; bac.update(ba); ex_add.append((k,a,b,ba))
        n+=1; rows_bad_rem+=rb; rows_bad_add+=ra
    print('\n==',stage,'rows',n,'rows w/ non-measure content removed',rows_bad_rem,'rows w/ words added',rows_bad_add)
    print('top removed non-measure:',brc.most_common(40))
    print('top added:',bac.most_common(40))
    res[stage]=(ex_rem,ex_add)
pickle.dump(res,open('ex.pkl','wb'))
