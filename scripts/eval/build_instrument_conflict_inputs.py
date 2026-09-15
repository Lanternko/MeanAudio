#!/usr/bin/env python3
"""Freeze explicit instrument mentions, per-row negative controls and blind sample."""
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path('/home/kojiek/MeanAudio')
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/instrument_conflict_20260908')
SOURCE = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
SEED = 20260908
FIDELITY = 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
# Explicit instrument nouns only; ambiguous bass, strings, brass and keys excluded.
LEXICON = {
    'piano': r'\bpianos?\b', 'guitar': r'\bguitars?\b',
    'drums': r'\b(?:drums|drum kit|drum set)\b',
    'violin': r'\b(?:violins?|fiddle)\b', 'cello': r'\bcellos?\b',
    'flute': r'\bflutes?\b', 'trumpet': r'\btrumpets?\b',
    'saxophone': r'\b(?:saxophones?|sax)\b', 'clarinet': r'\bclarinets?\b',
    'trombone': r'\btrombones?\b', 'harmonica': r'\bharmonicas?\b',
    'accordion': r'\baccordions?\b', 'banjo': r'\bbanjos?\b',
    'ukulele': r'\b(?:ukuleles?|ukeleles?)\b', 'harp': r'\bharps?\b',
    'sitar': r'\bsitars?\b', 'tabla': r'\btablas?\b',
    'xylophone': r'\bxylophones?\b', 'marimba': r'\bmarimbas?\b',
    'oboe': r'\boboes?\b', 'tuba': r'\btubas?\b',
    'mandolin': r'\bmandolins?\b', 'bagpipes': r'\bbagpipes?\b',
}
# Conservatively exclude entire rows with explicit negation tokens: no guessed absence.
NEGATION = re.compile(r"\b(?:no|not|without|absent|lack|lacks|lacking|neither|nor)\b", re.I)

def mentions(caption):
    return sorted(k for k, pattern in LEXICON.items() if re.search(pattern, caption, re.I))

def freeze(rows):
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('duplicate input IDs')
    rng = random.Random(SEED)
    assignments=[]
    for row in rows:
        found=mentions(row['caption'])
        if not found or NEGATION.search(row['caption']): continue
        target=rng.choice(found)
        assignments.append({'id':row['id'],'caption':row['caption'],'mentioned':found,'target':target})
    # Balance the actual negative term distribution exactly if a feasible permutation exists.
    # Bipartite augmenting paths match target-term slots to rows excluding that term.
    slots=[a['target'] for a in assignments]
    order=list(range(len(slots))); rng.shuffle(order)
    owners={k:[] for k in sorted(set(slots))}
    capacity=Counter(slots)
    chosen={}
    def augment(row_index, seen_rows, seen_terms):
        if row_index in seen_rows: return False
        seen_rows.add(row_index)
        terms=sorted(capacity, key=lambda t:hashlib.sha256(f'{SEED}:{row_index}:{t}'.encode()).hexdigest())
        for term in terms:
            if term in assignments[row_index]['mentioned'] or term in seen_terms: continue
            seen_terms.add(term)
            if len(owners[term])<capacity[term]:
                owners[term].append(row_index);chosen[row_index]=term;return True
            for old in list(owners[term]):
                if augment(old,seen_rows,seen_terms):
                    owners[term].remove(old);owners[term].append(row_index);chosen[row_index]=term;return True
        return False
    exact=all(augment(i,set(),set()) for i in order)
    if not exact:
        # Fully deterministic balanced fallback; disclose term-frequency mismatch in analysis.
        counts=Counter();chosen={}
        for i in order:
            allowed=[t for t in LEXICON if t not in assignments[i]['mentioned']]
            term=min(allowed,key=lambda t:(counts[t],hashlib.sha256(f'{SEED}:{i}:{t}'.encode()).hexdigest()))
            chosen[i]=term;counts[term]+=1
    for i,a in enumerate(assignments): a['unmentioned']=chosen[i]
    return assignments,exact

def main():
    OUT.mkdir(mode=0o700,parents=True,exist_ok=True)
    if (OUT/'assignments.json').exists(): raise SystemExit('already frozen; refusing overwrite')
    rows=list(csv.DictReader(SOURCE.open(),delimiter='\t'))
    assert len(rows)==5521
    assignments,exact=freeze(rows)
    for arm in ('fidelity8','fidelity8_conflict','fidelity8_unmentioned'):
        with (OUT/f'{arm}.tsv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','caption','negative_prompt'],delimiter='\t');w.writeheader()
            for a in assignments:
                term={'fidelity8':'','fidelity8_conflict':a['target'],'fidelity8_unmentioned':a['unmentioned']}[arm]
                w.writerow({'id':a['id'],'caption':a['caption'],'negative_prompt':FIDELITY+(', '+term if term else '')})
    blind=[];rng=random.Random(SEED+1)
    for target in sorted({a['target'] for a in assignments}):
        candidates=[a for a in assignments if a['target']==target];rng.shuffle(candidates)
        for a in candidates[:8]:
            arms=['fidelity8','fidelity8_conflict','fidelity8_unmentioned'];rng.shuffle(arms)
            blind.append({'id':a['id'],'target':target,'slots':dict(zip('ABC',arms))})
    spec={'seed':SEED,'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'lexicon':LEXICON,'negation_exclusion':NEGATION.pattern,'n':len(assignments),'assignments':assignments,
          'control_frequency_exact_match':exact,'target_counts':dict(Counter(a['target'] for a in assignments)),
          'unmentioned_counts':dict(Counter(a['unmentioned'] for a in assignments)),
          'listening_sample':blind,'listening_sampling':'up to 8 per target instrument, seed20260909, randomized ABC, keep all three audios'}
    (OUT/'assignments.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps({k:spec[k] for k in ['n','control_frequency_exact_match','target_counts','unmentioned_counts']}))
    print('listening_triplets',len(blind))
if __name__=='__main__':main()
