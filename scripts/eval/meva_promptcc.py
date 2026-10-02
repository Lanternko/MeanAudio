#!/usr/bin/env python3
"""095: frozen-input MEva shadow rescore; queue guest owns GPU and notifications."""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from meva_runtime import ROOT, RUNTIME, Evaluator, sha
sys.path.insert(0, str(ROOT / 'scripts/experiment_harness'))
from notification_receipts import atomic_secure_json

OUT = RUNTIME / 'results'
LOCK = ROOT / 'docs/experiments/meva_095_model_lock.json'
MANIFEST = RUNTIME / 'manifest.json'

def result_path(row):
    return OUT / row['set'] / (row['key'] + '.json')

def existing(row, binding):
    path = result_path(row)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get('input_sha256') != row['audio_sha256'] or record.get('model_binding') != binding:
        raise ValueError('Stale result binding: ' + str(path))
    if record.get('status') != 'scored' or not math.isfinite(record['meva_raw']):
        raise ValueError('Invalid previous score: ' + str(path))
    return record

def score(rows, binding):
    model = None
    for i, row in enumerate(rows):
        if existing(row, binding) is not None:
            continue
        if sha(row['audio_path']) != row['audio_sha256']:
            raise ValueError('Audio changed: ' + row['audio_path'])
        fs = os.statvfs(RUNTIME)
        if fs.f_bavail * fs.f_frsize < 50 * (1 << 30):
            raise RuntimeError('Storage hard stop: free bytes below 50 GiB')
        if model is None:
            model = Evaluator()
        start = time.monotonic()
        try:
            record = model.score(row['audio_path'])
        except Exception as exc:
            atomic_secure_json(RUNTIME / 'last_error.json',
                               {'set':row['set'], 'key':row['key'], 'error':str(exc)[:500]})
            raise
        record.update(set=row['set'], key=row['key'], model_binding=binding,
                      input_sha256=row['audio_sha256'], seconds=time.monotonic()-start)
        atomic_secure_json(result_path(row), record)
        atomic_secure_json(RUNTIME / 'progress.json',
                           {'last_key':row['key'], 'set':row['set'], 'index':i+1,
                            'updated_epoch':time.time()})
        if (i+1)%50 == 0:
            print(f"{row['set']} {i+1}/{len(rows)} score={record['meva_raw']:.4f}", flush=True)
    del model

def preflight():
    c = json.loads((ROOT/'docs/experiments/meva_095_contract.json').read_text())
    for item in c['raw_bindings']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Raw binding mismatch: '+item['path'])
    if sha(MANIFEST) != c['manifest_sha256']:
        raise ValueError('Manifest changed')
    if os.statvfs(RUNTIME).f_bavail * os.statvfs(RUNTIME).f_frsize < 50*(1<<30):
        return 75
    print('PASS: immutable inputs, runtime source and storage', flush=True)
    return 0

def summarize(rows, binding):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    all_records = [existing(r, binding) for r in rows]
    if any(r is None for r in all_records):
        raise ValueError('Missing scores; no completion')
    groups = sorted({r['set'] for r in rows})
    report = {'model_binding':binding, 'manifest_sha256':sha(MANIFEST), 'n_scored':len(rows),
              'mode':'shadow', 'aes_replaced':False, 'cells':{}, 'external':{}}
    for group in groups:
        pairs = [(r,v) for r,v in zip(rows,all_records) if r['set']==group]
        values = np.array([v['meva_raw'] for r,v in pairs])
        report['cells'][group]={'n':len(pairs),'meva_mean':float(values.mean()),
                               'meva_std':float(values.std()),
                               'min':float(values.min()),'max':float(values.max())}
        path = RUNTIME / 'tables' / (group+'.tsv');path.parent.mkdir(exist_ok=True)
        with path.open('w',newline='') as f:
            fields=['key','prompt_id','meva_raw','aes_pq','aes_ce','human_ovl','source_protocol']
            w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader()
            for r,v in pairs:
                w.writerow({k:(v['meva_raw'] if k=='meva_raw' else r.get(k,'')) for k in fields})
    # PAM labels are external to MEva's five training benchmarks. No fitting or selection here.
    pairs=[(r,v) for r,v in zip(rows,all_records) if r['set']=='pam' and r['system']!='real']
    if pairs:
        h=np.array([r['human_ovl'] for r,v in pairs]);m=np.array([v['meva_raw'] for r,v in pairs])
        pq=np.array([r['aes_pq'] for r,v in pairs]);ce=np.array([r['aes_ce'] for r,v in pairs])
        prompts=sorted({r['prompt_id'] for r,v in pairs})
        ix={p:np.array([i for i,(r,v) in enumerate(pairs) if r['prompt_id']==p]) for p in prompts}
        def metrics(indices):
            truth=h[indices]
            return np.array([spearmanr(x[indices],truth).statistic for x in [m,pq,ce]])
        point=metrics(np.arange(len(pairs)))
        rng=np.random.default_rng(20261003)
        samples=np.array([metrics(np.concatenate([ix[p] for p in rng.choice(prompts,len(prompts),replace=True)]))
                          for _ in range(2000)])
        diffs=samples[:,[0]]-samples[:,1:]
        ci=np.quantile(diffs,[.0125,.9875],axis=0) # Bonferroni simultaneous two-comparison 95% coverage.
        rho_ci=np.quantile(samples[:,0],[.025,.975])
        # Also compare same-prompt system pair judgments, avoiding between-system-only claims.
        def agreement(x):
            hit,total=0.,0
            for indices in ix.values():
                for j,a in enumerate(indices):
                    for b in indices[j+1:]:
                        if h[a]==h[b]: continue
                        total+=1
                        hit+=.5 if x[a]==x[b] else float((x[a]-x[b])*(h[a]-h[b])>0)
            return hit/total if total else None
        report['external']={'dataset':'PAM generated 400','n':len(pairs),'n_prompts':len(prompts),
           'spearman':dict(zip(['meva','aes_pq','aes_ce'],map(float,point))),
           'pearson':{name:float(pearsonr(x,h).statistic) for name,x in [('meva',m),('aes_pq',pq),('aes_ce',ce)]},
           'meva_rho_ci95':rho_ci.tolist(),'delta_rho':(point[0]-point[1:]).tolist(),
           'delta_rho_simultaneous_ci':ci.T.tolist(),
           'same_prompt_pair_agreement':{name:agreement(x) for name,x in [('meva',m),('aes_pq',pq),('aes_ce',ce)]},
           'external_gate':bool((ci[0]>0).all() and (point[0]-point[1:]>=.05).all() and rho_ci[0]>=.60),
           'adoption':'requires PromptCC human blind evaluation even if external gate passes'}
    atomic_secure_json(RUNTIME/'summary.json',report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',action='store_true');p.add_argument('--validate-only',action='store_true')
    p.add_argument('--limit',type=int);p.add_argument('--dataset',choices=['pam','all'],default='all')
    a=p.parse_args()
    if a.preflight: return preflight()
    binding=sha(LOCK)
    rows=json.loads(MANIFEST.read_text())
    if a.dataset=='pam':rows=[r for r in rows if r['set']=='pam']
    if a.limit:rows=rows[:a.limit]
    if not a.validate_only: score(rows,binding)
    report=summarize(rows,binding)
    print(json.dumps({'n':report['n_scored'],'external':report['external']},indent=2),flush=True)
    return 0

if __name__=='__main__': raise SystemExit(main())
