#!/usr/bin/env python3
"""Contract-bound canonical baseline and paired gain sensitivity experiment."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path
import numpy as np

ROOT = Path('/home/kojiek/MeanAudio')
CONTRACT = ROOT / 'docs/experiments/loudness_aes_cfg3_20260911_contract.json'
AXES = ('CE', 'CU', 'PC', 'PQ')
GAINS = (0, -3, -6, -9)
NEGATIVE = 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def atomic(p, value):
    p = Path(p); p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    t = p.with_name('.' + p.name + '.tmp.' + str(os.getpid()))
    with t.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)
    fd = os.open(p.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def config(): return json.loads(CONTRACT.read_text())
def out(c): return Path(c['storage']['path'])
def binding(): return digest(CONTRACT)
def records(c):
    from score_musiccaps_per_item import read_musiccaps_tsv
    return read_musiccaps_tsv(Path(c['tsv']), expected_count=c['protocol']['rows'])
def stamp(c, phase, count=0):
    atomic(out(c)/'progress.json', {'phase': phase, 'count': count, 'contract_sha256': binding()})
def notify(c, event, summary):
    sys.path.insert(0, str(ROOT/'scripts/experiment_harness'))
    from notification_receipts import deliver_required
    return deliver_required(contract_path=CONTRACT, launcher_path=Path(os.environ.get('GPU_QUEUE_JOB_SCRIPT', c['bindings']['launcher'])),
        event=event, status='start', summary=summary, idempotency_key=f"{c['experiment_id']}:{c['run_id']}:{event}",
        notifier=Path(c['notification_receipts']['notifier']), python=Path(sys.executable), root=Path(c['notification_receipts']['root']))
def capacity(c):
    for path in c['resource_budget']['writable_filesystems']:
        fs=os.statvfs(path); free=fs.f_bavail*fs.f_frsize
        if free<c['storage']['hard_stop_free_bytes']: raise SystemExit(75)

def signal_stats(x, sr=16000):
    import pyloudnorm as pyln
    x=np.asarray(x,dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all(): raise ValueError('invalid waveform')
    rms=float(np.sqrt(np.mean(x*x))); peak=float(np.max(np.abs(x)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        lu=float(pyln.Meter(sr, filter_class='K-weighting', block_size=.4).integrated_loudness(x))
    window=np.hanning(len(x)); spec=np.abs(np.fft.rfft(x*window)); total=spec.sum()
    return {'lufs': lu if math.isfinite(lu) else None,
        'lufs_status': 'finite' if math.isfinite(lu) else 'below_gate_or_silence',
        'rms_dbfs':20*math.log10(rms) if rms>0 else None,
        'crest_linear':peak/rms if rms>0 else None,
        'crest_db':20*math.log10(peak/rms) if rms>0 else None,
        'peak':peak, 'peak_dbfs':20*math.log10(peak) if peak>0 else None,
        'clipped_fraction':float(np.mean(np.abs(x)>=.999)),
        'silence_fraction':float(np.mean(np.abs(x)<1e-3)),
        'centroid_hz':float(np.sum(spec*np.fft.rfftfreq(len(x),1/sr))/total) if total>0 else None}

def read_audio(p):
    import soundfile as sf
    x,sr=sf.read(p,dtype='float32')
    if sr!=16000 or x.ndim!=1 or not np.isfinite(x).all() or len(x)<6400: raise ValueError(f'invalid mono16k audio: {p}')
    return x

def gain_copy(source, target, gain):
    import soundfile as sf
    x=read_audio(source); y=x*np.float32(10**(gain/20))
    sf.write(target,y,16000,subtype='FLOAT')
    z=read_audio(target)
    if not np.array_equal(y,z): raise ValueError('float WAV roundtrip mismatch')
    a=float(np.sqrt(np.mean(x.astype(float)**2))); b=float(np.sqrt(np.mean(z.astype(float)**2)))
    if a:
        if abs(20*np.log10(b/a)-gain)>1e-5: raise ValueError('RMS gain mismatch')
        if abs(20*np.log10((np.max(np.abs(z))/b)/(np.max(np.abs(x))/a)))>1e-5: raise ValueError('crest changed')
    return z

def checked(p, c):
    v=json.loads(Path(p).read_text())
    if v.get('contract_sha256')!=binding(): raise ValueError(f'stale contract artifact: {p}')
    return v

def baseline(c):
    from score_musiccaps_per_item import validate_audio_directory
    rec=records(c); directory=out(c)/'_audio'/'baseline'; marker=out(c)/'audio_manifest.json'
    if marker.exists():
        manifest=checked(marker,c)
        if set(manifest['audio_sha256'])!={r.id for r in rec}: raise ValueError('audio ID drift')
        for i,h in manifest['audio_sha256'].items():
            if digest(directory/(i+'.flac'))!=h: raise ValueError('baseline audio drift')
        notify(c,'generation_pass','051 exact 5521 valid baseline audio files and hashes persisted; next: signal and AES scoring.')
        return manifest
    # Partial generation has no resumable RNG state. Restart the complete ordered seed42 generation.
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    expected={r.id for r in rec}
    for p in directory.iterdir():
        if p.is_symlink() or not p.is_file() or p.suffix!='.flac' or p.stem not in expected: raise ValueError('unsafe partial audio')
        p.unlink()
    capacity(c); notify(c,'generation_preflight_pass','051 canonical MusicCaps5521 MF25 CFG3 fidelity8 seed42 generation gate passed.')
    stamp(c,'generation')
    generated=subprocess.run(c['commands_generation']['fidelity8'],cwd=ROOT)
    if generated.returncode: raise SystemExit(generated.returncode if generated.returncode>0 else 128-generated.returncode)
    paths=validate_audio_directory(directory,rec)
    manifest={'contract_sha256':binding(),'audio_sha256':{i:digest(p) for i,p in paths.items()}}
    atomic(marker,manifest)
    notify(c,'generation_pass','051 exact 5521 valid baseline audio files and hashes persisted; next: signal and AES scoring.')
    return manifest

def item_valid(v, i, audio_hash, gain):
    if v.get('contract_sha256')!=binding() or v.get('id')!=i or v.get('source_sha256')!=audio_hash or v.get('gain_db')!=gain:
        raise ValueError('stale per-item score')
    if set(v['aes'])!=set(AXES) or not all(math.isfinite(float(v['aes'][k])) for k in AXES): raise ValueError('invalid AES')
    return v

def score_gain(c, gain):
    from score_musiccaps_per_item import load_aes_predictor, _aes_batch
    manifest=checked(out(c)/'audio_manifest.json',c); rec=records(c)
    source=out(c)/'_audio'/'baseline'; dest=out(c)/'items'/str(gain); temp=out(c)/'_gain_batch'
    dest.mkdir(parents=True,exist_ok=True); temp.mkdir(mode=0o700,exist_ok=True)
    todo=[]
    for r in rec:
        p=dest/(r.id+'.json')
        if p.exists(): item_valid(checked(p,c),r.id,manifest['audio_sha256'][r.id],gain)
        else: todo.append(r)
    predictor=None
    if todo:
        predictor=load_aes_predictor(Path(c['aes_snapshot']),device='cuda',batch_size=c['protocol']['scoring_batch_size'])
    for start in range(0,len(todo),c['protocol']['scoring_batch_size']):
        capacity(c); batch=todo[start:start+c['protocol']['scoring_batch_size']]; paths=[]; signals=[]
        for r in batch:
            src=source/(r.id+'.flac')
            if digest(src)!=manifest['audio_sha256'][r.id]: raise ValueError('source audio changed')
            p=temp/(r.id+'.wav'); x=gain_copy(src,p,gain)
            paths.append(p); signals.append(signal_stats(x))
        # Verify the exact padded model input retains the gain, before model inference.
        if predictor is not None:
            import torch
            from audiobox_aesthetics.infer import make_inference_batch
            for r,p in zip(batch,paths):
                x=torch.from_numpy(read_audio(source/(r.id+'.flac'))).unsqueeze(0)
                y=predictor._load_audio(p)
                px=make_inference_batch([x],10,10,sample_rate=16000)[0]
                py=make_inference_batch([y],10,10,sample_rate=16000)[0]
                if not all(torch.allclose(b,a*np.float32(10**(gain/20)),atol=1e-7,rtol=1e-6) for a,b in zip(px,py)):
                    raise ValueError('scorer input gain erased')
            if predictor.model.wavlm_model.cfg.normalize: raise ValueError('model normalizes waveform gain')
        values=_aes_batch(predictor,paths)
        for r,p,s,v in zip(batch,paths,signals,values):
            value={'contract_sha256':binding(),'id':r.id,'source_sha256':manifest['audio_sha256'][r.id],
                'gain_db':gain,'transformed_sha256':digest(p),'signal':s,'aes':v}
            item_valid(value,r.id,manifest['audio_sha256'][r.id],gain); atomic(dest/(r.id+'.json'),value)
            p.unlink()  # exact registered transient only, after atomic score commit
        stamp(c,'aes_'+str(gain),len(rec)-len(todo)+start+len(batch))
    # Existing per-item records are evidence; phase completion requires the exact ID set.
    if {p.stem for p in dest.glob('*.json')}!={r.id for r in rec}: raise ValueError('score ID mismatch')
    notify(c,'aes_'+str(gain)+'_pass',f'051 gain {gain} dB: all 5521 AES records persisted; next registered phase.')

def clap(c):
    from score_musiccaps_per_item import load_clap_model, _clap_batch
    target=out(c)/'clap.json'; rec=records(c)
    if target.exists():
        v=checked(target,c)
        if set(v['per_clip'])!={r.id for r in rec} or not all(math.isfinite(x) for x in v['per_clip'].values()): raise ValueError('invalid CLAP')
        notify(c,'clap_pass','051 canonical baseline CLAP5521 complete; gain contrasts use AES only.')
        return
    model=load_clap_model(Path(c['clap_checkpoint']),device='cuda',local_files_only=True); per={}
    import torch
    with torch.inference_mode():
        for start in range(0,len(rec),32):
            capacity(c); batch=rec[start:start+32]
            values=_clap_batch(model,[out(c)/'_audio'/'baseline'/(r.id+'.flac') for r in batch],[r.caption for r in batch])
            per.update({r.id:v for r,v in zip(batch,values)}); stamp(c,'clap',len(per))
    atomic(target,{'contract_sha256':binding(),'per_clip':per})
    notify(c,'clap_pass','051 canonical baseline CLAP5521 complete; gain contrasts use AES only.')

def mean_ci(values,rng,reps):
    x=np.asarray(values,dtype=float)
    if not len(x): return {'n':0,'mean':None,'ci95':None}
    b=[]
    for start in range(0,reps,100):
        ix=rng.integers(0,len(x),size=(min(100,reps-start),len(x)))
        b.extend(x[ix].mean(axis=1))
    return {'n':len(x),'mean':float(x.mean()),'ci95':np.quantile(b,[.025,.975]).tolist()}

def group_indices(values,groups):
    """Quantile cutpoints; ties stay together. Empty bins remain visible."""
    values=np.asarray(values,dtype=float); cuts=np.quantile(values,np.arange(1,groups)/groups)
    membership=np.searchsorted(cuts,values,side='left')
    return cuts.tolist(),[np.flatnonzero(membership==k) for k in range(groups)]

def independent_delta(high,low,rng,reps):
    high=np.asarray(high); low=np.asarray(low)
    if not len(high) or not len(low): return {'mean':None,'ci95':None,'status':'empty_bin'}
    means=[]
    for start in range(0,reps,100):
        n=min(100,reps-start)
        means.extend(high[rng.integers(0,len(high),(n,len(high)))].mean(1)-low[rng.integers(0,len(low),(n,len(low)))].mean(1))
    return {'mean':float(high.mean()-low.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}

def analyze_data(items,seed=20260911,reps=10000):
    from scipy.stats import spearmanr
    ids=sorted(items[0]); rng=np.random.default_rng(seed); base=items[0]
    result={'n':len(ids),'grouping':{},'gain_deltas':{},'crest_within_lufs':[],
            'inference':'Pointwise 95% intervals; fixed observed cutpoints; PQ primary within each of two separate questions; secondary outcomes exploratory.'}
    for measure in ('lufs','rms_dbfs','crest_db'):
        valid=[i for i in ids if base[i]['signal'][measure] is not None]
        values=np.array([base[i]['signal'][measure] for i in valid]); groups=[]; contrast={}; corr={}
        if len(valid):
            cuts,idx=group_indices(values,5)
            for k,ix in enumerate(idx):
                groups.append({'group':k+1,'n':len(ix),'range':[float(values[ix].min()),float(values[ix].max())] if len(ix) else None,
                    'aes':{a:mean_ci([base[valid[j]]['aes'][a] for j in ix],rng,reps) for a in AXES}})
            for a in AXES:
                y=np.array([base[i]['aes'][a] for i in valid])
                contrast[a]=independent_delta(y[idx[-1]],y[idx[0]],rng,reps)
                rho=float(spearmanr(values,y).statistic) if np.ptp(values)>0 and np.ptp(y)>0 else None
                corr[a]=rho
            if measure=='lufs':
                for k,ix in enumerate(idx):
                    subset=[valid[j] for j in ix if base[valid[j]]['signal']['crest_db'] is not None]
                    if not subset: continue
                    cc,ci=group_indices([base[i]['signal']['crest_db'] for i in subset],3)
                    result['crest_within_lufs'].append({'lufs_group':k+1,'crest_cuts_db':cc,'counts':[len(z) for z in ci],
                        'high_minus_low':{a:independent_delta([base[subset[j]]['aes'][a] for j in ci[-1]], [base[subset[j]]['aes'][a] for j in ci[0]],rng,reps) for a in AXES}})
        else: cuts=[]
        result['grouping'][measure]={'cuts':cuts,'excluded_ids':sorted(set(ids)-set(valid)),'groups':groups,'Q5_minus_Q1':contrast,'spearman':corr}
    for gain in GAINS[1:]:
        result['gain_deltas'][str(gain)]={a:mean_ci([items[gain][i]['aes'][a]-base[i]['aes'][a] for i in ids],rng,reps) for a in AXES}
    result['gain_deltas_by_baseline_lufs']={}
    valid=[i for i in ids if base[i]['signal']['lufs'] is not None]
    if valid:
        _,idx=group_indices([base[i]['signal']['lufs'] for i in valid],5)
        for k,ix in enumerate(idx):
            result['gain_deltas_by_baseline_lufs'][str(k+1)]={str(g):{a:mean_ci([items[g][valid[j]]['aes'][a]-base[valid[j]]['aes'][a] for j in ix],rng,reps) for a in AXES} for g in GAINS[1:]}
    result['primary']={'association_PQ_Q5_minus_Q1':result['grouping']['lufs']['Q5_minus_Q1'].get('PQ'),
        'gain_PQ_minus6_minus0':result['gain_deltas']['-6']['PQ']}
    return result

def load_items(c):
    manifest=checked(out(c)/'audio_manifest.json',c); ids={r.id for r in records(c)}; items={}
    for g in GAINS:
        d=out(c)/'items'/str(g)
        if {p.stem for p in d.glob('*.json')}!=ids: raise ValueError('missing/extra score ID')
        items[g]={i:item_valid(checked(d/(i+'.json'),c),i,manifest['audio_sha256'][i],g) for i in sorted(ids)}
    return items

def report(c):
    items=load_items(c); stamp(c,'analysis'); result=analyze_data(items,c['analysis']['bootstrap_seed'],c['analysis']['bootstrap_replicates'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids=sorted(items[0]); fig,axs=plt.subplots(2,4,figsize=(16,7))
    for ax,a in zip(axs[0],AXES):
        use=[i for i in ids if items[0][i]['signal']['lufs'] is not None]
        ax.scatter([items[0][i]['signal']['lufs'] for i in use],[items[0][i]['aes'][a] for i in use],s=2,alpha=.15)
        groups=result['grouping']['lufs']['groups']; nonempty=[g for g in groups if g['n']]
        x=[sum(g['range'])/2 for g in nonempty]; y=[g['aes'][a]['mean'] for g in nonempty]
        err=np.array([[g['aes'][a]['mean']-g['aes'][a]['ci95'][0],g['aes'][a]['ci95'][1]-g['aes'][a]['mean']] for g in nonempty]).T
        if nonempty: ax.errorbar(x,y,yerr=err,color='black',marker='o')
        ax.set(xlabel='Integrated LUFS',ylabel='AES '+a)
    for ax,a in zip(axs[1],AXES):
        means=[0]+[result['gain_deltas'][str(g)][a]['mean'] for g in GAINS[1:]]
        ax.plot(GAINS,means,marker='o'); ax.set(xlabel='Gain dB',ylabel='Paired delta '+a); ax.axhline(0,color='gray',ls='--')
    fig.tight_layout(); fig.savefig(out(c)/'loudness_aes.png',dpi=160); plt.close(fig)
    cp=checked(out(c)/'clap.json',c)
    if set(cp['per_clip'])!=set(ids) or not all(math.isfinite(x) for x in cp['per_clip'].values()): raise ValueError('CLAP coverage')
    table=out(c)/'per_clip.csv'
    with table.open('w',newline='') as f:
        cols=['id','gain_db','source_sha256','lufs','rms_dbfs','crest_linear','crest_db','peak_dbfs','silence_fraction','clipped_fraction','centroid_hz',*AXES,'baseline_clap']
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for g in GAINS:
            for i in ids:
                v=items[g][i]; w.writerow({'id':i,'gain_db':g,'source_sha256':v['source_sha256'],**{k:v['signal'][k] for k in cols[3:11]},**v['aes'],'baseline_clap':cp['per_clip'][i]})
    md=['# Loudness / AES — MusicCaps5521 MF25 CFG3 fidelity8 seed42 NoMask FP32',
        'Single checkpoint; exploratory secondary gain intervention. AES sensitivity is not a human quality judgment.',
        '', '| LUFS group | n | LUFS range | CE | CU | PC | PQ |','|---|---:|---|---:|---:|---:|---:|']
    for g in result['grouping']['lufs']['groups']:
        md.append('| '+str(g['group'])+' | '+str(g['n'])+' | '+str(g['range'])+' | '+' | '.join(str(g['aes'][a]['mean']) for a in AXES)+' |')
    md+=['','Primary effects (pointwise 95% bootstrap CI; no equivalence or automatic promotion claim):','```json',json.dumps(result['primary'],indent=2),'```','', 'Full intervals, RMS/crest groups, within-LUFS crest comparisons and gain-by-LUFS results: summary.json.', '![Analysis](loudness_aes.png)']
    (out(c)/'report.md').write_text('\n'.join(md)+'\n')
    result.update({'contract_sha256':binding(),'protocol':c['protocol'],'label':c['label'],
        'baseline_clap_mean':float(np.mean(list(cp['per_clip'].values()))),
        'artifacts_sha256':{str(p.relative_to(out(c))):digest(p) for p in [out(c)/'audio_manifest.json',out(c)/'clap.json',table,out(c)/'loudness_aes.png',out(c)/'report.md']},
        'item_hashes':{str(g):{i:digest(out(c)/'items'/str(g)/(i+'.json')) for i in ids} for g in GAINS},
        'decision':'analysis_complete_no_automatic_promotion'})
    atomic(out(c)/'summary.json',result); validate_all(c)
    notify(c,'analysis_pass','051 complete: LUFS/RMS/crest strata and paired AES gain contrasts with 95% CIs; no automatic promotion.')

def validate_all(c):
    v=checked(out(c)/'summary.json',c); items=load_items(c)
    if v['n']!=c['protocol']['rows'] or v['protocol']!=c['protocol'] or v['label']!=c['label']: raise ValueError('summary identity')
    for rel,h in v['artifacts_sha256'].items():
        if digest(out(c)/rel)!=h: raise ValueError('report artifact drift')
    for g in GAINS:
        if set(v['item_hashes'][str(g)])!=set(items[g]): raise ValueError('item hashes incomplete')
        for i,h in v['item_hashes'][str(g)].items():
            if digest(out(c)/'items'/str(g)/(i+'.json'))!=h: raise ValueError('item drift')
    manifest=checked(out(c)/'audio_manifest.json',c)
    for i,h in manifest['audio_sha256'].items():
        if digest(out(c)/'_audio'/'baseline'/(i+'.flac'))!=h: raise ValueError('retained audio drift')
    print(f"PASS: complete {c['protocol']['rows']} baseline / {4*c['protocol']['rows']} AES records / retained hashes / analysis artifacts")

def main():
    c=config()
    if '--validate-only' in sys.argv: validate_all(c); return
    if '--phase' in sys.argv:
        phase=sys.argv[sys.argv.index('--phase')+1]
        if phase=='clap': clap(c)
        elif phase=='report': report(c)
        else: score_gain(c,int(phase))
        return
    baseline(c)
    for phase in ['0','-3','-6','-9','clap','report']:
        capacity(c)
        completed=subprocess.run([sys.executable,__file__,'--phase',phase],cwd=ROOT)
        if completed.returncode: raise SystemExit(completed.returncode if completed.returncode>0 else 128-completed.returncode)
    validate_all(c)
if __name__=='__main__': main()
