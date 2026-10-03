"""Known-noise, dynamics, order and neural-stem interventions on replayed audio.
Derivatives are generated in memory; receipts bind source/recipe/waveform hashes.
"""
import os,json,csv,hashlib,time,gc,math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
import pyloudnorm as pyln
from generate import ROOT,sha,dump
SR=16000;AXES=['PQ','CE','CU','PC'];METER=pyln.Meter(SR)

def loudness(y):return float(METER.integrated_loudness(y))

def normalize(y):
    y=np.asarray(y,dtype=np.float32)
    gain=1.
    # Lift a nonzero quiet derivative above the BS1770 absolute gate first.
    rms=float(np.sqrt(np.mean(y.astype(float)**2)))
    if rms<1e-12:raise ValueError('zero-energy derivative')
    if not np.isfinite(loudness(y)):
        gain=10**((-40-20*np.log10(rms))/20);y=y*gain
    for _ in range(12):
        lu=loudness(y)
        if not np.isfinite(lu):raise ValueError('nonfinite loudness; cannot normalize')
        error=-23-lu
        if abs(error)<.005:return y.astype(np.float32),gain,lu
        g=10**(error/20);y=y*g;gain*=g
    raise ValueError('unstable BS1770 loudness gate')

def waveform_hash(y):return hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()

def features(y):
    z=y[:len(y)//320*320].reshape(-1,320);rms=np.sqrt(np.mean(z.astype(float)**2,1))
    lu=loudness(y)
    return dict(rms_db=float(20*np.log10(max(1e-12,np.sqrt(np.mean(y.astype(float)**2))))),frame_p10_db=float(20*np.log10(max(1e-12,np.percentile(rms,10)))),lufs=lu if np.isfinite(lu) else None,peak=float(np.max(np.abs(y))))

def interventions(y,seed):
    yield 'clean',y,'clean'
    for color in ['white','pink']:
        z=np.random.default_rng(seed).standard_normal(len(y))
        if color=='pink':
            s=np.fft.rfft(z);s/=np.sqrt(np.maximum(np.fft.rfftfreq(len(y),1/SR),20));s[0]=0;z=np.fft.irfft(s,n=len(y))
        z/=np.sqrt(np.mean(z*z));base=np.sqrt(np.mean(y.astype(float)**2))
        for snr in [50,40,30,20]:
            edited=(y+z*base*10**(-snr/20)).astype(np.float32)
            actual=20*np.log10(np.linalg.norm(y.astype(float))/np.linalg.norm((edited-y).astype(float)))
            assert abs(actual-snr)<.002
            yield f'{color}{snr}',edited,'clean'
    fr=y[:len(y)//320*320].reshape(-1,320);r=np.sqrt(np.mean(fr*fr,1));mask=(r<=np.percentile(r,20)).astype(float)
    env=signal.resample_poly(mask,320,1);env=np.pad(env,(0,max(0,len(y)-len(env))),mode='edge')[:len(y)];env=np.clip(env,0,1)
    for db in [6,12]:yield f'quiet_m{db}',(y*(1-env+env*10**(-db/20))).astype(np.float32),'clean'
    low=signal.sosfilt(signal.butter(2,1200,fs=SR,output='sos'),y)
    yield 'dark_m6',(low+(y-low)*10**(-6/20)).astype(np.float32),'clean'
    for seconds in [1,.25]:
        size=round(seconds*SR);end=len(y)//size*size;parts=y[:end].reshape(-1,size);order=np.random.default_rng(seed).permutation(len(parts));out=np.r_[parts[order].reshape(-1),y[end:]]
        assert np.isclose(np.sum(out.astype(float)**2),np.sum(y.astype(float)**2))
        yield f'local_shuffle{seconds}',out,'clean'
    yield 'gain_m6',(y*10**(-6/20)).astype(np.float32),'clean'

def stem_items(y,identifier,out,model):
    import torch
    from demucs_infer.apply import apply_model
    contract=json.loads((ROOT/'contract.json').read_text())
    path=Path(contract['parent_root'])/'cells'/out.name/'stems'/(identifier+'.npz')
    if path.exists():
        z=np.load(path);assert str(z['source_hash'])==waveform_hash(y);stems={name:z[name] for name in model.sources}
    else:
        stereo=np.repeat(signal.resample_poly(y,441,160)[:,None],2,axis=1).T.astype(np.float32)
        wav=torch.from_numpy(stereo).to('cuda');ref=wav.mean(0);mean=ref.mean();std=ref.std();assert std>1e-7
        separated=apply_model(model,((wav-mean)/std)[None],shifts=0,split=True,overlap=.25,progress=False,device='cuda')[0]
        separated=(separated*std+mean).cpu().numpy()
        stems={name:signal.resample_poly(separated[j].mean(0),160,441)[:len(y)].astype(np.float32) for j,name in enumerate(model.sources)}
    reconstructed=sum(stems.values());assert len(reconstructed)==len(y)
    residual=float(20*np.log10(max(1e-12,np.linalg.norm(reconstructed-y))/max(1e-12,np.linalg.norm(y))))
    yield 'stem_recompose',reconstructed.astype(np.float32),'clean',residual
    for name in model.sources:
        for db in [6,99]:
            case='stem_'+name+('_off' if db==99 else '_m6');amp=0 if db==99 else 10**(-db/20)
            yield case,(reconstructed+(amp-1)*stems[name]).astype(np.float32),'stem_recompose',residual

def main(cell_id,limit=None,smoke=False,stage="recovery"):
    assert stage in ["recovery","confirmatory"]
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import warnings;warnings.filterwarnings('ignore')
    import torch
    from audiobox_aesthetics.infer import initialize_predictor
    from demucs_infer.pretrained import get_model
    import metrics
    torch.set_num_threads(4);contract=json.loads((ROOT/'contract.json').read_text());out=ROOT/'cells'/cell_id
    rows={r['id']:r['caption'] for r in csv.DictReader(open(contract['tsv']),delimiter='\t')}
    ids=contract['discovery_ids'] if stage=='recovery' else contract['probe_ids']
    if smoke:ids=[p.stem for p in sorted((out/'audio').glob('*.flac'))][:limit or 2]
    elif limit:ids=ids[:limit]
    predictor=initialize_predictor();separator=get_model('htdemucs_6s',repo=Path(contract['demucs_model_repo'])).to('cuda').eval();clap=metrics.load_clap()
    cache=out/('smoke_probe.jsonl' if smoke else 'probe.jsonl');done={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r=json.loads(line);done[r['key']]=r
    exclusions=[];start=time.monotonic();scored=0;reused=0;invalidated=0;variant_exclusions=[];source_hashes={}
    with open(cache,'a',buffering=1) as f:
        for index,identifier in enumerate(ids):
            source=out/'audio'/(identifier+'.flac');assert source.exists();source_hash=sha(source);source_hashes[identifier]=source_hash;y,sr=sf.read(source,dtype='float32');assert sr==16000 and len(y)==contract['audio_frames']
            if not np.isfinite(loudness(y)):
                exclusions.append(dict(id=identifier,source_sha256=source_hash,reason='nonfinite LUFS; preserve canonical raw metrics, exclude normalized probe'))
                continue
            seed=int(hashlib.sha256(('20261002'+identifier).encode()).hexdigest()[:8],16)
            variants=[(c,z,b,None) for c,z,b in interventions(y,seed)]
            variants+=list(stem_items(y,identifier,out,separator))
            if stage=='confirmatory':
                for color in ['white','pink']:
                    z=np.random.default_rng(seed).standard_normal(len(y))
                    if color=='pink':
                        spec=np.fft.rfft(z);spec/=np.sqrt(np.maximum(np.fft.rfftfreq(len(y),1/SR),20));spec[0]=0;z=np.fft.irfft(spec,n=len(y))
                    z/=np.sqrt(np.mean(z*z))
                    for level in [-55,-45]:
                        edited=(y+z*10**(level/20)).astype(np.float32)
                        variants.append((f'{color}_absolute_m{-level}',edited,'clean',None))
            with torch.inference_mode():te=clap.get_text_embedding([rows[identifier]],use_tensor=True)
            prepared=[]
            for case,z,baseline,residual in variants:
                z=z.astype(np.float32);feat=features(z)
                for mode in ['raw','lufs23']:
                    key=f'{identifier}__{case}__{mode}'
                    try:
                        zz,gain,lu=(z,1.,feat['lufs']) if mode=='raw' else normalize(z)
                        zz=np.asarray(zz,dtype=np.float32)
                    except ValueError as exc:
                        # Record a missing normalized derivative, never substitute a score.
                        exclusion=dict(id=identifier,case=case,mode=mode,reason=str(exc))
                        variant_exclusions.append(exclusion)
                        if key not in done:
                            record=dict(key=key,id=identifier,case=case,baseline=baseline,mode=mode,source_sha256=source_hash,source_features=feat,gain=None,lufs_after=None,scored_waveform_sha256=None,reconstruction_relative_rms_db=residual,clap=None,status='excluded',reason=str(exc),**{a:None for a in AXES})
                            f.write(json.dumps(record,allow_nan=False)+'\n');done[key]=record
                        continue
                    if key in done:
                        assert done[key]['source_sha256']==source_hash
                        if done[key]['scored_waveform_sha256']==waveform_hash(zz):
                            reused+=1
                            continue
                        invalidated+=1
                    prepared.append((key,case,mode,zz,dict(id=identifier,case=case,baseline=baseline,mode=mode,source_sha256=source_hash,source_features=feat,gain=float(gain),lufs_after=float(lu) if lu is not None else None,scored_waveform_sha256=waveform_hash(zz),reconstruction_relative_rms_db=residual)))
            for offset in range(0,len(prepared),16):
                batch=prepared[offset:offset+16];scores=predictor.forward([dict(path=torch.from_numpy(z[None]),sample_rate=16000) for _,_,_,z,_ in batch])
                for (key,case,mode,z,meta),score in zip(batch,scores):
                    assert all(np.isfinite(score[a]) for a in AXES)
                    similarity=None
                    if mode=='lufs23':
                        np.random.seed(seed);torch.manual_seed(seed)
                        with torch.inference_mode():
                            inp=torch.from_numpy(signal.resample_poly(z,3,1).astype(np.float32))[None]
                            ae=clap.get_audio_embedding_from_data(x=inp,use_tensor=True)
                            similarity=float(torch.nn.functional.cosine_similarity(ae,te,dim=-1).item())
                    record=dict(key=key,**meta,**score,clap=similarity);f.write(json.dumps(record,allow_nan=False)+'\n');scored+=1
                    done[key]=record
            if index%8==0 or index==len(ids)-1:
                dump(out/(stage+'_progress.json'),dict(cell=cell_id,completed_prompts=index+1,total_prompts=len(ids),scored_now=scored,reused_scores=reused,invalidated_cached_scores=invalidated,stage=stage,elapsed_seconds=time.monotonic()-start,estimated_remaining_seconds=(time.monotonic()-start)/(index+1)*(len(ids)-index-1)))
                print('PROBE',cell_id,index+1,'/',len(ids),flush=True)
                assert __import__('shutil').disk_usage(ROOT).free>contract['hard_stop_free_bytes']
    dump(out/('smoke_probe_exclusions.json' if smoke else (stage+'_source_exclusions.json')),exclusions)
    dump(out/(stage+'_variant_exclusions.json'),variant_exclusions)
    # Check coverage and controls with the very same cached score records.
    cases=['clean','white50','white40','white30','white20','pink50','pink40','pink30','pink20','quiet_m6','quiet_m12','dark_m6','local_shuffle1','local_shuffle0.25','gain_m6','stem_recompose']+[f'stem_{n}{suffix}' for n in separator.sources for suffix in ['_m6','_off']]
    if stage=='confirmatory':cases += [f'{color}_absolute_m{level}' for color in ['white','pink'] for level in [55,45]]
    eligible=[i for i in ids if i not in {e['id'] for e in exclusions}]
    expected={f'{i}__{c}__{m}' for i in eligible for c in cases for m in ['raw','lufs23']};assert expected<=set(done)
    assert all(done[k]['source_sha256']==source_hashes[done[k]['id']] for k in expected)
    ctrl=max([abs(done[f'{i}__gain_m6__lufs23'][a]-done[f'{i}__clean__lufs23'][a]) for i in eligible for a in AXES if done[f'{i}__gain_m6__lufs23'][a] is not None and done[f'{i}__clean__lufs23'][a] is not None] or [0]);assert ctrl<.01
    if not smoke and not limit:
        dump(out/(stage+'_complete.json'),dict(n_prompts_requested=len(ids),n_prompts_eligible=len(eligible),n_cases=len(cases),n_scores=len(expected),n_cached_total=len(done),n_excluded_variants=len(variant_exclusions),scored_now=scored,reused_scores=reused,invalidated_cached_scores=invalidated,elapsed_seconds=time.monotonic()-start,gain_control_max_all_axes=ctrl,cache_bytes=cache.stat().st_size,cache_sha256=sha(cache),variant_exclusions_sha256=sha(out/(stage+'_variant_exclusions.json')),exclusions_sha256=sha(out/(stage+'_source_exclusions.json')),checkpoint_precision=str(next(predictor.model.parameters()).dtype),source_contract_sha256=sha(ROOT/'contract.json'),demucs_name='htdemucs_6s; shifts=0; recomposition baseline for source removal; other is not verified background noise'))
    print('PROBE COMPLETE',cell_id,len(done),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('cell');p.add_argument('--limit',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--stage',default='recovery');a=p.parse_args();main(a.cell,a.limit,a.smoke,a.stage)
