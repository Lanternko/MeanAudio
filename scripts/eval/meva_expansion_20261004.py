"""Expanded, preregistered MEva sanity and historical experiment comparisons."""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from meva_runtime import ROOT, Evaluator, sha
sys.path.insert(0,str(ROOT/'scripts/experiment_harness'))
from notification_receipts import atomic_secure_json as write
from meva_095_events import notify

RUN=ROOT/'runtime/meva_expansion_20261004'
C=ROOT/'docs/experiments/meva_expansion_20261004_contract.json'
REPORT=ROOT/'docs/experiments/results/meva_expansion_20261004.json'


def preflight():
    c=json.loads(C.read_text());b=Path(c['harn_bundle'])
    sc=json.loads((b/'contract.json').read_text())
    assert sha(C)==next(a['sha256'] for a in sc['corpus']['source_artifacts'] if a['path']==str(C))
    approval=json.loads((b/'preflight.json').read_text())['approval_evidence']
    assert datetime.now(timezone.utc)<datetime.fromisoformat(approval['expires_at'])
    assert sha(c['approval_record'])==approval['channel_record_sha256']
    for x in c['raw_bindings']:assert sha(x['path'])==x['sha256'],'Changed input: '+x['path']
    lock=json.loads((ROOT/'docs/experiments/meva_095_model_lock.json').read_text())
    for x in lock['files']:assert sha(x['path'])==x['sha256'],'Changed MEva runtime: '+x['path']
    from importlib.metadata import version
    for p,v in c['package_versions'].items():assert version(p)==v,'Package changed: '+p
    verify_generation_packages(c)
    st=Path('/home/kojiek/.config/meanaudio/discord_webhook_url').stat()
    assert st.st_uid==os.geteuid() and st.st_mode&0o777==0o600
    rows=json.loads((RUN/'manifest.json').read_text())
    assert len(rows)==c['existing_rows'] and len({(r['group'],r['key']) for r in rows})==len(rows)
    for r in rows:assert sha(r['audio_path'])==r['audio_sha256'],'Audio changed: '+r['audio_path']
    fs=os.statvfs(RUN)
    if fs.f_bavail*fs.f_frsize<50<<30:return 75
    print('PASS: all retained audio, source contracts, checkpoints, runtime and storage',flush=True)
    return 0


def existing(row,binding):
    p=Path(row.get('reuse_path') or RUN/'scores'/row['group']/(row['key']+'.json'))
    if not p.exists():return None
    v=json.loads(p.read_text())
    assert v.get('model_binding',v.get('binding'))==binding,'Stale model score: '+str(p)
    assert v.get('input_sha256')==row['audio_sha256'],'Stale input score: '+str(p)
    assert v['status']=='scored' and math.isfinite(v['meva_raw']),'Nonfinite/invalid score'
    return v


def progress(index,phase,key=''):
    write(RUN/'progress.json',dict(index=index,phase=phase,last_key=key,updated_epoch=time.time()))


def score(rows,binding):
    import torch
    torch.set_num_threads(4);model=None
    for i,r in enumerate(rows):
        v=existing(r,binding)
        if v is None:
            assert sha(r['audio_path'])==r['audio_sha256']
            fs=os.statvfs(RUN)
            if fs.f_bavail*fs.f_frsize<50<<30:raise RuntimeError('Storage hard stop')
            if model is None:model=Evaluator()
            v=model.score(r['audio_path']);v.update(input_sha256=r['audio_sha256'],binding=binding)
            write(RUN/'scores'/r['group']/(r['key']+'.json'),v)
        if (i+1)%100==0:
            progress(i+1,'score',r['group']+'/'+r['key']);print('MEva',i+1,len(rows),r['group'],flush=True)
    progress(len(rows),'scored')


def verify_generation_packages(c):
    code='import json;from importlib.metadata import version;print(json.dumps({p:version(p) for p in '+repr(list(c['generation_package_versions']))+'}))'
    actual=json.loads(subprocess.check_output(['/home/kojiek/venvs/dac/bin/python','-c',code],text=True))
    assert actual==c['generation_package_versions'],'Generation/scoring environment changed'


def phase_gate(c,key):
    # Repeat mutable provenance/storage checks before expanding GPU work.
    for x in c['raw_bindings']:assert sha(x['path'])==x['sha256'],'Changed phase input: '+x['path']
    fs=os.statvfs(RUN)
    assert fs.f_bavail*fs.f_frsize>=50<<30,'Storage hard stop before '+key
    verify_generation_packages(c)
    notify(c,Path(os.environ['GPU_QUEUE_JOB_SCRIPT']),'phase-'+key,'success',
           'MEva expansion phase '+key+': immutable inputs and storage PASS.',kind='gate_result',verdict='pass')


def generated_rows(c,g):
    label=g['experiment']+'_mc_mf25_cfg3_neg';out=RUN/'generated'/label
    report=json.loads((out/(label+'_REPORT.json')).read_text())
    assert report['status']=='passed' and float(report['cfg_strength'])==3 and report['num_steps']==25
    assert report['negative_prompt']==c['protocol']['negative_prompt']
    assert report['checkpoint_sha256']==g['checkpoint_sha256']
    assert report['seed']==42 and report['conditioning']=='no_q' and report['text_attention_mask'] is False
    assert report['gen_tsv_sha256']==c['musiccaps_sha256'] and report['score_tsv_sha256']==c['musiccaps_sha256']
    assert sha(report['metrics_path'])==report['metrics_sha256']
    per=list(csv.DictReader((out/label/'per_clip.tsv').open(),delimiter='\t'));assert len(per)==5521
    expected={r['id'] for r in csv.DictReader(open(c['musiccaps_tsv']),delimiter='\t')}
    assert {r['id'] for r in per}==expected
    assert {p.stem for p in (out/'audio').glob('*.flac')}==expected
    return [dict(group=g['experiment'],key=x['id'],audio_path=str(out/'audio'/(x['id']+'.flac')),
        audio_sha256=sha(out/'audio'/(x['id']+'.flac')),protocol='MusicCaps5521/MF25/CFG3/fidelity8/seed42/NoMask/fp32/NoQ') for x in per]


def generate(c):
    rows=[]
    for j,g in enumerate(c['generation']):
        phase_gate(c,'generate-'+g['experiment'])
        assert sha(g['checkpoint'])==g['checkpoint_sha256']
        label=g['experiment']+'_mc_mf25_cfg3_neg';out=RUN/'generated'/label
        env=dict(os.environ,OUT_ROOT=str(RUN/'generated'),HF_HOME='/home/kojiek/.cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        for key in ['SMOKE_ROWS','PYTHONPATH']:env.pop(key,None)
        log=RUN/(g['experiment']+'_generation.log')
        with log.open('a') as f:
            p=subprocess.Popen(g['command'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            try:
                while p.poll() is None:
                    progress(len(rows),'generate',g['experiment']+':'+str(len(list((out/'audio').glob('*.flac')))))
                    time.sleep(10)
                assert p.returncode==0,'Generation/evaluation failed: '+str(log)
            finally:
                if p.poll() is None:p.terminate();p.wait(timeout=30)
        seal=RUN/(g['experiment']+'_generated_manifest.json')
        new=generated_rows(c,g)
        assert len({x['key'] for x in new})==5521
        if seal.exists():assert json.loads(seal.read_text())==new,'Generated artifact drift'
        else:write(seal,new)
        rows.extend(new)
    return rows


def summarize(rows,c,partial=False):
    import numpy as np
    import pandas as pd
    values=[]
    for r in rows:
        v=existing(r,c['model_binding']);assert v is not None,'Missing score'
        values.append(dict(group=r['group'],key=r['key'],meva=v['meva_raw'],protocol=r['protocol']))
    d=pd.DataFrame(values);assert np.isfinite(d.meva).all()
    groups={g:v.set_index('key').meva for g,v in d.groupby('group')}
    means={g:dict(n=len(v),mean=float(v.mean()),std=float(v.std()),min=float(v.min()),max=float(v.max())) for g,v in groups.items()}
    comparisons=[];rng=np.random.default_rng(20261004)
    for pair in c['comparisons']:
        if pair['left'] not in groups or pair['right'] not in groups:
            assert partial,'Missing comparison group';continue
        a,b=groups[pair['left']],groups[pair['right']]
        common=a.index.intersection(b.index);assert len(common)==pair['n']
        delta=(a.loc[common]-b.loc[common]).to_numpy()
        wins=np.where(delta==0,.5,(delta>0).astype(float))
        boots=[]
        for _ in range(2000):
            ix=rng.integers(0,len(delta),len(delta));boots.append([delta[ix].mean(),wins[ix].mean()])
        ci=np.quantile(boots,[.025,.975],axis=0)
        comparisons.append(dict(**pair,mean_delta=float(delta.mean()),left_win_fraction=float(wins.mean()),
            delta_ci95=ci[:,0].tolist(),win_ci95=ci[:,1].tolist(),
            direction='left higher' if ci[0,0]>0 else 'right higher' if ci[1,0]<0 else 'inconclusive',
            caveat='Pointwise CI; descriptive, no human accuracy/adoption claim.'))
    result=dict(status='interim_existing' if partial else 'completed',n_scored=len(rows),groups=means,comparisons=comparisons,
        model_binding=c['model_binding'],coverage=json.loads((RUN/'coverage.json').read_text()),
        limitations=c['limitations'],aes_replaced=False)
    if partial:write(RUN/'interim_existing.json',result);return
    assert len(rows)==c['n_expected']
    d.to_csv(RUN/'per_clip.tsv',sep='\t',index=False)
    write(REPORT,result)
    lines=['# MEva 擴充實驗比較（2026-10-04）','',
        '固定 pooled-small-f03；quarter/full 為步數對照。10k/100k 為資料量＋步數＋歷史資料來源共同變化，非純資料量因果證據。',
        '所有預期方向事先登錄；不依結果改模型或篩選輸贏。無真人標籤，勝率指模型偏好。','',
        '| 對照（左−右） | n | MEva 平均差 [95% CI] | 左較高比例 | 方向 |','|---|---:|---:|---:|---|']
    for p in comparisons:
        lines.append(f'| {p["label"]} | {p["n"]} | {p["mean_delta"]:+.3f} [{p["delta_ci95"][0]:+.3f}, {p["delta_ci95"][1]:+.3f}] | {p["left_win_fraction"]:.1%} | {p["direction"]} |')
    lines+=['','## 限制','']+['- '+s for s in c['limitations']]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('COMPLETE',len(rows),'comparisons',len(comparisons),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    rc=preflight()
    if rc or a.preflight:return rc
    c=json.loads(C.read_text());rows=json.loads((RUN/'manifest.json').read_text())
    if not a.validate_only:
        score(rows,c['model_binding']);summarize(rows,c,partial=True)
        import torch
        torch.cuda.empty_cache()
        more=generate(c);phase_gate(c,'score-generated');score(more,c['model_binding'])
    else:
        more=[]
        for g in c['generation']:
            current=generated_rows(c,g)
            assert current==json.loads((RUN/(g['experiment']+'_generated_manifest.json')).read_text()),'Generated artifact drift'
            more.extend(current)
    summarize(rows+more,c)
    return 0


if __name__=='__main__':raise SystemExit(main())
