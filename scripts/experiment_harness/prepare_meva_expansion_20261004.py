"""Inventory retained outputs, freeze new evaluations, and prepare a tail queue entry."""
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path('/home/kojiek/MeanAudio');RUN=ROOT/'runtime/meva_expansion_20261004'
B=ROOT/'docs/experiments/harn/meva_expansion_20261004';C=ROOT/'docs/experiments/meva_expansion_20261004_contract.json'
sys.path[:0]=[str(ROOT/'scripts/eval'),str(ROOT/'scripts/experiment_harness')]
from meva_runtime import sha
from notification_receipts import atomic_secure_json as write,canonical_hash


def main():
    assert not C.exists(),'Registered contract is immutable'
    RUN.mkdir(parents=True,exist_ok=True);B.mkdir(parents=True,exist_ok=True)
    (B/'operator_request.txt').write_text('2026-10-04 responsible operator via Codex conversation:\n有沒有更多資訊，像是同一個方法，但是訓練步數不同的，例如 Quarter 對上 Four，然後我們的不同實驗，都給 MEVA 測一下。最好是測那些很明確一定有輸贏的，像是資料機越多通常就越好嘛。那 MEVA 最好就一定要判他贏。\nClarification: Quarter 對 full 資料量.\nScope: fixed MEva comparisons, retained outputs plus fresh canonical evaluation of four existing10k/100k checkpoints. No retraining, evaluator fitting, forced winner, queue reordering or AES replacement.\n')
    mc=Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
    ids=[x['id'] for x in csv.DictReader(mc.open(),delimiter='\t')];assert len(ids)==len(set(ids))==5521
    original=json.loads((ROOT/'runtime/meva_20261003/manifest.json').read_text())
    reuse={(x['set'],x['key']):x for x in original if x['set']!='pam'}
    inventory=json.loads((RUN/'inventory_nvme.json').read_text())
    cells=[(x['cell'],Path('/home/kojiek/eval_output_nvme')/x['cell']/'audio') for x in inventory if x['audio']==5521]
    cells += [(p.name,p/'audio') for p in sorted(Path('/home/kojiek/cfg0_eval_runtime/output').iterdir())
              if p.is_dir() and len(list((p/'audio').glob('*.flac')))==5521]
    rows=[];artifacts=[mc];coverage=[]
    for group,audio in cells:
        files={p.stem:p for p in audio.glob('*.flac')}
        assert set(files)==set(ids),'Incomplete or unmatched cell: '+group
        protocol=('Historical CFG0 native' if 'cfg0' in group else 'Historical CFG3+fidelity8 native')
        if group.endswith('_lvl30'):protocol+=';existing -30LUFS level-match diagnostic'
        # CFG0 reports supply independently validated generation/checkpoint metadata.
        reports=list(audio.parent.glob('*REPORT.json'))
        if not reports:reports=list(Path('/home/kojiek/cfg0_eval_runtime/reports').glob(group+'*REPORT.json'))
        artifacts+=reports
        mt=audio.parent/group/'metrics.txt'
        if mt.exists():artifacts.append(mt)
        coverage.append(dict(group=group,n=5521,protocol=protocol,modern_report_present=bool(reports),
            provenance='source report where retained; otherwise historical contract, directory label and aggregate metrics; descriptive only'))
        for key in ids:
            path=files[key];digest=sha(path)
            row=dict(group=group,key=key,audio_path=str(path),audio_sha256=digest,protocol=protocol)
            if (group,key) in reuse:
                assert digest==reuse[group,key]['audio_sha256'],'Old audio drift'
                row['reuse_path']=str(ROOT/'runtime/meva_20261003/results'/group/(key+'.json'))
            rows.append(row)
        print('frozen',group,len(ids),flush=True)
    degroot=Path('/home/kojiek/eval_output_nvme/d1_degrade_reference')
    deg=list(csv.DictReader((degroot/'degradations.tsv').open(),delimiter='\t'));artifacts.append(degroot/'degradations.tsv')
    clean=sorted({x['source_id'] for x in deg});assert len(clean)==192 and len(deg)==1728
    gold=[]
    for key in clean:
        p=Path('/home/kojiek/eval_output_nvme/d1_noise_probe_cfg0/audio')/(key+'.flac')
        gold.append(dict(group='d1_clean',key=key,audio_path=str(p),audio_sha256=sha(p),protocol='existing D1 clean source, native'))
    for x in deg:
        p=degroot/'audio'/(x['id']+'.flac')
        gold.append(dict(group='d1_'+x['degradation'],key=x['source_id'],audio_path=str(p),audio_sha256=sha(p),protocol='existing deterministic D1 degradation, source loudness matched'))
    # Within this new experiment, diagnostics and budget comparisons are computed first.
    rank=lambda x:0 if 'slot0nm_noq_full_' in x['group'] or 'slot0nm_noq_quarter_' in x['group'] else 1
    rows=gold+sorted(rows,key=rank)
    write(RUN/'manifest.json',rows)
    oldcoverage=json.loads((ROOT/'runtime/meva_20261003/coverage.json').read_text())
    write(RUN/'coverage.json',dict(retained_cells=coverage,n_retained_cells=len(cells),existing_rows=len(rows),
        known_degradation_rows=len(gold),excluded_prior_inventory=oldcoverage['cells'],
        scope='All complete MusicCaps cells retained on NVMe and CFG0 runtime; four fresh10k/100k cells; historical missing audio remains disclosed'))
    comparisons=[]
    add=lambda left,right,label,n,expectation:comparisons.append(dict(left=left,right=right,label=label,n=n,expectation=expectation))
    for cfg in ['mc_mf25_cfg3_neg','musiccaps_mf25_cfg0_noq']:
        add('phase8_qwen_caption2p0_slot0nm_noq_full_'+cfg,'phase8_qwen_caption2p0_slot0nm_noq_quarter_'+cfg,
            'slot0nm full−quarter / '+cfg,5521,'full expected higher; training budget only, no human-known winner')
    for name in sorted({x['degradation'] for x in deg}):
        add('d1_clean','d1_'+name,'clean−'+name,192,'clean expected higher; severe degradations strongest sanity tests, no human labels')
    complete={x[0] for x in cells}
    for name in sorted(complete):
        other=name.replace('_cfg3_neg','_cfg0')
        if '_cfg3_neg' in name and other in complete:add(name,other,name+' vs CFG0',5521,'direction unknown; jointly changes CFG and negative prompt')
    add('mf100k_noq_stage2_100000','mf10k_noq_fast_stage2_50000','MusicFlamingo100k−10k',5521,'100k expected higher; also doubles updates and changes historical source split')
    add('lpmc100k_noq_stage2_100000','lpmc10k_noq_fast_stage2_50000','LPMC100k−10k',5521,'100k expected higher; also doubles updates and changes historical source split')
    for k in ['10k','100k']:
        add('mf10k_noq_fast_stage2_50000' if k=='10k' else 'mf100k_noq_stage2_100000',
            'lpmc10k_noq_fast_stage2_50000' if k=='10k' else 'lpmc100k_noq_stage2_100000','MusicFlamingo−LPMC / '+k,5521,'method comparison; no human-known winner')
    action=ROOT/'scripts/eval/meva_expansion_20261004.py';guest=ROOT/'scripts/experiment_harness/meva_expansion_20261004_guest.py'
    launcher=RUN/'103_meva_expansion_20261004.sh'
    launcher.write_text('#!/bin/bash\n# GPU_QUEUE_CONTRACT='+str(C)+'\nexport GPU_QUEUE_JOB_SCRIPT="$0"\nexport GPU_QUEUE_CONTRACT='+str(C)+'\nexport PYTHONUNBUFFERED=1\nexport OMP_NUM_THREADS=4\nexport MKL_NUM_THREADS=4\nexec /home/kojiek/venvs/dac/bin/python '+str(guest)+'\n');launcher.chmod(0o700)
    test=ROOT/'scripts/tests/test_meva_expansion_20261004.py'
    acceptance=json.loads(subprocess.check_output(['/home/kojiek/venvs/dac/bin/python',str(test)],text=True))
    acceptance.update(source_sha256={str(p):sha(p) for p in [action,guest,test]},inherited_adapter_smoke=str(ROOT/'runtime/meva_20261003/smoke.json'))
    acceptance.update(d1_audit=json.loads((RUN/'d1_source_audit.json').read_text()),checkpoint_audit=json.loads((RUN/'checkpoint_cpu_audit.json').read_text()))
    write(B/'acceptance.json',acceptance)
    c=copy.deepcopy(json.loads((ROOT/'docs/experiments/musiceval_full_20261003_contract.json').read_text()))
    eid='meva-expansion-20261004';rid='run-20261004-meva103'
    generation=[]
    for name in ['mf10k_noq_fast_stage2_50000','mf100k_noq_stage2_100000','lpmc10k_noq_fast_stage2_50000','lpmc100k_noq_stage2_100000']:
        checkpoint=(ROOT/'exps'/name/(name+'_ema_final.pth')).resolve(strict=True);artifacts.append(checkpoint)
        generation.append(dict(experiment=name,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            command=['/bin/bash',str(ROOT/'scripts/eval/mc_mf25_eval.sh'),name,str(checkpoint),'--no_q','cfg3neg']))
    sources=[ROOT/'scripts/training_pipelines'/p for p in ['caption2p0_slot0nm_action.sh','train_pipeline_music_flamingo_10k.sh','train_pipeline_music_flamingo_100k.sh','train_pipeline_lpmc_10k_control.sh','train_pipeline_lpmc_100k_control.sh']]
    sources+=[ROOT/'docs/experiments'/p for p in ['caption2p0_slot0nm_quarter_cfg0_contract.json','caption2p0_slot0nm_full_cfg0_contract.json']]
    runtime=[ROOT/'eval.py',ROOT/'scripts/eval/mc_mf25_eval.sh',ROOT/'scripts/eval/eval_metrics.py',ROOT/'scripts/eval/meva_runtime.py',ROOT/'scripts/eval/d1_degrade_reference.py',
        ROOT/'meanaudio/model/mean_flow.py',ROOT/'meanaudio/model/networks.py',ROOT/'scripts/experiment_harness/meva_095_events.py',
        ROOT/'scripts/experiment_harness/preflight_capture.py',ROOT/'docs/experiments/meva_095_model_lock.json']
    c.update(experiment_id=eid,run_id=rid,document_kind='eval_only_meva_expansion_contract',approval_record=str(B/'operator_request.txt'),
        harn_bundle=str(B),queue_name=launcher.name,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),manifest=str(RUN/'manifest.json'),manifest_sha256=sha(RUN/'manifest.json'),
        existing_rows=len(rows),n_expected=len(rows)+4*5521,generation=generation,comparisons=comparisons,musiccaps_tsv=str(mc),musiccaps_sha256=sha(mc),
        generation_package_versions={p:version(p) for p in ['torch','torchaudio','transformers','numpy','soundfile','scipy','laion-clap','audiobox-aesthetics','librosa','pyloudnorm','safetensors']},
        question='Does fixed MEva recognize expected quality direction across training budgets, data-size regimes, methods and explicit degradations?',
        protocol=dict(new_generation=True,generated_content='evaluation audio only; captions are original non-generated MusicCaps annotations',
            generation='MusicCaps5521/MF25/CFG3/seed42/NoMask/fp32/NoQ',
            negative_prompt='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi',
            bootstrap_repeats=2000,bootstrap_seed=20261004,gate='all inputs bound/all scores finite/full coverage; performance directions descriptive; no adoption'),
        limitations=['Quarter/full slot0nm uses the same251598 rows; steps100k+50k vs400k+200k. This is a training-budget comparison.',
            '10k/100k historical checkpoints also change update budgets100k+50k vs200k+100k and original corpus splits. No pure data-size causal claim.',
            'Historical100k scripts reference a quarantined source TSV. Checkpoints are evaluated descriptively, never retrained or promoted as clean method evidence.',
            'Many retained legacy cells have aggregate metrics but no modern REPORT; directory/source contracts identify protocol. Preserve historical labels, no canonical relabeling.',
            'No PromptCC human labels: model preference fractions are not human accuracy. More data/updates are expectations, not guaranteed winners.',
            'D1 degradations are preexisting paired controls; severe noise and clipping are strongest checks, stylistic effects of mild distortion are subjective.',
            'Pointwise prompt/source bootstrap CIs are descriptive and do not include training-seed variation or establish multiplicity-controlled adoption.',
            'Only retained complete cells and four selected surviving data-size checkpoints are covered. Missing historical audio is disclosed; not all historic experiments are recovered.'],
        bindings=dict(launcher=str(Path('/home/kojiek/gpu_queue/p2/pending')/launcher.name),launcher_sha256=sha(launcher),harn=str(guest),harn_sha256=sha(guest),action=str(action),action_sha256=sha(action)),
        commands={n:[str(ROOT/'runtime/meva_20261003/venv/bin/python'),str(action)]+x for n,x in [('run',[]),('preflight',['--preflight']),('postflight',['--validate-only'])]},
        reports=[{'path':str(p)} for p in [ROOT/'docs/experiments/results/meva_expansion_20261004.json',ROOT/'docs/experiments/results/meva_expansion_20261004.md',RUN/'per_clip.tsv',RUN/'coverage.json']],
        completion_evidence=dict(eval_only=True,n_expected=len(rows)+4*5521,method='all input-bound finite cached/new MEva scores and canonical fresh generation reports'))
    c['launch_authorization']['scope']='Explicit user requested expanded MEva comparisons, data size and other experiments; existing checkpoints and audio only, no training'
    c['storage'].update(path=str(RUN),estimated_peak_additional_bytes=20<<30,cleanup='Only newly generated audio in the four registered output cells without a completed REPORT is discarded by mc_mf25_eval.sh on resume to restart the seed42 RNG stream. No retained inputs, complete cells, scores or checkpoints are deleted.')
    c['resource_budget'].update(max_active_seconds=48*3600,peak_vram_bytes=24<<30,additional_disk_bytes=20<<30)
    c['resume'].update(pause_progress=str(RUN/'pause_progress.json'),behavior='Reuse input/model-bound per-clip scores; completed generation reports are verified and reused; an incomplete new generation cell restarts from scratch to reproduce the seed42 RNG stream.')
    c['phases']=[dict(id='retained-rescore',output=str(RUN/'interim_existing.json'),resume='verified per-clip cache',next='generate-first'),
        *[dict(id='generate-'+g['experiment'],output=str(RUN/(g['experiment']+'_generated_manifest.json')),gate='registered REPORT fields, all5521 exact ids and raw audio hashes',resume='verified complete report reused, incomplete cell regenerated from seed42') for g in generation],
        dict(id='score-generated',resume='verified per-clip cache',gate='all22084 finite and bound'),
        dict(id='summarize',gate='all206197 finite scores and26 declared comparisons',output=c['reports'],resume='deterministic rebuild')]
    c['notifications'] += ['phase-generate-'+g['experiment'] for g in generation]+['phase-score-generated']
    c['queue_placement']='append_tail after existing AES101/102; no preemption'
    runtime += list((ROOT/'meanaudio').rglob('*.py'))
    artifacts += [ROOT/'weights'/name for name in ['v1-16.pth','best_netG.pt','music_speech_audioset_epoch_15_esc_89.98.pt','empty_string_t5.pth','empty_string_clap_c.pth']]
    hf=Path('/home/kojiek/.cache/huggingface/hub/models--google--flan-t5-large')
    artifacts.append(hf/'refs/main')
    artifacts += [p for p in (hf/'snapshots'/(hf/'refs/main').read_text().strip()).rglob('*') if p.is_file()]
    raw=[action,guest,test,Path(__file__),B/'operator_request.txt',RUN/'manifest.json',RUN/'coverage.json',ROOT/'docs/experiments/meva_expansion_20261004.md',RUN/'d1_source_audit.json',RUN/'checkpoint_cpu_audit.json',*sources,*runtime,*artifacts,
        Path(c['notification_receipts']['helper']),Path(c['notification_receipts']['notifier'])]
    c['raw_bindings']=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(raw))];write(C,c)
    build_bundle(c,launcher,guest,action,eid,rid)
    print('PREPARED, NOT SEATED:',len(rows),'existing +22084 generated; reused',sum('reuse_path' in x for x in rows),'comparisons',len(comparisons))


def build_bundle(c,launcher,guest,action,eid,rid):
    now=datetime.now(timezone.utc).isoformat();until=(datetime.now(timezone.utc)+timedelta(days=7)).isoformat()
    old=ROOT/'docs/experiments/harn/musiceval_full_20261003'
    sc=copy.deepcopy(json.loads((old/'contract.json').read_text()));sc.update(experiment_id=eid,run_id=rid)
    commands=[dict(action_id=n,argv=v,working_directory=str(ROOT),environment={'PYTHONUNBUFFERED':'1','CUDA_VISIBLE_DEVICES':'0'})
        for n,v in [('launch',['/bin/bash',str(launcher)]),('resume',['/bin/bash',str(launcher)]),('preflight',c['commands']['preflight']),('postflight',c['commands']['postflight'])]]
    sc['commands']=commands;sc['bindings']['runtime_sha256']=sha(guest);sc['bindings']['command_set_sha256']=canonical_hash({a['action_id']:a['argv'] for a in commands})
    sc['bindings']['policy_bundle_sha256']=hashlib.sha256(b''.join(p.read_bytes() for p in [ROOT/'AGENTS.md',ROOT/'docs/experiments/evaluation_policy.md',ROOT/'docs/experiments/experiment_notification_policy.md',ROOT/'docs/experiments/watcher_policy.md'])).hexdigest()
    artifacts=[dict(path=str(p),sha256=sha(p)) for p in [C,RUN/'manifest.json',RUN/'coverage.json',B/'acceptance.json',B/'operator_request.txt']]
    sc['corpus']['source_artifacts']=artifacts
    specs=[('retained-rescore',[str(RUN/'interim_existing.json'),str(RUN/'scores')])]
    specs += [('generate-'+g['experiment'],[str(RUN/'generated'/(g['experiment']+'_mc_mf25_cfg3_neg')),str(RUN/(g['experiment']+'_generated_manifest.json'))]) for g in c['generation']]
    specs += [('score-generated',[str(RUN/'scores')]),('summarize',[x['path'] for x in c['reports']])]
    sc['phases']=[dict(phase_id=name,action_id='launch',input_artifacts=artifacts,output_paths=outputs,completion_evidence=[dict(path=str(action),sha256=sha(action))],resume_action_id='resume') for name,outputs in specs]
    sc['filesystems']=[dict(path=str(RUN),hard_floor_bytes=50<<30,warning_floor_bytes=80<<30,peak_additional_bytes=20<<30,transient_bytes=2<<30,recovery_reserve_bytes=50<<30)]
    sc['notification_events']=c['notifications'];write(B/'contract.json',sc);ch=sha(B/'contract.json')
    pf=copy.deepcopy(json.loads((old/'preflight.json').read_text()));pf.update(experiment_id=eid,run_id=rid,contract_raw_sha256=ch,created_at=now)
    ap=pf['approval_evidence'];ap.update(evidence_id='approval-meva103-20261004',channel_record_id='codex-meva103-20261004',channel_record_sha256=sha(B/'operator_request.txt'),issued_at=now,expires_at=until,experiment_id=eid,run_id=rid)
    ap['bindings'].update(contract_raw_sha256=ch,policy_bundle_sha256=sc['bindings']['policy_bundle_sha256'],runtime_sha256=sha(guest),command_set_sha256=sc['bindings']['command_set_sha256'])
    for x in pf['checks']:x.update(observed_at=now,valid_until=until,evidence_sha256=sha(B/'acceptance.json'))
    fs=os.statvfs(RUN);free=fs.f_bavail*fs.f_frsize;assert free>=50<<30
    pf['storage']=[dict(path=str(RUN),measured_at=now,free_bytes=free,hard_floor_bytes=50<<30,peak_additional_bytes=20<<30,transient_bytes=2<<30,recovery_reserve_bytes=50<<30,verdict='pass')];write(B/'preflight.json',pf)
    event=dict(sequence=1,event_id='contract-register',idempotency_key=eid+':contract-register',event_kind='contract_registered',occurred_at=now,phase=None,verdict='none',relates_to_event_id=None,notification_status='not_applicable',previous_event_sha256=None);event['event_sha256']=canonical_hash(event)
    bind=dict(contract_raw_sha256=ch,preflight_report_raw_sha256=sha(B/'preflight.json'),schema_bundle_sha256=sc['bindings']['schema_bundle_sha256'])
    write(B/'ledger.json',dict(document_kind='event_ledger',schema_version='1.0.0',schema_bundle_id='harn-schema-v1',experiment_id=eid,run_id=rid,bindings=bind,events=[event]))
    write(B/'queue.json',dict(document_kind='queue_state',schema_version='1.0.0',schema_bundle_id='harn-schema-v1',queue_id='p2-meva103',updated_at=now,entries=[dict(entry_id='103-meva-expansion',position=1,experiment_id=eid,run_id=rid,status='ready',dependencies=[],assigned_resource=None,bindings={**bind,'ledger_raw_sha256':sha(B/'ledger.json')},terminal_notification_status='not_applicable')]))


if __name__=='__main__':main()
