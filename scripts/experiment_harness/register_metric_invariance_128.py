#!/usr/bin/env python3
"""Freeze 128 inputs, scripts and HARN documents. Installing P2 jobs is separate."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

R=Path('/home/kojiek/MeanAudio'); E=Path('/home/kojiek/eval_output_nvme')
O=Path('/mnt/seagate/kojiek_data/metric_invariance_128_20261017')
B=R/'docs/experiments/harn/metric_invariance_128'
PY='/home/kojiek/venvs/dac/bin/python'
sys.path.insert(0,str(R/'scripts/eval'))
import metric_invariance_128 as M


def write(p,v):M.write(p,v)
def sha(p):return M.sha(p)


def raw_record(path,ident,caption,lvl=None):
    p=Path(path)
    if not p.is_file():raise RuntimeError(f'missing audio: {p}')
    result={'id':ident,'caption':caption,'path':str(p),'sha256':sha(p)}
    if lvl is not None:
        result['raw_115']=True
        if Path(lvl).is_file():result['lvl30_sha256']=sha(lvl)
    return result


def manifest():
    import numpy as np
    caps=dict(M.A.items())
    fullcaps={r['id']:r['caption'] for r in csv.DictReader(open(M.A.TSV),delimiter='\t')}
    assert len(caps)==5131 and len(fullcaps)==5521
    ordered=np.random.default_rng(20261017).choice(sorted(caps),size=1000,replace=False).tolist()
    calibration={system:[raw_record(M.A.R115/name/'lvl30/audio'/f'{i}.flac',i,caps[i]) for i in ordered]
                 for system,name in M.A.SYSTEMS.items()}
    table_b={};references={};mir={}
    def add(name,raw,lvl,ref,mir_path=None):
        ids=sorted(p.stem for p in Path(raw).glob('*.flac'))
        if name!='musiccaps_real':assert len(ids)==5521,(name,len(ids))
        else:assert len(ids)==5131
        table_b[name]=[raw_record(Path(raw)/f'{i}.flac',i,fullcaps[i],Path(lvl)/f'{i}.flac') for i in ids]
        assert Path(ref).is_file(),str(ref)
        references[name]=str(ref)
        if mir_path and Path(mir_path).is_file():mir[name]=str(mir_path)
    c115=json.loads((R/'docs/experiments/ttm_metrics_115_20261009_contract.json').read_text())
    s115=json.loads((M.A.R115/'selection.json').read_text())['systems']
    for s in s115:
        name=s['name']; outname=name
        if name.startswith('meanaudio_'):outname=name.replace('meanaudio_','meanaudio_s14159265_',1)
        d=M.A.R115/name
        add(outname,s['audio'],d/'lvl30/audio',d/'lvl30/scores'/f'{name}_lvl30/per_clip.tsv',
            d/'raw/mir'/f'{name}_raw/mir_per_clip.tsv')
    c118=json.loads((R/'docs/experiments/ttm_metrics_seeds_118_20261011_contract.json').read_text())
    for s in c118['meanaudio_cells']:
        name=s['name'];d=E/'ttm_metrics_seeds_118_20261011'/name
        mp=Path(s['mir_json']).with_name('mir_per_clip.tsv')
        add(name,s['audio'],d/'lvl30/audio',d/'lvl30/scores'/f'{name}_lvl30/per_clip.tsv',mp)
    assert len(table_b)==10
    negmf={}
    for seed in (14159265,16180339,27182818):
        name=f'negmf_s{seed}_cfg0';raw=E/f'phase8_qwen_caption2p0_slot0clean_negmfn100_noq_full_s{seed}_mc_mf25_cfg0/audio'
        ids=sorted(p.stem for p in raw.glob('*.flac'));assert len(ids)==5521,(name,len(ids))
        negmf[name]=[raw_record(raw/f'{i}.flac',i,fullcaps[i],raw/'NOT_MATERIALIZED'/f'{i}.flac') for i in ids]
        control=f'meanaudio_s{seed}_cfg0';negmf[control]=table_b[control]
    stimuli=json.loads((M.A.EV/'metric_win_tie_126b_20261016/stimuli/items.json').read_text())
    midi={case:[] for case in M.MIDI_CASES}; keys={}
    for r in stimuli:
        if r['source']=='midi_generaluser' and r['case'] in midi:
            rec=raw_record(r['path'],r['excerpt'],r['caption']);midi[r['case']].append(rec)
            keys[r['key']]=r['excerpt']
    for case,rs in midi.items():
        rs.sort(key=lambda r:r['id']);assert len(rs)==402,(case,len(rs))
    midiref={}
    for p in (E/'metric_win_tie_126b_20261016/units').glob('midi_generaluser_*/scores.tsv'):
        for key,row in M.table(p,'key').items():
            if key in keys and key.endswith('__clean'):midiref[keys[key]]=row
    assert len(midiref)==402
    old=json.loads((R/'docs/experiments/results/metric_win_tie_126a_summary.json').read_text())
    ours=[f'meanaudio_s{s}_cfg3_neg' for s in (14159265,16180339,27182818)]
    controls=[f'meanaudio_s{s}_cfg0' for s in (14159265,16180339,27182818)]
    return {'seed':20261017,'calibration':calibration,'midi':midi,'midi_reference':midiref,
        'table_b':table_b,'negmf':negmf,'margins':{k:v*.1 for k,v in old['protocol']['S'].items()},
        'reference_115':{s:str(M.A.R115/n/'lvl30/scores'/f'{n}_lvl30/per_clip.tsv') for s,n in M.A.SYSTEMS.items()},
        'table_references':references,'real_mir':mir['musiccaps_real'],'mir':mir,
        'floor_mir':str(E/'metric_win_tie_126b_20261016/mir.json'),
        'claims':{'b':[['K1','PQ',ours,['musicgen_medium']*3],['K2','clap',ours,['musicgen_medium']*3],
                        ['K3','PQ',controls,['musicldm']*3]],
                  'c':[['K4','PQ',[f'negmf_s{s}_cfg0' for s in (14159265,16180339,27182818)],controls]]}}


def bundle(c,cp,acceptance):
    directory=Path(c['harn_bundle']);directory.mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc);at=now.isoformat();until=(now+timedelta(days=7)).isoformat()
    policy=hashlib.sha256(b''.join(p.read_bytes() for p in [R/'AGENTS.md',R/'docs/experiments/experiment_notification_policy.md',R/'docs/experiments/watcher_policy.md'])).hexdigest()
    schema=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((R/'docs/experiments/schemas').glob('*.json')))).hexdigest()
    commands=[{'action_id':k,'argv':v,'working_directory':str(R),'environment':{'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'}}
              for k,v in [('launch',['/bin/bash',c['bindings']['launcher']]),('resume',['/bin/bash',c['bindings']['launcher']]),
                          ('preflight',c['commands']['preflight']),('postflight',c['commands']['postflight'])]]
    bindings={'policy_bundle_sha256':policy,'schema_bundle_sha256':schema,
              'runtime_sha256':sha(c['bindings']['harn']),'command_set_sha256':M.hash_json({x['action_id']:x['argv'] for x in commands})}
    artifacts=[{'path':str(p),'sha256':sha(p)} for p in (cp,Path(c['manifest']),B/'operator_request.txt',acceptance)]
    contract={'document_kind':'experiment_contract','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1',
        'experiment_id':c['experiment_id'],'run_id':c['run_id'],'bindings':bindings,
        'approval_requirement':{'required':True,'responsible_role':'responsible-operator','trusted_channels':['operator_console']},
        'corpus':{'kind':'non_generated','source_artifacts':artifacts},'repair':{'enabled':False},
        'phases':[{'phase_id':'smoke-and-calibration' if c['part']=='a' else 'readjudication',
                   'action_id':'launch','input_artifacts':artifacts,'output_paths':[r['path'] for r in c['reports']],
                   'completion_evidence':[{'path':c['bindings']['action'],'sha256':sha(c['bindings']['action'])}],
                   'resume_action_id':'resume'}],
        'filesystems':c['filesystems'],'commands':commands,
        'required_preflight_checks':['policy','provenance','storage','acceptance','queue','watcher'],
        'notification_events':c['notifications']}
    write(directory/'contract.json',contract);ch=sha(directory/'contract.json')
    checks=[{'check_id':k,'verdict':'pass','observed_at':at,'valid_until':until,
             'evidence_sha256':sha(acceptance) if k in ('acceptance','watcher') else sha(cp)} for k in contract['required_preflight_checks']]
    measurements=[]
    for fs in c['filesystems']:
        st=os.statvfs(fs['path']);free=st.f_frsize*st.f_bavail
        assert free>=fs['hard_floor_bytes'],(fs['path'],free,fs['hard_floor_bytes'])
        measurements.append({k:fs[k] for k in ('path','hard_floor_bytes','peak_additional_bytes','transient_bytes','recovery_reserve_bytes')}|
                            {'measured_at':at,'free_bytes':free,'verdict':'pass'})
    preflight={'document_kind':'preflight_report','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1',
        'experiment_id':c['experiment_id'],'run_id':c['run_id'],'contract_raw_sha256':ch,
        'approval_evidence':{'evidence_id':'operator-start-128-20261017','source_kind':'trusted_operator_record','trusted_channel':'operator_console',
            'channel_record_id':'codex-chat-20261017-start-128','channel_record_sha256':sha(B/'operator_request.txt'),
            'approver_id':'responsible-operator','issued_at':at,'expires_at':until,'experiment_id':c['experiment_id'],'run_id':c['run_id'],
            'bindings':dict(bindings,contract_raw_sha256=ch,repair_envelope_sha256=None)},
        'checks':checks,'storage':measurements,'derived_verdict':'pass','created_at':at}
    write(directory/'preflight.json',preflight);ph=sha(directory/'preflight.json')
    event={'sequence':1,'event_id':'registered','idempotency_key':c['experiment_id']+':registered','event_kind':'contract_registered',
           'occurred_at':at,'phase':None,'verdict':'none','relates_to_event_id':None,'notification_status':'not_applicable',
           'previous_event_sha256':None,'event_sha256':M.hash_json({'contract':ch,'registered_at':at})}
    ledger={'document_kind':'event_ledger','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1',
            'experiment_id':c['experiment_id'],'run_id':c['run_id'],
            'bindings':{'contract_raw_sha256':ch,'preflight_report_raw_sha256':ph,'schema_bundle_sha256':schema},'events':[event]}
    write(directory/'ledger.json',ledger)
    queue={'document_kind':'queue_state','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1','queue_id':'p2-invariance-128',
           'updated_at':at,'entries':[{'entry_id':c['queue_name'][:-3],'position':1,'experiment_id':c['experiment_id'],'run_id':c['run_id'],
             'status':'ready','dependencies':[],'assigned_resource':None,
             'bindings':{'contract_raw_sha256':ch,'preflight_report_raw_sha256':ph,'ledger_raw_sha256':sha(directory/'ledger.json'),
                         'schema_bundle_sha256':schema},'terminal_notification_status':'not_applicable'}]}
    write(directory/'queue.json',queue)
    subprocess.run([PY,str(R/'scripts/validate_experiment_harness_documents.py'),
        '--contract',str(directory/'contract.json'),'--preflight',str(directory/'preflight.json'),
        '--ledger',str(directory/'ledger.json'),'--queue',str(directory/'queue.json')],check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest-only',action='store_true');args=ap.parse_args()
    B.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
    mf=O/'inputs.json'
    if not mf.exists():
        print('Hashing existing audio; no model execution.',flush=True);write(mf,manifest())
    if args.manifest_only:return
    acceptance=B/'acceptance.json';assert json.loads(acceptance.read_text())['status']=='passed'
    approval=B/'operator_request.txt'
    approval.write_text('2026-10-17 operator_console: 使用者提供 PR #2 的 128a/b/c/d 設計，並指示「開始實驗」。\n範圍：補腳本與 contracts、smoke、通過 gate 後依 p2 128→129→130 執行；128c 遵守 17h 順延規則。\n')
    common=[Path(__file__),R/'scripts/eval/metric_invariance_128.py',R/'scripts/experiment_harness/metric_invariance_128_guest.py',
        R/'scripts/eval/metric_win_tie_126a.py',R/'scripts/eval/ttm_metrics_115.py',R/'scripts/eval/eval_metrics.py',
        R/'scripts/eval/meva_score_dir.py',R/'scripts/eval/meva_runtime.py',R/'docs/experiments/meva_095_model_lock.json',
        R/'scripts/experiment_harness/notification_receipts.py',R/'scripts/experiment_harness/secondary_queue_controller.py',
        R/'scripts/notify_experiment_webhook.py',Path('/home/kojiek/gpu_queue/lib_scheduler.py'),
        R/'scripts/validate_experiment_harness_documents.py',Path(M.A.TSV),mf,acceptance,approval,
        R/'docs/experiments/metric_invariance_readjudication_128_20261017.md',
        R/'docs/experiments/results/metric_win_tie_126a_summary.json',
        R/'weights/music_speech_audioset_epoch_15_esc_89.98.pt']
    mfdata=json.loads(mf.read_text())
    common+=list(dict.fromkeys(Path(p) for p in list(mfdata['reference_115'].values())+list(mfdata['table_references'].values())+
        list(mfdata['mir'].values())+[mfdata['floor_mir']]))
    aes=Path('/home/kojiek/.cache/huggingface/hub/models--facebook--audiobox-aesthetics/snapshots')
    common += sorted(p for p in aes.glob('*/*') if p.name in ('model.safetensors','config.json'))
    common += sorted((O/'python_dependencies').rglob('*.py'))
    common += sorted((O/'python_dependencies').rglob('*.so'))
    common += [R/'docs/experiments/ttm_metrics_115_20261009_contract.json',
               R/'docs/experiments/ttm_metrics_seeds_118_20261011_contract.json',
               M.A.R115/'selection.json',M.A.R115/'summary.json',E/'ttm_metrics_seeds_118_20261011/summary.json']
    common += sorted((R/'docs/experiments/harn/negmf_full_120').glob('*contract.json'))
    for rs in list(mfdata['table_b'].values())+list(mfdata['negmf'].values()):
        common += sorted(Path(rs[0]['path']).parent.parent.glob('*REPORT.json'))
    common=list(dict.fromkeys(common))
    immutable={str(p):sha(p) for p in common}
    for number,part,label in [(128,'a','calibration'),(129,'b','table_b'),(130,'c','negmf')]:
        name=f'{number}_metric_invariance_{label}.sh';cp=R/f'docs/experiments/metric_invariance_128{part}_20261017_contract.json'
        launcher=R/'scripts/queue_candidates'/name;out=O/part;out.mkdir(exist_ok=True)
        (out/'tmp').mkdir(exist_ok=True)
        assert not (out/'started.json').exists(),'refuse replacing launched contract'
        launcher.write_text(f'''#!/bin/bash
# GPU_QUEUE_CONTRACT={cp}
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT={cp}
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR={out}/tmp
export PYTHONPATH=/mnt/seagate/kojiek_data/metric_invariance_128_20261017/python_dependencies
export PYTHONDONTWRITEBYTECODE=1
cd /home/kojiek/MeanAudio || exit 1
exec {PY} /home/kojiek/MeanAudio/scripts/experiment_harness/metric_invariance_128_guest.py
''');launcher.chmod(0o755)
        filesystems=[{'path':str(out),'hard_floor_bytes':50*2**30,'warning_floor_bytes':80*2**30,
            'peak_additional_bytes':6*2**30,'transient_bytes':512*2**20,'recovery_reserve_bytes':2*2**30},
            {'path':str(R),'hard_floor_bytes':50*2**30,'warning_floor_bytes':60*2**30,
             'peak_additional_bytes':100*2**20,'transient_bytes':10*2**20,'recovery_reserve_bytes':2*2**30}]
        action=R/'scripts/eval/metric_invariance_128.py';guest=R/'scripts/experiment_harness/metric_invariance_128_guest.py'
        c={'schema_version':1,'document_kind':'eval_only_metrics_contract','experiment_id':f'metric-invariance-128{part}-20261017',
            'run_id':f'run-20261017-{number}-metric-invariance','part':part,'status':'authorized_pending_preflight','launch_allowed':True,
            'priority':'P2','queue_role':'p2','queue_name':name,'dependencies':[] if part=='a' else
                ['128_metric_invariance_calibration']+(['129_metric_invariance_table_b'] if part=='c' else []),
            'ordering_dependencies':[] if part=='a' else [f'{number-1}_metric_invariance_'+('calibration' if part=='b' else 'table_b')],
            'approval_required':True,'launch_authorization':{'valid':True,'gpu_launch_allowed':True,'trusted_channel':'operator_console',
              'operator_instruction':'開始實驗','scope':'Implement and run PR2 design as explicitly labelled secondary historical-score invariance study; no training.'},
            'manifest':str(mf),'immutable_files':immutable,'harn_bundle':str(B/part),
            'design_document':str(R/'docs/experiments/metric_invariance_readjudication_128_20261017.md'),
            'protocol':{'classification':'separately_named_secondary_historical_metric_sensitivity',
              'historical_cfg0_label_preserved':True,'generation':False,'metrics':M.METRICS,'margins':mfdata['margins'],
              'normalization_lufs':-30,'limiter':False,'ensemble_shifts_ms':[0,40,80,120],'ensemble_polarities':[1,-1,1,-1],
              'variants':M.VARIANTS,'tie_variants':M.TIES,'degradation_ladders':M.LADDERS,'tie_pass':.8,
              'threshold_max_regression_steps':1,'wrong_direction_max_increase':.05,'identity_max_abs_diff':1e-4,
              'sample_rng_seed':20261017,'n_a_candidates':[1000,600,400],'budget_seconds_a':7*3600,
              'fallback':'C3 per metric; label sensitivity_only if no candidate passes',
              'clap_modes':['C0','C2','C3'],'meva_modes':['C0','C3'],'peak_over_fs_recorded':True,
              'budget_deferral_c_seconds':17*3600},
            'storage':{'path':str(out),'min_free_bytes':50*2**30},'filesystems':filesystems,
            'resume':{'kind':'from_scratch_with_autoresume','checkpoint':None,'iteration':0,
              'autoresume':str(out/'pause_progress.json'),'pause_progress':str(out/'pause_progress.json'),
              'behavior':'Hash-bound completed unit skipped; incomplete own temporary audio removed and redone; no upstream deletion.'},
            'watcher':{'stall_seconds':10800,'poll_seconds':5,'healthy_poll_model_calls':0,'automatic_repair':False},
            'calibration_summary':str(O/'a/summary.json'),'previous_summary':str(O/('a' if part=='b' else 'b')/'summary.json') if part!='a' else None,
            'scientific_dependencies':{} if part=='a' else {
                '128_metric_invariance_calibration':str(O/'a/summary.json'),
                **({'129_metric_invariance_table_b':str(O/'b/summary.json')} if part=='c' else {})},
            'commands':{'preflight':[PY,str(action),'--preflight'],'run':[PY,str(action)],'postflight':[PY,str(action),'--validate-only']},
            'reports':[{'path':str(out/'summary.json')}]+([{'path':str(out/'summary_d.json')}] if part=='b' else []),
            'bindings':{'launcher':str(Path('/home/kojiek/gpu_queue/p2/pending')/name),'launcher_sha256':sha(launcher),
                        'harn':str(guest),'harn_sha256':sha(guest),'action':str(action),'action_sha256':sha(action)},
            'notifications':['queue_handoff','start','preflight_pass','smoke_pass','calibration_gate','success','failure','held',
                'interrupted','paused','preflight_invalid','postflight_invalid','disk_hard_stop','disk_warning','storage_recovered',
                'stall','smoke_budget_invalid','budget_deferred','controller_incident'],
            'notification_receipts':{'required':True,'root':'/home/kojiek/gpu_queue/notification_receipts',
              'notifier':str(R/'scripts/notify_experiment_webhook.py'),'notifier_sha256':sha(R/'scripts/notify_experiment_webhook.py'),
              'helper':str(R/'scripts/experiment_harness/notification_receipts.py'),'helper_sha256':sha(R/'scripts/experiment_harness/notification_receipts.py')},
            'completion_evidence':{'eval_only':True,'method':'validated registered reports, terminal hash evidence and delivered receipt'},
            'cleanup_authorization':{'paths':[str(out/'tmp')],'behavior':'Only this run temporary scoring audio; upstream raw, lvl30, stimuli and cached weights retained.'}}
        write(cp,c);bundle(c,cp,acceptance);print(cp,flush=True)
    # Registration snapshot binds the whole chain, including blocked scientific
    # successors. Live eligibility is independently checked against host receipts.
    entries=[]
    for position,part in enumerate(('a','b','c'),1):
        q=json.loads((B/part/'queue.json').read_text());entry=q['entries'][0]
        entry['position']=position
        if position>1:
            entry['status']='blocked';entry['dependencies']=[entries[0]['entry_id']]
            if position==3:entry['dependencies'].append(entries[1]['entry_id'])
        entries.append(entry)
    for part in ('a','b','c'):
        directory=B/part;q=json.loads((directory/'queue.json').read_text());q['entries']=entries
        write(directory/'queue.json',q)
        subprocess.run([PY,str(R/'scripts/validate_experiment_harness_documents.py'),
            '--contract',str(directory/'contract.json'),'--preflight',str(directory/'preflight.json'),
            '--ledger',str(directory/'ledger.json'),'--queue',str(directory/'queue.json')],check=True)


if __name__=='__main__':main()
