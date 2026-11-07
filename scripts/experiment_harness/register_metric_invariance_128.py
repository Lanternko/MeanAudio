#!/usr/bin/env python3
"""Freeze 128 inputs, scripts and HARN documents. Installing P2 jobs is separate."""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

R=Path('/home/kojiek/MeanAudio'); E=Path('/home/kojiek/eval_output_nvme')
O=Path('/mnt/seagate/kojiek_data/metric_invariance_128_20261017')
B=R/'docs/experiments/harn/metric_invariance_128'
ORIGINAL_OUTPUT=O
DEPENDENCIES=ORIGINAL_OUTPUT/'python_dependencies'
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


def completed_calibration(cp):
    c=json.loads(cp.read_text());assert c['part']=='a'
    job=Path(c['queue_name']).stem
    launcher=Path('/home/kojiek/gpu_queue/p2/done')/c['queue_name']
    terminal=launcher.with_suffix('.terminal.json');t=json.loads(terminal.read_text())
    report=Path(c['reports'][0]['path']);s=json.loads(report.read_text())
    assert t['status']=='completed' and s['status']=='completed'
    assert any(r['path']==str(report) and r['sha256']==sha(report) for r in t['evidence']['reports'])
    sys.path.insert(0,str(R/'scripts/experiment_harness'))
    from notification_receipts import validate_delivered_receipt
    receipt=Path(t['notification_receipt']['path'])
    ok,reason=validate_delivered_receipt(receipt,contract_path=cp,launcher_path=launcher,event='success',status='success')
    assert ok,reason
    source=None;revision=None
    for rev in subprocess.check_output(['git','log','-40','--format=%H','--','scripts/eval/metric_invariance_128.py'],cwd=R,text=True).splitlines():
        blob=subprocess.check_output(['git','show',rev+':scripts/eval/metric_invariance_128.py'],cwd=R)
        if hashlib.sha256(blob).hexdigest()==c['bindings']['action_sha256']:
            source=blob.decode();revision=rev;break
    assert source is not None,'completed evaluator source unavailable from git'
    fingerprints=M.calibration_fingerprints(source)
    assert fingerprints==M.calibration_fingerprints((R/'scripts/eval/metric_invariance_128.py').read_text()),'calibration or scoring functions changed'
    critical=['metric_win_tie_126a.py','ttm_metrics_115.py','eval_metrics.py','meva_score_dir.py','meva_runtime.py']
    files=[cp,terminal,report,receipt]
    for name in critical:
        p=R/'scripts/eval'/name;assert sha(p)==c['immutable_files'][str(p)];files.append(p)
    started=report.parent/'started.json';files.append(started)
    elapsed=datetime.fromisoformat(t['written_at'].replace('Z','+00:00')).timestamp()-json.loads(started.read_text())['at']
    assert 0<elapsed<17*3600
    return {'source_contract':str(cp),'terminal':str(terminal),'report':str(report),'started':str(started),
            'job':job,'source_revision':revision,'source_action_sha256':c['bindings']['action_sha256'],
            'completed_calibration_seconds':elapsed,'calibration_fingerprints':fingerprints,
            'files':{str(p):sha(p) for p in files}}


def copy_completed_units(source, destination, inputs, action_sha):
    """Carry exact cached scores across a storage-only restart; ignore unfinished work."""
    import numpy as np
    copied=[]
    for done in sorted(source.glob('*/*/*/*/done.json')):
        group,system,variant,mode=done.parent.relative_to(source).parts
        if group not in ('identity','table') or system not in inputs['table_b']:
            raise RuntimeError('unexpected resume unit')
        records=inputs['table_b'][system]
        if group=='identity':
            if (variant,mode)!=('clean','C0'):raise RuntimeError('invalid identity resume unit')
            records=records[:min(len(records),224)]
        else:
            if not re.fullmatch(r'chunk\d{5}',variant) or mode not in ('C1','C2','C3'):
                raise RuntimeError('invalid table resume unit')
            offset=int(variant[5:]);records=records[offset:offset+1000]
        meta=json.loads(done.read_text());scores=done.parent/'scores.npz'
        expected=M.hash_json({'action':action_sha,'mode':mode,'system':system,'variant':variant,'records':records})
        if not records or meta['identity']!=expected or meta['n']!=len(records) or sha(scores)!=meta['score_sha256']:
            raise RuntimeError('resume unit provenance/hash mismatch')
        with np.load(scores) as z:
            required=[0,1,2,3]+([4] if mode!='C1' else [])+([5] if mode in ('C0','C3') else [])
            if z['ids'].tolist()!=[r['id'] for r in records] or z['values'].shape!=(len(records),6) or not np.isfinite(z['values'][:,required]).all():
                raise RuntimeError('resume score IDs/values invalid')
        out=destination/done.parent.relative_to(source);out.mkdir(parents=True,exist_ok=True)
        for p in (scores,done):
            dest=out/p.name
            if dest.exists():raise RuntimeError('refuse replacing resumed score')
            shutil.copy2(p,dest)
            if sha(dest)!=sha(p):raise RuntimeError('resume copy changed')
        copied.append({'source':str(done.parent),'destination':str(out),'n':len(records),
                       'scores_sha256':sha(scores),'done_sha256':sha(done)})
    return copied


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
        'approval_evidence':{'evidence_id':'operator-start-'+c['run_id'],'source_kind':'trusted_operator_record','trusted_channel':'operator_console',
            'channel_record_id':'codex-chat-'+c['run_id'],'channel_record_sha256':sha(B/'operator_request.txt'),
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
    queue={'document_kind':'queue_state','schema_version':'1.0.0','schema_bundle_id':'harn-schema-v1','queue_id':'p2-'+c['run_id'],
           'updated_at':at,'entries':[{'entry_id':c['queue_name'][:-3],'position':1,'experiment_id':c['experiment_id'],'run_id':c['run_id'],
             'status':'ready','dependencies':[],'assigned_resource':None,
             'bindings':{'contract_raw_sha256':ch,'preflight_report_raw_sha256':ph,'ledger_raw_sha256':sha(directory/'ledger.json'),
                         'schema_bundle_sha256':schema},'terminal_notification_status':'not_applicable'}]}
    write(directory/'queue.json',queue)
    subprocess.run([PY,str(R/'scripts/validate_experiment_harness_documents.py'),
        '--contract',str(directory/'contract.json'),'--preflight',str(directory/'preflight.json'),
        '--ledger',str(directory/'ledger.json'),'--queue',str(directory/'queue.json')],check=True)


def main():
    global O,B
    ap=argparse.ArgumentParser();ap.add_argument('--manifest-only',action='store_true')
    ap.add_argument('--run-suffix',default='',help='Unique retry label; preserves all previous runs.')
    ap.add_argument('--budget-hours-a',type=int,choices=(7,9),default=7,
                    help='Registered smoke feasibility ceiling; 9h explicitly approved by operator on 2026-10-17.')
    ap.add_argument('--reuse-calibration-contract',type=Path,help='Verified completed 128a; register only fresh 129/130.')
    ap.add_argument('--transient-root',type=Path,help='Dedicated scratch base; use output filesystem when NVMe headroom is low.')
    ap.add_argument('--output-root',type=Path,help='Explicit fresh score root; frozen input manifest stays unchanged.')
    ap.add_argument('--resume-contract',type=Path,help='Interrupted B contract; carry only hash-verified completed units.')
    args=ap.parse_args()
    if args.run_suffix and not re.fullmatch(r'[a-z0-9_]+',args.run_suffix):
        ap.error('run suffix must contain only lowercase letters, digits and underscores')
    suffix=('_'+args.run_suffix) if args.run_suffix else ''
    if suffix:
        O=Path(str(ORIGINAL_OUTPUT)+suffix);B=Path(str(B)+suffix)
    if args.output_root:O=args.output_root.resolve()
    def job(number,label):return f'{number}_metric_invariance_{label}{suffix}'
    B.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
    # Reuse the exact frozen input manifest, including the original sample draw.
    mf=ORIGINAL_OUTPUT/'inputs.json' if suffix else O/'inputs.json'
    if not mf.exists():
        print('Hashing existing audio; no model execution.',flush=True);write(mf,manifest())
    if args.manifest_only:return
    reuse=completed_calibration(args.reuse_calibration_contract.resolve()) if args.reuse_calibration_contract else None
    parts='bc' if reuse else 'abc'
    for part in parts:
        cp=R/f'docs/experiments/metric_invariance_128{part}_20261017{suffix}_contract.json'
        assert not cp.exists(),f'refuse replacing registered contract: {cp}'
    scans={part:str(B/'lufs_scans'/f'{part}.json') for part in 'abc'}
    M.require_lufs_scans(scans,mf)
    acceptance=B/'acceptance.json';accept=json.loads(acceptance.read_text());assert accept['status']=='passed'
    for path,expected in accept['bindings'].items():
        assert sha(R/path)==expected,f'acceptance evidence stale: {path}'
    approval=B/'operator_request.txt'
    approval.write_text('2026-10-17 operator_console: 使用者提供 PR #2 commit eff41ef，確認近靜音 LUFS 替代規則及兩個條件（fallback 排除敏感度分析、三份 CPU 全量掃描）；指示主機拉分支、掃描通過後重新登錄並以新 run 排 128→129→130。此指示承接先前「開始實驗」授權。\n範圍：C1/C3 非零 finite 且 LUFS=-inf 時以 unit-peak probe 再量測、正規化到 -30 LUFS；其他無效音訊仍失敗。所有 fallback ID/count 和排除後判定報告，不穩健者不用作結論。AES 每段最大差 0.01、各軸 signed mean 絕對值 0.001；CLAP/MEva 1e-4。m、80% 與科學判定不變；不套用 padding 3b1ead4，128c 遵守 17h 順延規則。舊 run 及 identity_r1 完整保留。\n')
    if args.budget_hours_a==9:
        approval.write_text(approval.read_text()+'2026-10-17 operator_console: 使用者對 quiet_r2 吞吐 gate 失敗後的具體提案回覆「同意」：只將 128a 預估上限由 7 h 改為 9 h，以 N=400 另登錄新 run（仍依新 smoke 在原候選集合選擇）；其餘規則不變。腳本讀取 contract 參數，三份 source-bound CPU 掃描必須重建通過；保留 quiet_r2 的全部證據。\n')
    if reuse:
        approval.write_text(approval.read_text()+'2026-10-18 operator_console: 使用者對各結論僅由所用指標阻斷、CU 等其他軸改記診斷警告的提案回覆「GO」。新 129/130 承接已驗證完成的 128a，只阻斷該階段 claims 使用的指標（b=PQ/CLAP，c=PQ）；所有軸的數值與警告保留，主要指標門檻不變。新 run 累計已完成 A 的耗時與本次 B 的時間，17h 預算數值保留，排除舊 run 終止後等待；不重跑 A，不改 m、80% 或敏感度標記。\n')
        write(B/'calibration_reuse.json',reuse)
    resume=None
    if args.resume_contract:
        assert reuse,'storage restart requires verified completed A'
        gate=json.loads((B/'archive_gate.json').read_text())
        assert gate['status']=='passed' and gate['verified_cold_cells'],'cold archive/capacity not verified'
        oldcp=args.resume_contract.resolve();old=json.loads(oldcp.read_text())
        terminal=Path('/home/kojiek/gpu_queue/p2/interrupted')/(Path(old['queue_name']).stem+'.terminal.json')
        t=json.loads(terminal.read_text());assert old['part']=='b' and t['status']=='interrupted'
        assert old['manifest']==str(mf) and old['bindings']['action_sha256']==sha(R/'scripts/eval/metric_invariance_128.py')
        sys.path.insert(0,str(R/'scripts/experiment_harness'))
        from notification_receipts import validate_delivered_receipt
        receipt=Path(t['notification_receipt']['path']);launcher=terminal.with_name(Path(old['queue_name']).name)
        ok,reason=validate_delivered_receipt(receipt,contract_path=oldcp,launcher_path=launcher,event='interrupted',status='interrupted')
        assert ok,reason
        source=Path(old['storage']['path']);(O/'b').mkdir(parents=True,exist_ok=True)
        copied=copy_completed_units(source,O/'b',json.loads(mf.read_text()),old['bindings']['action_sha256'])
        started=source/'started.json'
        resume={'source_contract':str(oldcp),'source_root':str(source),'units':copied,'original_b_started':str(started),
                'files':{str(p):sha(p) for p in (oldcp,terminal,receipt,started)}}
        write(B/'score_resume.json',resume)
        approval.write_text(approval.read_text()+'2026-10-18 operator_console: 使用者指出 Seagate 應只封存舊資料，新資料在 NVMe。修正先前熱資料放置：只封存已完成且不在現行輸入的 115 lvl23，逐檔驗證並保留 symlink；新分數、日誌與暫存回 NVMe。已完成 score unit 驗證後承接；科學參數與 17h 不變，仍用原 B started clock，包含本次搬移時間。\n')
    common=[Path(__file__),R/'scripts/eval/metric_invariance_128.py',R/'scripts/experiment_harness/metric_invariance_128_guest.py',
        R/'scripts/eval/metric_win_tie_126a.py',R/'scripts/eval/ttm_metrics_115.py',R/'scripts/eval/eval_metrics.py',
        R/'scripts/eval/meva_score_dir.py',R/'scripts/eval/meva_runtime.py',R/'docs/experiments/meva_095_model_lock.json',
        R/'scripts/experiment_harness/notification_receipts.py',R/'scripts/experiment_harness/secondary_queue_controller.py',
        R/'scripts/notify_experiment_webhook.py',Path('/home/kojiek/gpu_queue/lib_scheduler.py'),
        R/'scripts/validate_experiment_harness_documents.py',Path(M.A.TSV),mf,acceptance,approval,
        R/'docs/experiments/metric_invariance_readjudication_128_20261017.md',
        R/'docs/experiments/harn/metric_invariance_128/identity_gate_amendment_20261017.json',
        R/'docs/experiments/harn/metric_invariance_128/quiet_lufs_amendment_20261017.json',
        R/'docs/experiments/harn/metric_invariance_128/budget_amendment_20261017.json',
        R/'docs/experiments/harn/metric_invariance_128/claim_identity_amendment_20261018.json',
        R/'docs/experiments/results/metric_win_tie_126a_summary.json',
        R/'weights/music_speech_audioset_epoch_15_esc_89.98.pt']
    common += [Path(p) for p in scans.values()]
    if reuse:common += [B/'calibration_reuse.json']+[Path(p) for p in reuse['files']]
    if resume:
        common += [B/'score_resume.json',B/'storage_correction_plan.json',B/'archive_gate.json',B/'scan_reuse.json']
        for path,expected in gate['files'].items():
            assert sha(path)==expected,'archive verification evidence changed'
            common.append(Path(path))
        common += [Path(p) for p in resume['files']]
        common += [Path(u['destination'])/name for u in resume['units'] for name in ('scores.npz','done.json')]
    mfdata=json.loads(mf.read_text())
    common+=list(dict.fromkeys(Path(p) for p in list(mfdata['reference_115'].values())+list(mfdata['table_references'].values())+
        list(mfdata['mir'].values())+[mfdata['floor_mir']]))
    aes=Path('/home/kojiek/.cache/huggingface/hub/models--facebook--audiobox-aesthetics/snapshots')
    common += sorted(p for p in aes.glob('*/*') if p.name in ('model.safetensors','config.json'))
    common += sorted(DEPENDENCIES.rglob('*.py'))
    common += sorted(DEPENDENCIES.rglob('*.so'))
    common += [R/'docs/experiments/ttm_metrics_115_20261009_contract.json',
               R/'docs/experiments/ttm_metrics_seeds_118_20261011_contract.json',
               M.A.R115/'selection.json',M.A.R115/'summary.json',E/'ttm_metrics_seeds_118_20261011/summary.json']
    common += sorted((R/'docs/experiments/harn/negmf_full_120').glob('*contract.json'))
    for rs in list(mfdata['table_b'].values())+list(mfdata['negmf'].values()):
        common += sorted(Path(rs[0]['path']).parent.parent.glob('*REPORT.json'))
    common=list(dict.fromkeys(common))
    immutable={str(p):sha(p) for p in common}
    for number,part,label in [(128,'a','calibration'),(129,'b','table_b'),(130,'c','negmf')]:
        if part not in parts:continue
        name=job(number,label)+'.sh';cp=R/f'docs/experiments/metric_invariance_128{part}_20261017{suffix}_contract.json'
        launcher=R/'scripts/queue_candidates'/name;out=O/part;out.mkdir(exist_ok=True)
        tmp=(args.transient_root if args.transient_root else E/f'metric_invariance_128_20261017{suffix}_tmp')/part
        tmp.mkdir(parents=True,exist_ok=True);tmp.chmod(0o700)
        assert not (out/'started.json').exists(),'refuse replacing launched contract'
        launcher.write_text(f'''#!/bin/bash
# GPU_QUEUE_CONTRACT={cp}
export GPU_QUEUE_JOB_SCRIPT="$0"
export GPU_QUEUE_CONTRACT={cp}
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR={tmp}
export PYTHONPATH=/mnt/seagate/kojiek_data/metric_invariance_128_20261017/python_dependencies
export PYTHONDONTWRITEBYTECODE=1
cd /home/kojiek/MeanAudio || exit 1
exec {PY} /home/kojiek/MeanAudio/scripts/experiment_harness/metric_invariance_128_guest.py
''');launcher.chmod(0o755)
        on_output_fs=os.stat(tmp).st_dev==os.stat(out).st_dev
        filesystems=[{'path':str(out),'hard_floor_bytes':50*2**30,'warning_floor_bytes':80*2**30,
            'peak_additional_bytes':8*2**30 if on_output_fs else 2*2**30,'transient_bytes':6*2**30 if on_output_fs else 512*2**20,'recovery_reserve_bytes':2*2**30},
            {'path':str(R),'hard_floor_bytes':50*2**30,'warning_floor_bytes':60*2**30,
             'peak_additional_bytes':512*2**20 if on_output_fs else 6*2**30,'transient_bytes':100*2**20,'recovery_reserve_bytes':2*2**30}]
        if on_output_fs and os.stat(out).st_dev==os.stat(R).st_dev:
            # Score output and scratch share NVMe: one filesystem budget, including
            # 6 GiB audio plus reports; atomic duplication is counted separately.
            filesystems=[{'path':str(R),'hard_floor_bytes':50*2**30,'warning_floor_bytes':60*2**30,
                'peak_additional_bytes':7*2**30,'transient_bytes':100*2**20,'recovery_reserve_bytes':2*2**30}]
        action=R/'scripts/eval/metric_invariance_128.py';guest=R/'scripts/experiment_harness/metric_invariance_128_guest.py'
        calibration_report=reuse['report'] if reuse else str(O/'a/summary.json')
        calibration_job=reuse['job'] if reuse else job(128,'calibration')
        run_date=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
        c={'schema_version':1,'document_kind':'eval_only_metrics_contract','experiment_id':f'metric-invariance-128{part}-20261017{suffix}',
            'run_id':f'run-{run_date}-{number}-metric-invariance{suffix}','part':part,'status':'authorized_pending_preflight','launch_allowed':True,
            'priority':'P2','queue_role':'p2','queue_name':name,'dependencies':[] if part=='a' else
                [calibration_job]+([job(129,'table_b')] if part=='c' else []),
            'ordering_dependencies':[] if part=='a' else [calibration_job if part=='b' else job(129,'table_b')],
            'approval_required':True,'launch_authorization':{'valid':True,'gpu_launch_allowed':True,'trusted_channel':'operator_console',
              'operator_instruction':'GO (2026-10-18)' if reuse else '開始實驗','scope':'Implement and run PR2 design as explicitly labelled secondary historical-score invariance study; no training.'},
            'manifest':str(mf),'immutable_files':immutable,'harn_bundle':str(B/part),'lufs_scan_reports':scans,
            'design_document':str(R/'docs/experiments/metric_invariance_readjudication_128_20261017.md'),
            'protocol':{'classification':'separately_named_secondary_historical_metric_sensitivity',
              'historical_cfg0_label_preserved':True,'generation':False,'metrics':M.METRICS,'margins':mfdata['margins'],
              'normalization_lufs':-30,'limiter':False,
              'quiet_lufs_rule':{'fallback':'unit_peak_probe_then_-30_LUFS','invalid':'zero_empty_nonfinite_or_still_undefined',
                'sensitivity':'exclude_fallback_ids_rejudge','preflight_scan':'--scan-lufs',
                'amendment':'docs/experiments/harn/metric_invariance_128/quiet_lufs_amendment_20261017.json'},'ensemble_shifts_ms':[0,40,80,120],'ensemble_polarities':[1,-1,1,-1],
              'variants':M.VARIANTS,'tie_variants':M.TIES,'degradation_ladders':M.LADDERS,'tie_pass':.8,
              'threshold_max_regression_steps':1,'wrong_direction_max_increase':.05,'identity_gate':{'aes_max_abs':.01,'aes_abs_mean_signed':.001,'other_max_abs':1e-4,
                'amendment':'docs/experiments/harn/metric_invariance_128/identity_gate_amendment_20261017.json'},
              'sample_rng_seed':20261017,'n_a_candidates':[1000,600,400],'budget_seconds_a':args.budget_hours_a*3600,
              'fallback':'C3 per metric; label sensitivity_only if no candidate passes',
              'clap_modes':['C0','C2','C3'],'meva_modes':['C0','C3'],'peak_over_fs_recorded':True,
              'budget_deferral_c_seconds':17*3600},
            'storage':{'path':str(out),'transient_root':str(tmp),'min_free_bytes':50*2**30},'filesystems':filesystems,
            'resume':{'kind':'from_scratch_with_autoresume','checkpoint':None,'iteration':0,
              'autoresume':str(out/'pause_progress.json'),'pause_progress':str(out/'pause_progress.json'),
              'behavior':'Hash-bound completed unit skipped; incomplete own temporary audio removed and redone; no upstream deletion.'},
            'watcher':{'stall_seconds':10800,'poll_seconds':5,'healthy_poll_model_calls':0,'automatic_repair':False},
            'calibration_summary':calibration_report,'previous_summary':(calibration_report if part=='b' else str(O/'b/summary.json')) if part!='a' else None,
            'scientific_dependencies':{} if part=='a' else {
                calibration_job:calibration_report,
                **({job(129,'table_b'):str(O/'b/summary.json')} if part=='c' else {})},
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
            'cleanup_authorization':{'paths':[str(tmp)],'behavior':'Only this run temporary scoring audio; upstream raw, lvl30, stimuli and cached weights retained.'}}
        if part in 'bc':
            blocking=sorted({claim[1] for claim in mfdata['claims'][part]})
            c['protocol']['readjudication_identity_policy']={'blocking_metrics':blocking,
                'diagnostic_metrics':[m for m in ('PQ','CE','CU','PC','clap') if m not in blocking],
                'checked_n':200,'context_batch_size':32,
                'context_rule':'next real records complete the final batch; compare only the original first 200',
                'amendment':'docs/experiments/harn/metric_invariance_128/claim_identity_amendment_20261018.json'}
            # C may validly defer before scoring: its identity report is validated
            # through summary/postflight only when scoring actually takes place.
            if part=='b':c['reports'].append({'path':str(out/'identity_checks.json')})
            cells=mfdata['table_b'] if part=='b' else mfdata['negmf']
            c['notifications'] += ['identity_warning_'+name for name in cells]
        if reuse:
            c['calibration_reuse']=reuse
            c['budget_carry']={'completed_calibration_seconds':reuse['completed_calibration_seconds'],
                'readjudication_started_path':resume['original_b_started'] if resume else str(O/'b/started.json'),
                'scope':'completed A plus original B wall time, including storage relocation' if resume else 'completed A duration plus this fresh B run wall time; old terminal-to-restart idle excluded'}
            if resume:c['score_resume']=resume
            M.validate_calibration_reuse(c)
        write(cp,c);bundle(c,cp,acceptance);print(cp,flush=True)
    # Registration snapshot binds the whole chain, including blocked scientific
    # successors. Live eligibility is independently checked against host receipts.
    entries=[]
    for position,part in enumerate(parts,1):
        q=json.loads((B/part/'queue.json').read_text());entry=q['entries'][0]
        entry['position']=position
        if position>1:
            entry['status']='blocked';entry['dependencies']=[entries[0]['entry_id']]
            if position==3:entry['dependencies'].append(entries[1]['entry_id'])
        entries.append(entry)
    for part in parts:
        directory=B/part;q=json.loads((directory/'queue.json').read_text());q['entries']=entries
        write(directory/'queue.json',q)
        subprocess.run([PY,str(R/'scripts/validate_experiment_harness_documents.py'),
            '--contract',str(directory/'contract.json'),'--preflight',str(directory/'preflight.json'),
            '--ledger',str(directory/'ledger.json'),'--queue',str(directory/'queue.json')],check=True)


if __name__=='__main__':main()
