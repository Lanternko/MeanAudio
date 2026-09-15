#!/usr/bin/env python3
"""Prepare immutable 051 contract; does not install a launcher or launch compute."""
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime,timezone
R=Path('/home/kojiek/MeanAudio'); O=Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/loudness_aes_cfg3_20260911')
B=R/'docs/experiments/harn/loudness_aes_cfg3_20260911'; C=R/'docs/experiments/loudness_aes_cfg3_20260911_contract.json'
PYTHON='/home/kojiek/venvs/dac/bin/python'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
    O.mkdir(mode=0o700,parents=True,exist_ok=True);B.mkdir(parents=True,exist_ok=True)
    if (O/'audio_manifest.json').exists():raise SystemExit('refuse mutating launched contract')
    source=Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
    rows=list(csv.DictReader(source.open(),delimiter='\t'));assert len(rows)==len({r['id'] for r in rows})==5521
    neg='low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
    tsv=O/'musiccaps5521_cfg3_fidelity8.tsv'
    with tsv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id','caption','negative_prompt'],delimiter='\t');w.writeheader()
        for r in rows:w.writerow({'id':r['id'],'caption':r['caption'],'negative_prompt':neg})
    approval=B/'operator_request.txt';approval.write_text('設計實驗並加入 queue\n')
    old=json.loads((R/'docs/experiments/instrument_conflict_cfg3_20260908_contract.json').read_text())
    action=R/'scripts/eval/loudness_aes_cfg3_20260911.py';guest=R/'scripts/experiment_harness/loudness_aes_cfg3_20260911_guest.py';pre=R/'scripts/eval/validate_loudness_aes_cfg3_20260911.py'
    launcher=R/'scripts/queue_candidates/051_loudness_aes_cfg3_fidelity8.sh';live=Path('/home/kojiek/gpu_queue/p2/pending')/launcher.name
    snapshot=Path('/home/kojiek/.cache/huggingface/hub/models--facebook--audiobox-aesthetics/snapshots/9b1dd8e5df9af7216e836a98974fe3b82c56ded6')
    checkpoint=Path(old['inputs'][0]['path']);clap=R/'weights/music_speech_audioset_epoch_15_esc_89.98.pt'
    generation=[PYTHON,str(R/'scripts/eval/instrument_conflict_cfg3_20260908_generate.py'),'--variant','meanaudio_s','--model_path',str(checkpoint),'--output',str(O/'_audio'/'baseline'),'--tsv',str(tsv),'--use_meanflow','--num_steps','25','--cfg_strength','3','--no_text_attention_mask','--encoder_name','t5_clap','--text_c_dim','512','--seed','42','--full_precision','--no_q','--negative_prompt_column','negative_prompt']
    notifications=old['notification_receipts']
    paths=[checkpoint,clap,tsv,source,approval,action,guest,pre,launcher,R/'scripts/eval/instrument_conflict_cfg3_20260908_generate.py',R/'scripts/eval/score_musiccaps_per_item.py',R/'scripts/validate_experiment_harness_documents.py',R/'scripts/experiment_harness/secondary_queue_controller.py',Path(notifications['notifier']),Path(notifications['helper']),R/'weights/v1-16.pth',R/'weights/best_netG.pt',snapshot/'config.json',snapshot/'model.safetensors']
    paths += [R/'scripts/tests/selftest_loudness_aes_cfg3.py', R/'scripts/tests/run_loudness_queue_acceptance.py', Path(__file__).resolve(), R/'scripts/experiment_harness/bundle_loudness_aes_cfg3_20260911.py', R/'docs/experiments/loudness_aes_cfg3_20260911.md']
    paths+=sorted((R/'meanaudio').rglob('*.py'))
    for package in ('pyloudnorm','audiobox_aesthetics'):
        dist=importlib.metadata.distribution(package.replace('_','-'))
        paths += [Path(dist.locate_file(p)) for p in dist.files if str(p).endswith('.py') and str(p).startswith(package+'/')]
    # Freeze cached generation/text model files including refs: offline resolution is bound.
    cache=Path('/home/kojiek/.cache/huggingface/hub')
    for name in ('models--google--flan-t5-large','models--roberta-base'):
        d=cache/name
        paths += sorted(p for p in (d/'snapshots').rglob('*') if p.is_file())
        paths += sorted(p for p in (d/'refs').rglob('*') if p.is_file())
    paths=list(dict.fromkeys(paths))
    inputs=[{'kind':'immutable_input','path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
    inputs[0]['kind']='checkpoint'
    for i in inputs:
        if i['path']==str(tsv):i['kind']='evaluation_full_tsv'
    c={'schema_version':1,'document_kind':'preregistered_loudness_aes_evaluation_contract','experiment_id':'loudness-aes-cfg3-fidelity8-20260911','run_id':'run-20260911-051-loudness-aes-cfg3-fidelity8',
       'status':'operator_authorized_queue_tail','operator_instruction':'設計實驗並加入 queue','approval_record':str(approval),'launch_allowed':True,'approval_required':True,
       'launch_authorization':{'gpu_launch_allowed':True,'trusted_channel':'operator_console','operator':'responsible_operator','authorized_at':datetime.now(timezone.utc).isoformat(),'scope':'Design and append this loudness-stratified and paired-gain AES evaluation to the live queue tail.'},
       'queue_name':launcher.name,'queue_role':'p2','dependencies':[],'ordering_dependencies':[],
       'label':'c2p0_slot0_musiccaps5521_mf25_cfg3_fidelity8_seed42_nomask_fp32_loudness_aes',
       'protocol':{'classification':'canonical_baseline_with_separately_labeled_secondary_gain_interventions','dataset':'MusicCaps','rows':5521,'solver':'MeanFlow','steps':25,'cfg_strength':3,'negative_key':'fidelity8','negative_prompt':neg,'generation_seed':42,'mask':'NoMask','precision':'full','conditioning':'NoQ','scoring_batch_size':32,'metrics':['CLAP','AES_CE','AES_CU','AES_PC','AES_PQ']},
       'analysis':{'gains_db':[0,-3,-6,-9],'bootstrap_seed':20260911,'bootstrap_replicates':10000,'ci_percentiles':[2.5,97.5],'group_quantiles':[.2,.4,.6,.8],'ties':'keep together in lower bin; empty bins explicit','primary_association':'PQ LUFS Q5 minus Q1','primary_intervention':'paired PQ(-6dB)-PQ(0dB)','secondary':'RMS/crest grouping; CE/CU/PC; -3/-9dB; crest terciles within baseline LUFS; gain delta by baseline LUFS','multiple_comparisons':'Pointwise intervals. No global significance, equivalence or automatic promotion claim.','invalid_lufs':'null; excluded only from LUFS analyses; retain AES intervention'},
       'checkpoint':str(checkpoint),'clap_checkpoint':str(clap),'aes_snapshot':str(snapshot),'tsv':str(tsv),'source_tsv':str(source),'inputs':inputs,
       'reports':[{'negative_key':'fidelity8','path':str(O/'summary.json')}],'summary':str(O/'summary.json'),'harn_bundle':str(B),
       'resume':{'kind':'from_scratch_with_autoresume','iteration':0,'checkpoint':None,'autoresume':str(O/'resume_progress.json'),'behavior':'Partial unmanifested generation: remove own partial FLAC and regenerate seed42 original order. Manifested baseline: verify hashes. AES: verify per-ID contract/source/gain and score only missing. CLAP/report recompute if incomplete. Receipts persist.'},
       'storage':{'path':str(O),'transient_root':str(O/'_audio'),'estimated_peak_additional_bytes':6000000000,'warning_free_bytes':80000000000,'hard_stop_free_bytes':53687091200,'authorized_cleanup':'Only own unmanifested _audio/baseline/*.flac on interrupted generation; own _gain_batch/<expected-id>.wav after atomic per-ID score. Retain manifested baseline and all metrics/reports. No shared cache changes.'},
       'resource_budget':{'gpu':'P2 assigned GPU0 only','estimated_gpu_hours_range':[1.5,3],'generated_clips':5521,'aes_clip_conditions':22084,'clap_clips':5521,'writable_filesystems':[str(O),'/home/kojiek/logs','/home/kojiek/gpu_queue/notification_receipts',str(R)],'peak_additional_bytes':6000000000,'recovery_reserve_bytes':53687091200,'cpu_threads':2,'automatic_repair':False},
       'watcher':{'poll_seconds':2,'stall_seconds':7200,'healthy_poll_model_calls':0,'progress':'generation FLAC count/mtime; per-batch progress.json phase/count/mtime; completed summary; independent parent; 7200s hard stall','automatic_repair':False},
       'notifications':['queue_registration','generation_preflight_pass','generation_pass','aes_0_pass','aes_-3_pass','aes_-6_pass','aes_-9_pass','clap_pass','analysis_pass','start','terminal_success_failure_interruption','queue_handoff','unexpected_gpu_idle','disk_warning','disk_hard_stop','storage_recovered','stall'],
       'commands':{'preflight':[PYTHON,str(pre)],'run':[PYTHON,str(action)],'postflight':[PYTHON,str(action),'--validate-only']},'commands_generation':{'fidelity8':generation},
       'bindings':{'launcher':str(live),'launcher_sha256':sha(launcher),'harn':str(guest),'harn_sha256':sha(guest),'action':str(action),'action_sha256':sha(action)},'notification_receipts':notifications,
       'completion_evidence':old['completion_evidence']|{'authoritative_results':'summary.json + exact per-item AES/CLAP/audio hashes validated by commands.postflight','legacy_scheduler_compatibility':'historical CFG0 report is used only by old host terminal classifier, never as scientific comparator'},
       'queue_registration_context':{'prior_tail':'050_c2p0_slot4_no_digits_quarter.sh','policy':'append-only 051; preserve every active/pending/held entry; no scientific dependency'},
       'package_versions':{p:importlib.metadata.version(p) for p in ('numpy','scipy','soundfile','pyloudnorm','torch','audiobox-aesthetics')},
       'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'dirty_worktree':'pre-existing user edits preserved; executable/input raw hashes are authoritative'}
    c['phases']=[{'id':name,'pass':'verified exact expected artifacts, then required delivered gate event, then next phase','fail':'terminal failure, retain artifacts, outer host continues successors','invalid':'hold exact reason; no promotion','resource_wait':'75; poll capacity; repeat preflight and resume','resume':c['resume']['behavior']} for name in ['generate_canonical_baseline','aes_gain0','aes_gain_minus3','aes_gain_minus6','aes_gain_minus9','canonical_clap','analysis_report']]
    write(C,c);print(json.dumps({'contract':str(C),'inputs':len(inputs),'peak_additional_bytes':6000000000}))
if __name__=='__main__':main()
