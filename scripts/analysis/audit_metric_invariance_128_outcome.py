#!/usr/bin/env python3
"""Export host evidence or audit committed 128 results with standard Python only.

Default: no models, NumPy, host data, or writes. --export-host copies frozen
artifacts; --verify-audio additionally replays the 200 inputs on CPU only.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

R = Path(__file__).resolve().parents[2]
DEFAULT = R/'docs/experiments/harn/metric_invariance_128_budget_r3/outcome_20261018'
METRICS = ('PQ','CE','CU','PC','clap','meva')
MODES = ('C0','C1','C2','C3')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def tsv(p,rows):
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        w.writeheader();w.writerows(rows)


def export_host(out, verify_audio, scratch):
    import numpy as np
    sys.path.insert(0,str(R/'scripts/eval'))
    import metric_invariance_128 as M
    sys.path.insert(0,str(R/'scripts/experiment_harness'))
    from notification_receipts import validate_delivered_receipt
    O=Path('/mnt/seagate/kojiek_data/metric_invariance_128_20261017_budget_r3')
    Q=Path('/home/kojiek/gpu_queue')
    out.mkdir(parents=True,exist_ok=True)
    contracts={p:R/f'docs/experiments/metric_invariance_128{p}_20261017_budget_r3_contract.json' for p in 'abc'}
    cs={p:json.loads(f.read_text()) for p,f in contracts.items()}
    assert all(sha(c['bindings']['action'])==c['bindings']['action_sha256'] for c in cs.values())
    proc=subprocess.run([sys.executable,str(R/'scripts/eval/metric_invariance_128.py'),'--validate-only'],
        env=dict(os.environ,GPU_QUEUE_CONTRACT=str(contracts['a']),PYTHONPATH='/mnt/seagate/kojiek_data/metric_invariance_128_20261017/python_dependencies'),
        capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    write(out/'postflight_recheck.json',{'exit_code':proc.returncode,'action_sha256':cs['a']['bindings']['action_sha256'],
        'at':datetime.now(timezone.utc).isoformat(),'scope':'CPU: all completed unit hashes, ids/counts, required finite scores, 700 calibration and 44 MIDI units'})
    shutil.copyfile(O/'a/summary.json',out/'a_summary.json')
    summary=json.loads((out/'a_summary.json').read_text())
    rows=[]
    for mode,systems in summary['cells'].items():
        for system,metrics in systems.items():
            for metric,variants in metrics.items():
                for variant,rate in variants.items():rows.append({'mode':mode,'variant':variant,'metric':metric,**rate,'system':system})
    tsv(out/'calibration_rates.tsv',rows)
    tsv(out/'midi_rates.tsv',[{'mode':mode,**rate,'case':case} for mode,cases in summary['midi_pq'].items() for case,rate in cases.items()])
    manifest=json.loads(Path(cs['b']['manifest']).read_text())
    name='meanaudio_s14159265_cfg0';directory=O/'b/identity'/name/'clean/C0'
    meta=json.loads((directory/'done.json').read_text());assert sha(directory/'scores.npz')==meta['score_sha256']
    with np.load(directory/'scores.npz') as z:unit={k:z[k] for k in z.files}
    reference_path=Path(manifest['table_references'][name]);reference=M.table(reference_path)
    assert list(unit['ids'])==sorted(reference)[:200]
    rows=[]
    for k,i in enumerate(unit['ids']):
        row={'index':k,'id':str(i)}
        row.update({f'score_{m}':float(unit['values'][k,j]) for j,m in enumerate(METRICS)})
        row.update({f'reference_{m}':float(reference[str(i)][m]) for m in METRICS[:-1]})
        row['system']=name;rows.append(row)
    tsv(out/'b_identity_200.tsv',rows)
    write(out/'b_identity_provenance.json',{'source':str(directory),'score_sha256':meta['score_sha256'],
        'reference_path':str(reference_path),'reference_sha256':sha(reference_path),'reference_n':len(reference),
        'ids_equal_sorted_reference_prefix':True,'batch_size':32})
    shutil.copyfile(directory/'done.json',out/'b_identity_done.json')
    unit_manifest=[]
    for part in 'ab':
        for done in sorted((O/part).glob('*/*/*/*/done.json')):
            d=json.loads(done.read_text());p=done.parent/'scores.npz';assert sha(p)==d['score_sha256']
            unit_manifest.append({'part':part,'score_path':str(p),'score_sha256':d['score_sha256'],
                'done_path':str(done),'done_sha256':sha(done),'n':d['n'],'mode':d['mode']})
    assert sum(u['part']=='a' for u in unit_manifest)==764 and sum(u['part']=='b' for u in unit_manifest)==1
    write(out/'score_unit_manifest.json',unit_manifest)
    receipts={};states={}
    for part,state in [('a','done'),('b','failed'),('c','held')]:
        c=cs[part];launcher=Q/'p2'/state/c['queue_name'];terminal=launcher.with_suffix('.terminal.json')
        t=json.loads(terminal.read_text());states[part]=t['status'];shutil.copyfile(terminal,out/f'terminal_{part}.json')
        for receipt in (Path(c['notification_receipts']['root'])/c['experiment_id']).glob('*.json'):
            ok,reason=validate_delivered_receipt(receipt,contract_path=contracts[part],launcher_path=launcher,event=receipt.stem)
            assert ok,(str(receipt),reason);receipts[str(receipt)]={'valid':True,'sha256':sha(receipt)}
        for filename in ['controller_ledger.json','monitor.json']:
            shutil.copyfile(O/part/'supervisor'/filename,out/f'{part}_{filename}')
    assert states=={'a':'completed','b':'failed','c':'held'}
    pending=sorted(p.name for p in (Q/'p2/pending').glob('*.sh'));running=sorted(p.name for p in (Q/'p2/running').glob('*.sh'))
    assert not pending and not running
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    write(out/'terminal_snapshot.json',{'at_taipei':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'states':states,'p2_pending':pending,'p2_running':running,'gpu_compute_processes':compute.splitlines() if compute else [],'receipts':receipts})
    log=Path('/home/kojiek/logs/p2_host.log').read_text();start=log.rfind('P2_STARTING 129_metric_invariance_table_b_budget_r3 ')
    (out/'host_log_terminal_excerpt.txt').write_text(log[start:])
    if verify_audio:
        import soundfile as sf
        records=manifest['table_b'][name][:200];matches=[]
        with tempfile.TemporaryDirectory(prefix='128-b-audio-',dir=scratch) as td:
            for rec in records:
                got=M.prepare_one((rec,name,'clean','C0',td));p=Path(got['paths'][0])
                x,sr=sf.read(p,dtype='float32');original=reference_path.parents[2]/'audio'/f'{rec["id"]}.flac'
                y,sy=sf.read(original,dtype='float32')
                assert sha(original)==rec['lvl30_sha256'] and sr==sy and np.array_equal(x,y),rec['id']
                matches.append({'id':rec['id'],'raw_sha256':rec['sha256'],'lvl30_sha256':sha(original),
                    'decoded_float32_sha256':hashlib.sha256(x.tobytes()).hexdigest()});p.unlink()
        write(out/'b_audio_replay.json',{'n':len(matches),'all_decoded_float32_identical':True,
            'scope':'CPU only, exact registered raw-to-lvl30 + clean transform replay; no model or padding test','clips':matches})
    files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['provenance.json','audit.json']}
    write(out/'provenance.json',{'source_contract':str(contracts['b'].relative_to(R)),'source_contract_sha256':sha(contracts['b']),
        'source_manifest_sha256':sha(cs['b']['manifest']),'action_sha256':cs['b']['bindings']['action_sha256'],
        'auditor_sha256':sha(__file__),'files':files})


def decisions(cells,protocol):
    verdicts={};chosen={}
    systems=sorted(cells['C0'])
    for metric in ['PQ','CE','CU','clap']:
        verdicts[metric]={}
        for mode in MODES:
            if mode=='C1' and metric=='clap':continue
            failures=[]
            for system in systems:
                row,raw=cells[mode][system][metric],cells['C0'][system][metric]
                for variant in protocol['tie_variants']:
                    if not row[variant]['n'] or row[variant]['tie']<protocol['tie_pass']:failures.append(f'{system}/{variant}/tie')
                for name,ladder in protocol['degradation_ladders'].items():
                    a=next((i for i,v in enumerate(ladder) if raw[v]['worse']>=.8),len(ladder))
                    b=next((i for i,v in enumerate(ladder) if row[v]['worse']>=.8),len(ladder))
                    if b>a+protocol['threshold_max_regression_steps']:failures.append(f'{system}/{name}/threshold')
                    if b<len(ladder) and row[ladder[b]]['better']>raw[ladder[b]]['better']+protocol['wrong_direction_max_increase']:
                        failures.append(f'{system}/{name}/reversal')
            verdicts[metric][mode]={'passed':not failures,'failures':sorted(failures)}
        passing=[m for m in MODES if verdicts[metric].get(m,{}).get('passed')]
        chosen[metric]={'mode':passing[0] if passing else 'C3','sensitivity_only':not passing}
    return verdicts,chosen


def audit(out):
    provenance=json.loads((out/'provenance.json').read_text())
    assert all(sha(out/p)==h for p,h in provenance['files'].items())
    cp=R/provenance['source_contract'];assert sha(cp)==provenance['source_contract_sha256']
    protocol=json.loads(cp.read_text())['protocol'];s=json.loads((out/'a_summary.json').read_text())
    a=json.loads((out/'terminal_a.json').read_text())
    assert a['status']=='completed' and a['evidence']['reports'][0]['sha256']==sha(out/'a_summary.json')
    for cells,reported,chosen in [(s['cells'],s['verdicts'],s['chosen']),
        (s['lufs_fallback_sensitivity']['cells'],s['lufs_fallback_sensitivity']['verdicts_excl'],s['lufs_fallback_sensitivity']['chosen_excl'])]:
        verdicts,calculated=decisions(cells,protocol);assert calculated==chosen
        for metric,modes in verdicts.items():
            for mode,v in modes.items():
                assert v['passed']==reported[metric][mode]['passed'] and v['failures']==sorted(reported[metric][mode]['failures'])
    rows=list(csv.DictReader((out/'b_identity_200.tsv').open(),delimiter='\t'));assert len(rows)==200
    gate=protocol['identity_gate'];metrics={};failed=[]
    tail=len(rows)-len(rows)%32
    for metric in METRICS[:-1]:
        diff=[float(r['score_'+metric])-float(r['reference_'+metric]) for r in rows]
        mx=max(map(abs,diff));mean=sum(diff)/len(diff)
        bound=gate['other_max_abs'] if metric=='clap' else gate['aes_max_abs']
        stats={'max_abs':mx,'mean_signed':mean,'prefix_192_max_abs':max(map(abs,diff[:tail])),
            'tail_8_max_abs':max(map(abs,diff[tail:])),
            'indices_over_1e4':[i for i,d in enumerate(diff) if abs(d)>1e-4],
            'clips_over_registered_max':[{'index':i,'id':rows[i]['id'],'diff':d} for i,d in enumerate(diff) if abs(d)>bound]}
        metrics[metric]=stats
        if mx>bound or (metric!='clap' and abs(mean)>gate['aes_abs_mean_signed']):failed.append(metric)
    sens=s['lufs_fallback_sensitivity'];q3=s['midi_pq']['C1']['blastnote_x1']['better']<.5
    q3_excl=sens['midi_pq']['C1']['blastnote_x1']['better']<.5
    assert q3==s['q3_prediction_passed'] and q3_excl==sens['q3_prediction_passed_excl']
    return {'status':'audit_passed','n_a':s['n_a'],'chosen':s['chosen'],
        'passing_modes':{m:[k for k,v in vs.items() if v['passed']] for m,vs in s['verdicts'].items()},
        'midi_blastnote_x1':{mode:s['midi_pq'][mode]['blastnote_x1'] for mode in MODES},
        'q3_prediction_passed':q3,'q3_prediction_passed_excl':q3_excl,
        'fallback_excluded_counts':sens['excluded_counts'],'decisions_robust_to_fallback':sens['robust_to_lufs_fallback'],
        'b_identity':{'failed_metrics':failed,'prefix_n':tail,'tail_n':len(rows)-tail,'metrics':metrics},
        'scientific_scope':'128a completed. No Table B, NegMF or rhythm-stratification conclusion: b failed its identity gate; c held.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--snapshots',type=Path,default=DEFAULT)
    ap.add_argument('--export-host',action='store_true');ap.add_argument('--verify-audio',action='store_true')
    ap.add_argument('--scratch',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    if args.verify_audio and (not args.export_host or not args.scratch):ap.error('--verify-audio requires --export-host and --scratch')
    if args.export_host:export_host(args.snapshots,args.verify_audio,args.scratch)
    result=audit(args.snapshots)
    if args.output:write(args.output,result)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
