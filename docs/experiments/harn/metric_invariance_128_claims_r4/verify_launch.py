import csv,datetime,json,math,sys
from pathlib import Path
import numpy as np
R=Path('/home/kojiek/MeanAudio');Q=Path('/home/kojiek/gpu_queue');B=R/'docs/experiments/harn/metric_invariance_128_claims_r4'
O=Path('/mnt/seagate/kojiek_data/metric_invariance_128_20261017_claims_r4/b')
sys.path[:0]=[str(R/'scripts/eval'),str(R/'scripts/experiment_harness'),str(Q)]
import metric_invariance_128 as M
from lib_scheduler import pid_start_time
from notification_receipts import validate_delivered_receipt
cp=R/'docs/experiments/metric_invariance_128b_20261017_claims_r4_contract.json';c=json.loads(cp.read_text())
seat=json.loads((Q/'p2.running.json').read_text());monitor=json.loads((O/'supervisor/monitor.json').read_text())
assert seat['job_id']=='129_metric_invariance_table_b_claims_r4' and seat['phase']=='running'
assert pid_start_time(seat['pid'])==seat['start_time']
assert monitor['child_pid'] and pid_start_time(monitor['child_pid'])==monitor['child_start_time']
assert all(f['state']!='hard_stop' for f in monitor['filesystems'])
name='meanaudio_s14159265_cfg0';checks_path=O/'identity_checks.json'
unit=O/'table'/name/'chunk00000/C1'
if not checks_path.exists() or not (unit/'done.json').exists():
    print('Waiting for first identity and 1000-clip C1 unit');raise SystemExit(75)
checks=json.loads(checks_path.read_text());check=checks[name]
assert check['passed'] and not check['failed_blocking_metrics'] and check['blocking_metrics']==['PQ','clap']
assert (check['checked_n'],check['scored_context_n'])==(200,224)
manifest=json.loads(Path(c['manifest']).read_text());expected=manifest['table_b'][name]
proof=[]
for group,d,n in [('identity',O/'identity'/name/'clean/C0',224),('first_formal_C1_chunk',unit,1000)]:
    meta=json.loads((d/'done.json').read_text());scores=d/'scores.npz';assert M.sha(scores)==meta['score_sha256'] and meta['n']==n
    with np.load(scores) as z:
        assert z['values'].shape==(n,6) and z['ids'].tolist()==[r['id'] for r in expected[:n]]
        assert np.isfinite(z['values'][:,:4]).all()
        if group=='identity':
            got={'ids':z['ids'][:200],'values':z['values'][:200]}
            recheck=M.readjudication_identity(got,M.table(manifest['table_references'][name]),['PQ','clap'],c['protocol']['identity_gate'])
            assert recheck['metrics']==check['metrics'] and recheck['passed']
    M.write(B/f'live_{group}_metadata.json',meta)
    proof.append({'kind':group,'path':str(d),'n':n,'scores_sha256':M.sha(scores),'done_sha256':M.sha(d/'done.json')})
receipts=[]
for event in ('disk_warning','preflight_pass','queue_handoff','start'):
    p=Q/'notification_receipts'/c['experiment_id']/f'{event}.json'
    ok,reason=validate_delivered_receipt(p,contract_path=cp,launcher_path=Q/'p2/running'/c['queue_name'],event=event,status='start')
    assert ok,(event,reason)
    M.write(B/f'live_receipt_{event}.json',json.loads(p.read_text()))
    receipts.append({'event':event,'valid':ok,'reason':reason,'sha256':M.sha(p)})
assert (Q/'p2/pending/130_metric_invariance_negmf_claims_r4.sh').is_file()
assert not list((Q/'p2/pending').glob('128*claims_r4*'))
M.validate_calibration_reuse(c)
M.write(B/'live_identity_checks.json',checks)
M.write(B/'live_seat.json',seat);M.write(B/'live_monitor.json',monitor)
M.write(B/'live_controller_ledger.json',json.loads((O/'supervisor/controller_ledger.json').read_text()))
M.write(B/'live_b_started.json',json.loads((O/'started.json').read_text()))
result={'status':'running_verified','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'3977f766a42c0f9f4f4293616dc21001ff1abeab','contract_sha256':M.sha(cp),'seat':seat,'child_pid':monitor['child_pid'],'child_start_time':monitor['child_start_time'],'identity':check,'completed_units':proof,'notification_receipts':receipts,'next_job_pending':'130_metric_invariance_negmf_claims_r4','calibration_reused':c['calibration_summary'],'carried_calibration_seconds':c['budget_carry']['completed_calibration_seconds'],'elapsed_budget_seconds':M.prior_phase_elapsed(c),'final_scientific_results_available':False}
M.write(B/'launch_verification.json',result)
print(json.dumps(result,indent=2))
