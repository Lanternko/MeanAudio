import fcntl,os,json,hashlib,sys,datetime
from pathlib import Path
R=Path('/home/kojiek/MeanAudio');Q=Path('/home/kojiek/gpu_queue');B=R/'docs/experiments/harn/metric_invariance_128_claims_r4'
sys.path.insert(0,str(Q));from lib_scheduler import accept_guest
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert '128 preflight passed' in (B/'live_preflight_b.txt').read_text()
cs=[json.loads((R/f'docs/experiments/metric_invariance_128{part}_20261017_claims_r4_contract.json').read_text()) for part in 'bc']
staged=[];installed=[]
with (Q/'gpu0.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=sorted(p.name for p in (Q/'p2/pending').glob('*.sh'))
    assert not prior,'Existing P2 queue changed; preserve and append safely before installing'
    assert not list((Q/'p2/running').glob('*.sh')),'P2 running changed'
    try:
        for c in cs:
            assert c['protocol']['budget_seconds_a']==32400
            assert c['protocol']['readjudication_identity_policy']['blocking_metrics']==(['PQ','clap'] if c['part']=='b' else ['PQ'])
            assert c['calibration_reuse']['job']=='128_metric_invariance_calibration_budget_r3'
            assert c['storage']['transient_root'].startswith('/mnt/seagate/')
            name=c['queue_name'];src=R/'scripts/queue_candidates'/name;dst=Q/'p2/pending'/name
            assert not list((Q/'p2').glob('*/'+name)),f'duplicate {name}'
            assert sha(src)==c['bindings']['launcher_sha256']
            for fs in c['filesystems']:
                st=os.statvfs(fs['path']);assert st.f_bavail*st.f_frsize>=fs['hard_floor_bytes']
            tmp=dst.with_name('.'+dst.name+'.stage')
            with tmp.open('xb') as f:f.write(src.read_bytes());f.flush();os.fsync(f.fileno())
            tmp.chmod(0o755);staged.append(tmp)
            ok,reason=accept_guest(tmp);assert ok,reason
        for tmp,c in zip(staged,cs):
            dst=Q/'p2/pending'/c['queue_name'];os.replace(tmp,dst);installed.append(dst)
        fd=os.open(Q/'p2/pending',os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        record={'status':'installed','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'prior_p2_pending':prior,'order':[c['queue_name'] for c in cs],'launchers':{str(p):sha(p) for p in installed},'installation_guard':'exclusive shared GPU flock; empty P2 pending/running verified; exact launcher and duplicate checks; existing host retained'}
        (B/'queue_installation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    except BaseException:
        for p in staged:p.unlink(missing_ok=True)
        for p in installed:p.unlink(missing_ok=True)
        raise
