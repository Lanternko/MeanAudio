import hashlib,json,shutil,time
from pathlib import Path
R=Path('/home/kojiek/MeanAudio');B=R/'docs/experiments/harn/metric_invariance_128_nvme_r5'
p=B/'archive_results.json'
if not p.exists():raise SystemExit(75)
a=json.loads(p.read_text());free=shutil.disk_usage(R).free
required=50*2**30+7*2**30+100*2**20+2*2**30
if free<required:
    print(f'Archive capacity gate waiting: free {free}, required {required}');raise SystemExit(75)
files={};verified=[]
for name in a['completed_systems']:
    mp=B/f'archive_{name}.json';m=json.loads(mp.read_text())
    assert Path(m['source']).is_symlink() and Path(m['source']).resolve()==Path(m['destination']).resolve()
    files[str(mp)]=hashlib.sha256(mp.read_bytes()).hexdigest()
    verified.append({'system':name,'n':len(m['files']),'bytes':sum(r['bytes'] for r in m['files']),'source':m['source'],'archive':m['destination']})
assert verified
out=B/'archive_gate.json';assert not out.exists()
out.write_text(json.dumps({'status':'passed','at':time.time(),'free_bytes':free,'required_bytes':required,'hard_floor_bytes':50*2**30,'peak_additional_bytes':7*2**30,'transient_bytes':100*2**20,'recovery_reserve_bytes':2*2**30,'verified_cold_cells':verified,'files':files,'remaining_registered_cold_archive':'May continue in background; excludes all study inputs, source directories preserved until hashes/fsync/atomic exchange pass.'},indent=2)+'\n')
print(json.dumps({'status':'passed','verified_cells':len(verified),'free_GiB':free/2**30},indent=2))
