#!/usr/bin/env python3
"""Hash-verified cold archive; atomically retain source paths as symlinks."""
import concurrent.futures,ctypes,hashlib,json,os,shutil,stat,subprocess,sys,time
from pathlib import Path
R=Path('/home/kojiek/MeanAudio');B=R/'docs/experiments/harn/metric_invariance_128_nvme_r5'
plan=json.loads((B/'storage_correction_plan.json').read_text());floor=plan['hard_floor_bytes']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,p)
def fsync_dir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def capacity():
    free={str(p):shutil.disk_usage(p).free for p in (R,Path('/mnt/seagate'))}
    if min(free.values())<floor:raise RuntimeError('storage hard floor reached')
    return free
def notify(status,summary):
    subprocess.run(['/home/kojiek/venvs/dac/bin/python',str(R/'scripts/notify_experiment_webhook.py'),'--status',status,'--experiment','128-nvme-r5-cold-archive','--summary',summary],check=True,stdout=subprocess.DEVNULL)
def copy_one(pair):
    src,dst=pair;s0=src.stat()
    if not src.is_file() or src.is_symlink() or s0.st_uid!=os.getuid():raise RuntimeError('unexpected source ownership/type')
    capacity();h=hashlib.sha256();tmp=dst.with_name(dst.name+'.partial')
    if dst.exists():raise RuntimeError(f'refuse overwriting archive file {dst}')
    with src.open('rb') as f,tmp.open('xb') as g:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data);g.write(data)
        g.flush();os.fsync(g.fileno())
    want=h.hexdigest()
    if sha(tmp)!=want:raise RuntimeError(f'archive hash mismatch {src}')
    s1=src.stat()
    if (s1.st_ino,s1.st_size,s1.st_mtime_ns)!=(s0.st_ino,s0.st_size,s0.st_mtime_ns):raise RuntimeError('source changed during copy')
    shutil.copystat(src,tmp);os.replace(tmp,dst)
    return {'file':src.name,'bytes':s0.st_size,'sha256':want}

if __name__=='__main__':
    assert sha(__file__)==json.loads((B/'archive_cpu_contract.json').read_text())['driver_sha256']
    notify('start','128 storage correction: archive only completed 115 lvl23 audio (excluded from all current inputs), byte-verify and preserve paths; restore fresh scoring to NVMe.')
    completed=[];total_bytes=0;begin=time.time()
    try:
        for src_text in plan['cold_sources']:
            src=Path(src_text);system=src.parents[1].name;dst=Path(plan['archive_destination'])/system/'lvl23/audio'
            assert src.is_dir() and not src.is_symlink();inode=src.stat().st_ino
            files=sorted(src.iterdir());assert files and all(p.is_file() and not p.is_symlink() for p in files)
            assert not dst.exists();dst.mkdir(parents=True)
            rows=[];last=0
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for row in pool.map(copy_one,[(p,dst/p.name) for p in files]):
                    rows.append(row)
                    if time.monotonic()-last>5:
                        write(B/'archive_progress.json',{'system':system,'copied':len(rows),'total':len(files),'completed_systems':completed,'free_bytes':capacity(),'at':time.time()});last=time.monotonic()
            assert sorted(p.name for p in src.iterdir())==[p.name for p in files]
            assert sorted(p.name for p in dst.iterdir())==[p.name for p in files]
            manifest=B/f'archive_{system}.json';write(manifest,{'source':str(src),'destination':str(dst),'source_inode':inode,'files':rows})
            fsync_dir(dst);fsync_dir(manifest.parent)
            link=src.with_name('audio.nvme_r5_exchange');assert not link.exists();os.symlink(dst,link)
            libc=ctypes.CDLL(None,use_errno=True);exchange=libc.renameat2
            exchange.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
            if exchange(-100,os.fsencode(link),-100,os.fsencode(src),2):raise OSError(ctypes.get_errno(),'atomic exchange failed')
            fsync_dir(src.parent)
            assert src.is_symlink() and src.resolve()==dst.resolve() and link.is_dir() and not link.is_symlink() and link.stat().st_ino==inode
            assert sorted(p.name for p in link.iterdir())==[r['file'] for r in rows]
            for row in rows:
                if sha(link/row['file'])!=row['sha256'] or sha(src/row['file'])!=row['sha256']:raise RuntimeError('post-exchange verification failed; old copy retained')
            shutil.rmtree(link);fsync_dir(src.parent)
            completed.append(system);total_bytes+=sum(r['bytes'] for r in rows)
            write(B/'archive_results.json',{'status':'archiving','completed_systems':completed,'archived_bytes':total_bytes,'free_bytes':capacity(),'elapsed_seconds':time.time()-begin})
        write(B/'archive_results.json',{'status':'passed','completed_systems':completed,'archived_bytes':total_bytes,'free_bytes':capacity(),'elapsed_seconds':time.time()-begin})
        notify('success',f'128 cold archive verified: {len(completed)} completed lvl23 cells, {total_bytes} bytes; original paths preserved. New scoring can return to NVMe after registration.')
    except BaseException as e:
        write(B/'archive_incident.json',{'status':'held','error':str(e),'completed_systems':completed,'at':time.time()})
        notify('held','128 cold archive held; verified source/archive copies retained. No scoring launch until inspected.');raise
