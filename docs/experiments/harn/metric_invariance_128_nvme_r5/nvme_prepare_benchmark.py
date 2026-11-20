#!/usr/bin/env python3
"""CPU-only timing of unchanged waveform preparation, with owned NVMe scratch."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import concurrent.futures,json,multiprocessing as mp,shutil,sys,tempfile,time
from pathlib import Path
R=Path('/home/kojiek/MeanAudio');B=R/'docs/experiments/harn/metric_invariance_128_nvme_r5'
sys.path.insert(0,str(R/'scripts/eval'));import metric_invariance_128 as M

if __name__=='__main__':
    mf=Path('/mnt/seagate/kojiek_data/metric_invariance_128_20261017/inputs.json');m=json.loads(mf.read_text())
    tmp=Path(tempfile.mkdtemp(prefix='128-nvme-benchmark-',dir='/home/kojiek/eval_output_nvme'));rows=[]
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
            for name,records in m['table_b'].items():
                sample=[records[k*(len(records)-1)//31] for k in range(32)]
                warm=tmp/'warm';warm.mkdir()
                list(pool.map(M.prepare_one,[(r,name,'clean','C1',str(warm)) for r in sample[:4]],chunksize=1));shutil.rmtree(warm)
                for mode in ('C1','C2','C3'):
                    assert shutil.disk_usage(tmp).free>50*2**30+256*2**20
                    own=tmp/f'{name}_{mode}';own.mkdir();t=time.monotonic()
                    out=list(pool.map(M.prepare_one,[(r,name,'table',mode,str(own)) for r in sample],chunksize=8))
                    elapsed=time.monotonic()-t
                    row={'system':name,'mode':mode,'n':32,'seconds':elapsed,'seconds_per_clip':elapsed/32,'bytes':sum(Path(p).stat().st_size for v in out for p in v['paths'])}
                    rows.append(row);M.write(B/'nvme_prepare_timing.json',{'status':'benchmarking','action_sha256':M.sha(M.__file__),'manifest_sha256':M.sha(mf),'gpu':False,'workers':4,'rows':rows})
                    shutil.rmtree(own)
    finally:shutil.rmtree(tmp)
    M.write(B/'nvme_prepare_timing.json',{'status':'passed','action_sha256':M.sha(M.__file__),'manifest_sha256':M.sha(mf),'gpu':False,'workers':4,'rows':rows,'notes':'Preparation only, no scoring or study decision; 32 spread IDs per cell/mode, warm worker pool; small batches make ETA provisional.'})
    print('NVMe preparation benchmark complete',flush=True)
