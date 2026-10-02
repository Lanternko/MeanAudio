#!/usr/bin/env python3
"""Short P0 parity test on real audio, with shared GPU lock and queue checks."""
import fcntl
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from meva_runtime import ROOT,RUNTIME,Evaluator,sha
sys.path[:0]=['/home/kojiek/gpu_queue',str(ROOT/'scripts/experiment_harness')]
from lib_scheduler import probe_foreign
from notification_receipts import atomic_secure_json

def main():
    queue=Path('/home/kojiek/gpu_queue')
    with (queue/'gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any(list((queue/role/state).glob('*.sh')) for role in ['p1','p2'] for state in ['pending','running']):
            raise RuntimeError('Queue has pending/running work; short pilot must not compete')
        if probe_foreign()[0]!=1: raise RuntimeError('GPU resource probe not clear')
        import numpy as np
        import torch
        torch.set_num_threads(4)
        rows=json.loads((RUNTIME/'manifest.json').read_text())
        pam=[r for r in rows if r['set']=='pam' and r['system']!='real']
        selected=[next(r for r in pam if r['system']==s) for s in sorted({r['system'] for r in pam})[:3]]
        start=time.monotonic();model=Evaluator()
        spec=importlib.util.spec_from_file_location('official_extract',ROOT/'.external/MEva/scripts/features/extract_sae_features_musicdiscovery.py')
        official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        results=[]
        for r in selected:
            if sha(r['audio_path'])!=r['audio_sha256']:raise ValueError('Input changed')
            t=time.monotonic();pred=model.score(r['audio_path']);elapsed=time.monotonic()-t
            repeat=model.score(r['audio_path'])
            z=official._extract_batch(model.musicgen,model.sae,[{'audio_path':r['audio_path']}],
                                     'cuda',1500,4096,0.)[0]
            assert np.isfinite(z).all() and np.count_nonzero(z)>0
            pad=np.zeros((1500,4096),dtype=np.float32);pad[:len(z)]=z
            with torch.inference_mode():
                expected=model.cnn(None,torch.from_numpy(pad).unsqueeze(0).cuda(),
                                  sae_len=torch.tensor([len(z)],device='cuda')).item()
            parity=abs(pred['meva_raw']-expected);repro=abs(pred['meva_raw']-repeat['meva_raw'])
            assert parity<=1e-4 and repro<=1e-4,(parity,repro)
            results.append({'key':r['key'],'meva_raw':pred['meva_raw'],'parity_abs':parity,
                            'repeat_abs':repro,'score_seconds':elapsed,'frames':len(z)})
            print(results[-1],flush=True)
        report={'status':'passed','n_real_audio':len(selected),'results':results,
                'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'total_seconds':time.monotonic()-start,
                'runtime_source_sha256':sha(ROOT/'scripts/eval/meva_runtime.py'),
                'official_parity':'batch1 native short features + same exact upstream f03 architecture',
                'scope':'numerical adapter smoke, not human validity or original torch2.1 cross-hardware parity'}
        atomic_secure_json(RUNTIME/'smoke.json',report)
        print(json.dumps(report,indent=2),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
