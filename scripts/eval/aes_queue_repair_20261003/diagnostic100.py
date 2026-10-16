"""Bounded, registered diagnosis. No production score or threshold changes."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import soundfile as sf
SRC = Path('/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs/aes-followup-20261002')
FORMAL = SRC.parent/'aes-formal-20261003'
sys.path[:0] = [str(FORMAL), str(SRC), str(Path(__file__).parent)]
from common import spec, dump, sha, notify, load_records
import probe
from normalize_candidate import normalize_candidate

def main():
    import torch
    from audiobox_aesthetics.infer import initialize_predictor
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    c=spec(); root=Path(c['runtime_root']); root.mkdir(parents=True,exist_ok=True)
    predictor=initialize_predictor(); assert next(predictor.model.parameters()).dtype==torch.float32
    cache=root/'diagnosis.jsonl'; done=load_records(cache)
    settings=dict(torch=str(torch.__version__),cuda=str(torch.version.cuda),model_training=predictor.model.training,
                  encoder_training=predictor.model.wavlm_model.training,normalize=predictor.model.wavlm_model.cfg.normalize,
                  autocast=predictor.model.enable_autocast,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_tf32=torch.backends.cudnn.allow_tf32)
    axes=['PQ','CE','CU','PC'];started=time.monotonic()
    def forward(wavs):
        return predictor.forward([dict(path=torch.from_numpy(w[None]),sample_rate=16000) for w in wavs])
    with open(cache,'a',buffering=1) as f:
        for index,item in enumerate(c['diagnostic_sources']):
            key=item['cell']+'__'+item['id']
            if key in done:continue
            path=SRC/'cells'/item['cell']/'audio'/(item['id']+'.flac');assert sha(path)==item['sha256']
            y,sr=sf.read(path,dtype='float32');assert sr==16000 and len(y)==159744
            z=(y*10**(-6/20)).astype(np.float32)
            a=probe.normalize(y)[0];b=probe.normalize(z)[0]
            ca=normalize_candidate(y,probe.normalize)[0];cb=normalize_candidate(z,probe.normalize)[0]
            # Repeat the identical input; compare both near-identical gain controls
            # in isolation, together, and in a batch containing other waveforms.
            one_a=[forward([a])[0] for _ in range(3)];one_b=forward([b])[0]
            pair=forward([a,b]);mixed=forward([a,b]+[y,z]*7)
            candidate=[forward([ca])[0],forward([cb])[0]]
            candidate_pair=forward([ca,cb]);duplicate=forward([a,a])
            row=dict(key=key,source=item,settings=settings,
                     old_relative_error=float(np.linalg.norm(a.astype(float)-b)/np.linalg.norm(a)),
                     old_rms_delta_db=float(20*np.log10(np.linalg.norm(a.astype(float))/np.linalg.norm(b.astype(float)))),
                     candidate_relative_error=float(np.linalg.norm(ca.astype(float)-cb)/np.linalg.norm(ca)),
                     repeat_max=max(abs(one_a[0][axis]-v[axis]) for v in one_a[1:] for axis in axes),
                     singleton_control_max=max(abs(one_a[0][axis]-one_b[axis]) for axis in axes),
                     paired_control_max=max(abs(pair[0][axis]-pair[1][axis]) for axis in axes),
                     mixed_control_max=max(abs(mixed[0][axis]-mixed[1][axis]) for axis in axes),
                     batch_context_max=max(abs(one_a[0][axis]-v[axis]) for v in [pair[0],mixed[0]] for axis in axes),
                     duplicate_max=max(abs(duplicate[0][axis]-duplicate[1][axis]) for axis in axes),
                     candidate_control_max=max(abs(candidate[0][axis]-candidate[1][axis]) for axis in axes),
                     candidate_pair_max=max(abs(candidate_pair[0][axis]-candidate_pair[1][axis]) for axis in axes),
                     scores=dict(singleton_clean=one_a,singleton_gain=one_b,pair=pair,mixed=mixed[:2],candidate=candidate,candidate_pair=candidate_pair))
            f.write(json.dumps(row,allow_nan=False)+'\n');done[key]=row
            dump(root/'progress.json',dict(completed=len(done),total=len(c['diagnostic_sources']),elapsed_seconds=time.monotonic()-started))
            print('DIAGNOSTIC',len(done),len(c['diagnostic_sources']),key,flush=True)
    assert len(done)==len(c['diagnostic_sources'])
    result=dict(status='completed',diagnostic_only=True,production_promotion=False,n_sources=len(done),settings=settings,
                cache=str(cache),cache_sha256=sha(cache),contract_sha256=sha(os.environ['GPU_QUEUE_CONTRACT']),
                maxima={name:max(r[name] for r in done.values()) for name in ['repeat_max','singleton_control_max','paired_control_max','mixed_control_max','batch_context_max','duplicate_max','candidate_control_max','candidate_pair_max']})
    dump(root/'summary.json',result)
    notify('diagnostic_coverage_pass','start','099 diagnosis coverage/hash PASS; production experiments remain failed and blocked. Results distinguish loudness normalization, repeatability and batch context; no metric/gate changes. '+str(result['maxima']))

if __name__=='__main__':main()
