"""Admit an explicit scientific normalization revision, never an old gate bypass."""
import os,sys,json,runpy
from pathlib import Path
import numpy as np
import soundfile as sf
def main():
    c=json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())
    sys.path[:0]=[str(Path(__file__).parent),str(Path(c['original_preflight']).parent)]
    import probe101
    root=Path(c['corrected_science_root']);science=json.loads((root/'contract.json').read_text())
    assert science['normalization_revision']=='fixed_rms_seed_then_bs1770_minus23_v2'
    assert c['normalization_revision']==science['normalization_revision']
    import torch
    assert not torch.cuda.is_initialized()
    d=Path('/home/kojiek/MeanAudio/runtime/aes_queue_repair_20261003/diagnostic100/summary.json');r=json.loads(d.read_text())
    assert r['status']=='completed' and r['settings']['cudnn_tf32']==False and r['settings']['matmul_tf32']==False
    assert r['maxima']['batch_context_max']<.01 and r['maxima']['candidate_control_max']<.01
    for cell,identifier in [('N100_s27182818_mf25_cfg0_historical_null','QUB_vpjogmo_170'),('control_s14159265_mf25_cfg3_fidelity8','1cqcTbDxsHM_130')]:
        y,sr=sf.read(root/'cells'/cell/'audio'/(identifier+'.flac'),dtype='float32');a=probe101.normalize(y)[0]
        for db in [-40,-6,6,40]:
            b=probe101.normalize((y*10**(db/20)).astype(np.float32))[0]
            assert np.linalg.norm(a.astype(float)-b)/np.linalg.norm(a)<1e-5
    assert os.environ.get('PYTHONPATH')==c['runtime_environment']['PYTHONPATH']
    from demucs_infer.pretrained import get_model
    model=get_model('htdemucs_6s',repo=Path(science['demucs_model_repo'])).eval();assert next(model.parameters()).device.type=='cpu';del model
    assert not torch.cuda.is_initialized()
    runpy.run_path(c['original_preflight'],run_name='__main__')
if __name__=='__main__':main()
