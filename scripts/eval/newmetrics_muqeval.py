"""MuQ-Eval scorer (A1 frozen MuQ+attention-pool+MLP; A3a LoRA+ordinal CE).

score_muqeval(wav_path_or_array, sr, variant='A1') -> {'MI','TA'} (1-5 scale heads, MusicEval-trained).
Code: github.com/dgtql/MuQ-Eval @ 60a88f8 (MIT). Weights: HF zhudi2825/MuQ-Eval-A1 @ b082e23 (A1),
zhudi2825/MuQ-Eval @ cc0cb22 (A3a), both MIT; encoder OpenMuQ/MuQ-large-msd-iter @ 0562a57 (CC-BY-NC-4.0).
Input handling = upstream AudioProcessor(24000, 240000).process(mode='center'): mono mean, torchaudio
resample to 24 kHz, centre-crop to 10 s, zero-pad at the end if shorter. No loudness normalisation.
(README's `AudioProcessor.load_and_process` does not exist in the repo; 'center' chosen over upstream's
default 'random' crop for determinism.) TA head sees audio only: NOT a text-alignment score.
"""
import sys
import threading

import torch
from newmetrics_common import REPOS, HF_HOME, setup_env, load_mono

REV = {'A1': ('zhudi2825/MuQ-Eval-A1', 'b082e238827df51d99ac66bdd1b6f0d2406bc90e'),
       'A3a': ('zhudi2825/MuQ-Eval', 'cc0cb22ea4026f7d3fb1bae384927b870a24319f')}
_cache, _lock = {}, threading.Lock()


def _load(variant, device):
    key = (variant, device)
    with _lock:
        if key in _cache:
            return _cache[key]
        setup_env()
        repo = str(REPOS / 'MuQ-Eval')
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from omegaconf import OmegaConf
        from huggingface_hub import hf_hub_download
        from src.model import MusicQualityModel
        from src.data import AudioProcessor
        rid, rev = REV[variant]
        # HF config.yaml is a Hydra-style child ("defaults: [base]"): merge with the repo's base.yaml.
        base = OmegaConf.load(hf_hub_download(rid, 'base.yaml', revision=rev))
        cfg = OmegaConf.merge(base, OmegaConf.load(hf_hub_download(rid, 'config.yaml', revision=rev)))
        m = MusicQualityModel(cfg)
        m.load_state_dict(torch.load(hf_hub_download(rid, 'model_state_dict.pt', revision=rev),
                                     map_location='cpu', weights_only=False))
        m.to(device).eval()
        _cache[key] = (m, AudioProcessor(24000, 240000))
        return _cache[key]


@torch.no_grad()
def score_muqeval(x, sr=None, variant='A1', device='cuda'):
    m, proc = _load(variant, device)
    wav, sr = load_mono(x, sr)
    w = proc.process(wav, sr, mode='center').unsqueeze(0).to(device)
    out = m(w)
    # A3a heads are ordinal (5 logits): score = softmax expectation over bin centres 1..5, as upstream
    # OrdinalPredictionHead does; A1 heads are scalar.
    res = {}
    for k in ('MI', 'TA'):
        v = out[k]
        if v.numel() > 1:
            v = (torch.softmax(v.float(), -1) * torch.linspace(1, 5, v.shape[-1], device=v.device)).sum(-1)
        res[k] = float(v.item())
    return res
