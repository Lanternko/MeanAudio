"""PAM scorer (soham97/PAM @ ead6e7f; official implementation, MS-CLAP 2023 weights
microsoft/msclap CLAP_weights_2023.pth, MS-PL; text backbone openai-community/gpt2 @ 607a30d, MIT).

score_pam(wav_path_or_array, sr) -> {'pam': mean over 7-s chunks of P('clear and clean' | softmax over the two prompts),
                                     'pam_chunks': n_chunks}
Prompts (fixed upstream): 'the sound is clear and clean.' vs 'the sound is noisy and with artifacts.'
Input handling = upstream AudioDataset.readaudio/process_audio with repro=False (the default of run.py):
torchaudio resample to 44.1 kHz (torchaudio.load keeps channels; upstream reshape(-1) would INTERLEAVE a stereo file,
so we down-mix to mono by mean first -- the only deviation, a no-op for the mono benchmark files), then
  <7 s: tile-repeat to 7 s;  >=7 s: floor(len/7s) chunks plus one extra chunk of the LAST 7 s if there is a remainder.
Chunks are averaged. No loudness normalisation.
"""
import sys
import threading

import torch
from newmetrics_common import REPOS, setup_env, load_mono

_pam, _lock = {}, threading.Lock()
SAMPLES = 44100 * 7


def _load(device):
    with _lock:
        if device not in _pam:
            setup_env()
            sys.path.insert(0, str(REPOS / 'PAM'))
            from huggingface_hub import hf_hub_download
            from PAM import PAM
            w = hf_hub_download('microsoft/msclap', 'CLAP_weights_2023.pth',
                                revision='c47d441165daa21986ead0850660917636a81775')
            _pam[device] = PAM(model_fp=w, use_cuda=device.startswith('cuda'))
        return _pam[device]


def _chunks(a):
    import numpy as np
    if SAMPLES >= a.shape[0]:
        a = a.repeat(int(np.ceil(SAMPLES / a.shape[0])))[:SAMPLES]
    else:
        cut = a.shape[0] // SAMPLES
        init, rem = a[:cut * SAMPLES], a[cut * SAMPLES:]
        a = torch.cat([init, a[-SAMPLES:]]) if rem.shape[0] != 0 else init
    return a.reshape(-1, SAMPLES)


@torch.no_grad()
def score_pam(x, sr=None, device='cuda'):
    import torchaudio
    p = _load(device)
    wav, sr = load_mono(x, sr)
    t = torch.from_numpy(wav)
    if sr != 44100:
        t = torchaudio.transforms.Resample(sr, 44100)(t)
    ch = _chunks(t)
    avg, _ = p.evaluate(ch, [0, ch.shape[0]])
    return {'pam': float(avg[0]), 'pam_chunks': int(ch.shape[0])}
