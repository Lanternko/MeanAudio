"""CMI-RM scorer (Haiwen-Xia/CMI-RewardBench @ 9235ef5, code Apache-2.0; weights HF HaiwenXia/CMI-RM
@ 704f2dc, CC-BY-NC-4.0 -> non-commercial use only).

score_cmirm(wav_path_or_array, sr, prompt_mode='empty'|'caption', caption='') ->
    {'quality', 'alignment'}   (official `RewardModelInference.score`, mode='final'). Default bf16=False (fp32): upstream's bf16 autocast
    quantises the outputs (e.g. 1.75 / 0.396484375) and creates artificial ties; bf16=True reproduces upstream default.
Fixed protocol (decided before any benchmark run, never changed afterwards):
  * prompt_mode='empty'   : text='' , lyrics='', no reference audio.
  * prompt_mode='caption' : text=<original benchmark caption/prompt>, lyrics='', no reference audio.
    Benchmarks without a text prompt (MusicCaps real audio has one; PAM/MusicEval have per-clip prompts) pass it in.
  * Audio: mono mean, torch Resample to 24 kHz, truncated to max_dur=30 s, then 30 s chunk encoding.
    NO loudness normalisation, NO padding to 10 s (upstream does none).
'alignment' is conditioned on text; under 'empty' it is NOT a text-alignment measurement.
"""
import threading

import numpy as np
import torch
from newmetrics_common import REPOS, HF_HOME, setup_env, load_mono

REV = '704f2dc3a11a1d867cc9a683a49c17d9cc7d50e4'
_m, _lock = {}, threading.Lock()


def _load(device, bf16):
    with _lock:
        if (device, bf16) not in _m:
            # upstream code calls HfApi.model_info('xlm-roberta-base') -> needs hub reachability (weights are cached)
            setup_env(offline=False)
            import sys
            sys.path.insert(0, str(REPOS / 'CMI-RewardBench'))
            from huggingface_hub import hf_hub_download
            from baselines.inference import RewardModelInference
            ck = hf_hub_download('HaiwenXia/CMI-RM', 'model.safetensors', revision=REV)
            cf = hf_hub_download('HaiwenXia/CMI-RM', 'config.yaml', revision=REV)
            _m[(device, bf16)] = RewardModelInference(ck, config=cf, device=device, sr=24000, mode='final', bf16=bf16)
        return _m[(device, bf16)]


def score_cmirm(x, sr=None, prompt_mode='empty', caption='', device='cuda', bf16=False):
    m = _load(device, bf16)
    wav, sr = load_mono(x, sr)
    import torchaudio
    t = torch.from_numpy(wav)
    if sr != 24000:
        t = torchaudio.transforms.Resample(sr, 24000)(t)
    text = caption if prompt_mode == 'caption' else ''
    r = m.score(t, text=text, lyrics='', max_dur=30.0)
    return {'quality': r['quality'], 'alignment': r['alignment']}
