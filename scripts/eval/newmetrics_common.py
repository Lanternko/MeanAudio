"""Shared plumbing for the newmetrics_<model>.py scorers (newmodels_20261020).

Input contract (identical for every model): a path or a (samples,) / (samples, ch) / (ch, samples)
float array plus its sampling rate. Down-mix to mono by channel mean. NO loudness normalisation,
NO peak normalisation, NO clipping guard: each model applies only its own upstream resampling,
cropping and padding, documented in that model's module docstring.
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path('/mnt/seagate/newmodels_20261020')
REPOS = ROOT / 'repos'
HF_HOME = ROOT / 'hf'


def setup_env(offline=True):
    os.environ['HF_HOME'] = str(HF_HOME)
    if offline and not os.environ.get('NEWMODELS_ONLINE'):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'


def load_mono(x, sr=None):
    """-> (float32 mono 1-D array, sr). Path input is read with soundfile at native rate."""
    if isinstance(x, (str, Path)):
        import soundfile as sf
        a, sr = sf.read(str(x), dtype='float32', always_2d=True)
    else:
        if sr is None:
            raise ValueError('sr is required for array input')
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 1:
            a = a[:, None]
        elif a.shape[0] < a.shape[1] and a.shape[0] <= 8:  # (ch, samples)
            a = a.T
    return a.mean(axis=1).astype(np.float32), int(sr)
