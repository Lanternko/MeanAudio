"""Draft normalization revision; requires a new scientific contract before use.

Start from a fixed RMS so BS.1770 absolute gating cannot choose a different
normalization branch merely because the input has been multiplied by a gain.
The original samples, -23 LUFS target, and 0.005 LU tolerance are retained.
"""
import numpy as np


def normalize_candidate(y, original_normalize):
    y = np.asarray(y, dtype=np.float32)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if not np.isfinite(rms) or rms < 1e-12:
        raise ValueError("zero-energy or nonfinite derivative")
    initial_gain = 10 ** ((-23 - 20 * np.log10(rms)) / 20)
    seeded = (y.astype(np.float64) * initial_gain).astype(np.float32)
    result, gain, lu = original_normalize(seeded)
    return result, gain * initial_gain, lu
