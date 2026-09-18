# Verbatim from matchering/dsp.py 2.0.6
import numpy as np


def flip(array: np.ndarray) -> np.ndarray:
    return 1.0 - array


def rectify(array: np.ndarray, threshold: float) -> np.ndarray:
    rectified = np.abs(array).max(1)
    rectified[rectified <= threshold] = threshold
    rectified /= threshold
    return rectified


def max_mix(*args) -> np.ndarray:
    return np.maximum.reduce(args)
