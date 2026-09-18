"""Established open-source limiters behind one interface, for the 064 robustness arms.

Each takes float64 mono at 16 kHz already carrying the caller's pre-gain, and returns
float64 of the same length with its ceiling at `ceiling_db`.

  dpl        x42 dpl.lv2 Peaklim (Fons Adriaensen's DPL), true-peak mode, vendored
             in third_party/x42_dpl and built as an offline CLI
  alimiter   ffmpeg alimiter, sample-peak; level=0 (else it renormalises the output
             back to 0 dBFS) and latency=1 (else output is shifted by the attack time)
  hyrax      Matchering 2.0.6 Hyrax brickwall limiter, sample-peak, run at 16 kHz
  loudnorm   ffmpeg loudnorm (EBU R128) to integrated I with true-peak TP. Not a pure
             limiter: its dynamic mode is a loudness AGC followed by a true-peak limiter,
             i.e. the common real-world "normalise to -14 LUFS" pipeline
"""
from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np

SR = 16000
HERE = Path(__file__).resolve().parent
DPL = HERE / 'third_party/x42_dpl/dpl_cli'
HYRAX_ROOT = HERE / 'third_party/matchering_hyrax'
RELEASE_S = 0.05


def _pipe(cmd, x):
    r = subprocess.run(cmd, input=np.asarray(x, dtype='<f4').tobytes(), capture_output=True, check=True)
    y = np.frombuffer(r.stdout, dtype='<f4').astype(np.float64)
    return y


def dpl(x, ceiling_db=-1.0):
    y = _pipe([str(DPL), str(SR), str(ceiling_db), str(RELEASE_S), '1'], x)
    if len(y) != len(x):
        raise ValueError('dpl length mismatch')
    return y


def _ffmpeg(x, af, out_sr=SR):
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '1',
           '-i', 'pipe:0', '-af', af, '-ar', str(out_sr), '-f', 'f32le', '-ac', '1', 'pipe:1']
    return _pipe(cmd, x)


def alimiter(x, ceiling_db=-1.0):
    lim = 10 ** (ceiling_db / 20)
    y = _ffmpeg(x, f'alimiter=limit={lim:.6f}:attack=5:release={RELEASE_S * 1000:g}:level=0:latency=1')
    if len(y) != len(x):
        y = y[:len(x)] if len(y) > len(x) else np.concatenate([y, np.zeros(len(x) - len(y))])
    return y


def loudnorm(x, target_lufs=-14.0, tp_db=-1.0):
    # LRA is set wide so the AGC is not also asked to compress the loudness range.
    y = _ffmpeg(x, f'loudnorm=I={target_lufs:g}:TP={tp_db:g}:LRA=50,aresample={SR}')
    return y[:len(x)] if len(y) >= len(x) else np.concatenate([y, np.zeros(len(x) - len(y))])


_hyrax = None


def hyrax(x, ceiling_db=-1.0):
    global _hyrax
    if _hyrax is None:
        import sys
        sys.path.insert(0, str(HYRAX_ROOT))
        from matchering import Config
        from matchering.limiter.hyrax import limit
        _hyrax = (Config, limit)
    Config, limit = _hyrax
    cfg = Config(internal_sample_rate=SR, threshold=10 ** (ceiling_db / 20))
    y = limit(np.stack([x, x], 1), cfg)
    return np.asarray(y[:, 0], dtype=np.float64)


LIMITERS = {'dpl': dpl, 'alimiter': alimiter, 'hyrax': hyrax}
