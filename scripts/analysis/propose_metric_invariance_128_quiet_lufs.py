#!/usr/bin/env python3
"""Unapproved CPU-only LUFS fallback proposal; never used by the evaluator."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pyloudnorm
import soundfile as sf


def proposed_normalize(x, sr):
    """Keep finite-LUFS behavior; probe below-gate nonzero audio at unit peak."""
    if not x.size or not np.isfinite(x).all():
        raise ValueError('empty or nonfinite audio')
    peak = float(np.max(np.abs(x)))
    if peak == 0:
        raise ValueError('true silence remains invalid')
    meter = pyloudnorm.Meter(sr)
    level = float(meter.integrated_loudness(x))
    if level == -math.inf:
        x = x / peak
        level = float(meter.integrated_loudness(x))
    if not math.isfinite(level):
        raise ValueError('LUFS remains undefined')
    y = x * 10 ** ((-30 - level) / 20)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite normalized audio')
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    sr = 16000
    tone = np.sin(2*np.pi*440*np.arange(sr)/sr)*.1
    y = proposed_normalize(tone, sr)
    level = pyloudnorm.Meter(sr).integrated_loudness(tone)
    np.testing.assert_array_equal(y, tone*10**((-30-level)/20))
    for gain in (1, .01, .0001, 100):
        np.testing.assert_allclose(proposed_normalize(tone*gain, sr), y, atol=1e-12)
    for bad in (np.zeros(sr), np.full(sr, np.nan), np.array([])):
        try:
            proposed_normalize(bad, sr)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid audio accepted')
    path = Path('/home/kojiek/eval_output_nvme/ttm_metrics_115_20261009/musicgen_medium/lvl30/audio/RDCXvJPzT5k_520.flac')
    x, sr = sf.read(path)
    q = proposed_normalize(x, sr)
    assert abs(pyloudnorm.Meter(sr).integrated_loudness(q)+30) < 1e-8
    assert np.max(np.abs(q)) > 1  # No limiter or clipping.
    report = {'status':'cpu_proposal_checks_passed', 'approved':False,
              'production_applied':False, 'gpu_scored':False,
              'checks':['finite-LUFS output unchanged','gain invariance for tone across gate',
                        'zero/NaN/empty rejected','observed quiet clip reaches -30 LUFS',
                        'observed peak >1 preserved'],
              'observed_output_peak':float(np.max(np.abs(q))),
              'risk':'Changes the absolute-gate behavior for nonzero audio below the gate. Amplifies near-silent noise and may change invariance/degradation results. Requires explicit protocol amendment and new contracts.'}
    if args.output:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
