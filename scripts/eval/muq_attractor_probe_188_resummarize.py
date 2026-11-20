#!/usr/bin/env python3
"""188 rescue summarizer (post-hoc deviation D1, 2026-11-11).

p2 188 finished every scoring stage (A-F) and then crashed in the CPU-only summary step:
`part_f_summary` ran a least-squares fit on a feature matrix that contained NaN. The NaN comes from
`rms_cv`: `uniform_filter1d(w**2, 800)` is a running sum and returns tiny negative values on digitally
silent stretches, and `np.sqrt` of those is NaN (25 of the 2,000 Part F clips).

The action script is bound by the 188 contract and is not edited. This wrapper imports it, replaces only
the `rms_cv` line (clamp the mean-square at 0 before the square root; identical wherever the original was
finite), repairs the 25 NaN entries of F_feats.json (the original is kept as F_feats.nan_orig.json), and
then runs the script's own `--summary-only` path: summarize -> listening_pack -> validate.
Nothing is re-scored: the scorer is replaced by a function that raises.

Run:  cd ~/MeanAudio && OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      ~/venvs/dac/bin/python scripts/eval/muq_attractor_probe_188_resummarize.py
"""
import json
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import muq_attractor_probe_188 as M  # noqa: E402

H = M.H
_orig_feats = M.feats


def rms_cv_fixed(w):
    from scipy.ndimage import uniform_filter1d
    fr = uniform_filter1d(np.asarray(w, np.float32).astype(np.float64) ** 2, 800)[::400]
    rms = np.sqrt(np.maximum(fr, 0.0))
    return float(rms.std() / (rms.mean() + 1e-20))


def feats_fixed(w, partials=None):
    with np.errstate(invalid='ignore'):
        out, contour = _orig_feats(w, partials)
    out['rms_cv'] = rms_cv_fixed(w)
    return out, contour


def no_score(*a, **k):
    raise RuntimeError('rescue summarizer must not score anything: a per-clip cache is missing')


def repair_f_feats():
    fp = M.OUT / 'F_feats.json'
    bak = M.OUT / 'F_feats.nan_orig.json'
    fd = json.loads(fp.read_text())
    bad = [i for i, f in fd.items() if not all(math.isfinite(v) for v in f.values())]
    cols = sorted({k for i in bad for k, v in fd[i].items() if not math.isfinite(v)})
    M.log(f'rescue: F_feats has {len(bad)} clips with non-finite values in {cols}')
    if cols and cols != ['rms_cv']:
        raise SystemExit(f'unexpected non-finite features {cols}')
    # the fixed formula must reproduce the stored value wherever the stored value is finite
    good = [i for i in fd if i not in set(bad)]
    rng = np.random.default_rng(M.SEED)
    chk = [good[j] for j in rng.permutation(len(good))[:100]]
    dmax = max(abs(rms_cv_fixed(M.real30(i)) - fd[i]['rms_cv']) for i in chk)
    M.log(f'rescue: fixed rms_cv vs stored on {len(chk)} finite clips: max abs diff {dmax:.3e}')
    if dmax > 1e-9:
        raise SystemExit('fixed rms_cv does not reproduce stored finite values')
    if bad:
        if not bak.is_file():
            shutil.copy2(fp, bak)
        for i in bad:
            fd[i]['rms_cv'] = rms_cv_fixed(M.real30(i))
        M.jdump(fp, fd)
    M.jdump(M.OUT / 'rescue_D1.json', {'deviation': 'D1 rms_cv clamp before sqrt', 'n_repaired': len(bad), 'ids': bad,
                                      'repaired_rms_cv': {i: fd[i]['rms_cv'] for i in bad},
                                      'check_n': len(chk), 'check_max_abs_diff': dmax})
    return bad


def main():
    os.chdir(M.REPO)
    M.feats = feats_fixed            # forked feature workers inherit the patched module
    M.SC.score = no_score
    M.POOL = ThreadPoolExecutor(M.NTHREAD)
    H.load_scales()
    H.P = H.Paths(M.OUT / 'h183')
    att = json.loads((M.OUT / 'attractor.json').read_text())
    M.ATT.update(att)
    repair_f_feats()
    M.feat_many([M.RW(0)], att['main3'])
    M.summarize(att)
    M.listening_pack(att, M.load_selection())
    M.progress('done', '', 0, 0)
    return M.validate()


if __name__ == '__main__':
    sys.exit(main())
