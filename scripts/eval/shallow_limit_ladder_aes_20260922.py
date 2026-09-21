#!/usr/bin/env python3
"""072: the shallow half of the downward-limiting ladder, -18 dB to the original level.

Two experiments have measured limiting so far and they leave a hole between them:

  064  pushes level UP from the original (0 -> +6.8 LU) with x42-dpl, and decomposes
       each arm into a loudness part and a processing part against a loudness-matched
       twin. Nothing below 0.
  065  pushes level DOWN, but its D arms start at 18 dB of attenuation and go to 84,
       because its question was where the scorer, format and meter floors sit.

So for -18..0 dB - the range every real output actually lives in - there is no limiter
measurement at all, and the published D-m curve cannot be drawn continuously through
the original level. 065 also reported its D-m contrast for PQ and CLAP only, although
its analysis computes it for all four AES axes.

This run fills that hole. It is 065's machinery with two lists changed:

  ladder_db  [0, 3, 6, 9, 12, 15, 18]   scalar rungs (m3..m18), the matched controls
  limit_db   [3, 6, 9, 12, 15, 18]      downward-limited arms D3..D18 at the same L*

Everything else - the limiter (x42-dpl, true peak, -1 dBTP), the pre-gain bisection,
the F/Q scoring modes, the D<g>m processing twins, the 063 replication gates, the
bootstrap - is imported unchanged, so the new rungs are directly comparable to the
published 065 table rather than being a parallel measurement of the same thing.

D18 is measured again here on purpose: 065 already has it, so every clip's D18 is a
per-clip replication gate on this run (--validate-only). That gate is two-layered,
because 065's scorer is batch-composition sensitive:

  audio   content_sha256 must be bit-identical. The plan (pre-gain bisection) and the
          render are deterministic given the source, so this catches any real defect
          in the arms at zero tolerance.
  score   AES batches one condition across the clips still to do, so the grouping -
          and, through padding, the scores - depend on how many clips remain. A fresh
          full run reproduces 065's grouping exactly and lands at 0.0; a run resumed
          after preemption shifts the batch boundaries and drifts by about 1e-3 AES
          points. Measured on a 6-clip smoke run against 065: max 9.3e-4 on PQ with
          every content hash identical. So scores are gated at SCORE_TOLERANCE, which
          is two orders of magnitude below any effect this experiment reads
          (D18 - m18 is -0.571 PQ) and still catches a wrong arm or a wrong limiter.

Not applicable in this range, and registered as not-to-be-read:
  summary['floors']          the scorer and format floors are at -42 dB and below.
  summary['shape_float']     the ladder is truncated at -18, so its argmax sits on the
                             edge rung by construction; peak location was settled by
                             063 (-21 dB for PQ/CU, -6 for CE/PC) and is not re-asked.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import floor_ladder_aes_20260918 as fl  # noqa: E402
from gain_ladder_aes_20260918 import AXES  # noqa: E402

OUT = '/home/kojiek/nvme_experiment_artifacts/meanaudio/shallow_limit_ladder_aes_20260922'
FLOOR065 = '/home/kojiek/nvme_experiment_artifacts/meanaudio/floor_ladder_aes_20260918'

LADDER_DB = [0, 3, 6, 9, 12, 15, 18]
LIMIT_DB = [3, 6, 9, 12, 15, 18]
# the rung both runs share; its per-clip audio must reproduce 065 bit for bit and its
# scores to within the scorer's batch-composition drift (see the module docstring)
CROSS_CHECK_RUNG_DB = 18
AUDIO_TOLERANCE = 0
SCORE_TOLERANCE = {'aes': 5e-3, 'clap': 1e-3}


def configure():
    """Point 065's machinery at the shallow rungs and a fresh output root."""
    fl.CFG.update({
        'experiment': '072 shallow downward-limiting ladder AES',
        'out': os.environ.get('SHALLOW072_OUT', OUT),
        'rows': int(os.environ.get('SHALLOW072_ROWS', 5521)),
        'ladder_db': list(LADDER_DB),
        'limit_db': list(LIMIT_DB),
        # 065 gates its 063 replication at 1e-4, which is exact for a fresh full run
        # but not survivable across a preemption (see the module docstring). Same
        # reasoning and same headroom as SCORE_TOLERANCE.
        'replication_tolerance': SCORE_TOLERANCE['aes'],
    })
    return fl.CFG


def cross_check_065(summary):
    """The rung this run shares with 065, checked per clip in two layers.

    Audio is gated bit for bit: same source, same bisection, same render, so a single
    differing content hash means a defective arm. Scores are gated at SCORE_TOLERANCE
    because the AES batch grouping depends on how many clips remained to do.
    """
    here = Path(fl.CFG['out']) / 'items'
    there = Path(FLOOR065) / 'items'
    g = CROSS_CHECK_RUNG_DB
    keys = [fl.key(f'D{g}', m) for m in fl.MODES]
    keys += [fl.key(fl.rung(g), m) for m in fl.MODES]
    keys.append(fl.key(f'D{g}m', 'F'))
    metrics = (*AXES, 'clap')
    worst = {'aes': 0.0, 'clap': 0.0}
    worst_at = {'aes': None, 'clap': None}
    audio_mismatch = []
    ids = sorted(p.stem for p in here.glob('*.json'))
    for i in ids:
        a = json.loads((here / (i + '.json')).read_text())['conditions']
        b = json.loads((there / (i + '.json')).read_text())['conditions']
        for k in keys:
            if a[k]['content_sha256'] != b[k]['content_sha256']:
                audio_mismatch.append(f'{i}/{k}')
            for m in metrics:
                fam = 'clap' if m == 'clap' else 'aes'
                d = abs((a[k]['clap'] if m == 'clap' else a[k]['aes'][m])
                        - (b[k]['clap'] if m == 'clap' else b[k]['aes'][m]))
                if d > worst[fam]:
                    worst[fam], worst_at[fam] = d, f'{i}/{k}/{m}'
    ok = (len(audio_mismatch) <= AUDIO_TOLERANCE
          and all(worst[f] <= SCORE_TOLERANCE[f] for f in worst))
    return {'vs': '065 floor_ladder_aes_20260918', 'rung_db': g, 'conditions': keys,
            'n': len(ids), 'audio_mismatch_n': len(audio_mismatch),
            'audio_mismatch_examples': audio_mismatch[:5],
            'max_abs_diff': worst, 'worst_at': worst_at,
            'tolerance': {'audio_mismatch_n': AUDIO_TOLERANCE, **SCORE_TOLERANCE},
            'pass': ok}


def validate_only():
    """Postflight: the summary exists, describes this ladder, and reproduces 065."""
    configure()
    path = Path(fl.CFG['out']) / 'summary.json'
    problems = []
    if not path.is_file():
        print('INVALID: summary.json missing', flush=True)
        return 2
    s = json.loads(path.read_text())
    if s['config']['ladder_db'] != LADDER_DB or s['config']['limit_db'] != LIMIT_DB:
        problems.append('summary was produced with different rungs')
    if s['n'] != fl.CFG['rows']:
        problems.append(f'summary covers {s["n"]} clips, expected {fl.CFG["rows"]}')
    if not s.get('all_gates_pass'):
        failed = [k for k, v in s['gates'].items() if not v.get('pass')]
        problems.append('063 replication or construction gates failed: ' + ','.join(failed))
    for g in LIMIT_DB:
        if f'D{g}' not in s['limiting_vs_scalar']:
            problems.append(f'D{g} missing from limiting_vs_scalar')
    x = cross_check_065(s)
    if not x['pass']:
        problems.append(f'065 cross-check failed: {x["audio_mismatch_n"]} audio mismatches '
                        f'{x["audio_mismatch_examples"]}, scores {x["max_abs_diff"]} at {x["worst_at"]}')
    if 'cross_check_065' not in s:
        s['cross_check_065'] = x
        fl.atomic(path, s)
    if problems:
        for p in problems:
            print('INVALID: ' + p, flush=True)
        return 2
    print(f'PASS: {s["n"]} clips x {len(fl.conditions())} conditions, all gates pass, '
          f'065 D{CROSS_CHECK_RUNG_DB} cross-check: audio bit-identical, '
          f'scores max |diff| aes {x["max_abs_diff"]["aes"]:.2e} clap '
          f'{x["max_abs_diff"]["clap"]:.2e}', flush=True)
    return 0


def main():
    if '--validate-only' in sys.argv:
        return validate_only()
    configure()
    rec = fl.records()
    if '--analyze-only' not in sys.argv:
        plans = fl.plan_all(rec)
        fl.score_all(rec, plans)
    r = fl.analyze(rec)
    r['cross_check_065'] = cross_check_065(r)
    fl.atomic(Path(fl.CFG['out']) / 'summary.json', r)
    print(json.dumps({
        'gates': r['gates'],
        'all_gates_pass': r['all_gates_pass'],
        'cross_check_065': r['cross_check_065'],
        'limiter_minus_scalar_F': {
            f'D{g}': {q: r['limiting_vs_scalar'][f'D{g}']['metrics'][q]['limiter_minus_scalar_F']['mean']
                      for q in (*AXES, 'clap')} for g in LIMIT_DB},
    }, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
