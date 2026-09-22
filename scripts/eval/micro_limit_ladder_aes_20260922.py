#!/usr/bin/env python3
"""074: the last gap - downward limiting between the original level and -3 dB.

072 filled -18..-3 dB and found that the shallowest rung it could build already costs
-0.2365 PQ, because a limiter only removes peaks: dropping integrated L* by 3 dB
already demands 8.40 dB of gain reduction. So the published curve still does not reach
the original level, and the question 072 was preregistered to answer - how the
processing cost grows out of zero - was only half answered. Everything below 8.4 dB of
GR is unmeasured, and on the plotted curve the limiter branch stops 3 dB short of z0,
where the scalar and 064's upward branch both live.

This run closes that. Same machinery as 065/072, two lists changed:

  ladder_db  [0, 0.5, 1, 1.5, 2, 2.5, 3, 18]   scalar rungs, the matched controls
  limit_db   [0.5, 1, 1.5, 2, 2.5, 3]          downward-limited arms at the same L*

The rungs are chosen on the axis the curve is plotted on (L* relative to source), so
they land exactly in the visible gap, and they sample GR below 8.4 dB as a consequence
rather than by targeting GR directly - targeting GR would break the L*-matched twin
that makes the level/processing decomposition exact.

Two rungs are deliberate duplicates and act as per-clip replication gates:
  m18  063's rung, gated by 065's existing z0/m18 gates.
  D3   072's shallowest rung. Same source, same bisection, same render, so its audio
       must be bit-identical; its scores are gated at SCORE_TOLERANCE because AES
       batches one condition across the clips still to do, so the grouping - and
       through padding the scores - drift by ~1e-3 across a resumed run
       (see reference_aes_batch_composition_sensitivity).

Reachability: at pre-gain 0 the limiter sits at a -1 dBTP ceiling and these clips peak
below it, so the L* drop goes to 0 with G and every target in (0, 3] is bracketed by
[0, hi]. A clip whose source already peaks above -1 dBTP cannot reach the smallest
targets; those are reported as misses rather than silently clamped, and the hit-rate
gate is evaluated per rung.

Not applicable in this range, and registered as not-to-be-read:
  summary['floors']       every floor is at -42 dB and below; the ladder stops at -3.
  summary['shape_float']  the ladder is truncated, so its argmax sits on an edge rung
                          by construction; peaks were settled by 063/065.
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

OUT = '/home/kojiek/nvme_experiment_artifacts/meanaudio/micro_limit_ladder_aes_20260922'
SHALLOW072 = '/home/kojiek/nvme_experiment_artifacts/meanaudio/shallow_limit_ladder_aes_20260922'

LADDER_DB = [0, 0.5, 1, 1.5, 2, 2.5, 3, 18]
LIMIT_DB = [0.5, 1, 1.5, 2, 2.5, 3]
# the rung this run shares with 072
CROSS_CHECK_RUNG_DB = 3
AUDIO_TOLERANCE = 0
SCORE_TOLERANCE = {'aes': 5e-3, 'clap': 1e-3}


def configure():
    fl.CFG.update({
        'experiment': '074 micro downward-limiting ladder AES',
        'out': os.environ.get('MICRO074_OUT', OUT),
        'rows': int(os.environ.get('MICRO074_ROWS', 5521)),
        'ladder_db': list(LADDER_DB),
        'limit_db': list(LIMIT_DB),
        'replication_tolerance': SCORE_TOLERANCE['aes'],
    })
    return fl.CFG


def cross_check_072(summary):
    """D3 and m3, per clip, in two layers: audio bit-exact, scores within drift."""
    here = Path(fl.CFG['out']) / 'items'
    there = Path(SHALLOW072) / 'items'
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
    return {'vs': '072 shallow_limit_ladder_aes_20260922', 'rung_db': g, 'conditions': keys,
            'n': len(ids), 'audio_mismatch_n': len(audio_mismatch),
            'audio_mismatch_examples': audio_mismatch[:5],
            'max_abs_diff': worst, 'worst_at': worst_at,
            'tolerance': {'audio_mismatch_n': AUDIO_TOLERANCE, **SCORE_TOLERANCE},
            'pass': ok}


def per_rung_hit(rec):
    """Hit rate per D arm: the smallest targets are the ones at risk, so report them apart."""
    d = Path(fl.CFG['out']) / 'plan'
    plans = [json.loads((d / (r.id + '.json')).read_text()) for r in rec]
    out = {}
    for g in LIMIT_DB:
        arms = [p['arms'][f'D{g}'] for p in plans]
        hit = [a['hit'] for a in arms]
        out[f'D{g}'] = {
            'hit_rate': float(sum(hit)) / len(hit),
            'gr_max_db_mean': float(sum(a['gr_max_db'] for a in arms)) / len(arms),
            'pre_gain_db_mean': float(sum(a['pre_gain_db'] for a in arms)) / len(arms),
            'lstar_err_max_lu': max(abs(a['lstar'] - a['target_lstar']) for a in arms),
        }
    return out


def validate_only():
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
    x = cross_check_072(s)
    if not x['pass']:
        problems.append(f'072 cross-check failed: {x["audio_mismatch_n"]} audio mismatches '
                        f'{x["audio_mismatch_examples"]}, scores {x["max_abs_diff"]} at {x["worst_at"]}')
    if 'cross_check_072' not in s:
        s['cross_check_072'] = x
        fl.atomic(path, s)
    if problems:
        for p in problems:
            print('INVALID: ' + p, flush=True)
        return 2
    print(f'PASS: {s["n"]} clips x {len(fl.conditions())} conditions, all gates pass, '
          f'072 D{CROSS_CHECK_RUNG_DB} cross-check: audio bit-identical, '
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
        if '--plan-only' in sys.argv:
            print(json.dumps(per_rung_hit(rec), indent=1))
            return 0
        fl.score_all(rec, plans)
    r = fl.analyze(rec)
    r['per_rung_hit'] = per_rung_hit(rec)
    r['cross_check_072'] = cross_check_072(r)
    fl.atomic(Path(fl.CFG['out']) / 'summary.json', r)
    print(json.dumps({
        'gates': r['gates'],
        'all_gates_pass': r['all_gates_pass'],
        'per_rung_hit': r['per_rung_hit'],
        'cross_check_072': r['cross_check_072'],
        'processing_at_source_level_F': {
            f'D{g}': {q: r['limiting_vs_scalar'][f'D{g}']['metrics'][q]['processing_at_source_level']['mean']
                      for q in (*AXES, 'clap')} for g in LIMIT_DB},
    }, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
