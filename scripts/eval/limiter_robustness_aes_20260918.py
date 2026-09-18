#!/usr/bin/env python3
"""064b: does 064's limiter-processing effect survive a change of limiter?

064 uses x42-dpl (Adriaensen DPL, true-peak). Its level/processing split is only
publishable if other limiters give the same sign: the in-house lookahead limiter that
064 started with, ffmpeg alimiter and Matchering's Hyrax (both sample-peak). Same design, same 051 audio:
for each external limiter k, a fixed +6 dB arm (L6_k) and a -14 LUFS target arm
(T14_k), each with a twin scaled back to the source LUFS; plus ffmpeg loudnorm to
-14 LUFS (AGC + true-peak limiter, the everyday normalisation pipeline) and its twin.

064's dpl L6/T14 records are read from the 064 run and compared per clip; z0 is
re-scored here and must equal both 063 and 064.
"""
from __future__ import annotations
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/eval'))
import external_limiters as E  # noqa: E402
import limiter_loudness_aes_20260918 as M  # noqa: E402
from gain_ladder_aes_20260918 import AXES, SR, atomic, content_digest_array, digest, lufs, mean_ci, meter, read_audio, signal_stats  # noqa: E402

CFG = dict(M.CFG)
CFG.update({
    'experiment': '064b limiter robustness',
    'out': '/home/kojiek/nvme_experiment_artifacts/meanaudio/limiter_robustness_aes_20260918',
    'main_064_items': M.CFG['out'] + '/items',
    'external': ['own', 'alimiter', 'hyrax'],
    'ceiling_db': -1.0,
    'fixed_gain_db': 6.0,
    'target_lufs': -14.0,
    'loudnorm': {'I': -14.0, 'TP': -1.0, 'LRA': 50},
})
# sample-peak limiters are allowed their inter-sample overs; that is how they ship
# External limiters are scored as they ship: the only hard bound is no clipping, and
# their actual sample/true peaks are recorded per clip. loudnorm limits at 192 kHz and its
# resample back to 16 kHz lands peaks anywhere from -0.96 to -0.79 dBFS (TP up to -0.47).
# The in-house limiter keeps its construction bounds.
PEAK_LIMIT = {'own': -0.99, 'alimiter': -0.01, 'hyrax': -0.01, 'loudnorm': -0.01}
TP_LIMIT = {'own': -0.5, 'alimiter': 6.0, 'hyrax': 6.0, 'loudnorm': 6.0}


def own(y, ceiling_db):
    return M.limit(y, **{**M.CFG['own_limiter'], 'ceiling_dbfs': ceiling_db})[0]


LIMITERS = {**E.LIMITERS, 'own': own}


def arm_list():
    names = ['z0']
    for k in CFG['external']:
        names += [f'L6_{k}', f'L6_{k}m', f'T14_{k}', f'T14_{k}m']
    return names + ['LN14', 'LN14m']


def twin(met, y, src_lufs):
    back = 0.0
    for _ in range(6):
        back += src_lufs - lufs(met, y * 10 ** (back / 20))
        z = y * 10 ** (back / 20)
        if abs(lufs(met, z) - src_lufs) <= CFG['matched_tolerance_lu']:
            return z, back
    raise ValueError('matched twin missed source LUFS')


def to_target(f, x, met, src_lufs):
    y, pre, hit = M.hit_target(lambda v: f(v, CFG['ceiling_db']), x, met, src_lufs, CFG['target_lufs'])
    return y, {'pre_gain_db': float(pre), 'target_hit': hit}


def render_clip(x, met, src_lufs):
    out = {'z0': (x, {})}
    for k in CFG['external']:
        f = LIMITERS[k]
        g = CFG['fixed_gain_db']
        out[f'L6_{k}'] = (f(x * 10 ** (g / 20), CFG['ceiling_db']), {'pre_gain_db': g, 'limiter': k})
        y, info = to_target(f, x, met, src_lufs)
        out[f'T14_{k}'] = (y, {**info, 'limiter': k})
    ln = CFG['loudnorm']
    y = E.loudnorm(x, ln['I'], ln['TP'])
    out['LN14'] = (y, {'limiter': 'loudnorm'})
    for name in [n for n in list(out) if n != 'z0']:
        z, back = twin(met, out[name][0], src_lufs)
        out[name + 'm'] = (z, {'twin_of': name, 'scalar_db': float(back)})
    return out


def score_all():
    import soundfile as sf
    from score_musiccaps_per_item import load_aes_predictor, _aes_batch, load_clap_model, _clap_batch
    out = Path(CFG['out'])
    hashes = json.loads(Path(CFG['source_manifest']).read_text())['audio_sha256']
    rec = M.records()
    arms = arm_list()
    items = out / 'items'
    todo = [r for r in rec if not (items / (r.id + '.json')).exists()]
    print(f'{len(todo)} clips x {len(arms)} arms to score', flush=True)
    if not todo:
        return
    scratch = out / '_render'
    scratch.mkdir(parents=True, exist_ok=True)
    items.mkdir(parents=True, exist_ok=True)
    predictor = load_aes_predictor(Path(CFG['aes_snapshot']), device='cuda', batch_size=CFG['batch'])
    clap = load_clap_model(Path(CFG['clap_checkpoint']), device='cuda', local_files_only=True)
    done = 0
    for start in range(0, len(todo), CFG['batch']):
        batch = todo[start:start + CFG['batch']]
        staged = []
        for r in batch:
            src = Path(CFG['source_audio_root']) / (r.id + '.flac')
            if digest(src) != hashes[r.id]:
                raise ValueError('source audio drift: ' + r.id)
            x = read_audio(src)
            met = meter()
            base = signal_stats(x, met)
            per = {}
            for arm, (y, info) in render_clip(x, met, base['lufs']).items():
                if len(y) != len(x) or not np.isfinite(y).all():
                    raise ValueError(f'{r.id}/{arm}: bad render')
                st = signal_stats(y, met)
                st['true_peak_dbfs'] = M.true_peak_dbfs(y)
                lim = info.get('limiter')
                if lim == 'loudnorm' and st['peak_dbfs'] >= 0:
                    # loudnorm limits at 192 kHz; resampling to 16 kHz can overshoot full
                    # scale (FEuXIeWoCQQ_30: +0.21 dBFS, 4 samples). That is loudnorm as it
                    # ships at this rate: flag it and drop the clip from the loudnorm cell.
                    info = {**info, 'output_clips': True}
                elif lim and (st['peak_dbfs'] > PEAK_LIMIT[lim] or st['true_peak_dbfs'] > TP_LIMIT[lim]):
                    raise ValueError(f'{r.id}/{arm}: peak {st["peak_dbfs"]:.2f} / tp {st["true_peak_dbfs"]:.2f}')
                if arm.endswith('m') and st['peak_dbfs'] >= 0:
                    # A limiter that raised crest (loudnorm's AGC can) leaves a twin that
                    # would need >0 dBFS to reach the source LUFS. Score it, but drop the
                    # clip from that cell in analysis rather than measure out-of-range audio.
                    info = {**info, 'twin_valid': False}
                p = scratch / f'{r.id}__{arm}.wav'
                sf.write(p, y.astype(np.float32), SR, format='WAV', subtype='FLOAT')
                per[arm] = {'path': p, 'signal': st, 'info': info, 'content_sha256': content_digest_array(y)}
            staged.append((r, base, per))
        aes = {a: _aes_batch(predictor, [s[2][a]['path'] for s in staged]) for a in arms}
        for j, (r, base, per) in enumerate(staged):
            rowv = {'id': r.id, 'source_sha256': hashes[r.id], 'baseline': base, 'arms': {}}
            for a in arms:
                v = aes[a][j]
                if set(v) != set(AXES) or not all(math.isfinite(float(v[q])) for q in AXES):
                    raise ValueError('invalid AES ' + r.id + a)
                cl = float(_clap_batch(clap, [per[a]['path']], [r.caption])[0])
                if not math.isfinite(cl):
                    raise ValueError('invalid CLAP ' + r.id + a)
                rowv['arms'][a] = {'aes': {q: float(v[q]) for q in AXES}, 'clap': cl,
                                   'signal': per[a]['signal'], 'info': per[a]['info'],
                                   'content_sha256': per[a]['content_sha256']}
                per[a]['path'].unlink()
            atomic(items / (r.id + '.json'), rowv)
        done += len(batch)
        if done % (CFG['batch'] * 25) == 0 or done == len(todo):
            print(f'  {done}/{len(todo)}', flush=True)
            atomic(out / 'progress.json', {'count': done, 'of': len(todo)})
    shutil.rmtree(scratch, ignore_errors=True)


def analyze():
    out = Path(CFG['out'])
    ids = sorted(r.id for r in M.records())
    rows = {i: json.loads((out / 'items' / (i + '.json')).read_text()) for i in ids}
    main = {i: json.loads((Path(CFG['main_064_items']) / (i + '.json')).read_text()) for i in ids}
    metrics = (*AXES, 'clap')
    rng = np.random.default_rng(CFG['bootstrap_seed'])
    reps = CFG['bootstrap_replicates']

    def val(src, i, a, m):
        v = src[i]['arms'][a]
        return v['clap'] if m == 'clap' else v['aes'][m]

    worst = max(abs(val(rows, i, 'z0', m) - val(main, i, 'z0', m)) for i in ids for m in metrics)
    gate = {'max_abs_diff_vs_064_z0': worst, 'pass': worst <= CFG['replication_tolerance']}

    # (limiter, family) -> (source, arm, twin)
    cells = {('dpl', 'L6'): (main, 'L6', 'L6m'), ('dpl', 'T14'): (main, 'T14', 'T14m')}
    for k in CFG['external']:
        cells[(k, 'L6')] = (rows, f'L6_{k}', f'L6_{k}m')
        cells[(k, 'T14')] = (rows, f'T14_{k}', f'T14_{k}m')
    cells[('loudnorm', 'T14')] = (rows, 'LN14', 'LN14m')

    table = {}
    for (lim, fam), (src, arm, tw) in cells.items():
        all_ids = ids
        ids = [i for i in all_ids if src[i]['arms'][tw]['info'].get('twin_valid', True)
               and not src[i]['arms'][arm]['info'].get('output_clips', False)]
        e = {'n': len(ids), 'excluded_clipping': len(all_ids) - len(ids),'delta_lufs': float(np.mean([src[i]['arms'][arm]['signal']['lufs'] - src[i]['arms']['z0']['signal']['lufs'] for i in ids])),
             'crest_db': float(np.mean([src[i]['arms'][arm]['signal']['crest_db'] for i in ids])),
             'true_peak_dbfs': float(np.mean([src[i]['arms'][arm]['signal']['true_peak_dbfs'] for i in ids]))}
        for m in metrics:
            e[m] = {'total': mean_ci([val(src, i, arm, m) - val(src, i, 'z0', m) for i in ids], rng, reps),
                    'level': mean_ci([val(src, i, arm, m) - val(src, i, tw, m) for i in ids], rng, reps),
                    'processing': mean_ci([val(src, i, tw, m) - val(src, i, 'z0', m) for i in ids], rng, reps)}
        table[f'{fam}/{lim}'] = e
        ids = all_ids

    def sign(c):
        lo, hi = c['ci95']
        return 0 if lo <= 0 <= hi else (1 if c['mean'] > 0 else -1)

    agreement = {}
    for fam in ('L6', 'T14'):
        keys = [k for k in table if k.startswith(fam + '/') and not k.endswith('loudnorm')]
        agreement[fam] = {m: {part: sorted({sign(table[k][m][part]) for k in keys}) for part in ('total', 'level', 'processing')}
                          for m in metrics}
    result = {'config': CFG, 'n': len(ids), 'replication_gate': gate, 'cells': table,
              'sign_sets_across_limiters': agreement}
    atomic(out / 'summary.json', result)
    return result


def main():
    if '--analyze-only' not in sys.argv:
        score_all()
    r = analyze()
    print(json.dumps({'gate': r['replication_gate']}), flush=True)
    for k, e in r['cells'].items():
        print(f"{k:16s} dLUFS {e['delta_lufs']:+.2f} crest {e['crest_db']:.2f} | " + ' | '.join(
            f"{m} tot {e[m]['total']['mean']:+.4f} lev {e[m]['level']['mean']:+.4f} proc {e[m]['processing']['mean']:+.4f}"
            for m in ('PQ', 'CE', 'clap')))
    print(json.dumps(r['sign_sets_across_limiters'], indent=1))


if __name__ == '__main__':
    main()
