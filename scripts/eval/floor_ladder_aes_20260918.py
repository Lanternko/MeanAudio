#!/usr/bin/env python3
"""065: walk the level down to the floor, by scalar and by downward limiting.

063 stopped at -18 dB and read PQ as monotone (quieter scores higher). A 32-clip pilot
(2026-09-18) showed the curve turning over below -18 dB and every scorer reaching a
floor near -96 dB. This extends the ladder to that floor and separates the three
floors a quiet clip can hit:

  scorer floor   AES: WavLM's first conv has no bias and is followed by a per-channel
                 GroupNorm (eps 1e-5), so it is scale-invariant until the conv output
                 variance nears eps; below that the features, and the score, go to the
                 all-zero value. CLAP: log-mel with amin 1e-10.
  format floor   a deliverable PCM_16 file (what eval.py's sf.write produces) zeroes
                 everything under ~1 LSB (-90.3 dBFS).
  meter floor    BS.1770's -70 LUFS absolute gate. Loudness here is L*, LUFS with the
                 absolute gate moved to -270 (measure at +200 dB, subtract 200); the
                 gate is the meter's only non-scale-covariant step, so L* of a scalar
                 arm is exactly source L* minus the attenuation.

Every level is scored in two modes of the same waveform:
  F  float32 WAV; AES reads it as float; CLAP embeds the float 48 kHz resample with
     no quantisation anywhere. This is the level effect alone.
  Q  written as PCM_16 by soundfile, as eval.py does; AES reads it; CLAP uses the
     canonical filelist path. This is level plus the format floor.

Downward limiting (D arms): 064's primary limiter (x42-dpl, true-peak, -1 dBTP), driven
with a pre-gain G and scaled back down by G, i.e. limiting at an effective ceiling of
-1-G dBFS. G is bisected until L* drops by exactly g dB. Loud passages come down, quiet passages stay
where they were, so at the same L* a limited clip keeps its quiet content further
above the format floor than a scalar-attenuated one. D<g> is matched in L* to the
scalar rung m<g>, so D<g> - m<g> is limiter vs scalar at equal loudness, and D<g>m
(the same limited waveform scaled back to source L*) isolates the processing cost at
the original level, as in 064.

Reuses the 051 hash-verified baseline FLAC; audio is transient.
"""
from __future__ import annotations
import csv
import io
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from gain_ladder_aes_20260918 import (AXES, SR, atomic, content_digest_array, digest,  # noqa: E402
                                      mean_ci, meter, read_audio)
from limiter_loudness_aes_20260918 import reduction  # noqa: E402
from external_limiters import dpl  # noqa: E402

SRC_ROOT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/loudness_aes_cfg3_20260911')
CFG = {
    'experiment': '065 floor ladder AES',
    'source_audio_root': str(SRC_ROOT / '_audio/baseline'),
    'source_manifest': str(SRC_ROOT / 'audio_manifest.json'),
    'tsv': str(SRC_ROOT / 'musiccaps5521_cfg3_fidelity8.tsv'),
    'aes_snapshot': '/home/kojiek/.cache/huggingface/hub/models--facebook--audiobox-aesthetics/'
                    'snapshots/9b1dd8e5df9af7216e836a98974fe3b82c56ded6',
    'clap_checkpoint': str(ROOT / 'weights/music_speech_audioset_epoch_15_esc_89.98.pt'),
    'gain_ladder_063_items': '/home/kojiek/nvme_experiment_artifacts/meanaudio/gain_ladder_aes_20260918/items',
    'out': os.environ.get('FLOOR065_OUT',
                          '/home/kojiek/nvme_experiment_artifacts/meanaudio/floor_ladder_aes_20260918'),
    'rows': int(os.environ.get('FLOOR065_ROWS', 5521)),
    'batch': 16,
    # attenuation in dB below the source; 0 is z0
    'ladder_db': [0, 18, 21, 24, 27, 30, 33, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 108, 120],
    'limit_db': [18, 24, 36, 48, 60, 72, 84],
    'limiter': {'impl': 'x42-dpl (external_limiters.dpl)', 'ceiling_dbtp': -1.0},
    # Measure at +200 dB: the -70 gate then sits at -270 dBFS-equivalent, far below any
    # block of any rung (an 80 dB shift still gated source blocks under -54 LUFS at m96).
    'lstar_shift_db': 200.0,
    'pregain_search': {'lo_db': 0.0, 'hi_db': 160.0, 'max_iter': 40, 'tolerance_lu': 0.05},
    'plan_workers': 10,
    'bootstrap_seed': 20260918,
    'bootstrap_replicates': 10000,
    'replication_tolerance': 1e-4,
    # a Q-F gap, or a gap to the silence score, counts only past these (AES points / CLAP cosine)
    'tolerance': {'aes': 0.01, 'clap': 0.002},
}
MODES = ('F', 'Q')


def rung(g):
    return 'z0' if g == 0 else f'm{g}'


def conditions():
    """Every scored (arm, mode) pair for one clip, in scoring order."""
    out = [(rung(g), m) for g in CFG['ladder_db'] for m in MODES]
    out += [(f'D{g}', m) for g in CFG['limit_db'] for m in MODES]
    out += [(f'D{g}m', 'F') for g in CFG['limit_db']]
    out.append(('sil', 'F'))
    return out


def key(arm, mode):
    return f'{arm}.{mode}'


# ------------------------------------------------------------------ loudness

def lstar(met, x):
    """Integrated loudness with the -70 LUFS absolute gate moved to -270 LUFS."""
    import warnings
    s = CFG['lstar_shift_db']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        v = float(met.integrated_loudness(np.asarray(x, dtype=np.float64) * 10 ** (s / 20)))
    return v - s if math.isfinite(v) else None


def gated_lufs(met, x):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        v = float(met.integrated_loudness(np.asarray(x, dtype=np.float64)))
    return v if math.isfinite(v) else None


def stats(met, x):
    x = np.asarray(x, dtype=np.float64)
    rms = math.sqrt(float(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    return {
        'lstar': lstar(met, x),
        'lufs': gated_lufs(met, x),
        'rms_dbfs': 20 * math.log10(rms) if rms > 0 else None,
        'peak_dbfs': 20 * math.log10(peak) if peak > 0 else None,
        'crest_db': 20 * math.log10(peak / rms) if rms > 0 and peak > 0 else None,
    }


# ------------------------------------------------------------------ plan (CPU)

def limited(x, pre_db):
    """x-dpl at -1 dBTP on x * G, scaled back by G: limiting at a -1-G dBFS ceiling."""
    k = 10 ** (pre_db / 20)
    return dpl(x * k, CFG['limiter']['ceiling_dbtp']) / k


def plan_clip(clip_id):
    """Find, per D arm, the pre-gain G whose limited output sits exactly g dB below in L*.

    L* of the scaled-back output falls monotonically in G (more of the clip is pushed
    into the limiter), and at G = 0 the limiter barely acts, so [0, hi] brackets it.
    """
    x = read_audio(Path(CFG['source_audio_root']) / (clip_id + '.flac'))
    met = meter()
    src = lstar(met, x)
    ps = CFG['pregain_search']
    arms = {}
    for g in CFG['limit_db']:
        target = src - g
        lo, hi = ps['lo_db'], ps['hi_db']
        best = None
        for _ in range(ps['max_iter']):
            mid = (lo + hi) / 2
            got = lstar(met, limited(x, mid))
            if best is None or abs(got - target) < abs(best[1] - target):
                best = (mid, got)
            if abs(got - target) <= ps['tolerance_lu']:
                break
            if got > target:
                lo = mid
            else:
                hi = mid
        pre, got = best
        red = reduction(x * 10 ** (pre / 20), dpl(x * 10 ** (pre / 20), CFG['limiter']['ceiling_dbtp']))
        arms[f'D{g}'] = {'pre_gain_db': pre, 'effective_ceiling_dbfs': CFG['limiter']['ceiling_dbtp'] - pre,
                         'target_lstar': target, 'lstar': got,
                         'hit': abs(got - target) <= ps['tolerance_lu'], **red}
    return {'id': clip_id, 'source_lstar': src, 'arms': arms}


def plan_all(rec):
    from multiprocessing import Pool
    d = Path(CFG['out']) / 'plan'
    d.mkdir(parents=True, exist_ok=True)
    todo = [r.id for r in rec if not (d / (r.id + '.json')).exists()]
    print(f'plan: {len(todo)} clips to search', flush=True)
    done = 0
    with Pool(CFG['plan_workers']) as pool:
        for v in pool.imap_unordered(plan_clip, todo, chunksize=4):
            atomic(d / (v['id'] + '.json'), v)
            done += 1
            if done % 500 == 0:
                print(f'  plan {done}/{len(todo)}', flush=True)
    return {r.id: json.loads((d / (r.id + '.json')).read_text()) for r in rec}


# ------------------------------------------------------------------ render

def pcm16(y):
    """The waveform as a deliverable PCM_16 file would hold it (soundfile, as eval.py)."""
    import soundfile as sf
    b = io.BytesIO()
    sf.write(b, np.asarray(y, dtype=np.float64), SR, format='WAV', subtype='PCM_16')
    b.seek(0)
    return sf.read(b, dtype='float64')[0]


def render_clip(x, plan, met):
    """{arm: waveform} for every arm of one clip (mode applied later)."""
    out = {rung(g): x * 10 ** (-g / 20) for g in CFG['ladder_db']}
    for g in CFG['limit_db']:
        y = limited(x, plan['arms'][f'D{g}']['pre_gain_db'])
        out[f'D{g}'] = y
        back = plan['source_lstar'] - lstar(met, y)
        z = y * 10 ** (back / 20)
        if np.max(np.abs(z)) >= 1.0:
            raise ValueError(f'{plan["id"]}/D{g}m: loudness-matched twin clips')
        out[f'D{g}m'] = z
    out['sil'] = np.zeros_like(x)
    return out


# ------------------------------------------------------------------ scoring

def records():
    from score_musiccaps_per_item import read_musiccaps_tsv
    rec = read_musiccaps_tsv(Path(CFG['tsv']), expected_count=5521)
    return rec[:CFG['rows']]


def clap_float(model, y, caption):
    """CLAP on the float waveform: the filelist path minus its int16 quantisation."""
    import librosa
    import torch
    w = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=SR, target_sr=48000, res_type='soxr_hq')
    with torch.no_grad():
        a = model.get_audio_embedding_from_data(torch.from_numpy(w)[None].float(), use_tensor=True)
        t = model.get_text_embedding([caption], use_tensor=True)
    a = torch.nn.functional.normalize(a.float(), dim=-1)
    t = torch.nn.functional.normalize(t.float(), dim=-1)
    return float((a * t).sum())


def score_all(rec, plans):
    import soundfile as sf
    from score_musiccaps_per_item import load_aes_predictor, _aes_batch, load_clap_model, _clap_batch
    out = Path(CFG['out'])
    hashes = json.loads(Path(CFG['source_manifest']).read_text())['audio_sha256']
    conds = conditions()
    items = out / 'items'
    items.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rec if not (items / (r.id + '.json')).exists()]
    print(f'score: {len(todo)} clips x {len(conds)} conditions', flush=True)
    if not todo:
        return
    scratch = out / '_render'
    scratch.mkdir(parents=True, exist_ok=True)
    predictor = load_aes_predictor(Path(CFG['aes_snapshot']), device='cuda', batch_size=CFG['batch'])
    clap = load_clap_model(Path(CFG['clap_checkpoint']), device='cuda', local_files_only=True)
    done = 0
    for start in range(0, len(todo), CFG['batch']):
        free = shutil.disk_usage(out)
        if free.free < 20e9:
            raise SystemExit('less than 20 GB free on the output filesystem')
        batch = todo[start:start + CFG['batch']]
        staged = []
        for r in batch:
            src = Path(CFG['source_audio_root']) / (r.id + '.flac')
            if digest(src) != hashes[r.id]:
                raise ValueError('source audio drift: ' + r.id)
            x = read_audio(src)
            met = meter()
            waves = render_clip(x, plans[r.id], met)
            per = {}
            for arm, mode in conds:
                y = waves[arm] if mode == 'F' else pcm16(waves[arm])
                p = scratch / f'{r.id}__{arm}.{mode}.wav'
                sf.write(p, np.asarray(y, dtype=np.float32), SR, format='WAV', subtype='FLOAT')
                st = stats(met, y)
                if mode == 'Q':
                    st['zero_fraction'] = float(np.mean(y == 0))
                per[key(arm, mode)] = {'path': p, 'y': y, 'signal': st,
                                       'content_sha256': content_digest_array(y)}
            staged.append((r, per))
        # AES one call per condition over the same clips, which is 063's grouping, so the
        # z0.F and m18.F conditions can be checked against 063 per clip.
        aes = {key(a, m): _aes_batch(predictor, [s[1][key(a, m)]['path'] for s in staged]) for a, m in conds}
        for j, (r, per) in enumerate(staged):
            row = {'id': r.id, 'source_sha256': hashes[r.id], 'plan': plans[r.id], 'conditions': {}}
            for a, m in conds:
                k = key(a, m)
                v = aes[k][j]
                if set(v) != set(AXES) or not all(math.isfinite(float(v[q])) for q in AXES):
                    raise ValueError('invalid AES ' + r.id + k)
                if m == 'F' and a != 'sil':
                    cl = clap_float(clap, per[k]['y'], r.caption)
                else:
                    cl = float(_clap_batch(clap, [per[k]['path']], [r.caption])[0])
                if not math.isfinite(cl):
                    raise ValueError('invalid CLAP ' + r.id + k)
                row['conditions'][k] = {'aes': {q: float(v[q]) for q in AXES}, 'clap': cl,
                                        'signal': per[k]['signal'],
                                        'content_sha256': per[k]['content_sha256']}
                per[k]['path'].unlink()
            atomic(items / (r.id + '.json'), row)
        done += len(batch)
        if done % (CFG['batch'] * 10) == 0 or done == len(todo):
            print(f'  score {done}/{len(todo)}', flush=True)
            atomic(out / 'progress.json', {'count': done, 'of': len(todo)})
    shutil.rmtree(scratch, ignore_errors=True)


# ------------------------------------------------------------------ analysis

def analyze(rec):
    out = Path(CFG['out'])
    ids = sorted(r.id for r in rec)
    rows = {i: json.loads((out / 'items' / (i + '.json')).read_text()) for i in ids}
    metrics = (*AXES, 'clap')
    tol = lambda m: CFG['tolerance']['clap' if m == 'clap' else 'aes']  # noqa: E731
    rng = np.random.default_rng(CFG['bootstrap_seed'])
    reps = CFG['bootstrap_replicates']

    def val(i, k, m):
        v = rows[i]['conditions'][k]
        return v['clap'] if m == 'clap' else v['aes'][m]

    def arr(k, m):
        return np.array([val(i, k, m) for i in ids])

    # Replication gates: same samples, same scorer as 063.
    ref = Path(CFG['gain_ladder_063_items'])
    gates = {}
    for k, arm063, ms in (('z0.F', 'z0', AXES), ('m18.F', 'm18', AXES), ('z0.Q', 'z0', metrics)):
        worst = 0.0
        for i in ids:
            o = json.loads((ref / arm063 / (i + '.json')).read_text())
            for m in ms:
                worst = max(worst, abs(val(i, k, m) - (o['clap'] if m == 'clap' else o['aes'][m])))
        gates[k] = {'vs_063': arm063, 'metrics': list(ms), 'max_abs_diff': worst,
                    'pass': worst <= CFG['replication_tolerance']}
    # L* of a scalar rung must sit exactly g below the source (the meter-floor fix works)
    lerr = max(abs(rows[i]['conditions'][key(rung(g), 'F')]['signal']['lstar']
                   - (rows[i]['plan']['source_lstar'] - g)) for i in ids for g in CFG['ladder_db'])
    gates['lstar_scalar_covariance'] = {'max_abs_err_lu': lerr, 'pass': lerr <= 0.01}
    hit = float(np.mean([rows[i]['plan']['arms'][f'D{g}']['hit'] for i in ids for g in CFG['limit_db']]))
    gates['limiter_target_hit_rate'] = {'rate': hit, 'pass': hit >= 0.99}

    ladder = CFG['ladder_db']
    levels, cov = {}, {}
    for a, m in conditions():
        k = key(a, m)
        levels[k] = {q: mean_ci(arr(k, q), rng, reps) for q in metrics}
        sig = [rows[i]['conditions'][k]['signal'] for i in ids]
        cov[k] = {s: (float(np.mean([v[s] for v in sig if v.get(s) is not None]))
                      if any(v.get(s) is not None for v in sig) else None)
                  for s in ('lstar', 'lufs', 'peak_dbfs', 'crest_db', 'zero_fraction')}
        cov[k]['lufs_undefined_fraction'] = float(np.mean([v['lufs'] is None for v in sig]))

    # 1. shape of the float curve: where each metric peaks (argmax over rungs, bootstrapped)
    shape = {}
    n = len(ids)
    for q in metrics:
        mat = np.stack([arr(key(rung(g), 'F'), q) for g in ladder])  # rungs x clips
        means = mat.mean(1)
        am = []
        for _ in range(0, reps, 100):
            ix = rng.integers(0, n, size=(100, n))
            am.extend(np.argmax(mat[:, ix].mean(2), axis=0))
        am = np.array(am)
        steps = {f'{rung(a)}->{rung(b)}': mean_ci(mat[ladder.index(b)] - mat[ladder.index(a)], rng, reps)
                 for a, b in zip(ladder, ladder[1:])}
        shape[q] = {'peak_rung': rung(ladder[int(np.argmax(means))]),
                    'peak_attenuation_db': ladder[int(np.argmax(means))],
                    'peak_rung_bootstrap_share': {rung(ladder[j]): float(np.mean(am == j))
                                                  for j in sorted(set(am.tolist()))},
                    'steps_louder_to_quieter': steps}

    # 2. scorer floor: shallowest rung from which every deeper float rung is within tol of silence
    # 3. format floor: shallowest rung where the Q-F gap departs from its z0 value
    floors = {}
    for q in metrics:
        sil = arr('sil.F', q)
        near = [abs(float(np.mean(arr(key(rung(g), 'F'), q) - sil))) <= tol(q) for g in ladder]
        scorer = next((ladder[j] for j in range(len(ladder)) if all(near[j:])), None)
        gap0 = arr('z0.Q', q) - arr('z0.F', q)
        gaps, onset = {}, None
        for g in ladder:
            d = (arr(key(rung(g), 'Q'), q) - arr(key(rung(g), 'F'), q)) - gap0
            c = mean_ci(d, rng, reps)
            gaps[rung(g)] = c
            if onset is None and g > 0 and (c['ci95'][0] > 0 or c['ci95'][1] < 0) and abs(c['mean']) >= tol(q):
                onset = g
        floors[q] = {'scorer_floor_attenuation_db': scorer, 'silence_score': float(sil.mean()),
                     'format_floor_onset_attenuation_db': onset,
                     'z0_q_minus_f': mean_ci(gap0, rng, reps),
                     'q_minus_f_relative_to_z0': gaps}

    # 4. downward limiting vs scalar at matched L*
    limiting = {}
    for g in CFG['limit_db']:
        d, m_ = f'D{g}', rung(g)
        limiting[d] = {'matched_to': m_, 'metrics': {}}
        for q in metrics:
            limiting[d]['metrics'][q] = {
                'limiter_minus_scalar_Q': mean_ci(arr(key(d, 'Q'), q) - arr(key(m_, 'Q'), q), rng, reps),
                'limiter_minus_scalar_F': mean_ci(arr(key(d, 'F'), q) - arr(key(m_, 'F'), q), rng, reps),
                'format_damage_limiter': mean_ci(arr(key(d, 'Q'), q) - arr(key(d, 'F'), q), rng, reps),
                'format_damage_scalar': mean_ci(arr(key(m_, 'Q'), q) - arr(key(m_, 'F'), q), rng, reps),
                'processing_at_source_level': mean_ci(arr(key(d + 'm', 'F'), q) - arr('z0.F', q), rng, reps),
            }
        limiting[d]['effective_ceiling_dbfs_mean'] = float(np.mean(
            [rows[i]['plan']['arms'][d]['effective_ceiling_dbfs'] for i in ids]))
        limiting[d]['gr_max_db_mean'] = float(np.mean([rows[i]['plan']['arms'][d]['gr_max_db'] for i in ids]))

    result = {'config': CFG, 'n': len(ids), 'gates': gates, 'all_gates_pass': all(v['pass'] for v in gates.values()),
              'levels': levels, 'covariates': cov, 'shape_float': shape, 'floors': floors,
              'limiting_vs_scalar': limiting}
    atomic(out / 'summary.json', result)

    with (out / 'per_clip.csv').open('w', newline='') as f:
        cols = ['id', 'arm', 'mode', 'lstar', 'lufs', 'peak_dbfs', 'crest_db', 'zero_fraction', *AXES, 'clap']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in ids:
            for a, m in conditions():
                v = rows[i]['conditions'][key(a, m)]
                w.writerow({'id': i, 'arm': a, 'mode': m, **{s: v['signal'].get(s) for s in cols[3:8]},
                            **v['aes'], 'clap': v['clap']})
    plot(result)
    return result


def plot(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ladder = CFG['ladder_db']
    fig, axs = plt.subplots(1, 5, figsize=(24, 4.5))
    for ax, q in zip(axs, (*AXES, 'clap')):
        for mode, style in (('F', 'o-'), ('Q', 's--')):
            ax.plot([-g for g in ladder], [r['levels'][key(rung(g), mode)][q]['mean'] for g in ladder],
                    style, ms=3, label=f'scalar {mode}')
        lg = CFG['limit_db']
        ax.plot([-g for g in lg], [r['levels'][key(f'D{g}', 'Q')][q]['mean'] for g in lg], '^:', label='limiter Q')
        ax.axhline(r['levels']['sil.F'][q]['mean'], color='grey', lw=.8, label='silence')
        ax.set(title=q, xlabel='L* relative to source (dB)')
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(CFG['out']) / 'floor_ladder.png', dpi=150)
    plt.close(fig)


def main():
    rec = records()
    if '--analyze-only' not in sys.argv:
        plans = plan_all(rec)
        score_all(rec, plans)
    r = analyze(rec)
    print(json.dumps({'gates': r['gates'], 'all_gates_pass': r['all_gates_pass'],
                      'peaks': {q: r['shape_float'][q]['peak_rung'] for q in r['shape_float']},
                      'floors': {q: {k: v for k, v in r['floors'][q].items()
                                     if k.endswith('_db') or k == 'silence_score'} for q in r['floors']}},
                     indent=1))


if __name__ == '__main__':
    main()
