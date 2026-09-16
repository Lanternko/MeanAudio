#!/usr/bin/env python3
"""061: fixed-LUFS crest intervention on the 051 canonical baseline audio.

Answers whether the descriptive crest<->PQ association reported by 051 survives a
within-clip intervention that moves crest while holding integrated loudness fixed.
Reuses the 051 hash-verified baseline FLAC; generates no audio from a checkpoint.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path('/home/kojiek/MeanAudio')
CONTRACT = ROOT / 'docs/experiments/crest_intervention_cfg3_20260916_contract.json'
AXES = ('CE', 'CU', 'PC', 'PQ')
SR = 16000
sys.path.insert(0, str(ROOT / 'scripts/eval'))


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic(p, value):
    p = Path(p)
    p.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    t = p.with_name('.' + p.name + '.tmp.' + str(os.getpid()))
    with t.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def config():
    return json.loads(CONTRACT.read_text())


def binding():
    return digest(CONTRACT)


def out(c):
    return Path(c['storage']['path'])


def arms(c):
    return c['transform']['arms']


def records(c):
    from score_musiccaps_per_item import read_musiccaps_tsv
    return read_musiccaps_tsv(Path(c['tsv']), expected_count=c['protocol']['rows'])


def checked(p, c):
    v = json.loads(Path(p).read_text())
    if v.get('contract_sha256') != binding():
        raise ValueError(f'stale contract artifact: {p}')
    return v


def capacity(c):
    for path in c['resource_budget']['writable_filesystems']:
        fs = os.statvfs(path)
        if fs.f_bavail * fs.f_frsize < c['storage']['hard_stop_free_bytes']:
            raise SystemExit(75)


# ---------------------------------------------------------------- signal layer

def meter():
    import pyloudnorm as pyln
    return pyln.Meter(SR, filter_class='K-weighting', block_size=.4)


def lufs(met, x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        v = float(met.integrated_loudness(np.asarray(x, dtype=np.float64)))
    return v


def crest_db(x):
    x = np.asarray(x, dtype=np.float64)
    rms = math.sqrt(float(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    if rms <= 0 or peak <= 0:
        raise ValueError('degenerate waveform')
    return 20 * math.log10(peak / rms)


def env_follow(v, ta, tr):
    """Absolute-value envelope follower with separate attack/release constants."""
    aa = math.exp(-1 / (SR * ta))
    ar = math.exp(-1 / (SR * tr))
    out_ = np.empty_like(v)
    s = v[0]
    for i in range(len(v)):
        c = aa if v[i] > s else ar
        s = c * s + (1 - c) * v[i]
        out_[i] = s
    return out_


def one_pole(v, tau):
    a = math.exp(-1 / (SR * tau))
    out_ = np.empty_like(v)
    s = v[0]
    for i in range(len(v)):
        s = a * s + (1 - a) * v[i]
        out_[i] = s
    return out_


def shaper(x, ta, tr, ts):
    """Smoothed, RMS-centred level curve in dB.

    The applied gain for ratio r is clip((r-1) * shaper, +-clamp): the follower and
    the smoother are both linear in their input, so one curve serves every ratio and
    the bisection never repeats the O(N) filters.
    """
    x64 = np.asarray(x, dtype=np.float64)
    env = env_follow(np.abs(x64), ta, tr) + 1e-9
    ref = 20 * math.log10(math.sqrt(float(np.mean(x64 ** 2))) + 1e-12)
    return one_pole(20 * np.log10(env) - ref, ts)


def apply_gain(x, gain_db, met, target_lufs, tol=0.01, passes=4):
    """Apply a gain curve, then normalise onto the target integrated loudness.

    Gated loudness is not exactly scalar-equivariant: rescaling moves blocks across
    the relative gate, so a single correction can land ~0.1 LU off. Re-measure until
    the residual is inside tol; in practice the second pass is exact.
    """
    y = np.asarray(x, dtype=np.float64) * np.power(10.0, gain_db / 20.0)
    held = False
    for _ in range(passes):
        cur = lufs(met, y)
        if not math.isfinite(cur):
            break
        y = y * (10 ** ((target_lufs - cur) / 20.0))
        if abs(lufs(met, y) - target_lufs) <= tol:
            held = True
            break
    return y, held


def search_ratio(x, curve, clamp, met, target_lufs, base_crest, spec, grid_points, mod_cap):
    """Grid-search the ratio under a modulation cap.

    Crest is NOT monotone in the ratio: heavy processing in either direction can push
    a clip's peak-to-RMS back up, so bisection silently returns a bound that moves
    crest the wrong way. The grid is scanned exhaustively and ratios whose gain
    modulation exceeds the cap are rejected, which keeps every arm a mild transform
    and keeps the measured contrast about crest rather than about processing severity.
    """
    objective = spec['objective']
    grid = np.linspace(spec['ratio_lo'], spec['ratio_hi'], grid_points)
    best = None
    for r in grid:
        gain = np.clip((r - 1.0) * curve, -clamp, clamp)
        mod = float(np.sqrt(np.mean(gain ** 2)))
        if mod > mod_cap:
            continue
        delta = crest_db(apply_gain(x, gain, met, target_lufs)[0]) - base_crest
        if objective == 'target':
            score = -abs(delta - spec['target_delta_db'])
        elif objective == 'max':
            score = delta
        elif objective == 'min':
            score = -delta
        else:
            raise ValueError('unknown objective: ' + objective)
        if best is None or score > best[0]:
            best = (score, float(r), delta, mod)
    if best is None:
        return 1.0, 0.0, 0.0  # cap admits nothing; identity is the honest fallback
    return best[1], best[2], best[3]


def signal_stats(x, met):
    x = np.asarray(x, dtype=np.float64)
    rms = math.sqrt(float(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    window = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * window))
    total = spec.sum()
    return {
        'lufs': lufs(met, x),
        'rms_dbfs': 20 * math.log10(rms) if rms > 0 else None,
        'crest_db': 20 * math.log10(peak / rms) if rms > 0 else None,
        'peak_dbfs': 20 * math.log10(peak) if peak > 0 else None,
        'clipped_fraction': float(np.mean(np.abs(x) >= .999)),
        'silence_fraction': float(np.mean(np.abs(x) < 1e-3)),
        'centroid_hz': float(np.sum(spec * np.fft.rfftfreq(len(x), 1 / SR)) / total) if total > 0 else None,
    }


def read_audio(p):
    import soundfile as sf
    x, sr = sf.read(p, dtype='float32')
    if sr != SR or x.ndim != 1 or not np.isfinite(x).all() or len(x) < 6400:
        raise ValueError(f'invalid mono16k audio: {p}')
    return x


def transform_clip(job):
    """Build every arm for one clip. Pure function of (audio bytes, contract)."""
    import soundfile as sf
    clip_id, source, c = job
    t = c['transform']
    met = meter()
    x = read_audio(source)
    if digest(source) != c['_source_hashes'][clip_id]:
        raise ValueError('source audio drift')
    base = signal_stats(x, met)
    target_lufs = base['lufs'] + t['pad_db']
    clamp = t['gain_clamp_db']
    slow = shaper(x, *[t['slow_constants'][k] for k in ('attack_s', 'release_s', 'smooth_s')])
    fast = shaper(x, *[t['fast_constants'][k] for k in ('attack_s', 'release_s', 'smooth_s')])

    seed = int.from_bytes(hashlib.sha256((t['shift_seed_namespace'] + clip_id).encode()).digest()[:8], 'big')
    shift = int(np.random.default_rng(seed).integers(len(x)))

    gains, fitted = {}, {}
    # Derived arms read another arm's gain curve, so build sources first and stay
    # independent of the contract's key order.
    for name, spec in sorted(t['arms'].items(), key=lambda kv: kv[1]['kind'] == 'shifted_envelope'):
        kind = spec['kind']
        if kind == 'reference':
            gains[name] = np.zeros(len(x))
        elif kind == 'search_crest':
            curve = slow if spec['constants'] == 'slow' else fast
            r, delta, _mod = search_ratio(x, curve, clamp, met, target_lufs, base['crest_db'], spec,
                                          t['grid_points'], t['gain_mod_max_db'])
            fitted[name] = r
            gains[name] = np.clip((r - 1.0) * curve, -clamp, clamp)
        elif kind == 'shifted_envelope':
            gains[name] = np.roll(gains[spec['source_arm']], shift)
        else:
            raise ValueError('unknown arm kind: ' + kind)

    # Headroom is resolved per clip, not globally: the pad is a within-clip constant so
    # it cancels in every paired contrast, and only the few clips whose arms would clip
    # pay the extra attenuation.
    rendered = {name: apply_gain(x, g, met, target_lufs) for name, g in gains.items()}
    worst = max(20 * math.log10(float(np.max(np.abs(y)))) for y, _ in rendered.values())
    if worst > t['peak_ceiling_dbfs']:
        target_lufs -= (worst - t['peak_ceiling_dbfs']) + t['peak_headroom_margin_db']
        rendered = {name: apply_gain(x, g, met, target_lufs) for name, g in gains.items()}

    result = {'id': clip_id, 'contract_sha256': binding(), 'source_sha256': c['_source_hashes'][clip_id],
              'baseline': base, 'target_lufs': target_lufs, 'shift_samples': shift,
              'pad_db_effective': target_lufs - base['lufs'], 'fitted_ratio': fitted, 'arms': {}}
    ineligible = []
    for name, (y, held) in rendered.items():
        stats = signal_stats(y, met)
        stats['loudness_held'] = held
        stats['gain_mod_rms_db'] = float(np.sqrt(np.mean(gains[name] ** 2)))
        stats['gain_range_db'] = [float(gains[name].min()), float(gains[name].max())]
        stats['crest_delta_db'] = stats['crest_db'] - base['crest_db']
        stats['lufs_error'] = stats['lufs'] - target_lufs
        stats['centroid_ratio'] = stats['centroid_hz'] / base['centroid_hz'] if base['centroid_hz'] else None
        # Near-silent clips cannot be held at an arbitrary gated loudness: expansion pushes
        # their quiet majority under the gate, so the target becomes unreachable. Such clips
        # are flagged for exclusion rather than crashing the run or being silently kept.
        if not held or abs(stats['lufs_error']) > t['lufs_tolerance']:
            ineligible.append(f'{name}: loudness not held ({stats["lufs_error"]:+.3f} LU)')
        if stats['peak_dbfs'] > t['peak_ceiling_dbfs']:
            ineligible.append(f'{name}: peak {stats["peak_dbfs"]:+.2f} dBFS over ceiling')
        path = Path(c['storage']['audio_root']) / name / (clip_id + '.wav')
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name('.' + path.name + '.tmp')
        sf.write(tmp, y.astype(np.float32), SR, format='WAV', subtype='FLOAT')
        os.replace(tmp, path)
        stats['sha256'] = digest(path)
        result['arms'][name] = stats
    screen = (base['lufs'] < t['min_baseline_lufs'] or base['silence_fraction'] > t['max_silence_fraction'])
    result['pretreatment_screen_fail'] = bool(screen)
    result['analysis_eligible'] = not ineligible and not screen
    result['ineligible_reasons'] = ineligible
    return result


def transform(c):
    from multiprocessing import Pool
    manifest = json.loads(Path(c['source_manifest']).read_text())
    rec = records(c)
    ids = {r.id for r in rec}
    if set(manifest['audio_sha256']) != ids:
        raise ValueError('051 baseline manifest does not cover the registered rows')
    c = dict(c, _source_hashes=manifest['audio_sha256'])
    src_root = Path(c['source_audio_root'])
    dest = out(c) / 'transform'
    dest.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in rec:
        p = dest / (r.id + '.json')
        if p.exists():
            v = checked(p, c)
            if v['source_sha256'] != manifest['audio_sha256'][r.id] or set(v['arms']) != set(arms(c)):
                raise ValueError('stale transform record: ' + r.id)
        else:
            todo.append(r.id)
    if todo:
        capacity(c)
        print(f'transform: {len(todo)} clips x {len(arms(c))} arms', flush=True)
        with Pool(c['transform']['workers']) as pool:
            for n, value in enumerate(pool.imap_unordered(
                    transform_clip, [(i, src_root / (i + '.flac'), c) for i in todo], chunksize=4), 1):
                atomic(dest / (value['id'] + '.json'), value)
                if n % 250 == 0:
                    capacity(c)
                    atomic(out(c) / 'progress.json', {'phase': 'transform', 'count': n,
                                                      'contract_sha256': binding()})
                    print(f'  {n}/{len(todo)}', flush=True)
    if {p.stem for p in dest.glob('*.json')} != ids:
        raise ValueError('transform ID mismatch')
    atomic(out(c) / 'transform_manifest.json', {
        'contract_sha256': binding(),
        'arm_sha256': {a: {i: checked(dest / (i + '.json'), c)['arms'][a]['sha256'] for i in sorted(ids)}
                       for a in arms(c)}})
    print(f'transform complete: {len(ids)} clips', flush=True)


def load_transforms(c):
    rec = records(c)
    dest = out(c) / 'transform'
    return {r.id: checked(dest / (r.id + '.json'), c) for r in rec}


def score_arm(c, arm):
    from score_musiccaps_per_item import load_aes_predictor, _aes_batch
    manifest = checked(out(c) / 'transform_manifest.json', c)
    rec = records(c)
    audio = Path(c['storage']['audio_root']) / arm
    dest = out(c) / 'items' / arm
    dest.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in rec:
        p = dest / (r.id + '.json')
        if p.exists():
            v = checked(p, c)
            if v['arm'] != arm or v['audio_sha256'] != manifest['arm_sha256'][arm][r.id]:
                raise ValueError('stale score record: ' + r.id)
        else:
            todo.append(r)
    batch_size = c['protocol']['scoring_batch_size']
    predictor = load_aes_predictor(Path(c['aes_snapshot']), device='cuda', batch_size=batch_size) if todo else None
    for start in range(0, len(todo), batch_size):
        capacity(c)
        batch = todo[start:start + batch_size]
        paths = []
        for r in batch:
            p = audio / (r.id + '.wav')
            if digest(p) != manifest['arm_sha256'][arm][r.id]:
                raise ValueError('arm audio drift: ' + r.id)
            paths.append(p)
        for r, p, v in zip(batch, paths, _aes_batch(predictor, paths)):
            if set(v) != set(AXES) or not all(math.isfinite(float(v[k])) for k in AXES):
                raise ValueError('invalid AES: ' + r.id)
            atomic(dest / (r.id + '.json'), {'contract_sha256': binding(), 'id': r.id, 'arm': arm,
                                             'audio_sha256': manifest['arm_sha256'][arm][r.id], 'aes': v})
        atomic(out(c) / 'progress.json', {'phase': 'aes_' + arm, 'count': start + len(batch),
                                          'contract_sha256': binding()})
    if {p.stem for p in dest.glob('*.json')} != {r.id for r in rec}:
        raise ValueError('score ID mismatch')
    print(f'{arm}: {len(rec)} AES records', flush=True)


# --------------------------------------------------------------------- analysis

def mean_ci(values, rng, reps):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {'n': 0, 'mean': None, 'ci95': None}
    b = []
    for start in range(0, reps, 100):
        ix = rng.integers(0, len(x), size=(min(100, reps - start), len(x)))
        b.extend(x[ix].mean(axis=1))
    return {'n': len(x), 'mean': float(x.mean()), 'ci95': np.quantile(b, [.025, .975]).tolist()}


def slope_ci(dx, dy, rng, reps):
    """Paired per-clip slope dPQ/dcrest through the origin, bootstrapped over clips."""
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    keep = np.abs(dx) > 1e-6
    dx, dy = dx[keep], dy[keep]
    if len(dx) < 50:
        return {'n': int(len(dx)), 'slope': None, 'ci95': None}
    b = []
    for start in range(0, reps, 100):
        ix = rng.integers(0, len(dx), size=(min(100, reps - start), len(dx)))
        b.extend((dx[ix] * dy[ix]).sum(1) / (dx[ix] * dx[ix]).sum(1))
    return {'n': int(len(dx)), 'slope': float((dx * dy).sum() / (dx * dx).sum()),
            'ci95': np.quantile(b, [.025, .975]).tolist()}


def analyze(c, transforms, scores):
    a = c['analysis']
    rng = np.random.default_rng(a['bootstrap_seed'])
    reps = a['bootstrap_replicates']
    every = sorted(transforms)
    ids = [i for i in every if transforms[i]['analysis_eligible']]
    ref = a['reference_arm']
    result = {'n_total': len(every), 'n': len(ids), 'reference_arm': ref,
              'excluded': {'n': len(every) - len(ids),
                           'pretreatment_screen': sum(transforms[i]['pretreatment_screen_fail'] for i in every),
                           'loudness_or_peak_not_held': sum(bool(transforms[i]['ineligible_reasons']) for i in every),
                           'rule': a['exclusion_rule'],
                           'ids': [i for i in every if not transforms[i]['analysis_eligible']]}, 'arm_deltas': {}, 'achieved_crest': {},
              'dose_response': {}, 'covariates': {},
              'inference': 'Paired within-clip deltas against the reference arm; pointwise 95% '
                           'bootstrap CI over clips. Primary is the pooled dose-response slope on '
                           'ACHIEVED crest delta: the reachable crest change is clip-dependent, so a '
                           'nominal-target contrast would confound dose with processing severity.'}
    pooled_x, pooled_y = [], []
    low_x, low_y = [], []
    for arm in arms(c):
        dcrest = [transforms[i]['arms'][arm]['crest_delta_db'] - transforms[i]['arms'][ref]['crest_delta_db']
                  for i in ids]
        result['achieved_crest'][arm] = mean_ci(dcrest, rng, reps)
        result['covariates'][arm] = {
            k: mean_ci([transforms[i]['arms'][arm][k] for i in ids], rng, reps)
            for k in ('gain_mod_rms_db', 'lufs_error', 'peak_dbfs', 'centroid_ratio')}
        if arm == ref:
            continue
        dpq = {ax: [scores[arm][i]['aes'][ax] - scores[ref][i]['aes'][ax] for i in ids] for ax in AXES}
        result['arm_deltas'][arm] = {ax: mean_ci(dpq[ax], rng, reps) for ax in AXES}
        result['dose_response'][arm] = {ax: slope_ci(dcrest, dpq[ax], rng, reps) for ax in AXES}
        pooled_x.extend(dcrest)
        pooled_y.extend(dpq['PQ'])
        for k, i in enumerate(ids):
            if transforms[i]['arms'][arm]['gain_mod_rms_db'] <= a['low_modulation_db']:
                low_x.append(dcrest[k])
                low_y.append(dpq['PQ'][k])

    pooled = slope_ci(pooled_x, pooled_y, rng, reps)
    result['primary'] = {
        'endpoint': 'pooled paired dose-response slope dPQ per dB achieved crest delta, '
                    'all intervention arms against the reference arm',
        'observed_slope': pooled,
        'association_pq_per_db': a['association_pq_per_db'],
        'fraction_of_association': (pooled['slope'] / a['association_pq_per_db']
                                    if pooled['slope'] is not None else None),
        'low_modulation_slope': slope_ci(low_x, low_y, rng, reps),
        'decision': decide(pooled, a['association_pq_per_db'], a['causal_fraction_threshold']),
    }
    return result


def decide(pooled, association, threshold):
    if pooled['slope'] is None:
        return 'insufficient_dose: too few clips moved crest to estimate a slope'
    lo, hi = pooled['ci95']
    if lo <= 0 <= hi:
        return ('association_not_reproduced_by_intervention: crest is a proxy, not the lever; '
                'do not train toward crest')
    if pooled['slope'] <= 0:
        return 'intervention_opposes_association: unregistered direction; treat as exploratory'
    if pooled['slope'] >= threshold * association:
        return ('crest_is_a_causal_lever_on_PQ: training toward crest would hack AES; '
                're-read any PQ comparison whose arms differ in crest')
    return 'partial: intervention positive but far below the association-implied slope'


def report(c):
    transforms = load_transforms(c)
    ids = sorted(transforms)
    scores = {}
    for arm in arms(c):
        d = out(c) / 'items' / arm
        if {p.stem for p in d.glob('*.json')} != set(ids):
            raise ValueError('missing AES records for arm ' + arm)
        scores[arm] = {i: checked(d / (i + '.json'), c) for i in ids}
    result = analyze(c, transforms, scores)

    table = out(c) / 'per_clip.csv'
    with table.open('w', newline='') as f:
        cols = ['id', 'arm', 'baseline_crest_db', 'baseline_lufs', 'crest_db', 'crest_delta_db', 'lufs',
                'lufs_error', 'peak_dbfs', 'gain_mod_rms_db', 'centroid_ratio', 'fitted_ratio', *AXES]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm in arms(c):
            for i in ids:
                t = transforms[i]
                w.writerow({'id': i, 'arm': arm, 'baseline_crest_db': t['baseline']['crest_db'],
                            'baseline_lufs': t['baseline']['lufs'],
                            'fitted_ratio': t['fitted_ratio'].get(arm),
                            **{k: t['arms'][arm][k] for k in cols[4:11]},
                            **scores[arm][i]['aes']})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [a for a in arms(c) if a != c['analysis']['reference_arm']]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    x = [result['achieved_crest'][a]['mean'] for a in names]
    y = [result['arm_deltas'][a]['PQ']['mean'] for a in names]
    err = np.array([[y[k] - result['arm_deltas'][a]['PQ']['ci95'][0],
                     result['arm_deltas'][a]['PQ']['ci95'][1] - y[k]] for k, a in enumerate(names)]).T
    axs[0].errorbar(x, y, yerr=err, fmt='o')
    lim = np.linspace(min(x + [0]), max(x + [0]), 10)
    axs[0].plot(lim, c['analysis']['association_pq_per_db'] * lim, ls='--', color='gray',
                label='051 association-implied')
    for k, a in enumerate(names):
        axs[0].annotate(a, (x[k], y[k]))
    axs[0].set(xlabel='achieved crest delta (dB)', ylabel='paired PQ delta vs reference')
    axs[0].axhline(0, color='black', lw=.5)
    axs[0].legend()
    arm = c['analysis']['primary_arm']
    axs[1].scatter([transforms[i]['arms'][arm]['crest_delta_db'] for i in ids],
                   [scores[arm][i]['aes']['PQ'] - scores[c['analysis']['reference_arm']][i]['aes']['PQ']
                    for i in ids], s=2, alpha=.15)
    axs[1].set(xlabel=f'achieved crest delta (dB), {arm}', ylabel='paired PQ delta')
    axs[1].axhline(0, color='black', lw=.5)
    fig.tight_layout()
    fig.savefig(out(c) / 'crest_intervention.png', dpi=160)
    plt.close(fig)

    md = ['# 061 crest intervention at fixed LUFS — MusicCaps5521 (051 baseline audio)',
          '', 'Reference arm is a pure scalar to base LUFS %+g dB; every arm holds that loudness.'
          % c['transform']['pad_db'], '',
          '| arm | achieved crest delta dB | PQ delta | CE delta | CU delta | PC delta |',
          '|---|---:|---:|---:|---:|---:|']
    for a in names:
        md.append('| %s | %+.3f | ' % (a, result['achieved_crest'][a]['mean'])
                  + ' | '.join('%+.4f [%+.4f, %+.4f]' % (result['arm_deltas'][a][ax]['mean'],
                                                         *result['arm_deltas'][a][ax]['ci95']) for ax in
                               ('PQ', 'CE', 'CU', 'PC')) + ' |')
    md += ['', '```json', json.dumps(result['primary'], indent=2), '```', '',
           'Dose-response slopes, covariates and full intervals: summary.json.',
           '![Analysis](crest_intervention.png)']
    (out(c) / 'report.md').write_text('\n'.join(md) + '\n')

    result.update({'contract_sha256': binding(), 'protocol': c['protocol'], 'label': c['label'],
                   'transform': c['transform'],
                   'artifacts_sha256': {str(p.relative_to(out(c))): digest(p) for p in
                                        [table, out(c) / 'crest_intervention.png', out(c) / 'report.md']},
                   'decision': 'analysis_complete_no_automatic_promotion'})
    atomic(out(c) / 'summary.json', result)
    validate_all(c)


def validate_all(c):
    v = checked(out(c) / 'summary.json', c)
    if v['n_total'] != c['protocol']['rows'] or v['protocol'] != c['protocol'] or v['label'] != c['label']:
        raise ValueError('summary identity')
    if v['n'] < c['analysis']['min_eligible_rows']:
        raise ValueError(f'eligible rows {v["n"]} below the registered floor')
    for rel, h in v['artifacts_sha256'].items():
        if digest(out(c) / rel) != h:
            raise ValueError('report artifact drift')
    manifest = checked(out(c) / 'transform_manifest.json', c)
    ids = {r.id for r in records(c)}
    for arm in arms(c):
        if set(manifest['arm_sha256'][arm]) != ids:
            raise ValueError('transform manifest incomplete: ' + arm)
        if {p.stem for p in (out(c) / 'items' / arm).glob('*.json')} != ids:
            raise ValueError('AES records incomplete: ' + arm)
    print(f"PASS: {len(ids)} clips x {len(arms(c))} arms = {len(ids) * len(arms(c))} AES records; "
          f"{v['n']} eligible ({v['excluded']['n']} excluded); primary decision: {v['primary']['decision']}")


def main():
    c = config()
    if '--validate-only' in sys.argv:
        validate_all(c)
        return
    if '--phase' in sys.argv:
        phase = sys.argv[sys.argv.index('--phase') + 1]
        if phase == 'transform':
            transform(c)
        elif phase == 'report':
            report(c)
        else:
            score_arm(c, phase)
        return
    import subprocess
    for phase in ['transform', *arms(c), 'report']:
        capacity(c)
        done = subprocess.run([sys.executable, __file__, '--phase', phase], cwd=ROOT)
        if done.returncode:
            raise SystemExit(done.returncode if done.returncode > 0 else 128 - done.returncode)
    validate_all(c)


if __name__ == '__main__':
    main()
