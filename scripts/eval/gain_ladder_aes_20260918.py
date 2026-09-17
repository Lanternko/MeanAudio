#!/usr/bin/env python3
"""063: absolute-level ladder on the 051 canonical baseline audio.

051 measured attenuation only (0/-3/-6/-9 dB) and found PQ rising as the file gets
quieter. That leaves the amplification direction and the shape of the curve untested:
a monotone preference for quiet and an inverted-U with an interior optimum both fit
those four points. This walks a 3 dB ladder from -18 dB up to the original level, so
every step from the reference IS an amplification, and asks where PQ peaks.

Pure scalar gain: one multiplication per arm, no time-varying processing, no
per-sample nonlinearity. Crest is invariant under a scalar (peak and RMS scale
together), so unlike 061 this moves absolute level and nothing else.

Amplification never breaches full scale because the ladder tops out at the original
file. One secondary arm goes above it, restricted to clips with real headroom.

Reuses the 051 hash-verified baseline FLAC; generates no audio from a checkpoint.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path('/home/kojiek/MeanAudio')
CONTRACT = ROOT / 'docs/experiments/gain_ladder_aes_20260918_contract.json'
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
    return c['ladder']['arms']


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


def source_hashes(c):
    manifest = json.loads(Path(c['source_manifest']).read_text())
    return manifest['audio_sha256']


# ---------------------------------------------------------------- signal layer

def meter():
    import pyloudnorm as pyln
    return pyln.Meter(SR, filter_class='K-weighting', block_size=.4)


def lufs(met, x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        v = float(met.integrated_loudness(np.asarray(x, dtype=np.float64)))
    return v if math.isfinite(v) else None


def read_audio(p):
    import soundfile as sf
    x, sr = sf.read(p, dtype='float64')
    if sr != SR or x.ndim != 1 or not np.isfinite(x).all() or len(x) < 6400:
        raise ValueError(f'invalid mono16k audio: {p}')
    return x


def signal_stats(x, met):
    x = np.asarray(x, dtype=np.float64)
    rms = math.sqrt(float(np.mean(x * x)))
    peak = float(np.max(np.abs(x)))
    return {
        'lufs': lufs(met, x),
        'rms_dbfs': 20 * math.log10(rms) if rms > 0 else None,
        'peak_dbfs': 20 * math.log10(peak) if peak > 0 else None,
        'crest_db': 20 * math.log10(peak / rms) if rms > 0 and peak > 0 else None,
        'clipped_fraction': float(np.mean(np.abs(x) >= .999)),
        'silence_fraction': float(np.mean(np.abs(x) < 1e-3)),
    }


def content_digest_array(x, sr=SR):
    """Hash the decoded samples, not the file.

    libsndfile stamps a PEAK chunk carrying a Unix timestamp into float WAV headers,
    so identical audio hashes differently on every write. Hashing the samples makes a
    render-and-drop cycle verifiable against the record written the first time.
    """
    h = hashlib.sha256()
    h.update(str(sr).encode())
    h.update(np.ascontiguousarray(np.asarray(x, dtype=np.float32)).tobytes())
    return h.hexdigest()


def render(x, gain_db):
    return np.asarray(x, dtype=np.float64) * (10.0 ** (gain_db / 20.0))


def eligible_for(c, arm, base):
    """Per-arm admissibility, decided from the source clip alone.

    The ladder arms are admissible for every non-degenerate clip. The above-baseline
    arm is admissible only where the source has enough headroom that the amplified
    peak still clears the ceiling, so amplification is never measured through
    clipping distortion.
    """
    spec = arms(c)[arm]
    if base['peak_dbfs'] is None or base['rms_dbfs'] is None:
        return False, 'degenerate waveform'
    need = spec['gain_db'] + c['ladder']['peak_headroom_margin_db']
    if spec['gain_db'] > 0 and base['peak_dbfs'] + need > c['ladder']['peak_ceiling_dbfs']:
        return False, 'insufficient headroom for an above-baseline arm'
    return True, None


# ------------------------------------------------------------------- rendering

def score_arm(c, arm):
    """Render, hash, score and drop one arm, in item-sized batches.

    Audio is transient by design: a scalar-gain arm is exactly rebuildable from the
    source plus gain_db, so nothing is gained by keeping 28 GB of WAV on NVMe. The
    content hash recorded per item still lets a rebuild be checked against this run.
    """
    import soundfile as sf
    from score_musiccaps_per_item import load_aes_predictor, _aes_batch, load_clap_model, _clap_batch

    gain_db = arms(c)[arm]['gain_db']
    hashes = source_hashes(c)
    rec = records(c)
    dest = out(c) / 'items' / arm
    dest.mkdir(parents=True, exist_ok=True)
    todo = []
    for r in rec:
        p = dest / (r.id + '.json')
        if p.exists():
            v = checked(p, c)
            if v['arm'] != arm or v['source_sha256'] != hashes[r.id]:
                raise ValueError('stale score record: ' + r.id)
        else:
            todo.append(r)
    if not todo:
        print(f'{arm}: already complete', flush=True)
        return

    batch_size = c['protocol']['scoring_batch_size']
    scratch = Path(c['storage']['transient_root']) / arm
    scratch.mkdir(parents=True, exist_ok=True)
    predictor = load_aes_predictor(Path(c['aes_snapshot']), device='cuda', batch_size=batch_size)
    clap = load_clap_model(Path(c['clap_checkpoint']), device='cuda', local_files_only=True)
    print(f'{arm}: {len(todo)} clips at {gain_db:+g} dB', flush=True)
    done = 0
    for start in range(0, len(todo), batch_size):
        capacity(c)
        batch = todo[start:start + batch_size]
        staged = []
        for r in batch:
            src = Path(c['source_audio_root']) / (r.id + '.flac')
            if digest(src) != hashes[r.id]:
                raise ValueError('source audio drift: ' + r.id)
            x = read_audio(src)
            met = meter()
            base = signal_stats(x, met)
            ok, why = eligible_for(c, arm, base)
            y = render(x, gain_db)
            stats = signal_stats(y, met)
            # A scalar moves the peak by exactly the gain: that is the correctness check
            # every arm must pass. The absolute ceiling only constrains arms that amplify
            # past the source, because two of the 5,521 sources are themselves above it
            # (max -0.073 dBFS) and re-encoding an unamplified source cannot clip.
            if abs((stats['peak_dbfs'] - base['peak_dbfs']) - gain_db) > 1e-6:
                raise ValueError(f'{r.id}: scalar gain did not move the peak by {gain_db:+g} dB')
            if gain_db > 0 and ok and stats['peak_dbfs'] > c['ladder']['peak_ceiling_dbfs']:
                raise ValueError(f'{r.id}: admissible amplifying arm breached the peak ceiling')
            path = scratch / (r.id + '.wav')
            sf.write(path, y.astype(np.float32), SR, format='WAV', subtype='FLOAT')
            staged.append((r, path, base, stats, ok, why, content_digest_array(y)))

        aes = _aes_batch(predictor, [s[1] for s in staged])
        # CLAP one file at a time: batch size shifts CLAP scores and can reorder arms
        # (reference_clap_batch_size_sensitivity), so per-file is the only comparable form.
        clap_values = [_clap_batch(clap, [s[1]], [s[0].caption])[0] for s in staged]

        for (r, path, base, stats, ok, why, content), a, cl in zip(staged, aes, clap_values):
            if set(a) != set(AXES) or not all(math.isfinite(float(a[k])) for k in AXES):
                raise ValueError('invalid AES: ' + r.id)
            if not math.isfinite(float(cl)):
                raise ValueError('invalid CLAP: ' + r.id)
            atomic(dest / (r.id + '.json'), {
                'contract_sha256': binding(), 'id': r.id, 'arm': arm, 'gain_db': gain_db,
                'source_sha256': hashes[r.id], 'content_sha256': content,
                'baseline': base, 'signal': stats, 'arm_eligible': bool(ok),
                'ineligible_reason': why, 'aes': a, 'clap': float(cl)})
            path.unlink()
        done += len(batch)
        atomic(out(c) / 'progress.json', {'phase': 'score_' + arm, 'count': done,
                                          'contract_sha256': binding()})
        if done % (batch_size * 20) == 0:
            print(f'  {arm}: {done}/{len(todo)}', flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    if {p.stem for p in dest.glob('*.json')} != {r.id for r in rec}:
        raise ValueError('score ID mismatch: ' + arm)
    print(f'{arm}: {len(rec)} records complete', flush=True)


def load_scores(c):
    ids = [r.id for r in records(c)]
    values = {}
    for arm in arms(c):
        d = out(c) / 'items' / arm
        if {p.stem for p in d.glob('*.json')} != set(ids):
            raise ValueError('missing records for arm ' + arm)
        values[arm] = {i: checked(d / (i + '.json'), c) for i in ids}
    return values


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
    """Paired per-clip slope through the origin, bootstrapped over clips."""
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    keep = np.abs(dx) > 1e-9
    dx, dy = dx[keep], dy[keep]
    if len(dx) < 50:
        return {'n': int(len(dx)), 'slope': None, 'ci95': None}
    b = []
    for start in range(0, reps, 100):
        ix = rng.integers(0, len(dx), size=(min(100, reps - start), len(dx)))
        b.extend((dx[ix] * dy[ix]).sum(1) / (dx[ix] * dx[ix]).sum(1))
    return {'n': int(len(dx)), 'slope': float((dx * dy).sum() / (dx * dx).sum()),
            'ci95': np.quantile(b, [.025, .975]).tolist()}


def ladder_arms(c):
    """The ladder proper, quietest first. Above-baseline arms are secondary."""
    names = [a for a, s in arms(c).items() if s['gain_db'] <= 0]
    return sorted(names, key=lambda a: arms(c)[a]['gain_db'])


def shape(steps, rng_tol):
    """Classify the ladder's PQ profile from adjacent-step contrasts.

    Each step is the paired delta from one rung to the next louder rung, so a
    monotone preference for quiet makes every step negative.
    """
    signs = []
    for s in steps:
        lo, hi = s['ci95']
        signs.append(0 if lo <= 0 <= hi or abs(s['mean']) < rng_tol else (1 if s['mean'] > 0 else -1))
    if all(v <= 0 for v in signs) and any(v < 0 for v in signs):
        return 'monotone_quieter_scores_higher', signs
    if all(v >= 0 for v in signs) and any(v > 0 for v in signs):
        return 'monotone_louder_scores_higher', signs
    if all(v == 0 for v in signs):
        return 'flat_within_ci', signs
    up = [k for k, v in enumerate(signs) if v > 0]
    down = [k for k, v in enumerate(signs) if v < 0]
    if up and down and max(up) < min(down):
        return 'interior_optimum', signs
    return 'non_monotone_unclassified', signs


def analyze(c, scores):
    a = c['analysis']
    rng = np.random.default_rng(a['bootstrap_seed'])
    reps = a['bootstrap_replicates']
    every = sorted(scores[a['reference_arm']])
    ladder = ladder_arms(c)
    ref = a['reference_arm']
    ids = [i for i in every if all(scores[arm][i]['arm_eligible'] for arm in ladder)]
    metrics = (*AXES, 'clap')

    def value(arm, i, m):
        return float(scores[arm][i]['clap'] if m == 'clap' else scores[arm][i]['aes'][m])

    result = {
        'n_total': len(every), 'n': len(ids), 'reference_arm': ref,
        'ladder': [{'arm': arm, 'gain_db': arms(c)[arm]['gain_db']} for arm in ladder],
        'excluded': {'n': len(every) - len(ids),
                     'ids': [i for i in every if i not in set(ids)],
                     'rule': a['exclusion_rule']},
        'arm_deltas': {}, 'arm_levels': {}, 'adjacent_steps': {}, 'covariates': {},
        'inference': 'Paired within-clip deltas against the quietest rung; pointwise 95% bootstrap '
                     'CI over clips. Every contrast against the reference is an AMPLIFICATION. '
                     'Absolute CLAP here is not comparable to the canonical 48 kHz tables: it is '
                     'scored on the retained 16 kHz baseline audio, so only its response to gain is read.'}

    for arm in arms(c):
        sub = [i for i in every if scores[arm][i]['arm_eligible']] if arms(c)[arm]['gain_db'] > 0 else ids
        result['arm_levels'][arm] = {m: mean_ci([value(arm, i, m) for i in sub], rng, reps) for m in metrics}
        result['covariates'][arm] = {k: mean_ci([scores[arm][i]['signal'][k] for i in sub
                                                 if scores[arm][i]['signal'][k] is not None], rng, reps)
                                     for k in ('lufs', 'peak_dbfs', 'rms_dbfs', 'crest_db',
                                               'clipped_fraction')}
        if arm == ref:
            continue
        base = ref if arms(c)[arm]['gain_db'] <= 0 else a['above_baseline_reference_arm']
        result['arm_deltas'][arm] = {
            'against': base, 'n': len(sub), 'gain_delta_db': arms(c)[arm]['gain_db'] - arms(c)[base]['gain_db'],
            **{m: mean_ci([value(arm, i, m) - value(base, i, m) for i in sub], rng, reps) for m in metrics}}

    for lower, upper in zip(ladder, ladder[1:]):
        step = arms(c)[upper]['gain_db'] - arms(c)[lower]['gain_db']
        result['adjacent_steps'][f'{lower}->{upper}'] = {
            'gain_step_db': step,
            **{m: mean_ci([value(upper, i, m) - value(lower, i, m) for i in ids], rng, reps) for m in metrics}}

    pooled = {}
    for m in metrics:
        dx, dy = [], []
        for arm in ladder[1:]:
            g = arms(c)[arm]['gain_db'] - arms(c)[ref]['gain_db']
            dx.extend([g] * len(ids))
            dy.extend([value(arm, i, m) - value(ref, i, m) for i in ids])
        pooled[m] = slope_ci(dx, dy, rng, reps)

    steps = [result['adjacent_steps'][k]['PQ'] for k in
             [f'{lo}->{up}' for lo, up in zip(ladder, ladder[1:])]]
    verdict, signs = shape(steps, a['step_null_tolerance'])
    # 051 measured its gain arms against the original level, so the replication check is
    # read against that rung, not against this ladder's quietest reference.
    top = a['replication_reference_arm']
    replication = {}
    for arm, expected in a['replication_targets'].items():
        obs = mean_ci([value(arm, i, 'PQ') - value(top, i, 'PQ') for i in ids], rng, reps)
        replication[arm] = {'expected_pq_delta_vs_051': expected, 'observed': obs['mean'],
                            'ci95': obs['ci95'], 'against': top,
                            'within_tolerance': abs(obs['mean'] - expected) <= a['replication_tolerance']}

    result['primary'] = {
        'endpoint': 'pooled paired slope dPQ per dB of amplification from the quietest rung, '
                    'over the ladder arms',
        'observed_slope': pooled['PQ'],
        'attenuation_side_slope_051': a['slope_051_pq_per_db'],
        'secondary_slopes': {m: pooled[m] for m in metrics if m != 'PQ'},
        'ladder_shape': verdict,
        'adjacent_step_signs': signs,
        'replication_of_051': replication,
        'decision': decide(verdict, pooled['PQ'], replication, a),
    }
    return result


def decide(verdict, pooled, replication, a):
    if not all(v['within_tolerance'] for v in replication.values()):
        return ('replication_failed: the rungs that reproduce 051 fall outside the registered '
                'tolerance, so suspect a bug in this run before reading any shape')
    if pooled['slope'] is None:
        return 'insufficient_dose'
    if verdict == 'monotone_quieter_scores_higher':
        return ('loudness_is_a_monotone_lever: amplification lowers PQ across the whole registered '
                'range; arms must be loudness-matched before any PQ comparison')
    if verdict == 'interior_optimum':
        return ('interior_optimum: PQ peaks at a preferred absolute level, so PQ partly measures '
                'distance from that level rather than audio quality')
    if verdict == 'flat_within_ci':
        return ('no_level_response_outside_051_range: the 051 effect does not extend; re-examine '
                'before interpreting')
    if verdict == 'monotone_louder_scores_higher':
        return 'direction_reversed_versus_051: unregistered direction; treat as exploratory'
    return 'non_monotone_unclassified: exploratory'


def report(c):
    scores = load_scores(c)
    result = analyze(c, scores)
    ids = sorted(scores[c['analysis']['reference_arm']])

    table = out(c) / 'per_clip.csv'
    with table.open('w', newline='') as f:
        cols = ['id', 'arm', 'gain_db', 'arm_eligible', 'lufs', 'rms_dbfs', 'peak_dbfs', 'crest_db',
                'clipped_fraction', *AXES, 'clap']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm in arms(c):
            for i in ids:
                v = scores[arm][i]
                w.writerow({'id': i, 'arm': arm, 'gain_db': v['gain_db'],
                            'arm_eligible': int(bool(v['arm_eligible'])),
                            **{k: v['signal'][k] for k in cols[4:9]},
                            **v['aes'], 'clap': v['clap']})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ladder = ladder_arms(c)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    x = [arms(c)[arm]['gain_db'] for arm in ladder]
    for m, style in (('PQ', 'o-'), ('CE', 's--'), ('CU', '^--'), ('PC', 'v--')):
        y = [result['arm_levels'][arm][m]['mean'] for arm in ladder]
        axs[0].plot(x, [v - y[0] for v in y], style, label=m)
    axs[0].set(xlabel='absolute level (dB relative to the original file)',
               ylabel='AES change from the quietest rung')
    axs[0].axhline(0, color='black', lw=.5)
    axs[0].legend()
    y = [result['arm_levels'][arm]['clap']['mean'] for arm in ladder]
    axs[1].plot(x, y, 'o-')
    axs[1].set(xlabel='absolute level (dB relative to the original file)', ylabel='CLAP (per-file)')
    fig.tight_layout()
    fig.savefig(out(c) / 'gain_ladder.png', dpi=160)
    plt.close(fig)

    md = ['# 063 absolute-level ladder — MusicCaps5521 (051 baseline audio)', '',
          'Pure scalar gain; the reference is the quietest rung, so every contrast is an '
          'amplification. Crest is invariant under a scalar.', '',
          '| arm | level dB | PQ delta vs ref | CE delta | CU delta | PC delta | CLAP delta |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for arm in sorted(arms(c), key=lambda n: arms(c)[n]['gain_db']):
        if arm == c['analysis']['reference_arm']:
            continue
        d = result['arm_deltas'][arm]
        md.append('| %s | %+g | ' % (arm, arms(c)[arm]['gain_db'])
                  + ' | '.join('%+.4f [%+.4f, %+.4f]' % (d[m]['mean'], *d[m]['ci95'])
                               for m in ('PQ', 'CE', 'CU', 'PC', 'clap')) + ' |')
    md += ['', '```json', json.dumps(result['primary'], indent=2), '```', '',
           'Adjacent steps, per-arm levels, covariates and full intervals: summary.json.',
           '![Analysis](gain_ladder.png)']
    (out(c) / 'report.md').write_text('\n'.join(md) + '\n')

    result.update({'contract_sha256': binding(), 'protocol': c['protocol'], 'label': c['label'],
                   'ladder_config': c['ladder'],
                   'artifacts_sha256': {str(p.relative_to(out(c))): digest(p) for p in
                                        [table, out(c) / 'gain_ladder.png', out(c) / 'report.md']},
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
    ids = {r.id for r in records(c)}
    for arm in arms(c):
        if {p.stem for p in (out(c) / 'items' / arm).glob('*.json')} != ids:
            raise ValueError('records incomplete: ' + arm)
    print(f"PASS: {len(ids)} clips x {len(arms(c))} arms = {len(ids) * len(arms(c))} scored conditions; "
          f"{v['n']} eligible ({v['excluded']['n']} excluded); shape: {v['primary']['ladder_shape']}; "
          f"decision: {v['primary']['decision']}")


def main():
    c = config()
    if '--validate-only' in sys.argv:
        validate_all(c)
        return
    if '--phase' in sys.argv:
        phase = sys.argv[sys.argv.index('--phase') + 1]
        if phase == 'report':
            report(c)
        else:
            score_arm(c, phase)
        return
    capacity(c)
    for arm in sorted(arms(c), key=lambda a: arms(c)[a]['gain_db']):
        score_arm(c, arm)
    report(c)


if __name__ == '__main__':
    main()
