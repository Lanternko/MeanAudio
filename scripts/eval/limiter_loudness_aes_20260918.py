#!/usr/bin/env python3
"""064: raise loudness past the original through a peak limiter, then score AES + CLAP.

063 could only go louder than the original on a 4.7% headroom subset, because the
generated clips sit within ~1 dB of full scale. A lookahead peak limiter makes every
clip louder without clipping, at the cost of changing the waveform (crest drops).

To keep that cost separable from the loudness gain, every limited arm X has a twin Xm:
the *same limited waveform* scaled back to the source clip's integrated LUFS. Then
  X  - z0 : total effect of "make it louder with a limiter"
  X  - Xm : pure level effect (same waveform, scalar only; like 063 but above original)
  Xm - z0 : limiter processing effect at matched loudness
and the two parts sum exactly to the total.

Limiter (2026-09-18 revision): x42 dpl.lv2's Peaklim (Fons Adriaensen's DPL) in
true-peak mode, via the vendored offline CLI in external_limiters. The in-house
limiter below (`limit`) was the first choice; it is kept as one of the 064b robustness
limiters, and its partial run is in *_superseded_ownlimiter.

Arm families
  L<g>  fixed pre-gain +g dB into the limiter (every clip pushed by the same amount)
  T<n>  per-clip gain toward integrated -n LUFS, then limited; iterated so the limited
        output lands on the target (quiet clips get more gain, loud clips less)

Reuses the 051 hash-verified baseline FLAC, like 063. Audio is transient.
"""
from __future__ import annotations
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from gain_ladder_aes_20260918 import (AXES, SR, atomic, content_digest_array, digest, lufs,  # noqa: E402
                                      mean_ci, meter, read_audio, signal_stats)

SRC_ROOT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/loudness_aes_cfg3_20260911')
CFG = {
    'experiment': '064 limiter-loudness AES',
    'source_audio_root': str(SRC_ROOT / '_audio/baseline'),
    'source_manifest': str(SRC_ROOT / 'audio_manifest.json'),
    'tsv': str(SRC_ROOT / 'musiccaps5521_cfg3_fidelity8.tsv'),
    'aes_snapshot': '/home/kojiek/.cache/huggingface/hub/models--facebook--audiobox-aesthetics/'
                    'snapshots/9b1dd8e5df9af7216e836a98974fe3b82c56ded6',
    'clap_checkpoint': str(ROOT / 'weights/music_speech_audioset_epoch_15_esc_89.98.pt'),
    'gain_ladder_063_z0': '/home/kojiek/nvme_experiment_artifacts/meanaudio/gain_ladder_aes_20260918/items/z0',
    'out': '/home/kojiek/nvme_experiment_artifacts/meanaudio/limiter_loudness_aes_20260918',
    'rows': 5521,
    'batch': 16,
    'limiter': {'name': 'x42-dpl Peaklim, true-peak mode', 'ceiling_dbfs': -1.0, 'release_s': 0.05},
    # dpl's 4x TP filter is designed for 44.1/48 kHz; at 16 kHz typical overs are +0.2..+0.44
    # dBTP, and a T12 clip pinned at the +24 dB cap (13 dB mean gain reduction) reached +1.07
    'true_peak_max_dbtp': 2.0,
    'own_limiter': {'ceiling_dbfs': -1.0, 'lookahead_ms': 5.0, 'release_ms': 50.0},
    'target_tolerance_lu': 0.1,
    'target_max_iter': 8,
    'target_max_gain_db': 24.0,
    'matched_tolerance_lu': 0.05,
    'bootstrap_seed': 20260918,
    'bootstrap_replicates': 10000,
    'replication_tolerance': 1e-4,
}
FIXED = (3, 6, 9, 12)
TARGETS = (16, 14, 12)


def arm_list():
    names = ['z0']
    names += [f'L{g}' for g in FIXED] + [f'L{g}m' for g in FIXED]
    names += [f'T{t}' for t in TARGETS] + [f'T{t}m' for t in TARGETS]
    return names


# ------------------------------------------------------------------ limiter

def _release(m, coef):
    import numba

    @numba.njit(cache=True)
    def run(m, coef):
        g = np.empty_like(m)
        prev = 1.0
        for i in range(m.shape[0]):
            rec = prev + (1.0 - prev) * (1.0 - coef)
            v = m[i] if m[i] < rec else rec
            g[i] = v
            prev = v
        return g
    return run(m, coef)


def limit(y, ceiling_dbfs, lookahead_ms, release_ms):
    """Lookahead brickwall peak limiter; output |y| <= ceiling by construction.

    r[n] is the gain each sample needs. A forward min over the lookahead window,
    then a backward box average of the same length, gives a gain that is <= r[n] at
    every sample (each averaged term is a min over a window that contains n), so the
    ceiling is never exceeded and the gain ramps down smoothly ahead of each peak.
    Release is an exponential recovery applied before the box average, and it only
    ever lowers the gain, so it cannot break the bound.
    """
    from scipy.ndimage import minimum_filter1d
    ceil = 10 ** (ceiling_dbfs / 20)
    L = max(1, int(round(lookahead_ms * SR / 1000)))
    a = tp_envelope(y)
    r = np.where(a > ceil, ceil / np.maximum(a, 1e-30), 1.0)
    # forward-looking min over r[n .. n+L-1]
    pad = np.concatenate([r, np.ones(L)])
    m = minimum_filter1d(pad, size=L, origin=-(L // 2))[:len(r)]
    coef = math.exp(-1.0 / (release_ms * SR / 1000))
    m = _release(m, coef)
    # backward box average over m[n-L+1 .. n]; the lead-in pads with m[0], the min over
    # r[0 .. L-1], so the bound also holds for the first L samples (padding with 1 did not)
    c = np.concatenate([np.full(L - 1, m[0]), m])
    cs = np.cumsum(np.concatenate([[0.0], c]))
    g = (cs[L:] - cs[:-L]) / L
    out = y * g
    if np.max(np.abs(out)) > ceil * (1 + 1e-9):
        raise ValueError('limiter exceeded ceiling')
    return out, g


def tp_envelope(y):
    """Per-sample peak including inter-sample overs (4x oversampled, BS.1770 style).

    Detecting on sample values alone let reconstructed peaks reach +0.5 dBTP, and
    CLAP resamples to 48 kHz, where those overs become real sample values above full
    scale. Each sample takes the max of |x| over the oversampled points around it.
    """
    from scipy.signal import resample_poly
    u = np.abs(resample_poly(y, 4, 1))[:4 * len(y)]
    u = np.concatenate([u, np.zeros(4 * len(y) - len(u))])
    w = u.reshape(len(y), 4).max(1)
    prev = np.concatenate([[0.0], w[:-1]])
    return np.maximum(np.abs(y), np.maximum(w, prev))


def apply_limiter(y):
    import external_limiters as E
    lim = CFG['limiter']
    return E.dpl(y, lim['ceiling_dbfs'])


def reduction(pre, post):
    """Gain reduction in dB where the input has signal (|pre| above -60 dBFS)."""
    m = np.abs(pre) > 1e-3
    if not m.any():
        return {'gr_max_db': 0.0, 'gr_mean_db': 0.0}
    gr = -20 * np.log10(np.clip(np.abs(post[m]) / np.abs(pre[m]), 1e-6, None))
    return {'gr_max_db': float(np.percentile(gr, 99.9)), 'gr_mean_db': float(gr.mean())}


def true_peak_dbfs(x):
    from scipy.signal import resample_poly
    tp = float(np.max(np.abs(resample_poly(x, 4, 1))))
    return 20 * math.log10(tp) if tp > 0 else None


def hit_target(limiter, x, met, src_lufs, target):
    """Pre-gain that puts the limited output on `target` LUFS, by bisection.

    Output LUFS is monotone in pre-gain but its slope collapses under heavy limiting,
    so a fixed-point update (pre += target - out) overshoots and cycles. At
    pre = target - src the limiter can only remove loudness, so that is a lower
    bracket; the cap is the upper one. A clip that cannot reach the target even at the
    cap is kept at the cap and flagged.
    """
    tol = CFG['target_tolerance_lu']
    lo, hi = target - src_lufs, CFG['target_max_gain_db']
    if lo >= hi:
        y = limiter(x * 10 ** (hi / 20))
        return y, hi, abs(lufs(met, y) - target) <= tol
    y_hi = limiter(x * 10 ** (hi / 20))
    if lufs(met, y_hi) < target - tol:
        return y_hi, hi, False
    for _ in range(40):
        mid = (lo + hi) / 2
        y = limiter(x * 10 ** (mid / 20))
        lv = lufs(met, y)
        if abs(lv - target) <= tol:
            return y, mid, True
        lo, hi = (mid, hi) if lv < target else (lo, mid)
    return y, mid, False


def render_clip(x, met, src_lufs):
    """All arms for one clip. Returns {arm: (waveform, info)}."""
    out = {'z0': (x, {'pre_gain_db': 0.0})}
    for g in FIXED:
        y = apply_limiter(x * 10 ** (g / 20))
        out[f'L{g}'] = (y, {'pre_gain_db': float(g), **reduction(x * 10 ** (g / 20), y)})
    for t in TARGETS:
        target = -float(t)
        y, pre, hit = hit_target(apply_limiter, x, met, src_lufs, target)
        out[f'T{t}'] = (y, {'pre_gain_db': float(pre), 'target_lufs': target, 'target_hit': hit,
                            **reduction(x * 10 ** (pre / 20), y)})
    for name in [f'L{g}' for g in FIXED] + [f'T{t}' for t in TARGETS]:
        y, info = out[name]
        # Scalar moves LUFS 1:1 except where the -70 LUFS absolute gate admits or drops
        # blocks (near-silent clips), so iterate a few times.
        back = 0.0
        for _ in range(6):
            back += src_lufs - lufs(met, y * 10 ** (back / 20))
            z = y * 10 ** (back / 20)
            if abs(lufs(met, z) - src_lufs) <= CFG['matched_tolerance_lu']:
                break
        if abs(lufs(met, z) - src_lufs) > CFG['matched_tolerance_lu']:
            raise ValueError('matched twin missed source LUFS')
        out[name + 'm'] = (z, {'twin_of': name, 'scalar_db': float(back)})
    return out


# ------------------------------------------------------------------ scoring

def records():
    from score_musiccaps_per_item import read_musiccaps_tsv
    return read_musiccaps_tsv(Path(CFG['tsv']), expected_count=CFG['rows'])


def score_all():
    import soundfile as sf
    from score_musiccaps_per_item import load_aes_predictor, _aes_batch, load_clap_model, _clap_batch
    out = Path(CFG['out'])
    hashes = json.loads(Path(CFG['source_manifest']).read_text())['audio_sha256']
    rec = records()
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
    ceil = CFG['limiter']['ceiling_dbfs']
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
            rendered = render_clip(x, met, base['lufs'])
            per = {}
            for arm, (y, info) in rendered.items():
                st = signal_stats(y, met)
                st['true_peak_dbfs'] = true_peak_dbfs(y)
                # Limited arms must sit under the ceiling. Twins are the same waveform at the
                # source loudness, so they only have to stay out of clipping.
                limited = arm != 'z0' and not arm.endswith('m')
                if limited and (st['peak_dbfs'] > ceil + 0.01 or st['true_peak_dbfs'] > CFG['true_peak_max_dbtp']):
                    raise ValueError(f'{r.id}/{arm}: peak above ceiling')
                if arm.endswith('m') and st['peak_dbfs'] >= 0:
                    raise ValueError(f'{r.id}/{arm}: loudness-matched twin clips')
                p = scratch / f'{r.id}__{arm}.wav'
                sf.write(p, y.astype(np.float32), SR, format='WAV', subtype='FLOAT')
                per[arm] = {'path': p, 'signal': st, 'info': info, 'content_sha256': content_digest_array(y)}
            staged.append((r, base, per))
        # One AES call per arm over the same 16 clips: this is exactly 063's grouping, so
        # z0 can be compared to 063 per clip.
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


# ------------------------------------------------------------------ analysis

def analyze():
    out = Path(CFG['out'])
    ids = sorted(r.id for r in records())
    rows = {i: json.loads((out / 'items' / (i + '.json')).read_text()) for i in ids}
    if len(rows) != CFG['rows']:
        raise ValueError('incomplete')
    arms = arm_list()
    metrics = (*AXES, 'clap')
    rng = np.random.default_rng(CFG['bootstrap_seed'])
    reps = CFG['bootstrap_replicates']

    def val(i, a, m):
        v = rows[i]['arms'][a]
        return v['clap'] if m == 'clap' else v['aes'][m]

    # Replication gate: z0 must equal 063's z0 per clip (same audio, same scorer).
    ref = Path(CFG['gain_ladder_063_z0'])
    worst = 0.0
    for i in ids:
        o = json.loads((ref / (i + '.json')).read_text())
        for m in metrics:
            worst = max(worst, abs(val(i, 'z0', m) - (o['clap'] if m == 'clap' else o['aes'][m])))
    gate = {'max_abs_diff_vs_063_z0': worst, 'pass': worst <= CFG['replication_tolerance']}

    levels, cov = {}, {}
    for a in arms:
        levels[a] = {m: float(np.mean([val(i, a, m) for i in ids])) for m in metrics}
        sig = [rows[i]['arms'][a]['signal'] for i in ids]
        cov[a] = {k: float(np.mean([s[k] for s in sig if s[k] is not None]))
                  for k in ('lufs', 'peak_dbfs', 'true_peak_dbfs', 'crest_db', 'clipped_fraction')}
        info = [rows[i]['arms'][a]['info'] for i in ids]
        for k in ('pre_gain_db', 'gr_max_db', 'gr_mean_db'):
            if k in info[0]:
                cov[a][k] = float(np.mean([d[k] for d in info]))
        if 'target_hit' in info[0]:
            cov[a]['target_hit_rate'] = float(np.mean([d['target_hit'] for d in info]))

    decomp = {}
    for base in [f'L{g}' for g in FIXED] + [f'T{t}' for t in TARGETS]:
        twin = base + 'm'
        decomp[base] = {}
        for m in metrics:
            tot = [val(i, base, m) - val(i, 'z0', m) for i in ids]
            lev = [val(i, base, m) - val(i, twin, m) for i in ids]
            prc = [val(i, twin, m) - val(i, 'z0', m) for i in ids]
            decomp[base][m] = {'total': mean_ci(tot, rng, reps), 'level': mean_ci(lev, rng, reps),
                               'processing': mean_ci(prc, rng, reps)}
        dl = [rows[i]['arms'][base]['signal']['lufs'] - rows[i]['arms']['z0']['signal']['lufs'] for i in ids]
        decomp[base]['delta_lufs'] = float(np.mean(dl))
        # per-LU level slope (same waveform, scalar only)
        dpq = np.array([val(i, base, 'PQ') - val(i, twin, 'PQ') for i in ids])
        dcl = np.array([val(i, base, 'clap') - val(i, twin, 'clap') for i in ids])
        dl = np.array(dl)
        decomp[base]['level_slope_per_lu'] = {'PQ': float((dl * dpq).sum() / (dl * dl).sum()),
                                              'clap': float((dl * dcl).sum() / (dl * dl).sum())}

    # T arms: quiet vs loud source halves (the "quiet gets more gain" design)
    split = float(np.median([rows[i]['baseline']['lufs'] for i in ids]))
    halves = {}
    for t in TARGETS:
        a = f'T{t}'
        halves[a] = {}
        for name, sel in (('quiet_half', [i for i in ids if rows[i]['baseline']['lufs'] < split]),
                          ('loud_half', [i for i in ids if rows[i]['baseline']['lufs'] >= split])):
            halves[a][name] = {'n': len(sel),
                               'pre_gain_db': float(np.mean([rows[i]['arms'][a]['info']['pre_gain_db'] for i in sel])),
                               **{m: {'total': float(np.mean([val(i, a, m) - val(i, 'z0', m) for i in sel])),
                                      'level': float(np.mean([val(i, a, m) - val(i, a + 'm', m) for i in sel])),
                                      'processing': float(np.mean([val(i, a + 'm', m) - val(i, 'z0', m) for i in sel]))}
                                  for m in ('PQ', 'CE', 'clap')}}

    result = {'config': CFG, 'n': len(ids), 'replication_gate': gate, 'arm_levels': levels,
              'covariates': cov, 'decomposition_vs_z0': decomp,
              'target_arms_by_source_loudness': {'split_lufs': split, **halves}}
    atomic(out / 'summary.json', result)

    with (out / 'per_clip.csv').open('w', newline='') as f:
        cols = ['id', 'arm', 'lufs', 'peak_dbfs', 'true_peak_dbfs', 'crest_db', 'pre_gain_db', *AXES, 'clap']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in ids:
            for a in arms:
                v = rows[i]['arms'][a]
                w.writerow({'id': i, 'arm': a, **{k: v['signal'][k] for k in cols[2:6]},
                            'pre_gain_db': v['info'].get('pre_gain_db'), **v['aes'], 'clap': v['clap']})
    return result


def main():
    if '--analyze-only' not in sys.argv:
        score_all()
    r = analyze()
    print(json.dumps({'gate': r['replication_gate'],
                      'levels': {a: {k: round(v, 4) for k, v in r['arm_levels'][a].items()}
                                 for a in arm_list()},
                      'lufs': {a: round(r['covariates'][a]['lufs'], 2) for a in arm_list()}}, indent=1))


if __name__ == '__main__':
    main()
