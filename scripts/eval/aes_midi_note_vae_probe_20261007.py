#!/usr/bin/env python
"""006 AES synthetic-note x VAE round-trip probe (design: docs/experiments/aes_midi_note_vae_probe_20261007.md).

Groups: S single SoundFont notes, C controls (sine / silence / noise), M clean MIDI melodies
reused from the 10-01 Codex extended run, J Jamendo anchor from audit stage1.
Every clip -> pk (peak 0.95), vae (VAE+BigVGAN round trip), and -30 LUFS versions of both.

  --preflight      check pinned inputs and disk; exit 0
  (default)        build, score, summarise (skips when summary.json exists)
  --validate-only  check summary.json completeness and gates
"""
import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path('/home/kojiek/MeanAudio')
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/aes_midi_note_vae_probe_20261007')
ASSETS = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/aes_midi_assets')
EXT = Path('/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs/aes-extended-20261001')
STAGE1 = Path('/mnt/seagate/aes_audit/stage1')
SEED = 20261007
SR, N = 16_000, 160_000
SYN_SR = 44_100

PINS = {
    ROOT / 'scripts/eval/eval_metrics.py': '47406ee5bf30c837733a00be306e813d8c27301a2a8ea30de9f8f28b1dfec67d',
    ASSETS / 'soundfonts/GeneralUser-GS.sf2': '9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe',
    ASSETS / 'soundfonts/FluidR3_GM.sf2': '74594e8f4250680adf590507a306655a299935343583256f3b722c48a1bc1cb0',
    EXT / 'midi_items.json': 'b207f7a43665668502cfb23174239b982c7ee83744fa31b4a125f31a02613f37',
    STAGE1 / 'scores.tsv': '372fafba225dff1ac1d1d82134af1065b77e221811e9708c1d383b92facb67d4',
    ROOT / 'research/eval/aes_audit/aes_ref_vs_gen_audit.py': '6549c040a5827ecbff52d0c5d3f7d994d1b76063072b484408844243a3e0b11f',
    ROOT / 'research/eval/aes_audit/aes_audit_stage2.py': '73e411e933778e97f870301780c094ba5be71d529cd7db9b23abda36c736be2a',
}
BANKS = ['GeneralUser-GS', 'FluidR3_GM']
PROGRAMS = {0: 'piano', 24: 'nylon_guitar', 32: 'acoustic_bass', 40: 'violin', 73: 'flute', 89: 'warm_pad'}
PITCHES = [36, 48, 60, 72, 84]
ENVS = {'held': (0.5, 9.5), 'short': (0.5, 1.0)}
N_MELODY, N_JAM = 256, 64
VERSIONS = ['pk', 'vae', 'pk_lvl30', 'vae_lvl30']
REF = {'musiccaps_real': 6.90, 'jamendo': 7.55, 'jamendo_vae': 7.06, 'jamendo_vae_lvl30': 7.33,
       'arm081_hqlq_lvl30': 8.20, 'all_zero': 6.735}

sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/eval'), str(ROOT / 'research/eval/aes_audit'),
                str(ASSETS / 'midi-packages')]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def preflight():
    for p, want in PINS.items():
        if not p.exists() or sha(p) != want:
            sys.exit(f'pin mismatch: {p}')
    if shutil.disk_usage(OUT.parent).free < 10e9:
        sys.exit('disk < 10 GB free')
    import tinysoundfont, torchaudio, pyloudnorm, librosa  # noqa: F401
    print('preflight ok')


def peak_norm(x):
    m = np.abs(x).max()
    return x if m == 0 else (x / m * 0.95).astype(np.float32)


def to16k(x, sr):
    import torch
    import torchaudio
    return torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(x, np.float32)), sr, SR).numpy()


def render_note(bank, program, pitch, env):
    import tinysoundfont
    s = tinysoundfont.Synth(gain=-6, samplerate=SYN_SR)   # fresh synth per note: no voice carry-over
    sid = s.sfload(str(ASSETS / 'soundfonts' / f'{bank}.sf2'))
    s.program_select(0, sid, 0, program)
    on, off = ENVS[env]
    gen = lambda n: np.frombuffer(s.generate_simple(n), dtype=np.float32).reshape(-1, 2)
    a = gen(round(on * SYN_SR))
    s.noteon(0, pitch, 100)
    b = gen(round((off - on) * SYN_SR))
    s.noteoff(0, pitch)
    c = gen(SYN_SR * 10 - len(a) - len(b))
    x = np.concatenate([a, b, c]).mean(1)
    assert np.abs(x[:round(0.49 * SYN_SR)]).max() < 1e-8, 'leak before note-on'
    return to16k(x, SYN_SR)[:N]


def sine(pitch, env):
    on, off = ENVS[env]
    t = np.arange(N) / SR
    x = np.sin(2 * np.pi * 440 * 2 ** ((pitch - 69) / 12) * t)
    g = np.zeros(N)
    i0, i1, f = round(on * SR), round(off * SR), round(0.01 * SR)
    g[i0:i1] = 1
    g[i0:i0 + f] = np.linspace(0, 1, f)
    g[i1 - f:i1] = np.linspace(1, 0, f)
    return (x * g).astype(np.float32)


def noise(color):
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(N)
    if color == 'pink':
        spec = np.fft.rfft(x)
        spec /= np.sqrt(np.maximum(np.fft.rfftfreq(N, 1 / SR), 20))
        spec[0] = 0
        x = np.fft.irfft(spec, n=N)
    return x.astype(np.float32)


def build():
    items = []   # (id, group, meta, pk array)
    for bank in BANKS:
        for prog, name in PROGRAMS.items():
            for pitch in PITCHES:
                for env in ENVS:
                    x = render_note(bank, prog, pitch, env)
                    assert np.sqrt(np.mean(x ** 2)) > 1e-5, (bank, prog, pitch, env)
                    items.append((f'S__{bank}__{name}__p{pitch}__{env}', 'S',
                                  dict(bank=bank, instrument=name, pitch=pitch, env=env), peak_norm(x)))
    for pitch in PITCHES:
        for env in ENVS:
            items.append((f'C__sine__p{pitch}__{env}', 'C', dict(instrument='sine', pitch=pitch, env=env),
                          peak_norm(sine(pitch, env))))
    items.append(('C__silence', 'C', dict(instrument='silence'), np.zeros(N, np.float32)))
    for color in ('white', 'pink'):
        items.append((f'C__{color}_noise', 'C', dict(instrument=f'{color}_noise'), peak_norm(noise(color))))

    clean = sorted((r for r in json.loads((EXT / 'midi_items.json').read_text())
                    if r['domain'] == 'midi' and r['case'].endswith('_clean')), key=lambda r: r['id'])
    for r in random.Random(SEED).sample(clean, N_MELODY):
        x, sr = sf.read(r['path'], dtype='float32', always_2d=True)
        assert sr == SR and sha(r['path']) == r['sha256'], r['id']
        x = x.mean(1)[:N]
        items.append((f'M__{r["id"]}', 'M', dict(bank=r['bank'], instrument=r.get('instrument'),
                                                  template=r['template'], case=r['case']),
                      peak_norm(np.pad(x, (0, N - len(x))))))

    jam = sorted((STAGE1 / 'jam/audio').glob('*.flac'))
    for p in random.Random(SEED).sample(jam, N_JAM):
        x, sr = sf.read(p, dtype='float32')
        assert sr == SR
        items.append((f'J__{p.stem}', 'J', dict(stage1_id=p.stem), x[:N]))   # stage1 already peak-normed
    return items


def write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, SR, subtype='PCM_24')


def run():
    if (OUT / 'summary.json').exists():
        print('summary exists, skip')
        return
    if (OUT / 'audio').exists():
        shutil.rmtree(OUT / 'audio')    # atomic job: never top up a partial run
    import torch
    from aes_ref_vs_gen_audit import RoundTrip, features
    from aes_audit_stage2 import gain_one
    from eval_metrics import score_aes

    items = build()
    print(f'built {len(items)} clips', flush=True)
    rt = RoundTrip()
    arrays = {}
    for i in range(0, len(items), 32):
        chunk = items[i:i + 32]
        vae = rt([x for _, _, _, x in chunk])
        for (cid, _, _, x), v in zip(chunk, vae):
            arrays[(cid, 'pk')] = x
            arrays[(cid, 'vae')] = v
    del rt
    torch.cuda.empty_cache()

    paths, lvl_jobs = {}, []
    for (cid, ver), x in arrays.items():
        p = OUT / 'audio' / ver / f'{cid}.flac'
        write(p, x)
        paths[(cid, ver)] = p
        q = OUT / 'audio' / f'{ver}_lvl30' / f'{cid}.flac'
        lvl_jobs.append(((cid, f'{ver}_lvl30'), (str(p), str(q))))
        paths[(cid, f'{ver}_lvl30')] = q
    with Pool(16) as pool:
        lvl = dict(zip([k for k, _ in lvl_jobs], pool.map(gain_one, [j for _, j in lvl_jobs])))
    with Pool(24) as pool:
        keys = list(arrays)
        feats = dict(zip(keys, pool.map(features, [arrays[k] for k in keys], chunksize=8)))

    aes, failed = score_aes([str(p) for p in paths.values()])
    assert not failed, failed

    meta = {cid: (g, m) for cid, g, m, _ in items}
    rows = []
    for (cid, ver), p in paths.items():
        g, m = meta[cid]
        lufs_src, gain_db, status = lvl.get((cid, ver), (None, 0.0, 'native'))
        f = feats.get((cid, ver), {})
        rows.append({'id': cid, 'group': g, 'version': ver, **{k: m.get(k) for k in
                     ('bank', 'instrument', 'pitch', 'env', 'case', 'stage1_id')},
                     **aes[str(p)], 'lvl_gain_db': gain_db, 'lvl_status': status, **f})
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / 'scores.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    summarise(rows)


def boot_ci(v, rng, B=10_000):
    v = np.asarray(v, float)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def describe(v, rng):
    v = np.asarray(v, float)
    return {'n': len(v), 'mean': float(v.mean()), 'ci95': boot_ci(v, rng), 'median': float(np.median(v)),
            'p10': float(np.percentile(v, 10)), 'p90': float(np.percentile(v, 90)),
            'min': float(v.min()), 'max': float(v.max()),
            **{f'frac_gt_{k}': float((v > REF[k]).mean()) for k in ('musiccaps_real', 'jamendo', 'arm081_hqlq_lvl30')}}


def summarise(rows):
    rng = np.random.default_rng(SEED)
    by = {}
    for r in rows:
        by.setdefault((r['group'], r['version']), []).append(r)
    pq = lambda g, v, f=lambda r: True: [r['PQ'] for r in by[(g, v)] if f(r)]
    s = {'reference': REF, 'groups': {}}
    for (g, v), rs in sorted(by.items()):
        s['groups'][f'{g}_{v}'] = {k: describe([r[k] for r in rs], rng) for k in ('PQ', 'CE', 'CU', 'PC')}

    # paired VAE deltas
    s['delta_vae'] = {}
    for g in ('S', 'M', 'J'):
        for lv in ('', '_lvl30'):
            a = {r['id']: r['PQ'] for r in by[(g, 'pk' + lv)]}
            b = {r['id']: r['PQ'] for r in by[(g, 'vae' + lv)]}
            s['delta_vae'][f'{g}{lv}'] = describe([b[k] - a[k] for k in a], rng)

    # strata for single notes
    s['S_strata'] = {}
    for key in ('instrument', 'pitch', 'env', 'bank'):
        for ver in ('pk', 'vae'):
            vals = {}
            for r in by[('S', ver)]:
                vals.setdefault(str(r[key]), []).append(r['PQ'])
            s['S_strata'][f'{key}_{ver}'] = {k: {'n': len(v), 'mean': float(np.mean(v)), 'median': float(np.median(v))}
                                            for k, v in sorted(vals.items())}
    s['controls'] = {r['id']: {v: next(x['PQ'] for x in by[('C', v)] if x['id'] == r['id']) for v in VERSIONS}
                     for r in by[('C', 'pk')]}

    # gates
    st1 = {}
    with open(STAGE1 / 'scores.tsv') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['set'] in ('jam', 'jam_vae'):
                st1[(r['set'], r['id'])] = float(r['PQ'])
    g2 = {}
    for ver, sset in (('pk', 'jam'), ('vae', 'jam_vae')):
        d = [abs(r['PQ'] - st1[(sset, r['id'][3:])]) for r in by[('J', ver)]]
        g2[ver] = float(np.mean(d))
    sil = s['controls']['C__silence']['pk']
    finite = all(math.isfinite(r[k]) for r in rows for k in ('PQ', 'CE', 'CU', 'PC'))
    counts = {k: len(v) for k, v in by.items()}
    expect = {'S': 120, 'C': 13, 'M': N_MELODY, 'J': N_JAM}
    s['gates'] = {
        'G1_silence_pq': sil, 'G1_pass': abs(sil - REF['all_zero']) <= 0.01,
        'G2_jam_pk_mean_abs': g2['pk'], 'G2_jam_vae_mean_abs': g2['vae'],
        'G2_pass': g2['pk'] <= 0.03 and g2['vae'] <= 0.05,
        'G3_pass': finite and all(counts.get((g, v)) == n for g, n in expect.items() for v in VERSIONS),
    }
    s['gates']['all_pass'] = all(s['gates'][k] for k in ('G1_pass', 'G2_pass', 'G3_pass'))

    # pre-registered readouts
    S = s['groups']['S_pk']['PQ']
    Mv = s['groups']['M_vae']['PQ']
    sine = [r['PQ'] for r in by[('C', 'pk')] if r['instrument'] == 'sine']
    s['readouts'] = {
        'R1_single_note_beats_corpus': S['median'] >= REF['jamendo'] and S['ci95'][0] > REF['musiccaps_real'],
        'R1_S_pk_median': S['median'], 'R1_S_pk_mean_ci': S['ci95'],
        'R2_M_vae_mean': Mv['mean'], 'R2_M_vae_ci': Mv['ci95'],
        'R2_verdict': ('content_dependent_ceiling' if Mv['ci95'][0] > REF['jamendo_vae'] + 0.30 else
                       'vae_caps' if Mv['ci95'][1] < REF['jamendo_vae'] + 0.30 else 'inconclusive'),
        'R3_sine_pk_median': float(np.median(sine)),
        'R3_instrument_note_pk_median': S['median'],
    }
    (OUT / 'summary.json').write_text(json.dumps(s, indent=1))
    print(json.dumps({'gates': s['gates'], 'readouts': s['readouts']}, indent=1))


def validate():
    s = json.loads((OUT / 'summary.json').read_text())
    for g in ('S', 'C', 'M', 'J'):
        for v in VERSIONS:
            assert f'{g}_{v}' in s['groups'], (g, v)
    assert s['gates']['G3_pass'], 'incomplete / non-finite'
    print('valid; gates', s['gates'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    if a.preflight:
        preflight()
    elif a.validate_only:
        validate()
    else:
        run()
