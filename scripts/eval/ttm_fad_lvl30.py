#!/usr/bin/env python3
"""FAD at lvl30 for the TTM tables (2026-10-13 probe; direct run, not a queue job).

p2 115 computed FAD on raw only because its reference set was raw real audio and VGGish
log-mel features depend on level. Here the reference is levelled too (the lvl30 block of
musiccaps_real that 115 already built), so generated and reference audio sit at -30 LUFS.

Same extractor as eval_metrics.score_fad (frechet_audio_distance VGGish, no PCA, no
activation, 16 kHz mono); embeddings are cached per clip so sets can be regrouped.

Parts
  systems   9 generated cells, raw and lvl30, on eval_metrics' own 2,048-row seed-42 subset
            (raw is the replication check against the published numbers).
  ladder    white-noise ladder (4 sources x 6 conditions x 100 prompts + white-noise anchor),
            lvl30, against the full lvl30 reference (5,131 clips).
  midi      Codex extended round (19,531 clips, levelled on the fly as in p2 119) and the
            first-round karaoke clips (already levelled on disk), same reference.
  floor     real lvl30 subsets of n clips against the full real lvl30 set: what FAD reads
            for "the same distribution" at that n.

FAD is set-level and biased upward at small n, so the intervention rows are read as
(set after intervention) - (its clean baseline set) at equal n, with a clip-paired bootstrap.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')  # the log-mel front end runs in the worker pool, one thread each

import argparse
import collections
import csv
import json
import math
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(REPO / 'scripts/eval'))
EV = Path('/home/kojiek/eval_output_nvme')
OUT = EV / 'ttm_fad_lvl30_20261013'
REPORT = REPO / 'docs/experiments/results/ttm_fad_lvl30_20261013.json'
TSV = '/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv'
R115, R118 = EV / 'ttm_metrics_115_20261009', EV / 'ttm_metrics_seeds_118_20261011'
MA = 'phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full_s{}_mc_mf25_{}'
SR, TARGET, PEAK_CAP, N_BOOT = 16000, -30.0, 0.999, 300
PROGRAM = {0: 'piano', 24: 'nylon_guitar', 32: 'acoustic_bass', 40: 'violin', 73: 'flute', 89: 'warm_pad'}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def systems():
    out = [{'name': n, 'raw': EV / f'ttm_external_20261008/{n}/audio', 'lvl30': R115 / n / 'lvl30/audio'}
           for n in ('musicgen_medium', 'musicldm', 'musicldm_fid8')]
    for cell in ('cfg0', 'cfg3_neg'):
        out.append({'name': f'meanaudio_s14159265_{cell}', 'raw': EV / MA.format(14159265, cell) / 'audio',
                    'lvl30': R115 / f'meanaudio_{cell}' / 'lvl30/audio'})
        for seed in (27182818, 16180339):
            out.append({'name': f'meanaudio_s{seed}_{cell}', 'raw': EV / MA.format(seed, cell) / 'audio',
                        'lvl30': R118 / f'meanaudio_s{seed}_{cell}' / 'lvl30/audio'})
    return out


def load_one(job):
    """(key, path, level) -> (key, VGGish log-mel examples (frames, 1, 96, 64)); level=True applies the p2 119 gain rule."""
    key, path, level = job
    x, sr = sf.read(path, dtype='float64', always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR, quality='VHQ')
    if level:
        import pyloudnorm
        lufs, peak = float(pyloudnorm.Meter(SR).integrated_loudness(x)), float(np.max(np.abs(x)))
        if not np.isfinite(lufs) or peak <= 0:
            raise RuntimeError(f'{key}: unmeasurable loudness')
        g = 10 ** ((TARGET - lufs) / 20)
        x = x * (PEAK_CAP / peak if peak * g > PEAK_CAP else g)
    from torchvggish import vggish_input
    return key, vggish_input.waveform_to_examples(x.astype(np.float32), SR, True).detach().numpy()


class Embedder:
    def __init__(self):
        from frechet_audio_distance import FrechetAudioDistance
        self.fad = FrechetAudioDistance(use_pca=False, use_activation=False, verbose=False)
        self.fad.model.preprocess = False  # examples arrive already as log-mel from load_one
        self.pool = mp.Pool(10)

    def run(self, name, jobs):
        """jobs: [(key, path, level)] -> {key: (frames, 128) float32}; cached in OUT/emb/<name>.npz."""
        import torch
        cache = OUT / 'emb' / f'{name}.npz'
        if cache.is_file():
            z = np.load(cache)
            if set(z.files) == {k for k, _, _ in jobs}:
                return {k: z[k] for k in z.files}
        t0, emb = time.time(), {}
        with torch.no_grad():
            for i, (key, x) in enumerate(self.pool.imap(load_one, jobs, chunksize=8)):
                emb[key] = self.fad.model.forward(torch.from_numpy(x)).detach().cpu().numpy().astype(np.float32)
                if (i + 1) % 2000 == 0:
                    log(f'    {name}: {i + 1}/{len(jobs)}, {time.time() - t0:.0f}s')
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **emb)
        log(f'  emb {name}: {len(emb)} clips, {time.time() - t0:.0f}s')
        return emb


def stats(frames):
    return frames.mean(axis=0), np.cov(frames, rowvar=False)


def frechet(a, b):
    from scipy import linalg
    (m1, s1), (m2, s2) = a, b
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    d = m1 - m2
    return float(d.dot(d) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean.real))


def cat(emb, keys):
    return np.concatenate([emb[k] for k in keys])


def fad_of(emb, keys, ref):
    return frechet(stats(cat(emb, keys)), ref)


def paired_delta(emb, pairs, ref, rng):
    """pairs: [(treated key, baseline key)]; FAD(treated) - FAD(baseline), clip-paired bootstrap."""
    t, b = [p[0] for p in pairs], [p[1] for p in pairs]
    ft, fb = fad_of(emb, t, ref), fad_of(emb, b, ref)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(pairs), len(pairs))
        boots.append(fad_of(emb, [t[i] for i in idx], ref) - fad_of(emb, [b[i] for i in idx], ref))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'n': len(pairs), 'frames': int(sum(len(emb[k]) for k in t)), 'fad': ft, 'fad_base': fb,
            'd_fad': [ft - fb, float(lo), float(hi)]}


def part_systems(E, res):
    from eval_metrics import load_rows
    rows = load_rows(TSV)
    random.seed(42)
    sampled = [i for i, _ in random.sample(rows, 2048)]
    ref_raw, ref_lvl = R115 / 'musiccaps_real/raw_audio', R115 / 'musiccaps_real/lvl30/audio'
    ids = [i for i in sampled if (ref_raw / f'{i}.flac').exists()]
    log(f'systems: {len(ids)} pairs of 2048 sampled')
    full = sorted(p.stem for p in ref_lvl.glob('*.flac'))
    e_full = E.run('ref_lvl30_full', [(i, str(ref_lvl / f'{i}.flac'), False) for i in full])
    e_raw = E.run('ref_raw_sub', [(i, str(ref_raw / f'{i}.flac'), False) for i in ids])
    ref = {'raw': stats(cat(e_raw, ids)), 'lvl30': stats(cat(e_full, ids))}
    out = {'pairs': len(ids), 'cells': {}}
    for s in systems():
        rec = {}
        for block in ('raw', 'lvl30'):
            e = E.run(f'{s["name"]}_{block}', [(i, str(s[block] / f'{i}.flac'), False) for i in ids])
            rec[block] = frechet(stats(cat(e, ids)), ref[block])
        out['cells'][s['name']] = rec
        log(f'  {s["name"]}: raw {rec["raw"]:.3f}  lvl30 {rec["lvl30"]:.3f}')
    res['systems'] = out
    return e_full, full


def part_floor(e_full, full, ref, res, rng):
    out = {}
    for n in (23, 100, 128, 384, 1924):
        v = [fad_of(e_full, list(rng.choice(full, n, replace=False)), ref) for _ in range(10)]
        out[str(n)] = {'mean': float(np.mean(v)), 'min': float(min(v)), 'max': float(max(v))}
        log(f'  floor n={n}: {out[str(n)]["mean"]:.3f} [{min(v):.3f}, {max(v):.3f}]')
    res['floor_real_subset_vs_real_full'] = out


def part_ladder(E, ref, res, rng):
    d = EV / 'ttm_noise_ladder_20261012'
    man = list(csv.DictReader(open(d / 'manifest.tsv'), delimiter='\t'))
    e = E.run('ladder_lvl30', [(r['key'], str(d / 'lvl30/audio' / f'{r["key"]}.flac'), False) for r in man])
    by = collections.defaultdict(dict)
    for r in man:
        by[(r['source'], r['cond'])][r['id']] = r['key']
    out = {}
    for (src, cond), m in sorted(by.items()):
        if cond in ('clean', 'noise'):
            out[f'{src}|{cond}'] = {'n': len(m), 'fad': fad_of(e, list(m.values()), ref)}
        else:
            base = by[(src, 'clean')]
            out[f'{src}|{cond}'] = paired_delta(e, [(m[i], base[i]) for i in sorted(m)], ref, rng)
        log(f'  ladder {src}|{cond}: {json.dumps(out[f"{src}|{cond}"])}')
    res['ladder'] = out


def part_midi(E, ref, res, rng):
    from ttm_midi_extended_meva import items
    rows = items()
    e = E.run('codex_extended_lvl30', [(r['key'], r['path'], True) for r in rows])
    out, absolute = {}, collections.defaultdict(list)
    groups = collections.defaultdict(list)
    for r in rows:
        if r['set'] == 'karaoke' and r['case'] == 'clean':
            absolute[f'karaoke_clean_{r["duration"]}s'].append(r['key'])
        if r['set'] == 'midi' and r['case'].endswith('_clean'):
            absolute[f'midi_{PROGRAM[r["instrument"]]}_r{r["register"]}'].append(r['key'])
            absolute['midi_all_clean'].append(r['key'])
        if r['base']:
            tag = f'{r["duration"]}s' if r['set'] == 'karaoke' else 'all'
            groups[f'{r["set"]}|{r["family"]}|{r["condition"]}|{tag}'].append((r['key'], r['base']))
            if r['set'] == 'midi' and r['family'] == 'instrument':
                groups[f'midi|instrument|{r["condition"]}|r{r["register"]}'].append((r['key'], r['base']))
    res['extended_absolute'] = {k: {'n': len(v), 'frames': int(sum(len(e[x]) for x in v)), 'fad': fad_of(e, v, ref)}
                                for k, v in sorted(absolute.items())}
    log(f'  absolute: {json.dumps({k: round(v["fad"], 2) for k, v in res["extended_absolute"].items()})}')
    for name, pairs in sorted(groups.items()):
        out[name] = paired_delta(e, pairs, ref, rng)
        log(f'  ext {name}: n {out[name]["n"]} d {out[name]["d_fad"][0]:+.3f}')
    res['extended_paired'] = out

    d = EV / 'ttm_midi_karaoke_meva_20261012'
    keys = [r['key'] for r in csv.DictReader(open(d / 'manifest.tsv'), delimiter='\t') if r['key'].startswith('kar__')]
    e1 = E.run('karaoke_round1_lvl30', [(k, str(d / 'audio' / f'{k}.flac'), False) for k in keys])
    by = collections.defaultdict(dict)
    for k in keys:
        _, song, sec, dur, cond = k.split('__')
        by[(dur, cond)][(song, sec)] = k
    out = {}
    for (dur, cond), m in sorted(by.items()):
        if dur != '10':
            continue
        if cond == 'original':
            out['10|original'] = {'n': len(m), 'fad': fad_of(e1, list(m.values()), ref)}
        else:
            base = by[(dur, 'original')]
            out[f'10|{cond}'] = paired_delta(e1, [(m[c], base[c]) for c in sorted(m)], ref, rng)
            log(f'  kar1 {cond}: d {out[f"10|{cond}"]["d_fad"][0]:+.3f}')
    res['karaoke_round1'] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip_midi', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261013)
    res = {'document_kind': 'ttm_fad_lvl30_v1', 'written_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'protocol': {'extractor': 'frechet_audio_distance VGGish, use_pca False, use_activation False, 16 kHz mono',
                        'level': 'lvl30 = scalar gain to -30 LUFS, peak cap 0.999; reference levelled the same way',
                        'reference_systems': 'eval_metrics 2048-row seed-42 subset, rows with a real clip',
                        'reference_other': 'all real lvl30 clips (musiccaps_real/lvl30/audio)',
                        'n_boot': N_BOOT, 'bootstrap': 'clip-paired percentile, FAD(treated) - FAD(baseline)'}}
    E = Embedder()
    e_full, full = part_systems(E, res)
    ref = stats(cat(e_full, full))
    res['protocol']['reference_other_n'] = len(full)
    part_floor(e_full, full, ref, res, rng)
    part_ladder(E, ref, res, rng)
    if not args.skip_midi:
        part_midi(E, ref, res, rng)
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1))
    REPORT.write_text(json.dumps(res, indent=1))
    log(f'DONE -> {REPORT}')


if __name__ == '__main__':
    main()
