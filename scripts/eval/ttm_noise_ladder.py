#!/usr/bin/env python3
"""TTM plan table D, objective half: white-noise SNR ladder on paired clips (plan §表 D).

  ~/venvs/dac/bin/python scripts/eval/ttm_noise_ladder.py {build,score,analyze,all}

Sources (audio as scored by p2 115): musiccaps_real, meanaudio_cfg3_neg (s14159265),
musicgen_medium, musicldm. 100 base ids, the same for every source: ids whose 115 lvl30
gain status is `ok` in all four, sorted, rng(20261012) draw without replacement.

Per base clip: downmix to mono, resample to 16 kHz (soxr VHQ) BEFORE adding noise, then
clean + SNR 40/30/20/10/0 dB. One unit-power white Gaussian sequence per id (seeded by the
id, shared by every source and every SNR), scaled so mean(x^2)/mean(n^2) over the whole
clip equals the SNR. A pure-noise anchor (`whitenoise`) uses the same sequence.

Blocks:
  raw    one scalar gain per ladder (min(1, 0.999/max peak over its 6 files)), so the signal
         level is constant inside a ladder; anchor at the real clip's RMS.
  lvl30  every file gained on its own to -30 LUFS (BS.1770), peak rule as in 115.

Scoring reuses eval_metrics.py (CLAP batch 1 + AES) and meva_score_dir.py unchanged; ids
are `{source}__{cond}__{id}` in one directory per block. No listeners here: the human
columns of table D stay empty.
"""
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path('/home/kojiek/MeanAudio')
SRC115 = Path('/home/kojiek/eval_output_nvme/ttm_metrics_115_20261009')
OUT = Path('/home/kojiek/eval_output_nvme/ttm_noise_ladder_20261012')
TSV = '/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv'
PY = '/home/kojiek/venvs/dac/bin/python'
MEVA_PY = str(REPO / 'runtime/meva_20261003/venv/bin/python')
SOURCES = ('musiccaps_real', 'meanaudio_cfg3_neg', 'musicgen_medium', 'musicldm')
SNRS = (40, 30, 20, 10, 0)
CONDS = ('clean',) + tuple(f'snr{s}' for s in SNRS)
BLOCKS = ('lvl30', 'raw')
N_BASE, SEED, SR, PEAK_CAP, TARGET = 100, 20261012, 16000, 0.999, -30.0
AES = ('PQ', 'CE', 'CU', 'PC')
N_BOOT = 10000


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def source_dirs():
    sel = json.loads((SRC115 / 'selection.json').read_text())
    return {s['name']: Path(s.get('audio') or s['source']) for s in sel['systems'] if s['name'] in SOURCES}


def ok_ids(name):
    with open(SRC115 / name / 'lvl30' / 'gain_status.tsv') as f:
        return {r['id'] for r in csv.DictReader(f, delimiter='\t') if r['status'] == 'ok'}


def base_ids():
    common = sorted(set.intersection(*(ok_ids(s) for s in SOURCES)))
    pick = np.random.default_rng(SEED).choice(len(common), N_BASE, replace=False)
    return [common[i] for i in sorted(pick)], len(common)


def captions(ids):
    want, out = set(ids), {}
    with open(TSV, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['id'] in want:
                out[r['id']] = r['caption']
    assert len(out) == len(want)
    return out


def unit_noise(cid, n):
    seed = int(hashlib.sha256(cid.encode()).hexdigest()[:16], 16)
    z = np.random.default_rng(seed).standard_normal(n)
    return z / math.sqrt(float(np.mean(z ** 2)))


def lvl30_gain(x):
    import pyloudnorm
    lufs = float(pyloudnorm.Meter(SR).integrated_loudness(x))
    g = 10 ** ((TARGET - lufs) / 20)
    peak = float(np.max(np.abs(x)))
    capped = peak * g > PEAK_CAP
    return (PEAK_CAP / peak if capped else g), lufs, capped


def write(block, key, x):
    sf.write(OUT / block / 'audio' / f'{key}.flac', x, SR, format='FLAC', subtype='PCM_24')


def build():
    import soxr
    ids, n_common = base_ids()
    dirs = source_dirs()
    for b in BLOCKS:
        (OUT / b / 'audio').mkdir(parents=True, exist_ok=True)
    rows, real_rms = [], {}
    for name in SOURCES:
        for cid in ids:
            wav, sr = sf.read(dirs[name] / f'{cid}.flac', dtype='float64', always_2d=True)
            x = wav.mean(axis=1)
            if sr != SR:
                x = soxr.resample(x, sr, SR, quality='VHQ')
            p_sig = float(np.mean(x ** 2))
            z = unit_noise(cid, len(x))
            ladder = {'clean': x}
            for s in SNRS:
                ladder[f'snr{s}'] = x + z * math.sqrt(p_sig / 10 ** (s / 10))
            g_raw = min(1.0, PEAK_CAP / max(float(np.max(np.abs(v))) for v in ladder.values()))
            if name == 'musiccaps_real':
                real_rms[cid] = math.sqrt(p_sig) * g_raw
            for cond, v in ladder.items():
                key = f'{name}__{cond}__{cid}'
                write('raw', key, v * g_raw)
                g30, lufs, capped = lvl30_gain(v)
                write('lvl30', key, v * g30)
                rows.append({'key': key, 'source': name, 'cond': cond, 'id': cid, 'src_sr': sr,
                             'n_samples': len(x), 'raw_gain_db': 20 * math.log10(g_raw),
                             'mix_lufs': lufs, 'lvl30_gain_db': 20 * math.log10(g30),
                             'lvl30_peak_capped': int(capped),
                             'mix_peak': float(np.max(np.abs(v)))})
        log(f'BUILD {name}')
    for cid in ids:
        key = f'whitenoise__noise__{cid}'
        z = unit_noise(cid, SR * 10)
        raw = z * real_rms[cid]
        g_raw = min(1.0, PEAK_CAP / float(np.max(np.abs(raw))))
        write('raw', key, raw * g_raw)
        g30, lufs, capped = lvl30_gain(z * 0.01)
        write('lvl30', key, z * 0.01 * g30)
        rows.append({'key': key, 'source': 'whitenoise', 'cond': 'noise', 'id': cid, 'src_sr': SR,
                     'n_samples': len(z), 'raw_gain_db': 20 * math.log10(g_raw), 'mix_lufs': '',
                     'lvl30_gain_db': '', 'lvl30_peak_capped': int(capped), 'mix_peak': ''})
    # realised SNR from the written raw files (PCM_24 round trip)
    snr = {}
    for r in rows:
        if r['cond'].startswith('snr'):
            c, _ = sf.read(OUT / 'raw/audio' / f'{r["source"]}__clean__{r["id"]}.flac', dtype='float64')
            m, _ = sf.read(OUT / 'raw/audio' / f'{r["key"]}.flac', dtype='float64')
            snr[r['key']] = 10 * math.log10(float(np.mean(c ** 2)) / float(np.mean((m - c) ** 2)))
    worst = max(abs(snr[r['key']] - int(r['cond'][3:])) for r in rows if r['key'] in snr)
    if worst > 0.05:
        raise SystemExit(f'realised SNR off by {worst:.3f} dB')
    cap = captions(ids)
    with open(OUT / 'manifest.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0]) + ['snr_measured_db'], delimiter='\t')
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'snr_measured_db': snr.get(r['key'], '')})
    with open(OUT / 'ladder.tsv', 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['id', 'caption'])
        for r in sorted(rows, key=lambda r: (r['source'], r['cond'], r['id'])):
            w.writerow([r['key'], cap[r['id']]])
    (OUT / 'build.json').write_text(json.dumps({
        'n_common_ok_ids': n_common, 'n_base': N_BASE, 'seed': SEED, 'base_ids': ids,
        'sources': {k: str(v) for k, v in dirs.items()}, 'resample': 'soxr VHQ to 16 kHz, mono mean',
        'noise': 'unit-power white Gaussian, rng seeded by sha256(id)[:16], shared across sources and SNRs',
        'snr_window': 'whole clip, mean power', 'max_abs_snr_error_db': worst, 'n_files_per_block': len(rows),
        'lvl30_peak_capped': sum(r['lvl30_peak_capped'] for r in rows), 'format': 'FLAC PCM_24 16 kHz'}, indent=1))
    log(f'BUILD done: {len(rows)} files per block, max SNR error {worst:.4f} dB')


def score():
    n = sum(1 for _ in open(OUT / 'manifest.tsv')) - 1
    for b in BLOCKS:
        d = OUT / b
        mj = d / 'scores' / f'ladder_{b}' / 'metrics.json'
        if not mj.is_file():
            log(f'METRICS {b}')
            with open(d / 'metrics.log', 'w') as g:
                subprocess.run([PY, str(REPO / 'scripts/eval/eval_metrics.py'), '--gen_dir', str(d / 'audio'),
                                '--tsv', str(OUT / 'ladder.tsv'), '--exp_name', f'ladder_{b}',
                                '--out_dir', str(d / 'scores')], cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
        m = json.loads(mj.read_text())
        if not (m['n_present'] == n and m['metrics']['aes_n'] == n and m['metrics']['clap_n'] == n):
            raise SystemExit(f'{b}: metrics incomplete')
        ej = d / 'meva' / 'meva.json'
        if not ej.is_file():
            log(f'MEVA {b}')
            (d / 'meva').mkdir(exist_ok=True)
            with open(d / 'meva.log', 'w') as g:
                subprocess.run([MEVA_PY, str(REPO / 'scripts/eval/meva_score_dir.py'), '--audio_dir', str(d / 'audio'),
                                '--out_dir', str(d / 'meva')], cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
        if json.loads(ej.read_text())['n'] != n:
            raise SystemExit(f'{b}: MEva incomplete')
    log('SCORE done')


def ci(v, rng):
    v = np.asarray(v, float)
    boot = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(axis=1)
    return [float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def analyze(out_json):
    man = {r['key']: r for r in csv.DictReader(open(OUT / 'manifest.tsv'), delimiter='\t')}
    res = {'protocol': json.loads((OUT / 'build.json').read_text()), 'n_boot': N_BOOT, 'blocks': {}}
    res['protocol'].pop('base_ids')
    for b in BLOCKS:
        sc = {}
        for r in csv.DictReader(open(OUT / b / 'scores' / f'ladder_{b}' / 'per_clip.tsv'), delimiter='\t'):
            sc[r['id']] = {k: float(r[k]) for k in AES + ('clap', 'lufs')}
        for r in csv.DictReader(open(OUT / b / 'meva' / 'meva_per_clip.tsv'), delimiter='\t'):
            sc[r['id']]['MEva'] = float(r['meva_raw'])
        rng = np.random.default_rng(42)
        blk = {}
        for name in SOURCES:
            ids = sorted({m['id'] for m in man.values() if m['source'] == name})
            blk[name] = {}
            for cond in CONDS:
                cell = {'n': len(ids)}
                keys = [f'{name}__{cond}__{i}' for i in ids]
                for k in AES + ('MEva', 'clap', 'lufs'):
                    cell[k] = float(np.mean([sc[x][k] for x in keys]))
                if cond != 'clean':
                    cell['snr_measured_db'] = float(np.mean([float(man[x]['snr_measured_db']) for x in keys]))
                    for k in AES + ('MEva', 'clap'):
                        d = [sc[x][k] - sc[f'{name}__clean__{i}'][k] for x, i in zip(keys, ids)]
                        cell[f'd_{k}'] = ci(d, rng)
                        cell[f'clean_higher_{k}'] = float(np.mean(np.asarray(d) < 0))
                blk[name][cond] = cell
        keys = [k for k, m in man.items() if m['source'] == 'whitenoise']
        blk['whitenoise'] = {'noise': {'n': len(keys), **{k: ci([sc[x][k] for x in keys], rng)
                                                           for k in AES + ('MEva', 'clap', 'lufs')}}}
        res['blocks'][b] = blk
    Path(out_json).write_text(json.dumps(res, indent=1))
    for b in BLOCKS:
        print(f'\n== {b} ==')
        for name in SOURCES:
            for cond in CONDS:
                c = res['blocks'][b][name][cond]
                tail = '' if cond == 'clean' else (
                    f"  dPQ {c['d_PQ'][0]:+.3f} [{c['d_PQ'][1]:+.3f},{c['d_PQ'][2]:+.3f}] ({c['clean_higher_PQ']:.0%})"
                    f"  dCE {c['d_CE'][0]:+.3f}  dMEva {c['d_MEva'][0]:+.3f} [{c['d_MEva'][1]:+.3f},{c['d_MEva'][2]:+.3f}]"
                    f" ({c['clean_higher_MEva']:.0%})  dCLAP {c['d_clap'][0]:+.4f}")
                print(f"{name:20s} {cond:6s} PQ {c['PQ']:.3f} CE {c['CE']:.3f} MEva {c['MEva']:.3f} CLAP {c['clap']:.4f}{tail}")
        w = res['blocks'][b]['whitenoise']['noise']
        print(f"whitenoise           PQ {w['PQ'][0]:.3f} CE {w['CE'][0]:.3f} CU {w['CU'][0]:.3f} PC {w['PC'][0]:.3f} "
              f"MEva {w['MEva'][0]:.3f} CLAP {w['clap'][0]:.4f} LUFS {w['lufs'][0]:.2f}")


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage in ('build', 'all'):
        build()
    if stage in ('score', 'all'):
        score()
    if stage in ('analyze', 'all'):
        analyze(sys.argv[2] if len(sys.argv) > 2 else str(REPO / 'docs/experiments/results/ttm_noise_ladder_20261012.json'))
