#!/usr/bin/env python3
"""MEva + AES at lvl30 on the 2026-10-01 Codex extended AES-ablation round (p2 119).

Same re-leveling as ttm_midi_karaoke_meva.py (mono, soxr VHQ -> 16 kHz, scalar gain to
-30 LUFS, peak cap 0.999, FLAC PCM_24), on the third Codex round:

  midi     64 30-s templates (32 discovery / 32 validation) x 2 SoundFonts x 149 cases = 19,072
  karaoke  12 songs, 27 contexts (10 s / 30 s) x 17 cases = 459

Work is done shard by shard (build -> AES -> MEva -> drop the shard's audio), so a pause
loses at most one shard and peak disk stays near 2 GB. Paired deltas are item minus its
own baseline; bootstrap is over templates (midi, 64) or songs (karaoke, 12), 10,000 draws.

  (no flag)        run, then write <OUT>/summary.json
  --preflight      inputs, runtime and free space
  --validate-only  every shard and the summary are complete
"""
import argparse
import collections
import csv
import json
import math
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ttm_midi_karaoke_meva import MEVA_PY, METRICS, N_BOOT, PEAK_CAP, REPO, SR, SRC, TARGET, ci, log

EXT = SRC / 'aes-extended-20261001'
OUT = Path('/home/kojiek/eval_output_nvme/ttm_midi_extended_meva_119_20261012')
N_EXPECTED, SHARD, WORKERS, MIN_FREE = 19531, 2000, 12, 20 * 2 ** 30
PROGRAM = {0: 'piano', 24: 'nylon_guitar', 32: 'acoustic_bass', 40: 'violin', 73: 'flute', 89: 'warm_pad'}


def items():
    """One row per stimulus: key, set, cluster (bootstrap unit), base (key of its baseline), tags."""
    rows = []
    for it in json.loads((EXT / 'midi_items.json').read_text()):
        pre = f'midi__{it["bank"]}__{it["template"]}__'
        rows.append({'key': it['id'], 'set': 'midi', 'path': it['path'], 'cluster': str(it['template']),
                     'bank': it['bank'], 'split': it['split'], 'family': it['family'] or '',
                     'condition': str(it['condition']) if it['family'] else '', 'case': it['case'],
                     'instrument': it['instrument'], 'register': it['register'],
                     'base': pre + it['baseline'] if it['baseline'] else ''})
    for it in json.loads((EXT / 'karaoke_items.json').read_text()):
        rows.append({'key': it['id'], 'set': 'karaoke', 'path': it['path'], 'cluster': it['song'],
                     'bank': '', 'split': 'transfer', 'family': it['family'] or '',
                     'condition': str(it['condition']) if it['family'] else '', 'case': it['case'],
                     'instrument': None, 'register': None, 'duration': it['duration'],
                     'base': f'karaoke__{it["context_id"]}__{it["baseline"]}' if it['baseline'] else ''})
    return rows


def shards(rows):
    keys = sorted(r['key'] for r in rows)
    return [keys[i:i + SHARD] for i in range(0, len(keys), SHARD)]


def build_one(job):
    import pyloudnorm
    import soxr
    key, path, dst = job
    x, sr = sf.read(path, dtype='float64', always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality='VHQ')
    lufs, peak = float(pyloudnorm.Meter(SR).integrated_loudness(x)), float(np.max(np.abs(x)))
    if not np.isfinite(lufs) or peak <= 0:
        raise RuntimeError(f'{key}: unmeasurable loudness')
    g = 10 ** ((TARGET - lufs) / 20)
    capped = peak * g > PEAK_CAP
    g = PEAK_CAP / peak if capped else g
    sf.write(dst, x * g, SR, format='FLAC', subtype='PCM_24')
    return {'key': key, 'src': path, 'src_sr': sr, 'src_lufs': lufs, 'gain_db': 20 * math.log10(g), 'capped': int(capped)}


def shard_done(d, keys):
    try:
        aes = json.loads((d / 'aes.json').read_text())
        meva = {r['id']: float(r['meva_raw']) for r in csv.DictReader(open(d / 'meva' / 'meva_per_clip.tsv'), delimiter='\t')}
        return (json.loads((d / 'meva' / 'meva.json').read_text())['n'] == len(keys) and set(aes) == set(meva) == set(keys)
                and all(math.isfinite(v) for a in aes.values() for v in a.values()) and all(map(math.isfinite, meva.values())))
    except (OSError, ValueError, KeyError):
        return False


def run_shard(i, keys, path_of):
    from eval_metrics import score_aes
    d = OUT / 'shards' / f's{i:02d}'
    if shard_done(d, keys):
        return
    for sub in ('audio', 'meva'):
        shutil.rmtree(d / sub, ignore_errors=True)
        (d / sub).mkdir(parents=True)
    with Pool(WORKERS) as pool:
        man = pool.map(build_one, [(k, path_of[k], str(d / 'audio' / f'{k}.flac')) for k in keys], chunksize=16)
    with open(d / 'manifest.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(man[0]), delimiter='\t')
        w.writeheader()
        w.writerows(man)
    per, failed = score_aes([d / 'audio' / f'{k}.flac' for k in keys])
    if failed or len(per) != len(keys):
        raise SystemExit(f'shard {i}: AES incomplete {len(per)}/{len(keys)} failed={len(failed)}')
    (d / 'aes.json').write_text(json.dumps({Path(p).stem: v for p, v in per.items()}))
    with open(d / 'meva.log', 'w') as g:
        subprocess.run([MEVA_PY, str(REPO / 'scripts/eval/meva_score_dir.py'), '--audio_dir', str(d / 'audio'),
                        '--out_dir', str(d / 'meva')], cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    if not shard_done(d, keys):
        raise SystemExit(f'shard {i}: scores incomplete')
    shutil.rmtree(d / 'audio')
    log(f'shard {i} done n={len(keys)} capped={sum(m["capped"] for m in man)}')


def scores(rows):
    sc = {}
    for i, keys in enumerate(shards(rows)):
        d = OUT / 'shards' / f's{i:02d}'
        if not shard_done(d, keys):
            raise SystemExit(f'shard {i} incomplete')
        sc.update(json.loads((d / 'aes.json').read_text()))
        for r in csv.DictReader(open(d / 'meva' / 'meva_per_clip.tsv'), delimiter='\t'):
            sc[r['id']]['meva'] = float(r['meva_raw'])
    return sc


def contexts(r):
    """Pooling contexts a paired row contributes to."""
    if r['set'] == 'karaoke':
        return [f'{r["duration"]}s']
    out = ['ALL']
    if r['family'] == 'noise':
        out += [PROGRAM[r['instrument']], f'r{r["register"]}']
    if r['family'] == 'instrument':
        out += [f'r{r["register"]}']
    return out


def analyze():
    rows, rng = items(), np.random.default_rng(20261012)
    sc = scores(rows)
    capped = sum(int(r['capped']) for d in sorted((OUT / 'shards').glob('s*'))
                 for r in csv.DictReader(open(d / 'manifest.tsv'), delimiter='\t'))
    res = {'protocol': {'level': 'lvl30', 'sr': SR, 'n_boot': N_BOOT, 'n_clips': len(sc), 'capped': capped,
                        'bootstrap_unit': {'midi': 'template (64; 32 per split)', 'karaoke': 'song (12)'}},
           'absolute': {}, 'paired': {}}
    tree = lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

    groups = tree()
    for r in rows:
        names = []
        if r['set'] == 'karaoke' and r['case'] == 'clean':
            names = [f'karaoke_clean_{r["duration"]}s']
        if r['set'] == 'midi' and r['case'].endswith('_clean'):
            inst = PROGRAM[r['instrument']]
            names = [f'midi_{inst}_r{r["register"]}', f'midi_{inst}_r{r["register"]}_{r["bank"]}', f'midi_{inst}_r{r["register"]}_{r["split"]}']
        if r['set'] == 'midi' and r['case'] == 'pad_ref':
            names = ['midi_pad_ref']
        for name in names:
            for m in METRICS:
                groups[name][m][r['cluster']].append(sc[r['key']][m])
    for name, g in sorted(groups.items()):
        res['absolute'][name] = {m: ci(g[m], rng) for m in METRICS}

    pair = tree()
    for r in rows:
        if not r['base']:
            continue
        for ctx in contexts(r):
            stem = f'{r["set"]}|{r["family"]}|{r["condition"]}|{ctx}'
            names = [stem] if r['set'] == 'karaoke' else [stem + '|all', f'{stem}|{r["split"]}'] + ([f'{stem}|{r["bank"]}'] if ctx == 'ALL' else [])
            for name in names:
                for m in METRICS:
                    pair[name][m][r['cluster']].append(sc[r['key']][m] - sc[r['base']][m])
    for name, g in sorted(pair.items()):
        res['paired'][name] = {m: ci(g[m], rng) for m in METRICS}
        res['paired'][name]['n'] = sum(len(v) for v in g['PQ'].values())
        res['paired'][name]['clusters'] = len(g['PQ'])
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1))
    log(f'ANALYZE -> {OUT / "summary.json"}')


def preflight():
    rows = items()
    keys = {r['key'] for r in rows}
    missing = [r['path'] for r in rows if not Path(r['path']).is_file()]
    bad_base = [r['key'] for r in rows if r['base'] and r['base'] not in keys]
    OUT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUT).free
    ok = (len(rows) == len(keys) == N_EXPECTED and not missing and not bad_base and free >= MIN_FREE
          and Path(MEVA_PY).is_file())
    print(json.dumps({'n': len(rows), 'missing': len(missing), 'bad_base': len(bad_base), 'free_gb': free / 2 ** 30, 'ok': ok}))
    return 0 if ok else 1


def validate():
    rows = items()
    sc = scores(rows)
    summary = json.loads((OUT / 'summary.json').read_text())
    ok = len(sc) == N_EXPECTED == summary['protocol']['n_clips'] and bool(summary['paired']) and bool(summary['absolute'])
    print(json.dumps({'n_scored': len(sc), 'paired': len(summary['paired']), 'ok': ok}))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    if a.preflight:
        return preflight()
    if a.validate_only:
        return validate()
    rows = items()
    path_of = {r['key']: r['path'] for r in rows}
    for i, keys in enumerate(shards(rows)):
        run_shard(i, keys, path_of)
    analyze()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
