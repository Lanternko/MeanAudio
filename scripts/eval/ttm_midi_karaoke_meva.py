#!/usr/bin/env python3
"""MEva + AES at lvl30 on the 2026-10-01 Codex AES-ablation stimuli (MIDI and real songs).

The Codex rounds scored AES only, at -23 LUFS. This re-levels the same stimuli to the
TTM-table main block (-30 LUFS, scalar gain, peak cap 0.999, 16 kHz mono) and adds MEva, so
the instrument / register / noise / stem-removal contrasts can be read on both metrics.

  midi     aes-midi-ablation-20261001   8 templates x 2 SoundFonts, 2,160 clips (10 s)
  karaoke  aes-ablation-20261001        12 songs, 27 contexts x 33 arms, 891 clips (10 s / 30 s)

Stages: build -> score -> analyze (or `all`). Paired deltas are item minus its own baseline;
bootstrap is over templates (midi) or songs (karaoke), 10,000 draws. No captions, so no CLAP.
"""
import argparse
import collections
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path('/home/kojiek/MeanAudio')
SRC = Path('/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/outputs')
SETS = {'midi': SRC / 'aes-midi-ablation-20261001/items.json',
        'karaoke': SRC / 'aes-ablation-20261001/items.json'}
OUT = Path('/home/kojiek/eval_output_nvme/ttm_midi_karaoke_meva_20261012')
MEVA_PY = str(REPO / 'runtime/meva_20261003/venv/bin/python')
SR, TARGET, PEAK_CAP = 16000, -30.0, 0.999
METRICS = ('PQ', 'CE', 'CU', 'PC', 'meva')
N_BOOT = 10000


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def items():
    """One row per stimulus: key, set, cluster (bootstrap unit), base (key of its baseline), tags."""
    rows = []
    for it in json.loads(SETS['midi'].read_text()):
        pre = f'midi__{it["bank"]}__{it["template"]}__'
        for c in it['contrasts'] or [None]:
            rows.append({'key': pre + it['case'], 'set': 'midi', 'path': it['path'], 'cluster': str(it['template']),
                         'bank': it['bank'], 'duration': it['duration'],
                         'family': c['family'] if c else '', 'condition': c['condition'] if c else '',
                         'context': c['context'] if c else '', 'base': pre + c['baseline'] if c else ''})
    for it in json.loads(SETS['karaoke'].read_text()):
        pre = f'kar__{it["song"]}__{it["label"]}__{it["duration"]}__'
        rows.append({'key': pre + it['arm'], 'set': 'karaoke', 'path': it['path'], 'cluster': it['song'],
                     'bank': '', 'duration': it['duration'], 'family': 'arm', 'condition': it['arm'],
                     'context': f'{it["duration"]}s', 'base': '' if it['arm'] == it['baseline'] else pre + it['baseline']})
    return rows


def build():
    import pyloudnorm
    import soxr
    (OUT / 'audio').mkdir(parents=True, exist_ok=True)
    files = {r['key']: r['path'] for r in items()}
    meter, man = pyloudnorm.Meter(SR), []
    for i, (key, path) in enumerate(sorted(files.items())):
        x, sr = sf.read(path, dtype='float64', always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            x = soxr.resample(x, sr, SR, quality='VHQ')
        lufs, peak = float(meter.integrated_loudness(x)), float(np.max(np.abs(x)))
        if not np.isfinite(lufs) or peak <= 0:
            raise SystemExit(f'{key}: unmeasurable loudness')
        g = 10 ** ((TARGET - lufs) / 20)
        capped = peak * g > PEAK_CAP
        g = PEAK_CAP / peak if capped else g
        sf.write(OUT / 'audio' / f'{key}.flac', x * g, SR, format='FLAC', subtype='PCM_24')
        man.append({'key': key, 'src': path, 'src_sr': sr, 'src_lufs': lufs, 'gain_db': 20 * np.log10(g), 'capped': int(capped)})
        if (i + 1) % 500 == 0:
            log(f'build {i + 1}/{len(files)}')
    with open(OUT / 'manifest.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(man[0]), delimiter='\t')
        w.writeheader()
        w.writerows(man)
    log(f'BUILD done n={len(man)} capped={sum(m["capped"] for m in man)}')


def score():
    keys = [r['key'] for r in csv.DictReader(open(OUT / 'manifest.tsv'), delimiter='\t')]
    aj = OUT / 'aes_per_clip.json'
    if not aj.is_file():
        sys.path.insert(0, str(REPO / 'scripts/eval'))
        from eval_metrics import score_aes
        per, failed = score_aes([OUT / 'audio' / f'{k}.flac' for k in keys])
        if failed or len(per) != len(keys):
            raise SystemExit(f'AES incomplete: {len(per)}/{len(keys)} failed={len(failed)}')
        aj.write_text(json.dumps({Path(p).stem: v for p, v in per.items()}))
    ej = OUT / 'meva' / 'meva.json'
    if not ej.is_file():
        log('MEVA')
        (OUT / 'meva').mkdir(exist_ok=True)
        with open(OUT / 'meva.log', 'w') as g:
            subprocess.run([MEVA_PY, str(REPO / 'scripts/eval/meva_score_dir.py'), '--audio_dir', str(OUT / 'audio'),
                            '--out_dir', str(OUT / 'meva')], cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    if json.loads(ej.read_text())['n'] != len(keys):
        raise SystemExit('MEva incomplete')
    log('SCORE done')


def ci(by_cluster, rng):
    """Mean of cluster means with a cluster bootstrap."""
    m = np.array([np.mean(v) for v in by_cluster.values()])
    boot = m[rng.integers(0, len(m), (N_BOOT, len(m)))].mean(axis=1)
    return [float(m.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def analyze(out_json):
    sc = json.loads((OUT / 'aes_per_clip.json').read_text())
    for r in csv.DictReader(open(OUT / 'meva' / 'meva_per_clip.tsv'), delimiter='\t'):
        sc[r['id']]['meva'] = float(r['meva_raw'])
    rows, rng = items(), np.random.default_rng(20261012)
    res = {'protocol': {'level': 'lvl30', 'sr': SR, 'n_boot': N_BOOT, 'n_clips': len(sc),
                        'bootstrap_unit': {'midi': 'template (8)', 'karaoke': 'song (12)'}},
           'absolute': {}, 'paired': {}}

    # Absolute level of each baseline material.
    groups = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in rows:
        if r['set'] == 'karaoke' and r['condition'] == 'original':
            groups[f'karaoke_original_{r["context"]}'][r['cluster']]
            for m in METRICS:
                groups[f'karaoke_original_{r["context"]}'][m][r['cluster']].append(sc[r['key']][m])
        if r['set'] == 'midi' and r['family'] == 'instrument' and r['context'] == 'transpose_0':
            for name in (f'midi_{r["condition"]}', f'midi_{r["condition"]}_{r["bank"]}'):
                for m in METRICS:
                    groups[name][m][r['cluster']].append(sc[r['key']][m])
        if r['set'] == 'midi' and r['family'] == 'register' and r['context'] == 'piano' and r['condition'] == '12':
            for name in ('midi_piano', f'midi_piano_{r["bank"]}'):
                for m in METRICS:
                    groups[name][m][r['cluster']].append(sc[r['base']][m])
    for name, g in sorted(groups.items()):
        res['absolute'][name] = {m: ci(g[m], rng) for m in METRICS}

    # Paired deltas, pooled over contexts and per context.
    pair = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for r in rows:
        if not r['base']:
            continue
        names = [f'{r["set"]}|{r["family"]}|{r["condition"]}|{r["context"]}']
        if r['set'] == 'midi':
            names += [f'midi|{r["family"]}|{r["condition"]}|ALL', f'midi|{r["family"]}|{r["condition"]}|{r["context"]}|{r["bank"]}']
        for name in names:
            for m in METRICS:
                pair[name][m][r['cluster']].append(sc[r['key']][m] - sc[r['base']][m])
    for name, g in sorted(pair.items()):
        res['paired'][name] = {m: ci(g[m], rng) for m in METRICS}
        res['paired'][name]['n'] = sum(len(v) for v in g['PQ'].values())
    Path(out_json).write_text(json.dumps(res, indent=1))
    log(f'ANALYZE -> {out_json}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('build', 'score', 'analyze', 'all'))
    ap.add_argument('--out_json', default=str(REPO / 'docs/experiments/results/ttm_midi_karaoke_meva_20261012.json'))
    a = ap.parse_args()
    if a.stage in ('build', 'all'):
        build()
    if a.stage in ('score', 'all'):
        score()
    if a.stage in ('analyze', 'all'):
        analyze(a.out_json)
