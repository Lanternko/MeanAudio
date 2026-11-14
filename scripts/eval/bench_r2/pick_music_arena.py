#!/usr/bin/env python3
"""Pick the 2,000 pre-registered Music Arena battles (round-2 §2) and optionally download their mp3s.

Battle metadata = the per-month `rows_*.json` already in <ROOT>/meta/music_arena_audit (equivalent to the
battle_data/*/*.json files; checked field-by-field on a sample, see README).  Steps (all seed 20261025):
  1. drop battles whose audio_a/audio_b is null (sao / sao-small);
  2. stratify by (preference, month), proportional allocation with largest remainders;
     if usable TIE < 150 take every TIE first and fill the rest proportionally;
  3. norm subset = random 800 of the unique audio files (seed) for the -23 LUFS variant;
  4. write <ROOT>/meta/music_arena_pick.json.
`--download` fetches the mp3s with hf_hub_download (retry on 429) into <ROOT>/downloads/MusicArena.
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location('bench_r2_datasets', Path(__file__).with_name('bench_datasets.py'))
D = importlib.util.module_from_spec(_sp)  # not `import datasets`: would shadow the HuggingFace package
sys.modules['bench_r2_datasets'] = D
_sp.loader.exec_module(D)
ARENA_PICK, DL, META, SEED = D.ARENA_PICK, D.DL, D.META, D.SEED

REPO_ID = 'music-arena/music-arena-dataset'


def load_battles():
    out = []
    for f in sorted((META / 'music_arena_audit').glob('rows_*.json')):
        month = f.stem[len('rows_'):]
        for r in json.loads(f.read_text()):
            r['month'] = month
            out.append(r)
    return out


def allocate(strata, n):
    """strata: {key: [battles]} -> {key: count}, proportional, largest remainder, capped at stratum size."""
    tot = sum(len(v) for v in strata.values())
    exact = {k: n * len(v) / tot for k, v in strata.items()}
    alloc = {k: min(int(e), len(strata[k])) for k, e in exact.items()}
    rem = n - sum(alloc.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - int(exact[k])), str(k))):
        if rem <= 0:
            break
        if alloc[k] < len(strata[k]):
            alloc[k] += 1
            rem -= 1
    return alloc


def pick(battles, n=2000, n_norm=800):
    usable = [b for b in battles if b.get('audio_a') and b.get('audio_b')]
    rng = random.Random(SEED)
    usable.sort(key=lambda b: b['battle_uuid'])
    ties = [b for b in usable if b['preference'] == 'TIE']
    chosen = []
    pool = usable
    if len(ties) < 150:
        chosen = list(ties)
        pool = [b for b in usable if b['preference'] != 'TIE']
        n -= len(chosen)
    strata = defaultdict(list)
    for b in pool:
        strata[(b['preference'], b['month'])].append(b)
    alloc = allocate(strata, min(n, len(pool)))
    for k in sorted(strata):
        chosen += rng.sample(strata[k], alloc[k])
    chosen.sort(key=lambda b: b['battle_uuid'])
    audio = {}
    for b in chosen:
        for side in ('a', 'b'):
            audio.setdefault(b[f'audio_{side}'], dict(path=b[f'audio_{side}'], prompt=b['prompt'],
                                                       system=b[f'system_{side}']))
    paths = sorted(audio)
    norm = sorted(rng.sample(paths, min(n_norm, len(paths))))
    return usable, chosen, [audio[p] for p in paths], norm


def download(paths, workers=3):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    import os
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    miss = []

    def one(p):
        if (DL / 'MusicArena' / p).exists():
            return
        for i in range(60):
            try:
                hf_hub_download(REPO_ID, p, repo_type='dataset', local_dir=str(DL / 'MusicArena'))
                return
            except Exception as e:  # 429 -> wait, 404 -> record
                s = str(e)
                if '404' in s or 'Entry Not Found' in s:
                    miss.append(p)
                    return
                time.sleep(120 if '429' in s else 15)
        miss.append(p)

    with ThreadPoolExecutor(workers) as ex:
        for j, _ in enumerate(ex.map(one, paths)):
            if j % 200 == 0:
                print('arena dl', j, '/', len(paths), time.strftime('%T'), flush=True)
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--n_norm', type=int, default=800)
    ap.add_argument('--download', action='store_true')
    a = ap.parse_args()
    battles = load_battles()
    usable, chosen, audio, norm = pick(battles, a.n, a.n_norm)
    stats = dict(
        n_battles_all=len(battles), n_usable=len(usable), n_picked=len(chosen), n_audio=len(audio),
        pref_all=Counter(b['preference'] for b in battles), pref_usable=Counter(b['preference'] for b in usable),
        pref_picked=Counter(b['preference'] for b in chosen), month_picked=Counter(b['month'] for b in chosen),
        systems=Counter(s for b in chosen for s in (b['system_a'], b['system_b'])))
    keep = ('battle_uuid', 'date', 'prompt', 'audio_a', 'audio_b', 'system_a', 'system_b', 'preference', 'month',
            'is_instrumental', 'total_listening_time_a', 'total_listening_time_b')
    out = dict(seed=SEED, stats=stats, battles=[{k: b.get(k) for k in keep} for b in chosen], audio=audio,
               norm_subset=norm)
    ARENA_PICK.write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print(json.dumps(stats, indent=1))
    if a.download:
        miss = download([x['path'] for x in audio])
        (META / 'music_arena_missing.json').write_text(json.dumps(miss))
        print('downloaded; missing', len(miss))


if __name__ == '__main__':
    main()
