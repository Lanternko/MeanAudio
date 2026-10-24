#!/usr/bin/env python3
"""Round-2 benchmark screening: manifests, sampling and variant plan.

Design (preregistered): docs/experiments/benchmark_screening_r2_20261025.md
Only csv/json/numpy at import time (the MEva venv runs score_r2.py too). Heavy deps (pandas, pyarrow)
are imported lazily inside the builders.

Unit = (dataset, item_id, variant, loud).  variant in native|V5|V10|V30|VFULL, loud in raw|norm.
Manifest  : <ROOT>/manifests/<dataset>.tsv   one row per item (src, sha16, caption, labels the scorers need)
Plan      : PLAN[dataset][job] -> list of (variant, loud, subset)
Usage     : python bench_datasets.py build [dataset ...]   (CPU only, writes manifests)
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/mnt/seagate/benchmarks_r2')
DL = ROOT / 'downloads'
META = ROOT / 'meta'
DER = ROOT / 'derived'
SCORES = ROOT / 'scores'
MAN = ROOT / 'manifests'
REPO = Path('/home/kojiek/MeanAudio')
SEED = 20261025
MC522_TSV = REPO / 'research/eval/newmodels_20261020/bench_musiccaps522.tsv'
ARENA_PICK = META / 'music_arena_pick.json'
DATASETS = ('odaq', 'mc522', 'aime', 'singmos', 'songeval', 'arena')

# subset names: all | test (SingMOS official test) | s800 (SongEval stratified subset) | norm800 (Arena norm subset)
PLAN = {
    'odaq': {'J1': [('native', 'raw', 'all'), ('native', 'norm', 'all'), ('V5', 'raw', 'all'), ('V10', 'raw', 'all')]},
    'mc522': {'J1': [('V5', 'raw', 'all'), ('V10', 'raw', 'all'), ('V10', 'norm', 'all')]},
    'aime': {'J2': [('native', 'raw', 'all'), ('native', 'norm', 'all')]},
    'singmos': {'J3a': [('native', 'raw', 'all')], 'J3b': [('native', 'norm', 'all'), ('V5', 'raw', 'test')]},
    'songeval': {'J4a': [('VFULL', 'raw', 's800')], 'J4b': [('VFULL', 'norm', 's800')],
                 'J4c': [('V30', 'raw', 's800'), ('V5', 'raw', 's800'), ('V10', 'norm', 's800'), ('V10', 'raw', 'all')]},
    'arena': {'J5': [('native', 'raw', 'all'), ('native', 'norm', 'norm800')]},
}
ITEM_COLS = ['dataset', 'item_id', 'src', 'sha16', 'caption', 'group', 'subset', 'extra']


def sha16(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:16]


def safe(s):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(s))


def unit_id(dataset, item_id, variant, loud):
    return f'{dataset}|{item_id}|{variant}|{loud}'


def unit_key(item, variant, loud):
    """File-system key; includes the first 16 hex chars of the source sha256 (or the AIME track sha)."""
    return f'{safe(item["item_id"])}-{item["sha16"]}__{variant}__{loud}'


def variant_label(variant, loud):
    return f'{variant}_{loud}'


def write_manifest(dataset, items):
    MAN.mkdir(parents=True, exist_ok=True)
    with open(MAN / f'{dataset}.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=ITEM_COLS, delimiter='\t')
        w.writeheader()
        for it in items:
            w.writerow({k: it.get(k, '') for k in ITEM_COLS})


def read_manifest(dataset):
    with open(MAN / f'{dataset}.tsv', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def in_subset(item, subset):
    return subset == 'all' or subset in item['subset'].split(';')


def units(dataset, variant, loud, subset='all', items=None):
    items = read_manifest(dataset) if items is None else items
    return [dict(it, variant=variant, loud=loud, unit_id=unit_id(dataset, it['item_id'], variant, loud))
            for it in items if in_subset(it, subset)]


def plan_units(dataset, job):
    """All units of a job (list of dicts), in plan order."""
    items = read_manifest(dataset)
    out = []
    for variant, loud, subset in PLAN[dataset][job]:
        out += units(dataset, variant, loud, subset, items)
    return out


# ───────────────────────── builders ─────────────────────────
def build_odaq():
    import pandas as pd
    base = DL / 'ODAQ/ODAQ/ODAQ_listening_test'
    d = pd.read_csv(base / 'ODAQ_results.csv')
    items = []
    for (item, proc), g in d.groupby(['item', 'process'], sort=True):
        p = base / item / f'{proc}.wav'
        items.append(dict(dataset='odaq', item_id=f'{item}/{proc}', src=str(p), sha16=sha16(p), caption='',
                          group=item, subset='all',
                          extra=json.dumps({'trial': item, 'process': proc, 'method': g.method.iloc[0],
                                            'condition': g.condition.iloc[0]})))
    return items


def build_mc522():
    items = []
    for r in csv.DictReader(open(MC522_TSV), delimiter='\t'):
        items.append(dict(dataset='mc522', item_id=r['key'], src=r['path'], sha16=sha16(r['path']), caption=r['caption'],
                          group='real', subset='all', extra=json.dumps({'human_PQ': float(r['human_PQ'])})))
    return items


def aime_track_index():
    """track id -> (shard path, row index); cached in meta/aime_track_index.json (reads only id/model columns)."""
    cache = META / 'aime_track_index.json'
    if cache.exists():
        return json.loads(cache.read_text())
    import pyarrow.parquet as pq
    idx = {}
    for f in sorted((DL / 'AIME/data').glob('train-*.parquet')):
        t = pq.read_table(f).to_pylist()
        for i, r in enumerate(t):
            idx[r['id']] = [str(f), i, r['model'], r['description'], hashlib.sha256(r['audio']['bytes']).hexdigest()[:16]]
    cache.write_text(json.dumps(idx))
    return idx


def build_aime():
    import pandas as pd
    s = pd.read_parquet(META / 'aime_survey/train.parquet')
    idx = aime_track_index()
    snip = {}
    for side in ('1', '2'):
        sub = s[[f'model-{side}', f'track-{side}-id', f'track-{side}-begin', f'track-{side}-end', 'description']]
        for m, t, b, e, desc in sub.itertuples(index=False):
            snip[(m, int(t), b, e)] = desc
    items = []
    for (m, t, b, e), desc in sorted(snip.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        tid = f'{t:05d}'
        if tid not in idx:
            raise SystemExit(f'AIME track {tid} ({m}) not found in parquet shards')
        shard, row, pm, pdesc, tsha = idx[tid]
        items.append(dict(dataset='aime', item_id=f'{m}|{tid}|{b}-{e}', src=f'aime:{shard}:{row}:{b}:{e}',
                          sha16=tsha, caption=desc, group=m, subset='all',
                          extra=json.dumps({'track': tid, 'parquet_model': pm, 'begin': b, 'end': e})))
    return items


def build_singmos():
    meta = json.loads((META / 'singmos_pro/metadata.json').read_text())
    items = []
    for r in meta:
        p = DL / 'SingMOS-Pro' / r['wav']
        sub = 'all;' + ('test' if r['split'] == 'test' else 'train')
        items.append(dict(dataset='singmos', item_id=r['id'], src=str(p), sha16='', caption='', group=r['system_id'],
                          subset=sub, extra=json.dumps({'split': r['split'], 'system': r['system'], 'type': r['type'],
                                                        'dataset': r['dataset'], 'sr': r['sample_rate'],
                                                        'mos': sum(r['judge_score']) / len(r['judge_score']),
                                                        'n_judge': len(r['judge_score'])})))
    return items


def songeval_s800(rows):
    """Gender-stratified proportional sample of 800, pandas random_state=SEED, sorted by file_name."""
    import pandas as pd
    df = pd.DataFrame(rows).sort_values('file_name').reset_index(drop=True)
    n_tot, k = len(df), 800
    exact = df.gender.value_counts() * k / n_tot
    alloc = exact.astype(int)
    for g in (exact - alloc).sort_values(ascending=False).index[:k - int(alloc.sum())]:
        alloc[g] += 1
    picks = [df[df.gender == g].sample(n=int(n), random_state=SEED) for g, n in alloc.items()]
    return set(pd.concat(picks).file_name)


def build_songeval():
    rows = [json.loads(l) for l in open(META / 'songeval/metadata.jsonl')]
    s800 = songeval_s800([{'file_name': r['file_name'], 'gender': r['gender']} for r in rows])
    dims = ('Coherence', 'Musicality', 'Memorability', 'Clarity', 'Naturalness')
    items = []
    for r in sorted(rows, key=lambda r: r['file_name']):
        p = DL / 'SongEval' / r['file_name']
        ann = r['annotation']
        mean = {d: sum(a[d] for a in ann) / len(ann) for d in dims}
        items.append(dict(dataset='songeval', item_id=Path(r['file_name']).stem, src=str(p), sha16='', caption='',
                          group=r['gender'], subset='all' + (';s800' if r['file_name'] in s800 else ''),
                          extra=json.dumps({'gender': r['gender'], 'n_ann': len(ann), 'mean': mean})))
    return items


def build_arena():
    if not ARENA_PICK.exists():
        raise SystemExit('run pick_music_arena.py first')
    pick = json.loads(ARENA_PICK.read_text())
    norm = set(pick['norm_subset'])
    items = []
    for a in pick['audio']:
        p = DL / 'MusicArena' / a['path']
        if not p.exists():
            continue
        items.append(dict(dataset='arena', item_id=a['path'], src=str(p), sha16='', caption=a['prompt'],
                          group=a['system'], subset='all' + (';norm800' if a['path'] in norm else ''),
                          extra=json.dumps({'system': a['system']})))
    return items


BUILDERS = {'odaq': build_odaq, 'mc522': build_mc522, 'aime': build_aime, 'singmos': build_singmos,
            'songeval': build_songeval, 'arena': build_arena}


def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'build':
        raise SystemExit(__doc__)
    for ds in (sys.argv[2:] or DATASETS):
        items = BUILDERS[ds]()
        for it in items:
            if not it['sha16'] and Path(it['src']).is_file():
                it['sha16'] = sha16(it['src'])
        write_manifest(ds, items)
        subs = {}
        for it in items:
            for s in it['subset'].split(';'):
                subs[s] = subs.get(s, 0) + 1
        print(ds, len(items), subs, flush=True)


if __name__ == '__main__':
    main()
