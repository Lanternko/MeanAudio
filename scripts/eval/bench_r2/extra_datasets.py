#!/usr/bin/env python3
"""Extra benchmark datasets for the full metric x human-rating matrix (2026-10-27).

These are the datasets outside the sha-bound round-2 plan (J1-J5), so they do NOT
touch `bench_datasets.PLAN` / `analyze_r2.PRIMARY`:

  pam400      PAM human_eval, the 400 generated clips (4 systems x 100 prompts);
              human target pam_OVL (and pam_REL).  Rows come from the round-1 manifest
              research/eval/newmodels_20261020/bench_pam500.tsv, so the clips are exactly
              the ones the round-1 PAM400 cells used.
  me          MusicEval 2,748 clips (31 systems); target human_OVL / human_REL.  Rows come
              from research/eval/newmodels_20261020/bench_musiceval2748.tsv (same clips as
              the round-1 ME cells).
  musicprefs  i-need-sleep/musicprefs, 2,515 pairwise preferences (7 systems, 120 prompts,
              2 axes); audio is decoded from the HF parquet shards.

Unit convention is the round-2 one: unit = (dataset, item_id, 'native', 'raw'), whole clip,
source sample rate.  The source audio is never re-encoded for pam400/me (symlink); musicprefs
parquet bytes are written once to derived/musicprefs/.

sha16 is left empty for pam400/me (their item ids are already unique, and the source path is
kept in `src`); hashing 2 GB of source audio would add nothing but I/O.  musicprefs gets a
sha16 because its item ids are only "pair_side".

Usage:
  python -I extra_datasets.py build [pam400 me musicprefs]   # manifests (+ extract musicprefs)
  python -I extra_datasets.py prep  [pam400 me musicprefs]   # manifest + derived/<ds>/index.tsv
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_datasets as D  # noqa: E402

R1 = Path('/home/kojiek/MeanAudio/research/eval/newmodels_20261020')
MP_DL = D.DL / 'musicprefs'
MP_CSV = MP_DL / 'human_preference.csv'
EXTRA = ('pam400', 'me', 'musicprefs')
VARIANT, LOUD = 'native', 'raw'
INDEX_COLS = ['unit_id', 'key', 'variant', 'loud', 'sr', 'n_samples', 'dur', 'n_win', 'files', 'full',
              'invalid', 'gain_db', 'over_fs', 'peak', 'lufs_in', 'lufs_out']


def log(*a):
    print(*a, flush=True)


def _sha16(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:16]


# ───────────────────────── builders ─────────────────────────
def build_pam400():
    rows = [r for r in csv.DictReader(open(R1 / 'bench_pam500.tsv'), delimiter='\t') if r['system'] != 'real']
    if len(rows) != 400:
        raise SystemExit(f'expected 400 generated PAM clips, got {len(rows)}')
    items = []
    for r in rows:
        p = Path(r['path'])
        if not p.exists():
            raise SystemExit(f'missing audio {p}')
        items.append(dict(dataset='pam400', item_id=r['key'], src=str(p), sha16='',
                          caption=r['caption'], group=r['system'], subset='all',
                          extra=json.dumps({k: r[k] for k in
                                            ('prompt', 'pam_OVL', 'pam_REL', 'human_PQ', 'human_CE',
                                             'human_CU', 'human_PC', 'lufs')})))
    D.write_manifest('pam400', items)
    return items


def build_me():
    rows = list(csv.DictReader(open(R1 / 'bench_musiceval2748.tsv'), delimiter='\t'))
    if len(rows) != 2748:
        raise SystemExit(f'expected 2,748 MusicEval clips, got {len(rows)}')
    items = []
    for r in rows:
        p = Path(r['path'])
        if not p.exists():
            raise SystemExit(f'missing audio {p}')
        items.append(dict(dataset='me', item_id=r['key'], src=str(p), sha16='',
                          caption=r['caption'], group=r['system'], subset='all',
                          extra=json.dumps({k: r[k] for k in
                                            ('human_OVL', 'human_REL', 'split', 'shared', 'dur', 'prompt')})))
    D.write_manifest('me', items)
    return items


def build_musicprefs():
    """Decode every audio referenced by human_preference.csv from the HF parquet shards."""
    if not MP_CSV.exists():
        raise SystemExit(f'missing {MP_CSV}')
    wanted = set()
    for r in csv.DictReader(open(MP_CSV)):
        wanted.add(r['audio_a'])
        wanted.add(r['audio_b'])
    shards = sorted(MP_DL.glob('train-*-of-*.parquet'))
    if not shards:
        raise SystemExit(f'no parquet shards under {MP_DL}')
    import pyarrow.parquet as pq
    import soundfile as sf
    out = D.DER / 'musicprefs'
    out.mkdir(parents=True, exist_ok=True)
    seen, items = set(), []
    for sh in shards:
        tab = pq.read_table(sh, columns=['audio', 'prompt', 'system'])
        for rec in tab.to_pylist():
            name = Path(rec['audio']['path']).name
            if name not in wanted or name in seen:
                continue
            seen.add(name)
            data = rec['audio']['bytes']
            info = sf.info(io.BytesIO(data))
            item_id = Path(name).stem
            item = dict(dataset='musicprefs', item_id=item_id, src=name, sha16='',
                        caption=rec['prompt'], group=rec['system'], subset='all', extra='{}')
            tmp = out / f'{D.safe(item_id)}__native__raw.wav'
            if not tmp.exists():
                import numpy as np
                wav, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
                wav = wav.mean(axis=1) if wav.shape[1] > 1 else wav[:, 0]
                sf.write(tmp, wav.astype(np.float32), sr, subtype='FLOAT')
            item['sha16'] = _sha16(tmp)
            dst = out / f'{D.unit_key(item, VARIANT, LOUD)}.wav'  # unit_key = safe(id)-sha16__native__raw
            if not dst.exists():
                tmp.rename(dst)
            items.append(item)
        log(f'  {sh.name}: {len(seen)}/{len(wanted)} files')
        del tab
    missing = wanted - seen
    if missing:
        raise SystemExit(f'{len(missing)} referenced files not found in parquet (e.g. {sorted(missing)[:3]})')
    items.sort(key=lambda x: (int(x['item_id'].split('_')[0]), int(x['item_id'].split('_')[1])))
    D.write_manifest('musicprefs', items)
    return items


BUILDERS = {'pam400': build_pam400, 'me': build_me, 'musicprefs': build_musicprefs}


# ───────────────────────── index (decode_cache layout) ─────────────────────────
def prep(dataset):
    """Write the manifest (if missing) and derived/<dataset>/index.tsv for native/raw."""
    if not (D.MAN / f'{dataset}.tsv').exists():
        BUILDERS[dataset]()
    items = D.read_manifest(dataset)
    import soundfile as sf
    out = D.DER / dataset
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        key = D.unit_key(it, VARIANT, LOUD)
        dst = out / f'{key}.wav'
        if dataset in ('pam400', 'me'):
            src = Path(it['src'])
            if not dst.exists():
                dst.symlink_to(src)
        elif not dst.exists():
            raise SystemExit(f'missing {dst}; run build {dataset} first')
        info = sf.info(dst)
        rows.append(dict(unit_id=D.unit_id(dataset, it['item_id'], VARIANT, LOUD), key=key,
                         variant=VARIANT, loud=LOUD, sr=info.samplerate, n_samples=info.frames,
                         dur=round(info.frames / info.samplerate, 4), n_win=1, files=dst.name,
                         full='', invalid='', gain_db=0, over_fs=0, peak='', lufs_in='', lufs_out=''))
    with open(out / 'index.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS, delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    log(f'{dataset}: {len(rows)} units -> {out / "index.tsv"}')
    return rows


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('build', 'prep'):
        raise SystemExit(__doc__)
    for ds in (sys.argv[2:] or list(EXTRA)):
        if ds not in BUILDERS:
            raise SystemExit(f'unknown dataset {ds}')
        if sys.argv[1] == 'build':
            n = len(BUILDERS[ds]())
            log(f'{ds}: {n} items -> {D.MAN / (ds + ".tsv")}')
        else:
            prep(ds)


if __name__ == '__main__':
    main()
