#!/usr/bin/env python
"""Extract per-split baselines from the canonical full cells (no new generation).

For one split x checkpoint x cfg cell this records everything the gates compare
against: lvl30 CLAP/PQ/LUFS, raw level (silent/clipped/crest), lvl30 loudness
alignment (peak_capped_n, |LUFS+30|>0.05 count), the per-id raw flac sha256 map
(G0/G1), and FAD on the same subset (reference embeddings cached).

    baselines.py --split dev256 [--checkpoints s14159265,...] --scratch SCRATCH [--no-fad]

The FAD path is the canonical one (raw audio, 2048-sample rule, PCM_16 double
conversion); on full5521 s14159265/cfg3_neg it must reproduce 5.0836.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                   # noqa: E402
import fad_cache                                # noqa: E402


def _metrics_by_id(cell, exp_name):
    path = Path(cell) / exp_name / 'per_clip.tsv'
    if not path.is_file():
        raise SystemExit(f'[FAIL] missing {path}')
    recs = common.read_per_clip(path)
    if len({r['id'] for r in recs}) != len(recs):
        raise SystemExit(f'[FAIL] duplicate ids in {path}')
    return {r['id']: r for r in recs}


def build_one(scratch, split, ckpt, cfgkey, ids, rows, do_fad=True):
    raw = common.canonical_cell(ckpt, cfgkey, 'raw')
    lvl = common.canonical_cell(ckpt, cfgkey, 'lvl30')
    exp = common.EXP[ckpt]
    raw_metrics = _metrics_by_id(raw, f'{exp}_mc_mf25_{cfgkey}')
    lvl_metrics = _metrics_by_id(lvl, f'{exp}_mc_mf25_{cfgkey}_lvl30')
    missing_raw = [i for i in ids if i not in raw_metrics]
    missing_lvl = [i for i in ids if i not in lvl_metrics]
    if missing_raw or missing_lvl:
        raise SystemExit(f'[FAIL] {split}/{ckpt}/{cfgkey}: missing raw={missing_raw[:3]} '
                         f'lvl30={missing_lvl[:3]}')
    gain = {r['id']: r for r in common.read_gain_status(lvl / 'gain_status.tsv')}

    lvl_rows = [lvl_metrics[i] for i in ids]
    raw_rows = [raw_metrics[i] for i in ids]
    g_rows = [gain[i] for i in ids]
    clap, n_clap = common.mean(r['clap'] for r in lvl_rows)
    pq, _ = common.mean(r['PQ'] for r in lvl_rows)
    rec = {
        'split': split, 'checkpoint': ckpt, 'cfg': cfgkey, 'n': len(ids),
        'ids_sha256': common.sha256_text('\n'.join(ids)),
        'source': {'raw_cell': str(raw), 'lvl30_cell': str(lvl)},
        'lvl30': {
            'clap': clap, 'clap_n': n_clap,
            'PQ': pq, 'CE': common.mean(r['CE'] for r in lvl_rows)[0],
            'CU': common.mean(r['CU'] for r in lvl_rows)[0],
            'PC': common.mean(r['PC'] for r in lvl_rows)[0],
            'lufs': common.mean(r['lufs'] for r in lvl_rows)[0],
            'peak_capped_n': sum(r['status'] == 'peak_capped' for r in g_rows),
            'no_finite_n': sum(r['status'] == 'no_finite_loudness' for r in g_rows),
            'lufs_off_0p05_n': sum(1 for r in lvl_rows
                                   if r['lufs'] is not None and abs(r['lufs'] + 30.0) > 0.05),
        },
        'raw': {
            'clap': common.mean(r['clap'] for r in raw_rows)[0],   # present only on full cells
            'silent_n': sum(bool(r['silent']) for r in raw_rows),
            'clipped_n': sum((r['peak'] or 0.0) >= 0.999 for r in raw_rows),
            'crest_mean': common.mean(r['crest'] for r in raw_rows)[0],
            'lufs': common.mean(r['lufs'] for r in raw_rows)[0],
        },
        'written_at': common.now_iso(),
    }
    if do_fad:
        fad = fad_cache.score(raw / 'audio', rows, cache_dir=Path(scratch) / 'fad_cache')
        rec['fad'] = fad
    # per-id sha256 of the canonical raw flac (G0/G1); written next to baseline.json
    sha_path = common.baseline_path(scratch, split, ckpt, cfgkey).with_name('audio_sha256.tsv')
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sha_path, 'w', encoding='utf-8') as f:
        for i in ids:
            f.write(f'{i}\t{common.sha256(raw / "audio" / f"{i}.flac")}\n')
    rec['audio_sha256_tsv'] = str(sha_path)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--split', required=True)
    p.add_argument('--scratch', default=str(common.SCRATCH))
    p.add_argument('--checkpoints', default=','.join(common.SEEDS))
    p.add_argument('--cfgs', default='cfg0,cfg3_neg')
    p.add_argument('--no-fad', action='store_true')
    p.add_argument('--force', action='store_true')
    a = p.parse_args(argv)
    scratch = Path(a.scratch)
    rows = common.split_rows(scratch, a.split)
    ids = [i for i, _ in rows]
    lo, hi = common.SPLIT_RECORDS[a.split]
    if hi is not None and len(ids) != hi - lo:
        raise SystemExit(f'[FAIL] {a.split}: {len(ids)} rows, expected {hi - lo}')
    ckpts = [s for s in a.checkpoints.split(',') if s]
    for ckpt in ckpts:
        for cfgkey in a.cfgs.split(','):
            dst = common.baseline_path(scratch, a.split, ckpt, cfgkey)
            if dst.exists() and not a.force:
                print(f'[baselines] skip existing {dst}', flush=True)
                continue
            t0 = time.time()
            rec = build_one(scratch, a.split, ckpt, cfgkey, ids, rows, do_fad=not a.no_fad)
            rec['seconds'] = round(time.time() - t0, 1)
            common.atomic_json(dst, rec)
            print(json.dumps({k: rec[k] for k in ('split', 'checkpoint', 'cfg', 'n', 'seconds')}
                             | {'lvl30_clap': rec['lvl30']['clap'], 'lvl30_PQ': rec['lvl30']['PQ'],
                                'fad': rec.get('fad', {}).get('value')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
