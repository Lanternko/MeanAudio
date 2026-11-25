#!/usr/bin/env python3
"""§10.3 CPU checks for the round-2 benchmark cache (no GPU).

  python -I check_cache.py [--datasets odaq,mc522,...] [--out /mnt/seagate/benchmarks_r2/analysis/cache_check.json]

Per dataset: manifest/index counts, invalid units, ffprobe of 20 source files (seeded), and from index.tsv the
peak / LUFS statistics of the native-or-V10 raw rows (silence is the LUFS_in < -45 proxy, not a sample-level count).
Never executes anything from the data directories; paths are passed to ffprobe as arguments only.
"""
import argparse
import csv
import json
import random
import statistics as st
import subprocess
from pathlib import Path

ROOT = Path('/mnt/seagate/benchmarks_r2')
SEED = 20261025


def read_tsv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def probe(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                        'stream=sample_rate,channels,bit_rate,codec_name:format=duration,bit_rate', '-of', 'json',
                        str(path)], capture_output=True, text=True, timeout=60)
    try:
        j = json.loads(r.stdout)
        s, f = j['streams'][0], j['format']
        return dict(sr=int(s['sample_rate']), ch=int(s['channels']), codec=s['codec_name'],
                    bit_rate=int(s.get('bit_rate') or f.get('bit_rate') or 0), dur=float(f['duration']))
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e)[:120])


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def check(ds):
    man = read_tsv(ROOT / f'manifests/{ds}.tsv') if (ROOT / f'manifests/{ds}.tsv').exists() else []
    idx_p = ROOT / f'derived/{ds}/index.tsv'
    idx = read_tsv(idx_p) if idx_p.exists() else []
    out = dict(manifest_items=len(man), index_rows=len(idx), invalid=sum(1 for r in idx if r['invalid']))
    rng = random.Random(SEED)
    real = [m['src'] for m in man if m['src'].startswith('/') and Path(m['src']).exists()]
    if real:
        pick = rng.sample(real, min(20, len(real)))
        pr = [probe(p) for p in pick]
        ok = [x for x in pr if 'error' in x]
        good = [x for x in pr if 'error' not in x]
        out['ffprobe'] = dict(n=len(pr), errors=len(ok),
                              sr=sorted({x['sr'] for x in good}), ch=sorted({x['ch'] for x in good}),
                              codec=sorted({x['codec'] for x in good}),
                              bit_rate_kbps=[min(x['bit_rate'] for x in good) // 1000,
                                             max(x['bit_rate'] for x in good) // 1000] if good else None,
                              dur_s=[round(min(x['dur'] for x in good), 2),
                                     round(max(x['dur'] for x in good), 2)] if good else None)
    rows = [r for r in idx if r['loud'] == 'raw' and r['variant'] in ('native', 'V10')]
    if not rows:
        rows = [r for r in idx if r['loud'] == 'raw']
    if rows:
        pk = [float(r['peak']) for r in rows]
        lu = [float(r['lufs_in']) for r in rows if r['lufs_in'] not in ('', '-inf', 'nan')]
        out['stats'] = dict(n_rows=len(rows), sr=sorted({int(r['sr']) for r in rows}),
                            peak_max=max(pk), peak_ge_1=sum(1 for p in pk if p >= 1.0),
                            lufs_in_p05_p50_p95=[q(lu, .05), q(lu, .5), q(lu, .95)],
                            silent_lufs_lt_m45=sum(1 for x in lu if x < -45),
                            norm_gain_db_p50=st.median(float(r['gain_db']) for r in idx if r['loud'] == 'norm')
                            if any(r['loud'] == 'norm' for r in idx) else None)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='odaq,mc522,singmos,songeval,arena,aime')
    ap.add_argument('--out', default=str(ROOT / 'analysis/cache_check.json'))
    a = ap.parse_args()
    res = {d: check(d) for d in a.datasets.split(',')}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
