#!/usr/bin/env python3
"""Score every flac in a directory with the pinned MEva f03 evaluator (runs in the MEva venv).

  runtime/meva_20261003/venv/bin/python scripts/eval/meva_score_dir.py --audio_dir D --out_dir O
  runtime/meva_20261003/venv/bin/python scripts/eval/meva_score_dir.py --runtime_check N

Writes O/meva_per_clip.tsv (id, meva_raw, sae_frames, duration_sec, input_sha256) and
O/meva.json (n, mean, model_binding, protocol). --runtime_check re-scores the first N PAM
manifest rows whose 095 scores are on disk and fails if any differs by more than 1e-3,
so a changed runtime cannot silently shift the backfill scores.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meva_runtime import ROOT, RUNTIME, Evaluator, sha

LOCK = ROOT / 'docs/experiments/meva_095_model_lock.json'
BINDING = 'bd31dbfc306a5a379a087a6f79d917849f81da0748f16b321701b03341b0e0da'
TOL = 1e-3


def check_binding():
    got = sha(LOCK)
    if got != BINDING:
        raise SystemExit(f'model lock sha {got} != registered binding {BINDING}')


def runtime_check(n):
    rows = [r for r in json.loads((RUNTIME / 'manifest.json').read_text()) if r['set'] == 'pam'][:n]
    model, worst = Evaluator(), 0.0
    for r in rows:
        ref = json.loads((RUNTIME / 'results/pam' / (r['key'] + '.json')).read_text())
        if ref['model_binding'] != BINDING or sha(r['audio_path']) != ref['input_sha256']:
            raise SystemExit(f'stale reference score {r["key"]}')
        worst = max(worst, abs(model.score(r['audio_path'])['meva_raw'] - ref['meva_raw']))
    print(json.dumps({'runtime_check_n': len(rows), 'max_abs_diff': worst, 'tol': TOL}), flush=True)
    return 0 if len(rows) == n and worst <= TOL else 1


def score_dir(audio_dir, out_dir):
    files = sorted(Path(audio_dir).glob('*.flac'))
    model, rows, t0 = Evaluator(), [], time.time()
    for i, f in enumerate(files):
        v = model.score(f)
        rows.append({'id': f.stem, 'meva_raw': v['meva_raw'], 'sae_frames': v['sae_frames'],
                     'duration_sec': v['duration_sec'], 'input_sha256': sha(f)})
        if (i + 1) % 500 == 0:
            print(f'MEva {i + 1}/{len(files)} {time.time() - t0:.0f}s', flush=True)
    vals = [r['meva_raw'] for r in rows]
    if not vals or not all(math.isfinite(x) for x in vals):
        raise SystemExit('no or nonfinite MEva scores')
    out = Path(out_dir)
    with open(out / 'meva_per_clip.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rows)
    (out / 'meva.json').write_text(json.dumps({
        'n': len(vals), 'meva_mean': sum(vals) / len(vals), 'model_binding': BINDING,
        'protocol': v['protocol'], 'seconds': time.time() - t0}, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio_dir')
    ap.add_argument('--out_dir')
    ap.add_argument('--runtime_check', type=int)
    a = ap.parse_args()
    check_binding()
    if a.runtime_check:
        return runtime_check(a.runtime_check)
    return score_dir(a.audio_dir, a.out_dir)


if __name__ == '__main__':
    sys.exit(main())
