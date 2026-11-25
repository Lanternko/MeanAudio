#!/usr/bin/env python3
"""Score one (dataset, model) cell for datasets outside the sha-bound round-2 PLAN.

Reuses score_r2.py's scorer classes and aggregator verbatim (no copy of the model code):

  dac venv : python -I score_extra.py --dataset pam400 --models songeval,singmos,clap
  MEva venv: runtime/meva_20261003/venv/bin/python -I score_extra.py --dataset me --models meva

Units come from derived/<dataset>/index.tsv (native/raw only), so the row set is exactly
what extra_datasets.py prepared.  Output/resume convention is identical to score_r2:
<ROOT>/scores/<dataset>/<model>__<variant>_<loud>.tsv and ...err.tsv.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402

_sp = importlib.util.spec_from_file_location('bench_r2_datasets', HERE / 'bench_datasets.py')
D = importlib.util.module_from_spec(_sp)
sys.modules['bench_r2_datasets'] = D
_sp.loader.exec_module(D)
_ss = importlib.util.spec_from_file_location('score_r2_mod', HERE / 'score_r2.py')
S = importlib.util.module_from_spec(_ss)
sys.modules['score_r2_mod'] = S
_ss.loader.exec_module(S)


def log(*a):
    print(time.strftime('%T'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--variant', default='native')
    ap.add_argument('--loud', default='raw')
    a = ap.parse_args()
    ds = a.dataset
    idx = S.load_index(ds)
    items = {it['item_id']: it for it in D.read_manifest(ds)}
    us = []
    for uid, ix in idx.items():
        # unit_id = <ds>|<item…>|<variant>|<loud>; AIME item ids contain '|' themselves
        # (aime|AudioLDM 2 Large|01509|00:00:00-00:00:10|native|raw), so take the ends only.
        parts = uid.split('|')
        iid, variant, loud = '|'.join(parts[1:-2]), parts[-2], parts[-1]
        if variant != a.variant or loud != a.loud or ix['invalid']:
            continue
        it = items[iid]
        us.append(dict(item_id=iid, unit_id=uid, variant=variant, loud=loud, caption=it['caption'],
                       files=ix['files'], full=ix['full']))
    us.sort(key=lambda u: u['unit_id'])
    out_root = D.SCORES / ds
    out_root.mkdir(parents=True, exist_ok=True)
    for model in a.models.split(','):
        label = f'{model}__{D.variant_label(a.variant, a.loud)}'
        f_out, f_err = out_root / f'{label}.tsv', out_root / f'{label}.err.tsv'
        done, _ = S.read_tsv_ids(f_out)
        done_err, _ = S.read_tsv_ids(f_err)
        todo = [u for u in us if u['unit_id'] not in done and u['unit_id'] not in done_err]
        log(f'{ds} {label}: {len(us)} units, {len(done)} done, {len(done_err)} err, {len(todo)} todo')
        if not todo:
            continue
        scorer = S.build(model, ds, f'{ds}_extra_{model}')
        step = S.CHUNK.get(model, 50)
        fields = None
        fo = open(f_out, 'a', newline='')
        fe = open(f_err, 'a', newline='')
        if not f_err.stat().st_size:
            csv.writer(fe, delimiter='\t').writerow(['unit_id', 'error'])
        wr = None
        t0, n_ok, n_bad = time.time(), 0, 0
        for i in range(0, len(todo), step):
            res = scorer.score_units(todo[i:i + step])
            for u in todo[i:i + step]:
                r = res[u['unit_id']]
                ok = isinstance(r, list) and r
                if ok:
                    row = S.aggregate(r, scorer.cols, a.variant)
                    ok = all(math.isfinite(v) for v in row.values() if isinstance(v, float))
                if not ok:
                    n_bad += 1
                    csv.writer(fe, delimiter='\t').writerow(
                        [u['unit_id'], repr(r)[:300] if not isinstance(r, list) else 'nonfinite'])
                    continue
                row = dict(unit_id=u['unit_id'], item_id=u['item_id'], **row)
                if wr is None:
                    if not f_out.stat().st_size:
                        fields = list(row)
                    wr = csv.DictWriter(fo, fieldnames=fields, delimiter='\t')
                    if not f_out.stat().st_size:
                        wr.writeheader()
                wr.writerow(row)
                n_ok += 1
            fo.flush()
            fe.flush()
            if (i // step) % 5 == 0:
                log(f'  {label} {min(i + step, len(todo))}/{len(todo)} ok={n_ok} err={n_bad} '
                    f'{time.time() - t0:.0f}s')
        fo.close()
        fe.close()
        log(f'{label} done ok={n_ok} err={n_bad} {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
