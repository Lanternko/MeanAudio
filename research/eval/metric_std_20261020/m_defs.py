"""Compute m for the registered S definitions D1 (seed floor), D2 (human JND), D3 (system spread)."""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import EV, RES, NEURAL  # noqa: E402

R115 = EV / 'ttm_metrics_115_20261009'
R118 = EV / 'ttm_metrics_seeds_118_20261011'
HOME = Path('/home/kojiek/MeanAudio')


def seed_dirs(sysname):
    return [R115 / f'meanaudio_{sysname}/lvl30', R118 / f'meanaudio_s16180339_{sysname}/lvl30',
            R118 / f'meanaudio_s27182818_{sysname}/lvl30']


def load_seed(d):
    pc = next((d / 'scores').glob('*/per_clip.tsv'))
    rows = {r['id']: r for r in csv.DictReader(open(pc), delimiter='\t')}
    mv = {r['id']: float(r['meva_raw']) for r in csv.DictReader(open(d / 'meva/meva_per_clip.tsv'), delimiter='\t')}
    return {i: {'PQ': float(r['PQ']), 'CE': float(r['CE']), 'CU': float(r['CU']), 'PC': float(r['PC']),
                'clap': float(r['clap']), 'meva': mv[i]} for i, r in rows.items() if i in mv}


def d1():
    out, detail = {}, {}
    diffs = {m: [] for m in NEURAL}
    for sysname in ('cfg3_neg', 'cfg0'):
        seeds = [load_seed(d) for d in seed_dirs(sysname)]
        ids = sorted(set.intersection(*[set(s) for s in seeds]))
        detail[sysname] = {'n_ids': len(ids)}
        for a, b in itertools.combinations(range(3), 2):
            for m in NEURAL:
                diffs[m].append(np.array([seeds[a][i][m] - seeds[b][i][m] for i in ids]))
    for m in NEURAL:
        x = np.abs(np.concatenate(diffs[m]))
        out[m] = float(np.quantile(x, 0.8))
        detail[m] = {'n_pairs': int(len(x)), 'p50': float(np.median(x)), 'p80': out[m], 'sd': float(np.std(np.concatenate(diffs[m])))}
    return out, detail


def tab(path):
    return list(csv.DictReader(open(path), delimiter='\t'))


def d2():
    out, detail = {}, {}
    jnd = 0.5
    pam = tab(HOME / 'research/eval/output/aes_human_corr_pam/per_clip.tsv')
    pam = [r for r in pam if r['system'] != 'real']
    mc = tab(HOME / 'research/eval/output/aes_human_corr/per_clip.tsv')
    for ax, mname in (('PQ', 'PQ'), ('CE', 'CE'), ('CU', 'CU'), ('PC', 'PC')):
        for tag, rows in (('pam_gen', pam), ('musiccaps_real', mc)):
            h = np.array([float(r[f'human_{ax}']) for r in rows])
            s = np.array([float(r[f'raw_{ax}']) for r in rows])
            b = s.std() / h.std()
            detail.setdefault(mname, {})[tag] = {'n': len(h), 'sd_metric': float(s.std()), 'sd_human': float(h.std()),
                                                  'b': float(b), 'm': float(jnd * b),
                                                  'pearson': float(np.corrcoef(h, s)[0, 1])}
        out[mname] = detail[mname]['pam_gen']['m']
    me = tab(HOME / 'runtime/musiceval_full_20261003/per_clip.tsv')
    h = np.array([float(r['MI']) for r in me])
    s = np.array([float(r['MEva']) for r in me])
    b = s.std() / h.std()
    jme = 0.5 * 4 / 9
    detail['meva'] = {'musiceval': {'n': len(h), 'sd_metric': float(s.std()), 'sd_human_MI': float(h.std()), 'b': float(b),
                                    'jnd_h': jme, 'm': float(jme * b), 'pearson': float(np.corrcoef(h, s)[0, 1])}}
    out['meva'] = detail['meva']['musiceval']['m']
    return out, detail


def d3():
    rows = tab(HOME / 'runtime/musiceval_full_20261003/per_system.tsv')
    out, detail = {}, {}
    for k, m in (('PQ', 'PQ'), ('CE', 'CE'), ('CU', 'CU'), ('MEva', 'meva')):
        v = np.array([float(r[k]) for r in rows])
        q1, q3 = np.quantile(v, [0.25, 0.75])
        out[m] = float(0.1 * (q3 - q1))
        detail[m] = {'n_sys': len(v), 'iqr': float(q3 - q1), 'range': float(np.ptp(v)), 'm': out[m]}
    return out, detail


if __name__ == '__main__':
    res = {}
    for name, f in (('D1', d1), ('D2', d2), ('D3', d3)):
        res[name], res[name + '_detail'] = f()
    Path(__file__).with_name('m_defs.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.endswith('detail')}, indent=1))
