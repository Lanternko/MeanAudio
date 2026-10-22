"""A: sensitivity of 126a/126b/126c pass/fail to the S (m) definition.  CPU only, reads existing per-clip scores."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/kojiek/MeanAudio/scripts/eval')
from common import *  # noqa: E402,F401,F403
import common as C  # noqa: E402
import metric_win_tie_126a as A  # noqa: E402
import metric_win_tie_126b as B  # noqa: E402
import metric_win_tie_126c as CC  # noqa: E402

MD = json.loads((HERE / 'm_defs.json').read_text())
DEFS = ['D0', 'D1', 'D2', 'D3']


def m_for(d, metric, S):
    if d == 'D0':
        return 0.1 * S[metric]
    return MD[d].get(metric)  # None when undefined


def cells_126a():
    data = C.load_126a()
    S = C.S_of('126a')
    clean = data['clean']
    out = []
    for v, per in data.items():
        for s, rec in per.items():
            if rec['kind'] not in ('tie', 'lose'):
                continue
            assert rec['ids'] == clean[s]['ids']
            out.append({'ds': '126a', 'cell': f'{v}|{s}', 'ladder_step': v, 'system': s, 'kind': rec['kind'],
                        'd': {m: rec['m'][m] - clean[s]['m'][m] for m in C.NEURAL}})
    return out, S


def cells_keyed(tag, root, items_path, mir_path, S, cell_key):
    sc, rows = C.load_keyed(root, items_path, mir_path)
    groups = {}
    for r in rows:
        if r['base'] and r['kind'] in ('tie', 'soft', 'lose'):
            groups.setdefault(cell_key(r), []).append(r)
    out = []
    for key, rs in groups.items():
        d = {}
        for m in list(C.NEURAL) + list(C.MIR):
            d[m] = np.array([sc[r['key']].get(m, np.nan) - sc[r['base']].get(m, np.nan) for r in rs])
        out.append({'ds': tag, 'cell': '|'.join(key), 'ladder_step': key[-2] if tag == '126b' else key[0],
                    'system': key[1] if tag == '126b' else 'musdb', 'kind': key[-1], 'd': d})
    return out


def evaluate(cells, S):
    """per cell × metric × D: rates and pass"""
    res = []
    for c in cells:
        for m, d in c['d'].items():
            if m not in S:
                continue
            row = {'ds': c['ds'], 'cell': c['cell'], 'step': c['ladder_step'], 'system': c['system'], 'kind': c['kind'],
                   'metric': m, 'n': int(np.isfinite(d).sum())}
            for D in DEFS:
                mm = m_for(D, m, S)
                if mm is None:
                    row[D] = None
                    continue
                r = C.judge_rates(d, mm)
                row[D] = {'m': mm, **(r or {}), 'pass': C.passes(r, c['kind'], m)}
            res.append(row)
    return res


def flip_table(rows):
    tab = {}
    for r in rows:
        if r['D0'] is None or r['D0']['pass'] is None:
            continue
        cls = 'tie' if r['kind'] in ('tie', 'soft') else 'lose'
        for D in DEFS[1:]:
            if r[D] is None:
                continue
            t = tab.setdefault((r['ds'], r['metric'], cls, D), {'n': 0, 'p0': 0, 'pD': 0, 'to_pass': 0, 'to_fail': 0})
            t['n'] += 1
            t['p0'] += bool(r['D0']['pass'])
            t['pD'] += bool(r[D]['pass'])
            t['to_pass'] += (not r['D0']['pass']) and r[D]['pass']
            t['to_fail'] += r['D0']['pass'] and (not r[D]['pass'])
    return tab


def monotone_violations(rows):
    bad = []
    for r in rows:
        if r['D0'] is None:
            continue
        for D in DEFS[1:]:
            if r[D] is None:
                continue
            up = r[D]['m'] >= r['D0']['m']
            if r['kind'] in ('tie', 'soft'):
                ok = (r[D]['tie'] >= r['D0']['tie'] - 1e-12) if up else (r[D]['tie'] <= r['D0']['tie'] + 1e-12)
            else:
                ok = (r[D]['worse'] <= r['D0']['worse'] + 1e-12) if up else (r[D]['worse'] >= r['D0']['worse'] - 1e-12)
            if not ok:
                bad.append((r['ds'], r['cell'], r['metric'], D))
    return bad


def ladder_readouts(rows, ladders, ds_filter, majority=True):
    """ladders: {name: [steps]}; per metric × D: tolerance (tie) / detection (lose) step; per-step pass = majority of systems"""
    out = {}
    by = {}
    for r in rows:
        if r['ds'] != ds_filter:
            continue
        by.setdefault((r['step'], r['metric']), []).append(r)
    for lad, steps in ladders.items():
        for metric in {r['metric'] for r in rows if r['ds'] == ds_filter}:
            for D in DEFS:
                ps = []
                kind = None
                for st in steps:
                    rs = [x for x in by.get((st, metric), []) if x[D] is not None and x[D]['pass'] is not None]
                    if not rs:
                        ps.append(None)
                        continue
                    kind = 'tie' if rs[0]['kind'] in ('tie', 'soft') else 'lose'
                    ps.append(np.mean([bool(x[D]['pass']) for x in rs]) >= 0.5)
                if any(p is None for p in ps) or kind is None:
                    continue
                if kind == 'tie':
                    ff = next((i for i, p in enumerate(ps) if not p), None)
                    val = None if not ps[0] else (steps[ff - 1] if ff else steps[-1])
                else:
                    val = next((st for st, p in zip(steps, ps) if p), None)
                out.setdefault(f'{lad}|{metric}', {'kind': kind})[D] = val
    return out


if __name__ == '__main__':
    allrows, ladders = [], {}
    c, S = cells_126a()
    r = evaluate(c, S)
    allrows += r
    ladders['126a'] = ladder_readouts(r, A.LADDERS, '126a')
    S126a = S
    # reproduce stored 126a pass flags at D0
    summ = json.loads((C.RES / 'metric_win_tie_126a_summary.json').read_text())['cells']
    mism = 0
    tot = 0
    for x in r:
        if x['D0']['pass'] is None:
            continue
        v, s = x['step'], x['system']
        st = summ[v][s]['metrics'][x['metric']]['at_m']['pass']
        tot += 1
        mism += bool(st) != bool(x['D0']['pass'])
    print('126a D0 reproduce mismatches', mism, '/', tot)
    for tag, root, ip, mp, sname, keyf, lad in (
            ('126b', C.EV / 'metric_win_tie_126b_20261016', C.EV / 'metric_win_tie_126b_20261016/stimuli/items.json',
             C.EV / 'metric_win_tie_126b_20261016/mir.json', '126b', lambda r: (r['block'], r['source'], r['case'], r['kind']),
             {**B.MIDI_LADDERS, **A.LADDERS}),
            ('126c', C.EV / 'metric_win_tie_126c_20261019', C.EV / 'metric_win_tie_126c_20261019/stimuli/items.json',
             C.EV / 'metric_win_tie_126c_20261019/mir.json', '126c', lambda r: (r['case'], r['kind']), CC.LADDERS)):
        S = C.S_of(sname)
        cells = cells_keyed(tag, root, ip, mp, S, keyf)
        r = evaluate(cells, S)
        # reproduce stored D0 flags
        ssum = json.loads((C.RES / f'metric_win_tie_{tag}_summary.json').read_text())['cells']
        mism = tot = 0
        for x in r:
            if x['D0']['pass'] is None:
                continue
            parts = x['cell'].split('|')
            try:
                if tag == '126b':
                    st = ssum[f'{parts[0]}/{parts[1]}'][parts[2]]['metrics'][x['metric']]['at_m']['pass']
                else:
                    st = ssum[parts[0]]['metrics'][x['metric']]['at_m']['pass']
            except KeyError:
                continue
            tot += 1
            mism += bool(st) != bool(x['D0']['pass'])
        print(tag, 'D0 reproduce mismatches', mism, '/', tot)
        allrows += r
        if tag == '126b':
            # 126b pooled over sources: collapse ladder steps by case (system=source) majority across sources
            ladders[tag] = ladder_readouts(r, lad, tag)
        else:
            ladders[tag] = ladder_readouts(r, lad, tag)
    ft = flip_table(allrows)
    mv = monotone_violations(allrows)
    print('monotone violations', len(mv), mv[:5])
    out = {'flip_table': {'|'.join(k): v for k, v in ft.items()}, 'ladders': ladders, 'monotone_violations': mv,
           'n_rows': len(allrows)}
    (HERE / 'sensitivity.json').write_text(json.dumps(out, indent=1))
    # slim per-cell dump for evidence table (D0 only rates)
    import gzip
    with gzip.open(HERE / 'cells_rates.json.gz', 'wt') as f:
        json.dump(allrows, f)
    print('rows', len(allrows))
