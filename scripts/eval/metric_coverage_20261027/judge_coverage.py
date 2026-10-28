#!/usr/bin/env python3
"""Job 177 (CPU): preregistered judging -> coverage_summary.json + matrix + deck data.

Design: docs/experiments/metric_coverage_completion_20261027.md §5 (fixed before any
score existed).  This module only applies the rules; it never re-picks m, S, the 80 %
band, the sample size or the OOD markers.

Rule summary (§5.1-§5.4):
  * d = sign * (score(case) - score(base)) per paired clip, sign per metric.
  * m = 0.10 * S, S from the original protocol (or job 162 for the new models).
  * A judgment unit passes a tie when >= 80 % of its pairs have |d| <= m, and a lose
    when >= 80 % have d < -m (only for DIRECTED metrics).
  * A cell passes when both its tie units and its lose units are >= 80 % passing;
    one of the two -> partial; neither -> fail.
  * Units whose domain mismatch is an OOD marker (OOD-MIDI / OOD-instrumental /
    OOD-stems / OOD-synthetic / OOD-10s-instrumental) are reported but excluded from
    the verdict; a cell left with OOD units only is `ood_only`.  OOD-extreme is an
    annotation only (it is the protocol's own extreme band).
  * Constant scores / all-zero d -> degenerate, never a pass.
  * Unmeasurable S -> inconclusive (MuQ may borrow the 133 S, but stays inconclusive).
  * Descriptive kinds (soft / descriptive / ctl / transition / ref / floor) never pass.

Outputs (all under $EV/metric_coverage_20261027):
  coverage_summary.json, coverage_matrix.md, coverage_matrix.html, deck_data.json,
  jobs/job_done_177.json.
"""
import collections
import csv
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import common as C  # noqa: E402

OUT = C.OUT_ROOT
DIRECTED_A = ('PQ', 'CE', 'CU', 'meva', 'clap')
DIRECTED_B = DIRECTED_A + ('pulse_clarity', 'ibi_cv', 'key_cnn_conf', 'chroma_entropy')
EXCLUDE_OOD = {'OOD-MIDI', 'OOD-instrumental', 'OOD-stems', 'OOD-synthetic',
               'OOD-10s-instrumental'}
LABELS = ('pass', 'partial', 'fail', 'degenerate', 'ood_only', 'inconclusive', 'n/a')


def log(*a):
    C.log('[177]', *a)


# ── pure judging primitives (unit-tested) ────────────────────────────────────
def judge_unit(ds, m, kind, directed=True, ood=''):
    """One §5.2 judgment unit.  ds = list of sign-corrected d values."""
    ds = [float(x) for x in ds if isinstance(x, (int, float)) and math.isfinite(x)]
    base = {'kind': kind, 'n': len(ds), 'ood': ood, 'exclude': ood in EXCLUDE_OOD}
    if m is None or not ds:
        return {**base, 'pass': None, 'status': 'inconclusive'}
    tie = sum(1 for x in ds if abs(x) <= m) / len(ds)
    worse = sum(1 for x in ds if x < -m) / len(ds)
    better = sum(1 for x in ds if x > m) / len(ds)
    degenerate = len(ds) >= 2 and all(abs(x) <= 1e-12 for x in ds)
    passed = (tie >= C.PASS) if kind == 'tie' else ((worse >= C.PASS) if directed else None)
    if degenerate:
        passed = False
    return {**base, 'tie': tie, 'worse': worse, 'better': better,
            'mean_d': sum(ds) / len(ds), 'pass': passed, 'degenerate': degenerate,
            'status': 'degenerate' if degenerate else 'ok'}


def atm_unit(at_m, kind, directed=True, ood='', n=None):
    """Unit whose pair rates were precomputed by an original protocol summary."""
    if not isinstance(at_m, dict):
        return {'kind': kind, 'n': n, 'ood': ood, 'exclude': ood in EXCLUDE_OOD,
                'pass': None, 'status': 'inconclusive'}
    tie = float(at_m.get('tie', float('nan')))
    worse = float(at_m.get('worse', float('nan')))
    if not math.isfinite(tie):
        return {'kind': kind, 'n': n, 'ood': ood, 'exclude': ood in EXCLUDE_OOD,
                'pass': None, 'status': 'inconclusive'}
    passed = (tie >= C.PASS) if kind == 'tie' else ((worse >= C.PASS) if directed else None)
    return {'kind': kind, 'n': n, 'ood': ood, 'exclude': ood in EXCLUDE_OOD,
            'tie': tie, 'worse': worse, 'pass': passed, 'degenerate': False,
            'status': 'ok'}


def _rates(units, kind):
    us = [u for u in units if u.get('kind') == kind and u.get('pass') is not None]
    if not us:
        return None
    return sum(1 for u in us if u['pass']) / len(us)


def cell_from_units(units):
    """Aggregate one dim's units into the §5.4 label."""
    units = [u for u in units or [] if isinstance(u, dict)]
    usable = [u for u in units if u.get('pass') is not None]
    excluded = [u for u in usable if u.get('exclude')]
    in_domain = [u for u in usable if not u.get('exclude')]
    out = {'n_units': len(usable), 'excluded_units': len(excluded),
           'ood_units': sorted({u['ood'] for u in excluded}),
           'degenerate_units': sum(1 for u in in_domain if u.get('degenerate'))}

    def verdict(us):
        tie_r, lose_r = _rates(us, 'tie'), _rates(us, 'lose')
        out.update({'tie_units': sum(1 for u in us if u.get('kind') == 'tie'),
                    'lose_units': sum(1 for u in us if u.get('kind') == 'lose'),
                    'tie_pass_rate': tie_r, 'lose_pass_rate': lose_r})
        if tie_r is None and lose_r is None:
            return 'inconclusive'
        if all(u.get('degenerate') for u in us):
            return 'degenerate'
        ok_t = tie_r is None or tie_r >= C.PASS
        ok_l = lose_r is None or lose_r >= C.PASS
        if ok_t and ok_l:
            return 'pass'
        if ok_t or ok_l:
            return 'partial'
        return 'fail'

    if not units:
        out.update({'label': 'n/a', 'judged_label': 'n/a'})
        return out
    if not in_domain:
        judged = verdict(usable) if usable else 'inconclusive'
        out['judged_label'] = judged
        out['label'] = 'ood_only' if usable else 'inconclusive'
        return out
    judged = verdict(in_domain)
    out['judged_label'] = judged
    out['label'] = judged
    return out


def combine_dims(labels):
    """One matrix cell from its dimensions: no averaging, no best-dim cherry-pick.
    pass only when every measured dim passes; partial when some do; n/a when none."""
    labels = [x for x in labels if x]
    if not labels or all(x == 'n/a' for x in labels):
        return 'n/a'
    if all(x == 'pass' for x in labels):
        return 'pass'
    if any(x == 'pass' for x in labels):
        return 'partial'
    if any(x == 'inconclusive' for x in labels):
        return 'inconclusive'
    if all(x == 'ood_only' for x in labels):
        return 'ood_only'
    if any(x == 'fail' for x in labels):
        return 'fail'
    if any(x == 'degenerate' for x in labels):
        return 'degenerate'
    return labels[0]


def glyph(label):
    return {'pass': '●', 'partial': '●', 'fail': '●', 'degenerate': '●',
            'ood_only': '●', 'inconclusive': '◐', 'n/a': '—'}.get(label, '○')


# ── S lookup + MuQ fallback (§8.1) ───────────────────────────────────────────
MUQ_FALLBACK_S = 0.108679


def s_info(model, protocol, dim):
    m, S = C.m_for(model, protocol, dim)
    info = {'S': S, 'm': m, 'S_source': None, 'S_fallback': False}
    if m is not None:
        if model == 'aes' or model == 'meva' or model == 'clap':
            src = C.S_134_SUMMARY if protocol == '134' else C.S_126A_SUMMARY
        elif model == 'mir':
            src = C.S_FILE / 's_values_mir.json'
        elif model in ('mert', 'muq'):
            src = C.S_133_SUMMARY
        elif model == 'songeval':
            src = C.S_126D_SUMMARY
        else:
            src = C.S_FILE / f's_values_{model}.json'
        info['S_source'] = str(src)
        return info
    if model == 'muq' and isinstance(MUQ_FALLBACK_S, float):
        info.update({'S': MUQ_FALLBACK_S, 'm': C.M_FRAC * MUQ_FALLBACK_S,
                     'S_source': '133', 'S_fallback': True})
    return info


# ── IO helpers ───────────────────────────────────────────────────────────────
def read_unit_tsv(path, cols):
    return C.parse_score_tsv(path, cols)


def unit_files(model, protocol):
    return C.final_tsv_files(OUT / model / protocol, protocol, model)


def _one_tsv(model, protocol):
    files = unit_files(model, protocol)
    if not files:
        return None
    if len(files) > 1:
        # combine shards (126b/126c/134); 126a uses the per-unit reader instead
        merged = {}
        for f in files:
            merged.update(read_unit_tsv(f, C.MODELS[model]['cols']))
        return merged
    return read_unit_tsv(files[0], C.MODELS[model]['cols'])


# ── 126a ─────────────────────────────────────────────────────────────────────
def ood_126a(model):
    return {'singmos': 'OOD-instrumental', 'songeval': 'OOD-10s-instrumental'}.get(model, '')


def new_126a(model):
    """Per-unit (variant x system) verdicts from the job's own 126a TSVs.
    A unit with a `__full__` TSV (ambiguous full rerun) replaces its sampled one."""
    out_dir = OUT / model / '126a'
    cols, signs = C.MODELS[model]['cols'], C.MODELS[model]['signs']
    directed = C.MODELS[model]['directed']
    dims = {c: [] for c in cols}
    ood = ood_126a(model)
    for v in C.VARIANTS:
        if v['kind'] not in ('tie', 'lose'):
            continue
        for s in C.SYSTEMS:
            name = f"{v['name']}__{s}"
            clean_name = f'clean__{s}'
            case_path = C.full_file(out_dir, '126a', model, name)
            clean_path = C.full_file(out_dir, '126a', model, clean_name)
            if not case_path.is_file():
                case_path = C.shard_file(out_dir, '126a', model, 0, name)
            if not clean_path.is_file():
                clean_path = C.shard_file(out_dir, '126a', model, 0, clean_name)
            if not case_path.is_file() or not clean_path.is_file():
                continue
            case = read_unit_tsv(case_path, cols)
            clean = read_unit_tsv(clean_path, cols)
            prefix = f"{v['name']}__{s}__"
            pairs = collections.defaultdict(list)
            for key, sc in case.items():
                if not key.startswith(prefix):
                    continue
                cid = key[len(prefix):]
                b = clean.get(f'{clean_name}__{cid}')
                if not b:
                    continue
                for c, sg in zip(cols, signs):
                    if c in sc and c in b:
                        pairs[c].append(sg * (sc[c] - b[c]))
            for c, d in pairs.items():
                dims[c].append(judge_unit(d, C.m_for(model, '126a', c)[0], v['kind'],
                                          directed[cols.index(c)], ood))
    return {'dims': dims}


def existing_126a():
    """AES/MEva/CLAP from the original 126a summary's at_m; unit = variant x system."""
    summ = C.read_json(C.S_126A_SUMMARY, {})
    cells = summ.get('cells') or {}
    dims = {k: [] for k in ('PQ', 'CE', 'CU', 'PC', 'meva', 'clap')}
    for case, systems in cells.items():
        kind = C.VARIANT_KIND.get(case)
        if kind not in ('tie', 'lose'):
            continue
        for system, cell in systems.items():
            for metric in dims:
                at_m = ((cell.get('metrics') or {}).get(metric) or {}).get('at_m')
                if at_m is None:
                    continue
                dims[metric].append(atm_unit(at_m, kind, metric in DIRECTED_A))
    return {'dims': dims}


def songeval_126a():
    """Per-unit SongEval from the 126d 126a TSVs (the 126d summary is pooled over
    systems; §5.2 wants variant x system units)."""
    root = C.PROTOCOL_126D / '126a'
    cols = C.MODELS['songeval']['cols']
    dims = {c: [] for c in cols}
    for v in C.VARIANTS:
        if v['kind'] not in ('tie', 'lose'):
            continue
        for s in C.SYSTEMS:
            cpath = root / v['name'] / s / 'songeval.tsv'
            bpath = root / 'clean' / s / 'songeval.tsv'
            if not cpath.is_file() or not bpath.is_file():
                continue
            case, clean = read_unit_tsv(cpath, cols), read_unit_tsv(bpath, cols)
            for c in cols:
                d = [case[k][c] - clean[k][c] for k in case
                     if k in clean and c in case[k] and c in clean[k]]
                dims[c].append(judge_unit(d, C.m_for('songeval', '126a', c)[0],
                                          v['kind'], True, 'OOD-10s-instrumental'))
    return {'dims': dims}


def mert_126a():
    """MERT 126a comes from the 133 summary (unit = variant x system)."""
    summ = C.read_json(C.S_133_SUMMARY, {})
    units = []
    for variant, systems in (summ.get('126a') or {}).items():
        kind = C.VARIANT_KIND.get(variant)
        if kind not in ('tie', 'lose'):
            continue
        for system, cell in systems.items():
            if not isinstance(cell, dict) or 'tie' not in cell:
                continue
            units.append(atm_unit({'tie': cell['tie'], 'worse': cell.get('worse', 0.0)},
                                  kind, True, n=cell.get('n')))
    return {'dims': {'mert_cosim': units}}


# ── 126b / rec / floor ───────────────────────────────────────────────────────
def _group_items_126b(model):
    items = C.load_items_file('126b')
    groups = collections.defaultdict(list)
    for it in items:
        if it.get('kind') not in ('tie', 'lose') or not it.get('base'):
            continue
        groups[(it.get('block'), it.get('source'), it.get('case'), it.get('kind'))].append(it)
    return groups


def tsv_126b(model, path):
    """MIDI and rec units (source x case) from one per-clip TSV."""
    cols, signs = C.MODELS[model]['cols'], C.MODELS[model]['signs']
    scores = read_unit_tsv(path, cols)
    if not scores:
        return None
    midi = {c: [] for c in cols}
    rec = {c: [] for c in cols}
    groups = _group_items_126b(model)
    for (block, source, case, kind), its in groups.items():
        ood = C.item_domain(model, '126b', its[0])
        target = rec if block == 'rec' else midi
        for c, sg in zip(cols, signs):
            d = [sg * (scores[it['key']][c] - scores[it['base']][c]) for it in its
                 if it['key'] in scores and it['base'] in scores
                 and c in scores[it['key']] and c in scores[it['base']]]
            if not d:
                continue
            target[c].append(judge_unit(d, C.m_for(model, '126b', c)[0], kind, True, ood))
    return {'midi': midi, 'rec': rec}


def floor_126b_tsv(model, path):
    cols = C.MODELS[model]['cols']
    scores = read_unit_tsv(path, cols)
    if not scores:
        return {}
    items = C.load_items_file('126b')
    floors = [it for it in items if it.get('block') == 'floor']
    out = {}
    for case in sorted({it['case'] for it in floors}):
        ks = [it['key'] for it in floors if it['case'] == case and it['key'] in scores]
        if not ks:
            continue
        out[case] = {}
        for c in cols:
            v = [scores[k][c] for k in ks if c in scores[k]]
            if not v:
                continue
            mean = sum(v) / len(v)
            mv = [scores[it['key']][c] for it in items
                  if it.get('kind') == 'ref' and it.get('block') in ('midi', 'rec')
                  and it['key'] in scores and c in scores[it['key']]]
            out[case][c] = {'mean': mean,
                            'frac_music_below_floor': (sum(1 for x in mv if x < mean) / len(mv)
                                                       if mv else None)}
    return out


def existing_126b():
    """AES/MEva/CLAP from the original 126b summary (at_m per case)."""
    summ = C.read_json(C.S_126B_SUMMARY, {})
    cols = ('PQ', 'CE', 'CU', 'PC', 'meva', 'clap')
    midi = {c: [] for c in cols}
    rec = {c: [] for c in cols}
    floor = {}
    for cell_key, cases in (summ.get('cells') or {}).items():
        block, _, source = cell_key.partition('/')
        target = rec if block == 'rec' else (midi if block == 'midi' else None)
        if target is None or source == 'soundfont_swap':
            continue
        for case, cell in cases.items():
            kind = cell.get('kind')
            if kind not in ('tie', 'lose'):
                continue
            for c in cols:
                at_m = ((cell.get('metrics') or {}).get(c) or {}).get('at_m')
                if at_m is None:
                    continue
                target[c].append(atm_unit(at_m, kind, c in DIRECTED_B, n=cell.get('n')))
    for case, metrics in (summ.get('floors') or {}).items():
        floor[case] = {c: {k: (metrics.get(c) or {}).get(k)
                           for k in ('mean', 'frac_music_below_floor')}
                       for c in cols if metrics.get(c)}
    return {'midi': midi, 'rec': rec, 'floor': floor}


def mert_muq_126b(model):
    """MERT/MuQ 126b from the 133 per-clip cosine TSV."""
    path = C.PROTOCOL_133 / f'126b_{model}_cosim.tsv'
    return tsv_126b(model, path) if path.is_file() else None


def songeval_126b():
    return tsv_126b('songeval', C.PROTOCOL_126D / '126b_songeval.tsv')


# ── 126c ─────────────────────────────────────────────────────────────────────
def _group_items_126c():
    groups = collections.defaultdict(list)
    for it in C.load_items_file('126c'):
        if it.get('kind') not in ('tie', 'lose') or not it.get('base'):
            continue
        groups[(it.get('case'), it.get('kind'))].append(it)
    return groups


def tsv_126c(model, path):
    cols, signs = C.MODELS[model]['cols'], C.MODELS[model]['signs']
    scores = read_unit_tsv(path, cols)
    if not scores:
        return None
    dims = {c: [] for c in cols}
    for (case, kind), its in _group_items_126c().items():
        ood = C.item_domain(model, '126c', its[0])
        for c, sg in zip(cols, signs):
            d = [sg * (scores[it['key']][c] - scores[it['base']][c]) for it in its
                 if it['key'] in scores and it['base'] in scores
                 and c in scores[it['key']] and c in scores[it['base']]]
            if d:
                dims[c].append(judge_unit(d, C.m_for(model, '126c', c)[0], kind, True, ood))
    return {'dims': dims}


def existing_126c():
    summ = C.read_json(C.S_126C_SUMMARY, {})
    cols = ('PQ', 'CE', 'CU', 'PC', 'meva', 'clap', 'pulse_clarity', 'ibi_cv',
            'key_cnn_conf', 'chroma_entropy')
    dims = {c: [] for c in cols}
    for case, cell in (summ.get('cells') or {}).items():
        kind = cell.get('kind')
        if kind not in ('tie', 'lose'):
            continue
        for c in cols:
            at_m = ((cell.get('metrics') or {}).get(c) or {}).get('at_m')
            if at_m is None:
                continue
            dims[c].append(atm_unit(at_m, kind, c in DIRECTED_B, n=cell.get('n')))
    return {'dims': dims}


def mert_muq_126c(model):
    path = C.PROTOCOL_133 / f'126c_{model}_cosim.tsv'
    return tsv_126c(model, path) if path.is_file() else None


def songeval_126c():
    return tsv_126c('songeval', C.PROTOCOL_126D / '126c_songeval.tsv')


# ── 134 ──────────────────────────────────────────────────────────────────────
def _items134_by_cell():
    out = {}
    for it in C.load_items_file('134'):
        out[(it.get('source'), it.get('impl'), it.get('case'))] = it
    return out


def _ood_134(model, it):
    if model == 'singmos':
        return 'OOD-instrumental'
    if model == 'songeval':
        return 'OOD-10s-instrumental'
    try:
        v, axis = float(it.get('value')), it.get('axis')
    except (TypeError, ValueError):
        return ''
    if it.get('kind') == 'lose' and (
            (axis == 'tempo' and (v <= 0.80 or v >= 1.25))
            or (axis == 'pitch' and abs(v) >= 5)):
        return 'OOD-extreme'
    return ''


def tsv_134(model, path):
    cols, signs = C.MODELS[model]['cols'], C.MODELS[model]['signs']
    scores = read_unit_tsv(path, cols)
    if not scores:
        return None
    groups = collections.defaultdict(list)
    for it in C.load_items_file('134'):
        if it.get('kind') not in ('tie', 'lose') or not it.get('base'):
            continue
        groups[(it.get('axis'), it.get('impl'), it.get('value'), it.get('kind'),
                it.get('source'))].append(it)
    dims = {c: [] for c in cols}
    for (axis, impl, value, kind, source), its in groups.items():
        ood = _ood_134(model, its[0])
        for c, sg in zip(cols, signs):
            d = [sg * (scores[it['key']][c] - scores[it['base']][c]) for it in its
                 if it['key'] in scores and it['base'] in scores
                 and c in scores[it['key']] and c in scores[it['base']]]
            if d:
                u = judge_unit(d, C.m_for(model, '134', c)[0], kind, True, ood)
                u['axis'], u['impl'], u['value'], u['source'] = axis, impl, value, source
                dims[c].append(u)
    return {'dims': dims}


def existing_134():
    summ = C.read_json(C.S_134_SUMMARY, {})
    cols = ('PQ', 'CE', 'CU', 'PC', 'meva', 'clap', 'pulse_clarity', 'ibi_cv')
    by_cell = _items134_by_cell()
    dims = {c: [] for c in cols}
    for key, cell in (summ.get('cells') or {}).items():
        kind = cell.get('kind')
        if kind not in ('tie', 'lose'):
            continue
        source, impl, case = key.split('/', 2)
        it = by_cell.get((source, impl, case), {})
        for c in cols:
            at_m = ((cell.get('metrics') or {}).get(c) or {}).get('at_m')
            if at_m is None:
                continue
            ood = ('OOD-instrumental' if c in ('pulse_clarity', 'ibi_cv', 'key_cnn_conf',
                                               'chroma_entropy') else '')
            u = atm_unit(at_m, kind, c in DIRECTED_B, ood=ood, n=cell.get('n'))
            u['axis'] = it.get('axis')
            dims[c].append(u)
    return {'dims': dims}


# ── dispatch + cache ─────────────────────────────────────────────────────────
_CACHE = {}


def evaluate(model, protocol):
    key = (model, protocol)
    if key in _CACHE:
        return _CACHE[key]
    if protocol == '126a':
        if model in ('aes', 'meva', 'clap'):
            res = existing_126a()
        elif model in ('pam', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'singmos'):
            res = new_126a(model)
        elif model == 'songeval':
            res = songeval_126a()
        elif model == 'mert':
            res = mert_126a()
        elif model == 'muq':
            res = new_126a('muq')
        elif model == 'mir':
            res = new_126a('mir')
        else:
            res = {'dims': {}}
    elif protocol == '126b':
        if model in ('aes', 'meva', 'clap'):
            res = existing_126b()
        elif model in ('pam', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'singmos'):
            path = _single_path(model, '126b')
            res = tsv_126b(model, path)
            if res:
                res['floor'] = floor_126b_tsv(model, path)
        elif model == 'songeval':
            res = songeval_126b()
            if res:
                res['floor'] = floor_126b_tsv('songeval',
                                              C.PROTOCOL_126D / '126b_songeval.tsv')
        elif model in ('mert', 'muq'):
            res = mert_muq_126b(model)
        elif model == 'mir':
            res = existing_126b()  # MIR is in the original 126b summary
        else:
            res = None
    elif protocol == '126c':
        if model in ('aes', 'meva', 'clap', 'mir'):
            res = existing_126c()
        elif model in ('pam', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'singmos'):
            res = tsv_126c(model, _single_path(model, '126c'))
        elif model == 'songeval':
            res = songeval_126c()
        elif model in ('mert', 'muq'):
            res = mert_muq_126c(model)
        else:
            res = None
    elif protocol == '134':
        if model in ('aes', 'meva', 'clap', 'mir'):
            res = existing_134() if model != 'mir' else tsv_134('mir', _single_path('mir', '134'))
        elif model in ('pam', 'muqeval_A1', 'muqeval_A3a', 'cmirm', 'singmos', 'muq',
                       'mert', 'songeval'):
            res = tsv_134(model, _single_path(model, '134'))
        else:
            res = None
    else:
        res = None
    _CACHE[key] = res
    return res


def _single_path(model, protocol):
    files = unit_files(model, protocol)
    return files[0] if files else OUT / model / protocol / 'missing.tsv'


# ── matrix assembly ──────────────────────────────────────────────────────────
def row_model(row_id, default):
    return C.EXISTING.get(row_id) or default


MULTI_DIM_MODELS = {'muqeval_A1', 'muqeval_A3a', 'cmirm'}


def row_dims(row_id, model, dim):
    """Multi-dim model rows judge every dimension separately; every other row
    (AES axes, SongEval dims, MIR dims, single-output models) judges its own dim."""
    if model in MULTI_DIM_MODELS:
        return list(C.MODELS[model]['cols'])
    return [dim]


def build():
    t0 = time.time()
    cells = {}
    for n_done, (row_id, model, dim, _label) in enumerate(C.MATRIX_ROWS, 1):
        src_model = row_model(row_id, model)
        dims = row_dims(row_id, src_model, dim) if src_model else [dim]
        cells[row_id] = {}
        for protocol in C.PROTOCOLS:
            res = evaluate(src_model, protocol)
            if res is None:
                cells[row_id][protocol] = {'label': 'inconclusive', 'dims': {},
                                           'reason': 'no scores for this source'}
                continue
            block = (res.get('midi') or {}) if protocol == '126b' else res.get('dims', {})
            dim_out = {}
            for d in dims:
                units = (block or {}).get(d) or []
                cell = cell_from_units(units)
                si = s_info(src_model, protocol, d)
                cell.update({'S': si['S'], 'm': si['m'], 'S_source': si['S_source'],
                             'S_fallback': si['S_fallback']})
                if si['S_fallback']:
                    cell['label'] = 'inconclusive'
                dim_out[d] = cell
            labels = [dim_out[d]['label'] for d in dim_out]
            cells[row_id][protocol] = {
                'label': combine_dims(labels), 'dims': dim_out,
                'source_model': src_model, 'protocol': protocol,
            }
            if protocol == '126b' and dim_out and not any(
                    dim_out[d].get('n_units') for d in dim_out):
                cells[row_id][protocol]['label'] = 'inconclusive'
        # separate rec column
        res = evaluate(src_model, '126b')
        rec_block = (res or {}).get('rec') or {}
        dim_out = {}
        for d in dims:
            cell = cell_from_units(rec_block.get(d) or [])
            si = s_info(src_model, '126b', d)
            cell.update({'S': si['S'], 'm': si['m'], 'S_source': si['S_source'],
                         'S_fallback': si['S_fallback']})
            if si['S_fallback']:
                cell['label'] = 'inconclusive'
            dim_out[d] = cell
        cells[row_id]['rec'] = {'label': combine_dims([c['label'] for c in dim_out.values()]),
                                'dims': dim_out, 'source_model': src_model}
        # floor column (descriptive only)
        fl = (res or {}).get('floor') or {}
        floor_dims = {}
        for d in dims:
            per_case = {case: metrics.get(d) for case, metrics in fl.items()
                        if isinstance(metrics, dict) and metrics.get(d)}
            floor_dims[d] = {'label': 'n/a', 'cases': per_case,
                             'note': 'descriptive percentiles only, no pass/fail (§5.1)'}
        cells[row_id]['floor'] = {
            'label': 'n/a' if src_model in ('mert', 'muq') else 'ood_only',
            'dims': floor_dims, 'source_model': src_model,
            'note': ('N/A for reference-based metrics (§1.4)' if src_model in ('mert', 'muq')
                     else 'descriptive only, not a model verdict')}
        # the guest's stall watchdog only sees files move; judging writes no TSVs
        C.write_json_atomic(OUT / 'judge_heartbeat.json',
                            {'document_kind': 'metric_coverage_judge_heartbeat_v1',
                             'written_at': C.written_at(), 'row': row_id,
                             'n_rows_done': n_done, 'n_rows_total': len(C.MATRIX_ROWS)})
    summary = {
        'document_kind': 'metric_coverage_summary_v1',
        'generated_at': C.written_at(),
        'preregistered': str(C.DOCS / 'metric_coverage_completion_20261027.md'),
        'rules': {'m_frac': C.M_FRAC, 'pass': C.PASS, 'ambiguous_band': C.AMBIG_BAND,
                  'directed_126a': list(DIRECTED_A),
                  'directed_126b_c_134': list(DIRECTED_B),
                  'excluded_ood': sorted(EXCLUDE_OOD),
                  'note': 'OOD-extreme is an annotation, not an exclusion'},
        'sample': {'agreement_sha256': _sha(OUT / 'sample_agreement_20261027.json'),
                   'manifest_sha256': _sha(C.sample_manifest_path()),
                   'agreement': C.sample_agreement()},
        'cells': cells,
        'counts': count_cells(cells),
        'elapsed_s': round(time.time() - t0, 1),
    }
    return summary


def _sha(p):
    return C.sha256_file(p) if Path(p).is_file() else None


def count_cells(cells):
    c = collections.Counter()
    for row in cells.values():
        for protocol in C.PROTOCOLS:
            c[row.get(protocol, {}).get('label', 'missing')] += 1
    c['total'] = len(cells) * len(C.PROTOCOLS)
    c['filled_glyph'] = sum(c[k] for k in ('pass', 'partial', 'fail', 'degenerate',
                                           'ood_only'))
    return dict(c)


def render_md(summary):
    lines = ['# 指標 × 介入覆蓋矩陣（job 177 產生）', '',
             '> 由 `judge_coverage.py` 從 `coverage_summary.json` 產生；'
             '●=有判定、◐=有分數無判定、○=無分數、—=N/A。', '',
             '| # | 指標 | 126a | 126b | rec | floor | 126c | 134 |',
             '|---:|---|---|---|---|---|---|---|']
    for i, (row_id, _m, _d, label) in enumerate(C.MATRIX_ROWS, 1):
        row = summary['cells'][row_id]
        cols = []
        for protocol in C.PROTOCOLS:
            cell = row.get(protocol) or {}
            cell_label = cell.get('label', 'inconclusive')
            cols.append(f'{glyph(cell_label)} {cell_label}')
        lines.append(f"| {i} | {label} | " + ' | '.join(cols) + ' |')
    c = summary['counts']
    lines += ['', f"計數：● {c.get('filled_glyph', 0)}、◐ {c.get('inconclusive', 0)}、"
                  f"— {c.get('n/a', 0)}（總 {c.get('total', 0)} 格）", '']
    return '\n'.join(lines)


def render_html(summary):
    rows = []
    for i, (row_id, _m, _d, label) in enumerate(C.MATRIX_ROWS, 1):
        tds = [f'<td class="lbl">{label}</td>']
        for protocol in C.PROTOCOLS:
            cell = summary['cells'][row_id].get(protocol) or {}
            lab = cell.get('label', 'inconclusive')
            dims = cell.get('dims') or {}
            tip = '; '.join(f'{k}={v.get("label")}' for k, v in dims.items())
            tds.append(f'<td class="{lab}" title="{tip}">{glyph(lab)} {lab}</td>')
        rows.append(f'<tr><td class="num">{i}</td>' + ''.join(tds) + '</tr>')
    head = ''.join(f'<th>{p}</th>' for p in C.PROTOCOLS)
    return f"""<!doctype html><meta charset="utf-8">
<title>metric coverage 2026-10-27</title>
<style>
body{{font:14px/1.4 system-ui,sans-serif;margin:24px;color:#111}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:4px 8px}}
th{{background:#f3f3f3}}.lbl{{text-align:left}}.num{{color:#888}}
td.pass{{background:#d9f2d9}}td.partial{{background:#fff0c2}}td.fail{{background:#f8d7da}}
td.degenerate{{background:#e9d8fd}}td.ood_only{{background:#e2e8f0}}
td.inconclusive{{background:#f5f5f5}}td.n/a{{background:#fff;color:#aaa}}
</style><h2>指標 × 介入覆蓋（22 × 6）</h2>
<table><thead><tr><th>#</th><th>指標</th>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p>●=有判定、◐=有分數無判定、○=無分數、—=N/A。滑過格子可看各維度標籤。</p>
"""


def deck_data(summary):
    rows = []
    for row_id, _m, _d, label in C.MATRIX_ROWS:
        row = {'row': row_id, 'label': label, 'protocols': {}}
        for protocol in C.PROTOCOLS:
            cell = summary['cells'][row_id].get(protocol) or {}
            row['protocols'][protocol] = {
                'label': cell.get('label'), 'glyph': glyph(cell.get('label')),
                'dims': {k: {'label': v.get('label'),
                             'tie_pass_rate': v.get('tie_pass_rate'),
                             'lose_pass_rate': v.get('lose_pass_rate'),
                             'n_units': v.get('n_units')}
                         for k, v in (cell.get('dims') or {}).items()},
            }
        rows.append(row)
    return {'document_kind': 'metric_coverage_deck_v1', 'generated_at': C.written_at(),
            'counts': summary['counts'], 'rows': rows}


# ── job entry points ─────────────────────────────────────────────────────────
def require_inputs():
    need = [OUT / 'prep_161.json', OUT / 'sample_agreement_20261027.json',
            C.S_FILE / 's_values_mir.json']
    for j in range(163, 177):
        need.append(OUT / 'jobs' / f'job_done_{j}.json')
    for model, protocols in (('muqeval_A1', ('126a', '126b', '126c', '134')),
                             ('muqeval_A3a', ('126a', '126b', '126c', '134')),
                             ('cmirm', ('126a', '126b', '126c', '134')),
                             ('pam', ('126a', '126b', '126c', '134')),
                             ('singmos', ('126a', '126b', '126c', '134')),
                             ('muq', ('126a', '134')),
                             ('mert', ('134',)), ('songeval', ('134',)),
                             ('mir', ('126a', '134'))):
        for p in protocols:
            if C.final_tsv_files(OUT / model / p, p, model):
                continue
            need.append(OUT / model / p)
    missing = [str(p) for p in need if not Path(p).exists()]
    if missing:
        print('PREREQUISITE MISSING:', file=sys.stderr)
        for m in missing:
            print('  ' + m, file=sys.stderr)
        raise SystemExit(4)


def run(dry_run=False):
    if dry_run:
        require_inputs()
        log('dry-run: all judge inputs present')
        return 0
    require_inputs()
    summary = build()
    C.write_json_atomic(OUT / 'coverage_summary.json', summary)
    (OUT / 'coverage_matrix.md').write_text(render_md(summary), encoding='utf-8')
    (OUT / 'coverage_matrix.html').write_text(render_html(summary), encoding='utf-8')
    C.write_json_atomic(OUT / 'deck_data.json', deck_data(summary))
    C.write_json_atomic(OUT / 'jobs' / 'job_done_177.json',
                        {'job': 177, 'written_at': C.written_at(),
                         'counts': summary['counts']})
    log(f"counts: {summary['counts']}")
    return 0


def run_cli(dry_run=False):
    return run(dry_run=dry_run)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    return run_cli(dry_run=a.dry_run)


if __name__ == '__main__':
    sys.exit(main())
