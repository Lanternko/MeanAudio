#!/usr/bin/env python3
"""p2 139 action: segment-length sensitivity of the table-B metrics (EXPLORATORY, no claim, no threshold).

Spec (pre-registered, frozen): docs/experiments/seed_robustness_table_b_135_20261021.md §4.
Material  seed-42 lvl30 audio of the existing runs: ours CFG3+neg x 3 training seeds (115 / 118),
          MusicGen-medium, MusicLDM, MusicCaps real recordings (5,131 clips with a reference file).
Versions  full 10 s | first 5 s | last 5 s (each cut from the lvl30 file, then re-gained to -30 LUFS
          with the 115 rule: silent clips are not gained, peak cap 0.999, FLAC PCM_24) | mean of the
          two 5 s scores (computed in the analysis).
Metrics   AES CE/CU/CU/PC/PQ + CLAP batch 1 (eval_metrics.py), MEva (meva_score_dir.py),
          SongEval se_* (126d runner functions) and MIR pulse_clarity / key_cnn_conf (mir_metrics.py).
          Scores of the full version: AES/CLAP/MEva come from the 115/118 lvl30 tables (identical
          protocol); SongEval and MIR are scored here on the same lvl30 files.
Report    (a) per-clip Spearman full vs mean-of-halves; (b) system ranking (Kendall tau, sign of every
          pairwise system-mean difference, on the 5,131 clips every system has); (c) the table-B H1 / H2
          differences (ours mean of 3 training seeds minus MusicGen: PQ, CLAP) per version;
          (d) first-minus-last half difference per system and metric. Paired prompt bootstrap
          (2000, percentile 95%) for (c) and (d). Only valid "under inference seed 42".

Usage   segment_length_139.py [--root DIR] [--limit N] [--report-json P --report-md P]
        or through the queue guest with GPU_QUEUE_CONTRACT (--preflight / --validate-only).
"""
import argparse
import csv
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'songeval_src'))
import seed_robustness_135 as sr  # noqa: E402
import ttm_metrics_115 as p115  # noqa: E402

ROOT = sr.ROOT / 'seg139'
REPORT_JSON = sr.REPO / 'docs/experiments/results/segment_length_139_summary.json'
REPORT_MD = sr.REPO / 'docs/experiments/results/segment_length_139_tables.md'
VERSIONS = ('first5', 'last5')
SEG_SECONDS = 5.0
SE_DIMS = ('se_coh', 'se_mem', 'se_nat', 'se_cla', 'se_mus')
MIR_COLS = ('pulse_clarity', 'key_cnn_conf')
SCORES = ('PQ', 'CE', 'CU', 'PC', 'clap', 'meva', *SE_DIMS, *MIR_COLS)
N_BOOT = 2000
BOOT_SEED = 20261021
REAL_LVL30 = sr.M115 / 'musiccaps_real' / 'lvl30'


def entities():
    """name -> (existing lvl30 dir, sampling rate, kind)."""
    out = {}
    for t in sr.TRAIN_SEEDS:
        d, n = sr.existing_lvl30_dir(sr.ours_share(t, 'cfg3_neg', 42))
        out[f'ours_{t}_cfg3_neg'] = (d, 16000)
    for s in ('musicgen_medium', 'musicldm'):
        d, n = sr.existing_lvl30_dir(sr.ext_share(s, 42))
        out[s] = (d, sr.EXTERNAL[s]['sr'])
    out['real'] = (REAL_LVL30, None)
    return out


def tsv_for(name, root, ids_tsv, limit):
    """TSV with exactly the ids this entity has (real: the 5,131 with a reference), first `limit` if set."""
    with open(sr.TSV, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f, delimiter='\t')
        fields, rows = rd.fieldnames, list(rd)
    if name == 'real':
        have = {p.stem for p in (REAL_LVL30 / 'audio').glob('*.flac')}
        rows = [r for r in rows if r['id'] in have]
    if limit:
        rows = rows[:limit]
    dst = Path(ids_tsv)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    return dst, [r['id'] for r in rows]


def contract(root, tsv):
    return sr.c115(root, tsv)


def sysrec(name, audio):
    return {'name': name, 'kind': 'external', 'audio': str(audio), 'sampling_rate': 16000, 'fad': False}


def cut_done(d, ids):
    return d.is_dir() and all((d / f'{i}.flac').is_file() for i in ids)


def cut_segments(src_audio, dst, ids, which):
    """Cut the first / last 5 s of every clip; atomic per directory."""
    if cut_done(dst, ids):
        return
    tmp = dst.with_name(dst.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.mkdir(parents=True)
    for i in ids:
        wav, rate = sf.read(str(src_audio / f'{i}.flac'), dtype='float64', always_2d=True)
        n = int(round(SEG_SECONDS * rate))
        seg = wav[:n] if which == 'first5' else wav[-n:]
        sf.write(str(tmp / f'{i}.flac'), seg, rate, format='FLAC', subtype='PCM_24')
    tmp.rename(dst)


def tsv_rows(path, key='id'):
    with open(path, newline='') as f:
        return {r[key]: r for r in csv.DictReader(f, delimiter='\t')}


def extra_done(path, ids, cols):
    if not path.is_file():
        return False
    rows = tsv_rows(path)
    return all(i in rows and all(rows[i].get(c, '') != '' for c in cols) for i in ids)


def write_extra(path, ids, vals, cols):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['id', *cols])
        for i in ids:
            w.writerow([i, *[repr(float(vals[i][c])) for c in cols]])
    tmp.rename(path)


def score_extras(audio_dir, ids, out_dir, muq_gen):
    """SongEval + MIR for one directory of lvl30 clips."""
    out_dir.mkdir(parents=True, exist_ok=True)
    se, mir = out_dir / 'songeval.tsv', out_dir / 'mir.tsv'
    paths = [str(Path(audio_dir) / f'{i}.flac') for i in ids]
    if not extra_done(se, ids, SE_DIMS):
        import metric_win_tie_126d as S
        muq, gen = muq_gen()
        t0 = time.time()
        res = S.score_files(paths, muq, gen, 'cuda', progress=False)
        write_extra(se, ids, {i: dict(zip(SE_DIMS, res[p])) for i, p in zip(ids, paths)}, SE_DIMS)
        p115.log(f'SONGEVAL {out_dir.parent.name}/{out_dir.name}: {len(ids)} clips, {time.time() - t0:.0f}s')
    if not extra_done(mir, ids, MIR_COLS):
        import mir_metrics as M
        t0 = time.time()
        res = M.score_mir(paths)
        write_extra(mir, ids, {i: res[p] for i, p in zip(ids, paths)}, MIR_COLS)
        p115.log(f'MIR {out_dir.parent.name}/{out_dir.name}: {len(ids)} clips, {time.time() - t0:.0f}s')


def make_muq_loader():
    cache = {}

    def load():
        if 'm' not in cache:
            import metric_win_tie_126d as S
            cache['m'] = S.load_songeval('cuda')
        return cache['m']
    return load


def unit_tables(c, name, ids, kind):
    """Per-clip dict of all SCORES for one version of one entity (lvl30 block already scored)."""
    s = sysrec(name, '')
    pc = tsv_rows(p115.metrics_json(c, s, 'lvl30').parent / 'per_clip.tsv')
    mv = tsv_rows(p115.meva_json(c, s, 'lvl30').parent / 'meva_per_clip.tsv')
    base = p115.root(c) / name / 'lvl30'
    se, mir = tsv_rows(base / 'extras' / 'songeval.tsv'), tsv_rows(base / 'extras' / 'mir.tsv')
    out = np.empty((len(SCORES), len(ids)))
    for j, i in enumerate(ids):
        for k, m in enumerate(SCORES):
            if m in ('PQ', 'CE', 'CU', 'PC', 'clap'):
                out[k, j] = float(pc[i][m])
            elif m == 'meva':
                out[k, j] = float(mv[i]['meva_raw'])
            elif m in SE_DIMS:
                out[k, j] = float(se[i][m])
            else:
                out[k, j] = float(mir[i][m])
    return out


def full_tables(root, entity, ids, d):
    """The full-version table: AES/CLAP/MEva from the 115/118 blocks, SongEval/MIR from this run."""
    pc = tsv_rows(next((d / 'scores').glob('*/per_clip.tsv')))
    mv = tsv_rows(d / 'meva' / 'meva_per_clip.tsv')
    se, mir = tsv_rows(root / entity / 'full' / 'extras' / 'songeval.tsv'), tsv_rows(root / entity / 'full' / 'extras' / 'mir.tsv')
    out = np.empty((len(SCORES), len(ids)))
    for j, i in enumerate(ids):
        for k, m in enumerate(SCORES):
            if m in ('PQ', 'CE', 'CU', 'PC', 'clap'):
                out[k, j] = float(pc[i][m])
            elif m == 'meva':
                out[k, j] = float(mv[i]['meva_raw'])
            elif m in SE_DIMS:
                out[k, j] = float(se[i][m])
            else:
                out[k, j] = float(mir[i][m])
    return out


def run_units(root, limit):
    loader = make_muq_loader()
    ents = entities()
    ids_by = {}
    for name, (d, rate) in ents.items():
        tsv, ids = tsv_for(name, root, root / name / 'ids.tsv', limit)
        ids_by[name] = ids
        c = contract(root, tsv)
        # full version: only SongEval + MIR are new
        score_extras(d / 'audio', ids, root / name / 'full' / 'extras', loader)
        for v in VERSIONS:
            vname = f'{name}__{v}'
            cut = root / name / f'cut_{v}'
            t0 = time.time()
            cut_segments(d / 'audio', cut, ids, v)
            s = sysrec(vname, cut)
            s_audio = root / vname / 'lvl30' / 'audio'
            if not p115.gain_done(c, s, 'lvl30'):
                p115.make_block(c, s, 'lvl30')
            if not p115.metrics_done(c, s, 'lvl30'):
                p115.run_metrics(c, s, 'lvl30')
            if not p115.meva_done(c, s, 'lvl30'):
                p115.run_meva(c, s, 'lvl30')
            score_extras(s_audio, ids, root / vname / 'lvl30' / 'extras', loader)
            p115.log(f'UNIT {vname}: {len(ids)} clips, {time.time() - t0:.0f}s (incl. cached stages)')
    return ents, ids_by


def load_all(root, ents, ids_by):
    T = {}
    for name, (d, rate) in ents.items():
        ids = ids_by[name]
        T[name] = {'full': full_tables(root, name, ids, d)}
        for v in VERSIONS:
            vname = f'{name}__{v}'
            c = contract(root, root / name / 'ids.tsv')
            T[name][v] = unit_tables(c, vname, ids, 'seg')
        T[name]['mean5'] = (T[name]['first5'] + T[name]['last5']) / 2
    return T


# ── analysis ─────────────────────────────────────────────
def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    s = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float('nan')
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau(a, b):
    n, s = len(a), 0.0
    for i, j in itertools.combinations(range(n), 2):
        s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float(s / (n * (n - 1) / 2))


def nanmean(x, axis=-1):
    return np.nanmean(x, axis=axis)


def analyse(T, ids_by):
    rng = np.random.default_rng(BOOT_SEED)
    names = list(T)
    # restrict every entity to the ids every entity has (the real set is the smallest)
    pos = {n: {i: j for j, i in enumerate(ids_by[n])} for n in names}
    real_ids = [i for i in ids_by['real'] if all(i in pos[n] for n in names)]
    C = {}
    for n in names:
        sel = [pos[n][i] for i in real_ids]
        C[n] = {v: T[n][v][:, sel] for v in ('full', 'first5', 'last5', 'mean5')}
    P = len(real_ids)
    res = {'document_kind': 'segment_length_139_summary_v1', 'exploratory': True,
           'inference_seed': 42, 'n_common_prompts': P, 'n_boot': N_BOOT, 'entities': names,
           'scores': list(SCORES)}
    # (a) Spearman
    res['spearman_full_vs_mean5'] = {n: {m: spearman(T[n]['full'][k], T[n]['mean5'][k]) for k, m in enumerate(SCORES)}
                                     for n in names}
    res['spearman_first5_vs_last5'] = {n: {m: spearman(T[n]['first5'][k], T[n]['last5'][k]) for k, m in enumerate(SCORES)}
                                       for n in names}
    # (b) ranking
    means = {v: np.array([[nanmean(C[n][v][k]) for n in names] for k in range(len(SCORES))])
             for v in ('full', 'first5', 'last5', 'mean5')}
    res['system_means_common_ids'] = {v: {m: dict(zip(names, means[v][k].tolist())) for k, m in enumerate(SCORES)}
                                      for v in means}
    rank = {}
    for k, m in enumerate(SCORES):
        row = {'kendall_tau_full_vs_mean5': kendall_tau(means['full'][k], means['mean5'][k]),
               'kendall_tau_full_vs_first5': kendall_tau(means['full'][k], means['first5'][k]),
               'kendall_tau_full_vs_last5': kendall_tau(means['full'][k], means['last5'][k])}
        flips = []
        for (i, a), (j, b) in itertools.combinations(list(enumerate(names)), 2):
            d_full = means['full'][k][i] - means['full'][k][j]
            d_5 = means['mean5'][k][i] - means['mean5'][k][j]
            flips.append({'pair': f'{a} - {b}', 'full': float(d_full), 'mean5': float(d_5),
                          'sign_changed': bool(np.sign(d_full) != np.sign(d_5))})
        row['pairs'] = flips
        row['n_sign_changes'] = sum(f['sign_changed'] for f in flips)
        rank[m] = row
    res['ranking'] = rank
    # (c) H1 / H2 differences per version, (d) first - last
    ours = [n for n in names if n.startswith('ours_')]
    W = rng.multinomial(P, np.full(P, 1.0 / P), size=N_BOOT).T / P

    def boot(vec):  # vec: per-prompt paired difference (P,)
        v = np.nan_to_num(vec)
        r = v @ W
        return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
    h = {}
    for m in ('PQ', 'clap'):
        k = SCORES.index(m)
        h[m] = {}
        for v in ('full', 'first5', 'last5', 'mean5'):
            o = np.mean([C[n][v][k] for n in ours], axis=0)
            dvec = o - C['musicgen_medium'][v][k]
            h[m][v] = {'mean_diff': float(np.nanmean(dvec)), 'ci95_prompt_bootstrap': boot(dvec)}
    res['h1_h2_ours_minus_musicgen'] = h
    half = {}
    for n in names:
        half[n] = {}
        for k, m in enumerate(SCORES):
            dvec = C[n]['first5'][k] - C[n]['last5'][k]
            half[n][m] = {'first_minus_last': float(np.nanmean(dvec)), 'ci95': boot(dvec)}
    res['first_minus_last'] = half
    return res


def f(x, nd=4):
    return 'nan' if x is None or not np.isfinite(x) else f'{x:+.{nd}f}'


def markdown(r):
    t = ['# 139 segment-length sensitivity (EXPLORATORY: seed 42 only, no claim)', '',
         f'common prompts {r["n_common_prompts"]} (those the real set has); bootstrap {r["n_boot"]}.', '',
         '## (a) per-clip Spearman, full 10 s vs mean of the two 5 s scores', '',
         '| entity | ' + ' | '.join(r['scores']) + ' |', '|---|' + '---|' * len(r['scores'])]
    for n, row in r['spearman_full_vs_mean5'].items():
        t.append(f'| {n} | ' + ' | '.join(f(row[m], 3) for m in r['scores']) + ' |')
    t += ['', '## (b) system ranking (system means on the common ids)', '',
          '| metric | tau full~mean5 | tau full~first5 | tau full~last5 | pair sign changes (of 15) |', '|---|---|---|---|---|']
    for m, row in r['ranking'].items():
        t.append(f'| {m} | {f(row["kendall_tau_full_vs_mean5"], 3)} | {f(row["kendall_tau_full_vs_first5"], 3)} | '
                 f'{f(row["kendall_tau_full_vs_last5"], 3)} | {row["n_sign_changes"]} |')
    t += ['', '## (c) table-B H1 / H2 difference (ours mean of 3 training seeds CFG3+neg - MusicGen) by version', '',
          '| metric | version | mean diff | 95% CI |', '|---|---|---|---|']
    for m, d in r['h1_h2_ours_minus_musicgen'].items():
        for v, x in d.items():
            t.append(f'| {m} | {v} | {f(x["mean_diff"])} | [{f(x["ci95_prompt_bootstrap"][0])}, {f(x["ci95_prompt_bootstrap"][1])}] |')
    t += ['', '## (d) first 5 s minus last 5 s', '', '| entity | ' + ' | '.join(r['scores']) + ' |',
          '|---|' + '---|' * len(r['scores'])]
    for n, row in r['first_minus_last'].items():
        t.append(f'| {n} | ' + ' | '.join(f(row[m]['first_minus_last']) for m in r['scores']) + ' |')
    return '\n'.join(t) + '\n'


def write_report(r, jp, mp):
    Path(jp).parent.mkdir(parents=True, exist_ok=True)
    Path(jp).write_text(json.dumps(r, indent=1))
    Path(mp).write_text(markdown(r))


# ── queue interface ──────────────────────────────────────
def preflight(c):
    b = c['bindings']
    for k in ('action', 'seed_robustness_135', 'ttm_metrics_115', 'eval_metrics', 'mir_metrics', 'songeval_runner'):
        if sr.sha256_file(b[k]) != b[f'{k}_sha256']:
            sys.exit(f'{k} sha mismatch')
    for name, (d, _) in entities().items():
        if not (d / 'audio').is_dir():
            sys.exit(f'{name}: seed-42 lvl30 audio missing: {d}')
    free = shutil.disk_usage(sr.ROOT.parent).free
    if free < c['fixed_protocol']['min_free_bytes']:
        sys.exit(f'free space {free} below the contract floor')
    print('preflight ok')


def validate(c):
    r = json.loads(Path(c['reports'][0]['path']).read_text())
    if not r.get('exploratory') or r.get('n_boot') != N_BOOT or len(r['entities']) != 5:
        sys.exit('bad 139 summary')
    if r['n_common_prompts'] < 5000:
        sys.exit('139 summary covers too few prompts (limit run?)')
    print(json.dumps({'n_common_prompts': r['n_common_prompts'],
                      'H1_PQ_by_version': {v: x['mean_diff'] for v, x in r['h1_h2_ours_minus_musicgen']['PQ'].items()}}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--report-json')
    ap.add_argument('--report-md')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    if os.environ.get('GPU_QUEUE_CONTRACT') and not a.limit:
        c = json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())
        if a.preflight:
            preflight(c)
            return 0
        if a.validate_only:
            validate(c)
            return 0
        a.report_json, a.report_md = c['reports'][0]['path'], c['reports'][1]['path']
    root = Path(a.root)
    if a.limit and Path(a.root) == ROOT:
        sys.exit('--limit requires a different --root (pilot output must not mix with the real run)')
    ents, ids_by = run_units(root, a.limit)
    r = analyse(load_all(root, ents, ids_by), ids_by)
    r['limit'] = a.limit or None
    r['analysis_sha256'] = sr.sha256_file(__file__)
    jp = a.report_json or root / 'summary.json'
    mp = a.report_md or root / 'tables.md'
    write_report(r, jp, mp)
    p115.log(f'139 report: {jp}, {mp}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
