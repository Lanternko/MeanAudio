#!/usr/bin/env python3
"""126b: per-pair win/tie/lose on famous pieces (MIDI / real / poor performances) and floors.

Design: docs/experiments/metric_win_tie_126_20261016.md (MIDI block, recording block, floors).
Stimuli: built by metric_win_tie_126b_stimuli.py (CPU) into STIM/items.json; this script only scores.

Scoring: items are grouped into units of <= UNIT clips (one source per unit), symlinked into a unit
audio dir and scored with AES (4 axes), MEva, CLAP (batch 1, group caption) and VGGish frames
(float16, for per-case FAD). MIR (pulse clarity, IBI CV, key CNN confidence, chroma entropy) runs on
the clips flagged mir (MIDI cases, clean recording crops, floors) as a CPU subprocess next to the GPU loop.

Reading (analyze), same rule as 126a: d = score(case) - score(its clean base);
tie if |d| <= m, m = 0.10 S; S = the 126a S (range of the 5 TTM systems' clean means) for the
neural metrics and the SD of all clean clips here for MIR; pass = 80 % of pairs.
kind: tie (126a signal ties) and soft (transpose +-1, tempo +-2/5 %, velocity x0.85, SoundFont swap)
should tie; lose should be worse. Cross comparisons and floors are descriptive win rates.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.pop('MKL_NUM_THREADS', None)  # MKL_NUM_THREADS hangs the forked loader pool in the dac venv

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(REPO / 'scripts/eval'))
import metric_win_tie_126a as A  # noqa: E402  (judge rule, signal ladders, MEva path)

OUT = Path('/home/kojiek/eval_output_nvme/metric_win_tie_126b_20261016')
STIM = OUT / 'stimuli'
ROOT = OUT
REPORT = REPO / 'docs/experiments/results/metric_win_tie_126b_summary.json'
REPORT_126A = REPO / 'docs/experiments/results/metric_win_tie_126a_summary.json'
UNIT = 3000
MIN_FREE = 20 * 2 ** 30
NEURAL = A.METRICS  # PQ CE CU PC meva clap
MIR = {'pulse_clarity': 1, 'key_cnn_conf': 1, 'ibi_cv': -1, 'chroma_entropy': -1}  # sign: +1 higher is better
DIRECTED = (*A.DIRECTED, *MIR)
M_FRAC, PASS, M_CURVE = A.M_FRAC, A.PASS, A.M_CURVE
MIDI_LADDERS = {
    'wrong': ['wrong1_semi', 'wrong1_tritone', 'wrong5pct', 'wrong10pct', 'wrong30pct', 'wrong100pct'],
    'randpitch': ['randpitch10pct', 'randpitch30pct', 'randpitch100pct'],
    'posshuffle': ['posshuffle10pct', 'posshuffle30pct', 'posshuffle100pct'],
    'pitchshuffle': ['pitchshuffle2s', 'pitchshuffle_all'],
    'jitter': ['jitter10ms', 'jitter30ms', 'jitter80ms', 'jitter200ms'],
    'unsteady': ['unsteady5pct', 'unsteady15pct', 'unsteady30pct'],
    'blockshuffle': ['blockshuffle3.0s', 'blockshuffle1.5s', 'blockshuffle0.5s'],
    'drop': ['drop10pct', 'drop30pct', 'drop60pct'], 'extra': ['extra10pct', 'extra30pct'],
    'blastnote': ['blastnote_x1', 'blastnote_x3', 'blastnote_x10'],
    'transpose': ['transpose-1', 'transpose+1'], 'tempo': ['tempo_x0.98', 'tempo_x1.02', 'tempo_x0.95', 'tempo_x1.05'],
}
log = A.log


def items(limit=None):
    rows = json.loads((STIM / 'items.json').read_text())
    if limit:  # smoke: first `limit` excerpts per source, all their cases, plus all floors
        keep, seen = set(), {}
        for r in rows:
            if r['block'] == 'floor':
                continue
            ex = seen.setdefault(r['source'], [])
            if r['excerpt'] not in ex and len(ex) < limit:
                ex.append(r['excerpt'])
            if r['excerpt'] in ex:
                keep.add(r['key'])
        rows = [r for r in rows if r['block'] == 'floor' or r['key'] in keep]
    return rows


def units(rows):
    by = {}
    for r in rows:
        by.setdefault(r['source'], []).append(r)
    out = []
    for src in sorted(by):
        rs = by[src]
        for k in range(0, len(rs), UNIT):
            out.append((f'{src}_{k // UNIT:03d}', rs[k:k + UNIT]))
    return out


def unit_done(d, keys):
    try:
        got = [r['key'] for r in csv.DictReader(open(d / 'scores.tsv'), delimiter='\t')]
        meta = json.loads((d / 'done.json').read_text())
    except (OSError, ValueError):
        return False
    return got == keys and meta.get('n') == len(keys) and (d / 'vggish_frames.npz').is_file()


def run_unit(name, rs, embedder, clap_model):
    from eval_metrics import score_aes, score_clap
    d, keys = ROOT / 'units' / name, [r['key'] for r in rs]
    if unit_done(d, keys):
        return
    t0 = time.time()
    audio = d / 'audio'
    shutil.rmtree(d, ignore_errors=True)
    audio.mkdir(parents=True)
    for r in rs:
        (audio / f'{r["key"]}.flac').symlink_to(r['path'])
    meva_out = d / 'meva'
    meva_out.mkdir()
    meva = subprocess.Popen([A.MEVA_PY, str(REPO / 'scripts/eval/meva_score_dir.py'), '--audio_dir', str(audio),
                             '--out_dir', str(meva_out)], cwd=REPO, stdout=open(d / 'meva.log', 'w'), stderr=subprocess.STDOUT)
    aes, failed = score_aes([audio / f'{k}.flac' for k in keys], progress=False)
    if failed or len(aes) != len(keys):
        raise SystemExit(f'{name}: AES incomplete {len(aes)}/{len(keys)}')
    aes = {Path(p).name[:-5]: v for p, v in aes.items()}
    clap = score_clap([(r['key'], r['caption']) for r in rs], audio, model=clap_model, progress=False)
    if len(clap) != len(keys):
        raise SystemExit(f'{name}: CLAP incomplete {len(clap)}/{len(keys)}')
    import torch
    from ttm_fad_lvl30 import load_one
    emb = {}
    with torch.no_grad():
        for key, x in embedder.pool.imap(load_one, [(k, str(audio / f'{k}.flac'), False) for k in keys], chunksize=8):
            emb[key] = embedder.fad.model.forward(torch.from_numpy(x)).detach().cpu().numpy().astype(np.float16)
    lens = np.array([len(emb[k]) for k in keys])
    np.savez(d / 'vggish_frames.npz', frames=np.concatenate([emb[k] for k in keys]), lens=lens, keys=np.array(keys))
    if meva.wait() != 0:
        raise SystemExit(f'{name}: MEva failed, see {d / "meva.log"}')
    mv = {r['id']: float(r['meva_raw']) for r in csv.DictReader(open(meva_out / 'meva_per_clip.tsv'), delimiter='\t')}
    if set(mv) != set(keys):
        raise SystemExit(f'{name}: MEva incomplete {len(mv)}/{len(keys)}')
    with open(d / 'scores.tsv', 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['key', *NEURAL])
        for k in keys:
            vals = [aes[k]['PQ'], aes[k]['CE'], aes[k]['CU'], aes[k]['PC'], mv[k], clap[k]]
            if not all(map(math.isfinite, vals)):
                raise SystemExit(f'{name}: nonfinite score for {k}')
            w.writerow([k, *[f'{x:.6f}' for x in vals]])
    (d / 'done.json').write_text(json.dumps({'n': len(keys), 'seconds': time.time() - t0}))
    shutil.rmtree(audio)
    log(f'unit done {name} n={len(keys)} {time.time() - t0:.0f}s')


def mir_worker(limit):
    from mir_metrics import score_mir
    rows = [r for r in items(limit) if r['mir']]
    per = score_mir([r['path'] for r in rows], madmom_workers=6, chroma_workers=3)
    by_path = {r['path']: r['key'] for r in rows}
    (ROOT / 'mir.json').write_text(json.dumps({by_path[p]: v for p, v in per.items()}))
    return 0


def run(rows, limit):
    import ttm_fad_lvl30 as F
    embedder = F.Embedder()  # forks the worker pool first, while this process is single-threaded
    from eval_metrics import load_clap
    clap_model = load_clap()
    mir = None
    if not (ROOT / 'mir.json').is_file():
        cmd = [sys.executable, __file__, '--mir-worker'] + (['--limit', str(limit)] if limit else [])
        mir = subprocess.Popen(cmd, stdout=open(ROOT / 'mir.log', 'w'), stderr=subprocess.STDOUT)
    todo = units(rows)
    for k, (name, rs) in enumerate(todo):
        if shutil.disk_usage(ROOT).free < MIN_FREE // 2:
            raise SystemExit('disk below half the preflight minimum')
        log(f'[{k + 1}/{len(todo)}] {name}')
        run_unit(name, rs, embedder, clap_model)
    embedder.pool.close()
    if mir is not None:
        log('waiting for MIR')
        if mir.wait() != 0:
            raise SystemExit(f'MIR failed, see {ROOT / "mir.log"}')


# ── analysis ──
def load_all(rows):
    sc, frames = {}, {}
    for name, rs in units(rows):
        d = ROOT / 'units' / name
        for r in csv.DictReader(open(d / 'scores.tsv'), delimiter='\t'):
            sc[r['key']] = {m: float(r[m]) for m in NEURAL}
        z = np.load(d / 'vggish_frames.npz')
        off = np.concatenate([[0], np.cumsum(z['lens'])])
        for i, k in enumerate(z['keys']):
            frames[str(k)] = z['frames'][off[i]:off[i + 1]].astype(np.float64)
    mir = json.loads((ROOT / 'mir.json').read_text())
    for k, v in mir.items():
        for m, sign in MIR.items():
            x = v.get(m, float('nan'))
            sc[k][m] = sign * x if math.isfinite(x) else float('nan')  # stored sign-flipped: higher = better for all
    return sc, frames


def fad(fa, fb):
    a, b = np.concatenate(fa), np.concatenate(fb)
    return A.frechet({'mu': a.mean(0), 'sigma': np.cov(a, rowvar=False)}, {'mu': b.mean(0), 'sigma': np.cov(b, rowvar=False)})


def judge(d, m, metric, kind):
    """126a.judge with 126b's DIRECTED (adds the MIR features)"""
    out = {'tie': float(np.mean(np.abs(d) <= m)), 'worse': float(np.mean(d < -m)), 'better': float(np.mean(d > m))}
    if kind == 'tie':
        out['pass'] = out['tie'] >= PASS
    elif metric in DIRECTED and kind == 'lose':
        out['pass'] = out['worse'] >= PASS
    return out


def cell(pairs, sc, S, kind):
    out = {'kind': kind, 'n': len(pairs), 'metrics': {}}
    for m in S:
        d = np.array([sc[a][m] - sc[b][m] for a, b in pairs if m in sc[a] and m in sc[b]])
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        mk = 'tie' if kind in ('tie', 'soft') else kind
        out['metrics'][m] = {'n': len(d), 'mean_delta': float(d.mean()), 'median_delta': float(np.median(d)),
                             'abs_delta_p90': float(np.quantile(np.abs(d), 0.9)),
                             'at_m': judge(d, M_FRAC * S[m], m, mk),
                             'curve': {str(f): judge(d, f * S[m], m, mk) for f in M_CURVE}}
    return out


def winrate(xs, ys, m):
    """unpaired: fraction of all (x, y) pairs judged x better / tie / worse at tolerance m"""
    x, y = np.asarray(xs)[:, None], np.asarray(ys)[None, :]
    d = (x - y).ravel()
    d = d[np.isfinite(d)]
    return {'better': float(np.mean(d > m)), 'tie': float(np.mean(np.abs(d) <= m)), 'worse': float(np.mean(d < -m)),
            'mean_x': float(np.nanmean(xs)), 'mean_y': float(np.nanmean(ys)), 'n_x': len(xs), 'n_y': len(ys)}


def analyze(rows):
    src = REPORT_126A if ROOT == OUT else Path('/home/kojiek/eval_output_nvme/metric_win_tie_126a_20261016/smoke/report.json')
    if not src.is_file():
        raise SystemExit(f'need 126a S from {src}')
    S = dict(json.loads(src.read_text())['protocol']['S'])
    sc, frames = load_all(rows)
    by_key = {r['key']: r for r in rows}
    clean = [r for r in rows if r['kind'] == 'ref']
    for m in MIR:
        v = np.array([sc[r['key']].get(m, np.nan) for r in clean])
        S[m] = float(np.nanstd(v))
    # per (block/source/case) cells, pooled and per group
    cells, groups = {}, {}
    for r in rows:
        if r['base']:
            groups.setdefault((r['block'], r['source'], r['case'], r['kind']), []).append(r)
    for (block, src, case, kind), rs in sorted(groups.items()):
        pairs = [(r['key'], r['base']) for r in rs]
        c = cell(pairs, sc, S, kind)
        c['ladder'] = rs[0]['ladder']
        c['by_group'] = {g: cell([(r['key'], r['base']) for r in rs if r['group'] == g], sc, S, kind)['metrics']
                         for g in sorted({r['group'] for r in rs})}
        c['fad_vs_clean'] = fad([frames[a] for a, _ in pairs], [frames[b] for _, b in pairs])
        c['clips_over_fs'] = int(sum(r.get('over_fs', 0) > 0 for r in rs))
        cells.setdefault(f'{block}/{src}', {})[case] = c
    # SoundFont swap: same excerpt, same notes, other bank (should tie)
    fl = {r['excerpt']: r['key'] for r in clean if r['source'] == 'midi_fluidr3'}
    gu = {r['excerpt']: r['key'] for r in clean if r['source'] == 'midi_generaluser'}
    sf_pairs = [(fl[e], gu[e]) for e in sorted(set(fl) & set(gu))]
    cells.setdefault('midi/soundfont_swap', {})['fluidr3_vs_generaluser'] = cell(sf_pairs, sc, S, 'soft')
    ladders = {}
    for src, cs in cells.items():
        for lad, steps in {**MIDI_LADDERS, **A.LADDERS}.items():
            if not all(st in cs for st in steps):
                continue
            kind = cs[steps[0]]['kind']
            for m in S:
                passes = [cs[st]['metrics'].get(m, {}).get('at_m', {}).get('pass') for st in steps]
                if any(p is None for p in passes):
                    continue
                if kind in ('tie', 'soft'):
                    ff = next((i for i, p in enumerate(passes) if not p), None)
                    r = {'tolerance': None if not passes[0] else (steps[ff - 1] if ff else steps[-1]), 'n_pass': int(sum(passes))}
                else:
                    r = {'detection': next((st for st, p in zip(steps, passes) if p), None)}
                ladders.setdefault(src, {}).setdefault(lad, {})[m] = r
    # cross comparisons (unpaired, per piece): pro vs poor, pro vs MIDI, poor vs MIDI; CSD real vs MIDI (paired)
    cross = {}
    clean_by = {}
    for r in clean:
        clean_by.setdefault((r['source'], r['piece']), []).append(r['key'])
    pieces = sorted({r['piece'] for r in clean if r['group'] == 'classical'})
    for p in pieces + ['__all_classical__']:
        def keys(src):
            return [k for (s, pc), ks in clean_by.items() if s == src and (pc == p or (p == '__all_classical__' and pc in pieces))
                    for k in ks]
        comp = {}
        for a, b in (('pro', 'poor'), ('pro', 'midi_generaluser'), ('poor', 'midi_generaluser'), ('midi_fluidr3', 'midi_generaluser')):
            ka, kb = keys(a), keys(b)
            if ka and kb:
                comp[f'{a}_vs_{b}'] = {m: winrate([sc[k][m] for k in ka], [sc[k][m] for k in kb], M_FRAC * S[m]) for m in S}
        cross[f'classical/{p}'] = comp
    csd = {r['excerpt']: r['key'] for r in clean if r['source'] == 'csd_real'}
    paired = [(csd[e], gu[e]) for e in sorted(set(csd) & set(gu))]
    cross['nursery/csd_real_vs_midi_paired'] = cell(paired, sc, S, 'descriptive')
    ps = [r['key'] for r in clean if r['source'] == 'poor_sing']
    if ps:
        cross['nursery/csd_real_vs_poor_sing'] = {m: winrate([sc[k][m] for k in csd.values()], [sc[k][m] for k in ps],
                                                             M_FRAC * S[m]) for m in S}
        cross['nursery/poor_sing_vs_midi'] = {m: winrate([sc[k][m] for k in ps],
                                                         [sc[gu[e]][m] for e in gu if by_key[gu[e]]['group'] == 'nursery'],
                                                         M_FRAC * S[m]) for m in S}
    # floors: how much real music does each floor out-score; floor vs white noise at the same level
    music = [r['key'] for r in clean if r['block'] in ('midi', 'rec')]
    floors = {}
    fl_rows = [r for r in rows if r['block'] == 'floor']
    for case in sorted({r['case'] for r in fl_rows}):
        ks = [r['key'] for r in fl_rows if r['case'] == case]
        lvl = case.rsplit('_', 1)[1]
        white = [r['key'] for r in fl_rows if r['case'] == f'white_{lvl}']
        f = {}
        for m in S:
            v = np.array([sc[k].get(m, np.nan) for k in ks])
            mv = np.array([sc[k].get(m, np.nan) for k in music])
            if not np.isfinite(v).any():
                continue
            fm = float(np.nanmean(v))
            f[m] = {'mean': fm, 'frac_music_below_floor': float(np.nanmean(mv < fm)),
                    'vs_white': winrate(v, [sc[k].get(m, np.nan) for k in white], M_FRAC * S[m])}
        f['fad_vs_all_music'] = fad([frames[k] for k in ks], [frames[k] for k in music])
        floors[case] = f
    summary = {'protocol': {'n_clips': len(rows), 'm_frac': M_FRAC, 'pass_rate': PASS, 'S': S,
                            'S_definition': 'neural metrics: 126a S (range of the 5 TTM systems clean means); '
                                            'MIR: SD of all 126b clean clips; MIR stored sign-flipped so higher = better',
                            'n_by_source_kind': {f'{s}/{k}': sum(1 for r in rows if r['source'] == s and r['kind'] == k)
                                                 for s, k in sorted({(r['source'], r['kind']) for r in rows})}},
               'cells': cells, 'ladders': ladders, 'cross': cross, 'floors': floors}
    for p in (ROOT / 'summary.json', REPORT if ROOT == OUT else ROOT / 'report.json'):
        p.write_text(json.dumps(summary, indent=1))
    log(f'ANALYZE -> {ROOT / "summary.json"} ({sum(len(x) for x in cells.values())} cells)')


def preflight():
    rows = items()
    OUT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUT).free
    missing = [r['path'] for r in rows if not Path(r['path']).is_file()]
    absent = [p for p in [A.MEVA_PY, '/home/kojiek/venvs/madmom/bin/python', '/home/kojiek/venvs/mir/bin/python']
              if not Path(p).exists()]
    keys = [r['key'] for r in rows]
    ok = len(rows) > 50000 and not missing and not absent and free >= MIN_FREE and len(keys) == len(set(keys))
    print(json.dumps({'n': len(rows), 'n_units': len(units(rows)), 'missing': len(missing), 'absent': absent,
                      'free_gb': free / 2 ** 30, 'n_mir': sum(r['mir'] for r in rows), 'ok': ok}))
    return 0 if ok else 1


def validate(rows):
    for name, rs in units(rows):
        if not unit_done(ROOT / 'units' / name, [r['key'] for r in rs]):
            print(json.dumps({'ok': False, 'incomplete': name}))
            return 1
    summ = json.loads((ROOT / 'summary.json').read_text())
    n_cells = sum(len(x) for x in summ['cells'].values())
    ok = summ['protocol']['n_clips'] == len(rows) and n_cells > 100 and len(summ['floors']) > 10
    print(json.dumps({'n': len(rows), 'cells': n_cells, 'floors': len(summ['floors']), 'ok': ok}))
    return 0 if ok else 1


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--mir-worker', action='store_true')
    ap.add_argument('--limit', type=int, help='smoke: first N excerpts per source')
    a = ap.parse_args()
    if a.preflight:
        return preflight()
    if a.limit:
        ROOT = OUT / 'smoke'
        ROOT.mkdir(parents=True, exist_ok=True)
    if a.mir_worker:
        return mir_worker(a.limit)
    rows = items(a.limit)
    if a.validate_only:
        return validate(rows)
    if not a.analyze_only:
        run(rows, a.limit)
    analyze(rows)
    return validate(rows)


if __name__ == '__main__':
    sys.exit(main())
