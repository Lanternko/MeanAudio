#!/usr/bin/env python3
"""126c: per-pair win/tie/lose on MUSDB18-HQ stem edits (remove / solo / rebalance / misalign / detune / swap).

Design: docs/experiments/metric_win_tie_126c_musdb_20261019.md.
Stimuli: built by metric_win_tie_126c_stimuli.py (CPU + a short demucs pass) into STIM/items.json.

Scoring is 126b's unit loop unchanged (imported): units of <= 3000 clips symlinked into a unit dir,
AES (4 axes), MEva, CLAP (batch 1, caption "music"), VGGish frames (float16, per-case FAD); MIR on the
clips flagged mir as a CPU subprocess next to the GPU loop.

Reading (analyze), same rule as 126a/126b: d = score(case) - score(mix); tie if |d| <= m, m = 0.10 S;
S = the 126a S for the neural metrics, SD of the 126c mix clips for MIR; pass = 80 % of pairs.
Extra 126c readouts:
  separation   dmx_no_<stem> vs no_<stem> (paired, same window): what htdemucs removal adds beyond removal
  detune_vs_pv vocals_detune<c> vs vocal_pv0 (paired): the pitch error with the rubberband processing held fixed
  human        AES_natural_music.jsonl rates the 50 MUSDB test mixtures and their 4 stems (10 raters, 4 axes):
               per track, human mean(stem) - mean(mixture) against the metric's mean d(only_<stem>) over the
               track's windows (sign agreement and Spearman over track x stem). Descriptive: the human clips
               are not our windows.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.pop('MKL_NUM_THREADS', None)  # MKL_NUM_THREADS hangs the forked loader pool in the dac venv

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(REPO / 'scripts/eval'))
import metric_win_tie_126a as A  # noqa: E402
import metric_win_tie_126b as B  # noqa: E402  (unit scorer, judge/cell/winrate, FAD)

OUT = Path('/home/kojiek/eval_output_nvme/metric_win_tie_126c_20261019')
REPORT = REPO / 'docs/experiments/results/metric_win_tie_126c_summary.json'
HUMAN = REPO / 'research/eval/aes_human_ratings/AES_natural_music.jsonl'
HUMAN_AXES = {'PQ': 'Production_Quality', 'CE': 'Content_Enjoyment', 'CU': 'Content_Usefulness', 'PC': 'Production_Complexity'}
STEMS = ('vocals', 'drums', 'bass', 'other')
MIN_FREE = 15 * 2 ** 30
N_MIN = 25000
LADDERS = {
    'drums_lag': ['drums_lag20ms', 'drums_lag50ms', 'drums_lag100ms', 'drums_lag250ms'],
    'vocals_lag': ['vocals_lag50ms', 'vocals_lag100ms', 'vocals_lag250ms'],
    'vocals_detune': ['vocals_detune25c', 'vocals_detune50c', 'vocals_detune100c'],
    **{f'rebal_{s}_up': [f'rebal_{s}_+1dB', f'rebal_{s}_+3dB', f'rebal_{s}_+6dB'] for s in STEMS},
    **{f'rebal_{s}_down': [f'rebal_{s}_-1dB', f'rebal_{s}_-3dB', f'rebal_{s}_-6dB'] for s in STEMS},
    'noise': ['noise_snr40', 'noise_snr30', 'noise_snr20', 'noise_snr10'],
    'gain_down': ['gain-1', 'gain-3'],
}
log = A.log


def bind(root):
    """point the imported 126b scorer at this job's stimuli and output root"""
    B.OUT, B.STIM, B.ROOT, B.MIN_FREE = OUT, OUT / 'stimuli', root, MIN_FREE


def mir_worker(limit):
    from mir_metrics import score_mir
    rows = [r for r in B.items(limit) if r['mir']]
    per = score_mir([r['path'] for r in rows], madmom_workers=6, chroma_workers=3)
    by_path = {r['path']: r['key'] for r in rows}
    (B.ROOT / 'mir.json').write_text(json.dumps({by_path[p]: v for p, v in per.items()}))
    return 0


def run(rows, limit):
    import ttm_fad_lvl30 as F
    embedder = F.Embedder()  # forks the worker pool first, while this process is single-threaded
    from eval_metrics import load_clap
    clap_model = load_clap()
    mir = None
    if not (B.ROOT / 'mir.json').is_file():
        cmd = [sys.executable, __file__, '--mir-worker'] + (['--limit', str(limit)] if limit else [])
        mir = subprocess.Popen(cmd, stdout=open(B.ROOT / 'mir.log', 'w'), stderr=subprocess.STDOUT)
    todo = B.units(rows)
    for k, (name, rs) in enumerate(todo):
        if shutil.disk_usage(B.ROOT).free < MIN_FREE // 2:
            raise SystemExit('disk below half the preflight minimum')
        log(f'[{k + 1}/{len(todo)}] {name}')
        B.run_unit(name, rs, embedder, clap_model)
    embedder.pool.close()
    if mir is not None:
        log('waiting for MIR')
        if mir.wait() != 0:
            raise SystemExit(f'MIR failed, see {B.ROOT / "mir.log"}')


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b)
    return float(r.statistic) if np.isfinite(r.statistic) else None


def human_anchor(rows, sc, S):
    hum = {}
    for line in HUMAN.read_text().splitlines():
        r = json.loads(line)
        parts = r['data_path'].split('/')
        if 'MUSDB18' not in parts:
            continue
        track, stem = (parts[-3], parts[-2]), parts[-1][:-4]
        hum.setdefault(track, {})[stem] = {ax: float(np.mean(r[k])) for ax, k in HUMAN_AXES.items()}
    met = {}
    for r in rows:
        if r['case'].startswith('only_') and (r['group'], r['piece']) in hum:
            met.setdefault(((r['group'], r['piece']), r['case'][5:]), []).append((r['key'], r['base']))
    out = {'n_tracks_rated': len(hum), 'n_tracks_matched': len({t for t, _ in met})}
    for ax in HUMAN_AXES:
        hs, ms, per_stem = [], [], {}
        for (t, s), pairs in sorted(met.items()):
            if s not in hum[t] or 'mixture' not in hum[t]:
                continue
            h = hum[t][s][ax] - hum[t]['mixture'][ax]
            d = float(np.mean([sc[a][ax] - sc[b][ax] for a, b in pairs]))
            hs.append(h)
            ms.append(d)
            per_stem.setdefault(s, []).append((h, d))
        hs, ms = np.array(hs), np.array(ms)
        if not len(hs):
            continue
        out[ax] = {'n': len(hs), 'spearman': spearman(hs, ms), 'sign_agree': float(np.mean(np.sign(hs) == np.sign(ms))),
                   'per_stem': {s: {'human_mean_delta': float(np.mean([h for h, _ in v])),
                                    'metric_mean_delta': float(np.mean([d for _, d in v])),
                                    'metric_mean_delta_over_S': float(np.mean([d for _, d in v]) / S[ax]),
                                    'sign_agree': float(np.mean([np.sign(h) == np.sign(d) for h, d in v])), 'n': len(v)}
                                for s, v in sorted(per_stem.items())}}
    return out


def analyze(rows):
    smoke = B.ROOT != OUT
    src = (Path('/home/kojiek/eval_output_nvme/metric_win_tie_126a_20261016/smoke/report.json') if smoke
           else B.REPORT_126A)
    S = dict(json.loads(src.read_text())['protocol']['S'])
    sc, frames = B.load_all(rows)
    clean = [r for r in rows if r['kind'] == 'ref']
    for m in B.MIR:
        S[m] = float(np.nanstd([sc[r['key']].get(m, np.nan) for r in clean]))
    groups = {}
    for r in rows:
        if r['base']:
            groups.setdefault((r['case'], r['kind']), []).append(r)
    cells = {}
    for (case, kind), rs in sorted(groups.items()):
        pairs = [(r['key'], r['base']) for r in rs]
        c = B.cell(pairs, sc, S, kind)
        c['ladder'] = rs[0]['ladder']
        c['by_group'] = {g: B.cell([(r['key'], r['base']) for r in rs if r['group'] == g], sc, S, kind)['metrics']
                         for g in sorted({r['group'] for r in rs})}
        c['fad_vs_mix'] = B.fad([frames[a] for a, _ in pairs], [frames[b] for _, b in pairs])
        c['clips_over_fs'] = int(sum(r.get('over_fs', 0) > 0 for r in rs))
        c['clips_capped'] = int(sum(r.get('capped', 0) for r in rs))
        cells[case] = c
    ladders = {}
    for lad, steps in LADDERS.items():
        if not all(st in cells for st in steps):
            continue
        kinds = {cells[st]['kind'] for st in steps}
        for m in S:
            passes = [cells[st]['metrics'].get(m, {}).get('at_m', {}).get('pass') for st in steps]
            if any(p is None for p in passes):
                continue
            if kinds <= {'tie', 'soft'}:
                ff = next((i for i, p in enumerate(passes) if not p), None)
                r = {'tolerance': None if not passes[0] else (steps[ff - 1] if ff else steps[-1]), 'n_pass': int(sum(passes))}
            else:
                r = {'detection': next((st for st, p in zip(steps, passes) if p), None)}
            ladders.setdefault(lad, {})[m] = r
    by_win = {}
    for r in rows:
        by_win.setdefault(r['excerpt'], {})[r['case']] = r['key']
    cross = {}
    for s in STEMS:
        pairs = [(v[f'dmx_no_{s}'], v[f'no_{s}']) for v in by_win.values() if f'dmx_no_{s}' in v and f'no_{s}' in v]
        cross[f'separation/dmx_no_{s}_vs_no_{s}'] = B.cell(pairs, sc, S, 'descriptive')
    for c in LADDERS['vocals_detune']:
        pairs = [(v[c], v['vocal_pv0']) for v in by_win.values() if c in v and 'vocal_pv0' in v]
        cross[f'detune_vs_pv/{c}_vs_vocal_pv0'] = B.cell(pairs, sc, S, 'lose')
    summary = {'protocol': {'n_clips': len(rows), 'n_windows': len(by_win), 'n_tracks': len({r['piece'] for r in rows}),
                            'm_frac': B.M_FRAC, 'pass_rate': B.PASS, 'S': S,
                            'S_definition': 'neural metrics: 126a S (range of the 5 TTM systems clean means); '
                                            'MIR: SD of all 126c mix clips; MIR stored sign-flipped so higher = better',
                            'n_by_kind': {k: sum(1 for r in rows if r['kind'] == k) for k in sorted({r['kind'] for r in rows})}},
               'cells': cells, 'ladders': ladders, 'cross': cross, 'human': human_anchor(rows, sc, S)}
    for p in (B.ROOT / 'summary.json', REPORT if not smoke else B.ROOT / 'report.json'):
        p.write_text(json.dumps(summary, indent=1))
    log(f'ANALYZE -> {B.ROOT / "summary.json"} ({len(cells)} cells)')


def preflight():
    bind(OUT)
    rows = B.items()
    free = shutil.disk_usage(OUT).free
    missing = [r['path'] for r in rows if not Path(r['path']).is_file()]
    absent = [p for p in [A.MEVA_PY, '/home/kojiek/venvs/madmom/bin/python', '/home/kojiek/venvs/mir/bin/python', str(HUMAN)]
              if not Path(p).exists()]
    keys = [r['key'] for r in rows]
    ok = len(rows) > N_MIN and not missing and not absent and free >= MIN_FREE and len(keys) == len(set(keys))
    print(json.dumps({'n': len(rows), 'n_units': len(B.units(rows)), 'missing': len(missing), 'absent': absent,
                      'free_gb': free / 2 ** 30, 'n_mir': sum(r['mir'] for r in rows), 'ok': ok}))
    return 0 if ok else 1


def validate(rows):
    for name, rs in B.units(rows):
        if not B.unit_done(B.ROOT / 'units' / name, [r['key'] for r in rs]):
            print(json.dumps({'ok': False, 'incomplete': name}))
            return 1
    summ = json.loads((B.ROOT / 'summary.json').read_text())
    n_cells = len(summ['cells'])
    ok = summ['protocol']['n_clips'] == len(rows) and n_cells >= 60 and 'PQ' in summ['human']
    print(json.dumps({'n': len(rows), 'cells': n_cells, 'human_n': summ['human'].get('PQ', {}).get('n'), 'ok': ok}))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--mir-worker', action='store_true')
    ap.add_argument('--limit', type=int, help='smoke: first N windows (reads the smoke stimuli)')
    a = ap.parse_args()
    if a.preflight:
        return preflight()
    root = OUT
    bind(root)
    if a.limit:
        root = OUT / 'smoke'
        root.mkdir(parents=True, exist_ok=True)
        bind(root)
        B.STIM = OUT / 'stimuli' / 'smoke'
    if a.mir_worker:
        return mir_worker(a.limit)
    rows = B.items(a.limit)
    if a.validate_only:
        return validate(rows)
    if not a.analyze_only:
        run(rows, a.limit)
    analyze(rows)
    return validate(rows)


if __name__ == '__main__':
    sys.exit(main())
