#!/usr/bin/env python3
"""181: MuQ-Eval rescore of the 081 (quality-label prefix) and 084 N100 (NegMF) cells.

Design (frozen before scoring): docs/experiments/muq_081_084_rescore_181_20261105.md.
Question: do training-time PQ-targeted methods also raise MuQ-Eval A1 MI?

Same 36 cells as 112 (meva_081_084_backfill.py), regenerated with their original
eval.py flags. Per cell:
  generate -> level (eval_metrics, no CLAP/AES) -> identity gate vs stored per_clip.tsv
  -> lvl30 copies (level_match_rescore._gain_one, -30 LUFS, peak cap 0.999) with a gain
  cross-check vs the stored gain_status.tsv -> MuQ-Eval A1 / A3a x MI / TA x raw / lvl30
  in a subprocess (CUDA stays out of the process that forks the gain pool)
  -> per_cell/<label>.tsv -> delete audio.
Batch 1 (per clip): batch 16 drifted up to 3.2e-3 (A1) / 3.6e-3 (A3a) from batch 1 on
64 clips, over the registered 1e-3 limit (deviation D1).

Modes: (default) run pending cells then write summary.json; --preflight;
--validate-only; --score-cell LABEL (internal); --summary-only.
Resumable at cell level; a partial cell is wiped and regenerated.
"""
import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import meva_081_084_backfill as B  # noqa: E402
from meva_081_084_backfill import (ARM, CTRL, N100, QL_CELLS, SEEDS, ROWS, PY, REPO,  # noqa: E402
                                   EVAL_ROOT, MC_TSV, cells, eval_args, eval_cmd, read_clips,
                                   stored_clips)

OUT = EVAL_ROOT / 'muq_081_084_rescore_181'
CELLS_DIR = OUT / 'cells'
PER_CELL = OUT / 'per_cell'
SUMMARY = OUT / 'summary.json'
PROGRESS = OUT / 'progress.json'
MIN_FREE = 20 * 1024**3
TOL_PEAK, TOL_LUFS, TOL_GAIN, MIN_MATCH = 1e-4, 0.01, 0.01, 0.999
TARGET, PEAK_CAP = -30.0, 0.999
VARIANTS, HEADS, CONDS = ('A1', 'A3a'), ('MI', 'TA'), ('raw', 'lvl30')
READINGS = [f'{v}_{h}_{c}' for c in CONDS for v in VARIANTS for h in HEADS]
PRIMARY = 'A1_MI_lvl30'

# Pre-registered (design §4): S = range of the 5 TTM-system means (11-03 Part B), m = 0.10 S.
S_MUQ = {'A1': 3.541 - 2.990, 'A3a': 3.716 - 3.093}
M_MUQ = {'A1': 0.055, 'A3a': 0.062}
TTM5 = ('musicgen_medium', 'meanaudio_cfg3_neg', 'musicldm_fid8', 'musicldm', 'meanaudio_cfg0')
TTM115 = EVAL_ROOT / 'ttm_metrics_115_20261009/summary.json'
PARTB = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/muq_attack_fix_20261103')
NBOOT, BOOT_SEED = 2000, 181
DEVIATIONS = {
    'D1': 'MuQ-Eval scored per clip (batch 1), not batch 16: on 64 lvl30 clips batch 16 differed '
          'from batch 1 by up to 3.2e-3 (A1 MI) / 3.6e-3 (A3a MI), over the registered 1e-3 limit.',
    'D2': 'PQ_lvl30 (only used for R and the Spearman descriptions) is AES PQ recomputed on the lvl30 audio '
          'regenerated here, not the stored lvl30 cells: 4 control cells (nmv2pair s16180339/s27182818 '
          'cfg0, cfg3_neg) never had a stored lvl30 copy. Where a stored copy exists the gain gate and a '
          'recorded PQ cross-check still compare against it.',
}


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def cdir(label):
    return CELLS_DIR / label


def cell_tsv(label):
    return PER_CELL / f'{label}.tsv'


def cell_done(label):
    p, d = cell_tsv(label), cdir(label) / 'done.json'
    if not (p.is_file() and d.is_file()):
        return False
    return json.loads(d.read_text()).get('passed') is True and len(read_clips(p)) == ROWS


def lvl30_stored(label):
    return EVAL_ROOT / f'{label}_lvl30'


# ── per-cell pipeline ────────────────────────────────────
def identity(label):
    old = stored_clips(label)
    new = read_clips(cdir(label) / 'level' / 'per_clip.tsv')
    common = sorted(set(old) & set(new))
    match, wp, wl = 0, 0.0, 0.0
    for k in common:
        dp = abs(float(old[k]['peak']) - float(new[k]['peak']))
        lo, ln = old[k]['lufs'], new[k]['lufs']
        na = ('', 'None', 'nan')
        if lo in na and ln in na:
            dl = 0.0
        elif lo in na or ln in na:
            dl = math.inf
        else:
            dl = abs(float(lo) - float(ln))
        wp, wl = max(wp, dp), max(wl, dl)
        match += int(dp <= TOL_PEAK and dl <= TOL_LUFS)
    frac = match / ROWS
    return {'n_old': len(old), 'n_new': len(new), 'n_common': len(common), 'n_match': match,
            'match_frac': frac, 'max_abs_dpeak': wp, 'max_abs_dlufs': wl if math.isfinite(wl) else 'inf',
            'passed': len(common) == ROWS and frac >= MIN_MATCH}


def make_lvl30(label, workers=16):
    from level_match_rescore import _gain_one
    src = cdir(label) / 'audio'
    dst = cdir(label) / 'lvl30'
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(dst / p.name), TARGET, PEAK_CAP) for p in sorted(src.glob('*.flac'))]
    with Pool(workers) as pool:
        res = pool.map(_gain_one, jobs, chunksize=32)
    gs = lvl30_stored(label) / 'gain_status.tsv'
    new = {stem: (status, float(gain)) for stem, status, _lufs, gain in res}
    if not gs.is_file():
        # 4 control cells (nmv2pair s16180339/s27182818 cfg0, cfg3_neg) never had a stored lvl30 copy
        return new, {'n': len(new), 'stored': False, 'passed': bool(len(new) == ROWS)}
    stored = read_clips(gs)
    new = {stem: (status, float(gain)) for stem, status, _lufs, gain in res}
    match, worst, status_mismatch = 0, 0.0, 0
    for k, (status, gain) in new.items():
        if k not in stored:
            continue
        d = abs(gain - float(stored[k]['gain_db']))
        worst = max(worst, d)
        status_mismatch += int(status != stored[k]['status'])
        match += int(d <= TOL_GAIN)
    gate = {'n': len(new), 'stored': True, 'n_stored': len(stored), 'n_match': match, 'match_frac': match / ROWS,
            'max_abs_dgain_db': worst, 'status_mismatch': status_mismatch,
            'passed': bool(len(new) == ROWS and match / ROWS >= MIN_MATCH)}
    return new, gate


def run_cell(label):
    d = cdir(label)
    if d.exists():
        shutil.rmtree(d)
    cell_tsv(label).unlink(missing_ok=True)
    audio = d / 'audio'
    audio.mkdir(parents=True)
    args = eval_args(label)
    cmd = eval_cmd(args, audio)
    (d / 'eval_cmd.json').write_text(json.dumps({'source_args': args, 'cmd': cmd}, indent=1, default=str))
    log(f'GEN {label}  cfg={args["cfg_strength"]} neg={args.get("negative_prompt")!r} tsv={Path(args["tsv"]).name}')
    with open(d / 'gen.log', 'w') as g:
        subprocess.run(cmd, cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    n = len(list(audio.glob('*.flac')))
    if n != ROWS:
        raise RuntimeError(f'{label}: {n} flac, expected {ROWS}')
    log(f'LEVEL {label}')
    with open(d / 'level.log', 'w') as g:
        subprocess.run([str(PY), 'scripts/eval/eval_metrics.py', '--gen_dir', str(audio), '--tsv', str(MC_TSV),
                        '--exp_name', 'level', '--out_dir', str(d), '--skip_clap', '--skip_aes'],
                       cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    ident = identity(label)
    if not ident['passed']:
        (d / 'done.json').write_text(json.dumps({'identity': ident, 'passed': False}, indent=1))
        raise RuntimeError(f'{label}: regenerated audio does not match the original: {ident}')
    log(f'LVL30 {label}')
    gains, ggate = make_lvl30(label)
    if not ggate['passed']:
        (d / 'done.json').write_text(json.dumps({'identity': ident, 'gain': ggate, 'passed': False}, indent=1))
        raise RuntimeError(f'{label}: lvl30 gains do not match the stored ones: {ggate}')
    with open(d / 'gains.tsv', 'w') as f:
        f.write('id\tstatus\tgain_db\n')
        for k in sorted(gains):
            f.write(f'{k}\t{gains[k][0]}\t{gains[k][1]!r}\n')
    log(f'SCORE {label}')
    with open(d / 'score.log', 'w') as g:
        subprocess.run([str(PY), '-I', str(Path(__file__).resolve()), '--score-cell', label],
                       cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    rows = read_clips(cell_tsv(label))
    if len(rows) != ROWS:
        raise RuntimeError(f'{label}: scored {len(rows)} rows')
    pqx = pq_crosscheck(label, rows)
    shutil.rmtree(audio)
    shutil.rmtree(d / 'lvl30')
    (d / 'done.json').write_text(json.dumps({'identity': ident, 'gain': ggate, 'pq_lvl30_vs_stored': pqx,
                                             'passed': True}, indent=1))
    a1 = statistics.fmean(float(r[PRIMARY]) for r in rows.values())
    log(f'DONE {label}  {PRIMARY} {a1:.4f}  identity {ident["n_match"]}/{ROWS}  gain {ggate["n_match"]}/{ROWS}')


def pq_crosscheck(label, rows):
    """Recorded, not gated: our AES PQ on lvl30 vs the stored lvl30 cell (AES batch drift ~1e-3)."""
    sp = lvl30_stored(label) / f'{label}_lvl30' / 'per_clip.tsv'
    if not sp.is_file():
        return {'stored': False}
    st = read_clips(sp)
    ds = [float(rows[k]['PQ_lvl30']) - float(st[k]['PQ']) for k in rows if k in st]
    return {'stored': True, 'n': len(ds), 'mean_d': statistics.fmean(ds), 'max_abs_d': max(abs(x) for x in ds)}


def score_cell(label):
    """Subprocess entry: MuQ-Eval on raw and lvl30 audio of one cell."""
    sys.path[:0] = [str(REPO), str(REPO / 'scripts/eval')]
    from muq_gen_steer_179_score import muq_hf_cache
    from newmetrics_muqeval import score_muqeval
    d = cdir(label)
    ids = sorted(p.stem for p in (d / 'audio').glob('*.flac'))
    if len(ids) != ROWS:
        raise SystemExit(f'{label}: {len(ids)} raw clips')
    rec = {i: {} for i in ids}
    with muq_hf_cache():
        for cond, sub in (('raw', 'audio'), ('lvl30', 'lvl30')):
            for v in VARIANTS:
                t = time.time()
                for i in ids:
                    s = score_muqeval(str(d / sub / f'{i}.flac'), None, variant=v)
                    for h in HEADS:
                        rec[i][f'{v}_{h}_{cond}'] = s[h]
                log(f'{label} {v} {cond} {len(ids) / (time.time() - t):.1f} clip/s')
    from eval_metrics import score_aes
    paths = [str(d / 'lvl30' / f'{i}.flac') for i in ids]
    aes, failed = score_aes(paths, progress=False)
    if failed:
        raise SystemExit(f'{label}: AES failed on {len(failed)} clips')
    for i, pth in zip(ids, paths):
        rec[i]['PQ_lvl30'] = aes[pth]['PQ']
    PER_CELL.mkdir(parents=True, exist_ok=True)
    tmp = cell_tsv(label).with_suffix('.tmp')
    with open(tmp, 'w') as f:
        cols = READINGS + ['PQ_lvl30']
        f.write('id\t' + '\t'.join(cols) + '\n')
        for i in ids:
            vals = [rec[i][k] for k in cols]
            if not all(math.isfinite(x) for x in vals):
                raise SystemExit(f'{label}: non-finite score on {i}')
            f.write(i + '\t' + '\t'.join(repr(x) for x in vals) + '\n')
    tmp.replace(cell_tsv(label))
    return 0


# ── summary ──────────────────────────────────────────────
def load_cell(label):
    muq = read_clips(cell_tsv(label))
    raw = stored_clips(label)
    mv_path = EVAL_ROOT / f'{label}_meva' / f'{label}_meva' / 'meva_per_clip.tsv'
    mv = read_clips(mv_path) if mv_path.is_file() else {}
    ids = sorted(muq)
    out = {}
    for i in ids:
        r = {k: float(muq[i][k]) for k in READINGS}
        r['PQ_raw'] = float(raw[i]['PQ'])
        r['PQ_lvl30'] = float(muq[i]['PQ_lvl30'])
        r['meva'] = float(mv[i]['meva_raw']) if i in mv else math.nan
        out[i] = r
    return out


def judge(per_seed, m):
    mean = statistics.fmean(per_seed)
    if mean >= m and all(x > 0 for x in per_seed):
        return 'follows'
    if abs(mean) < m:
        return 'does_not_follow'
    if mean <= -m and all(x < 0 for x in per_seed):
        return 'opposite'
    return 'undecided'


def summarize():
    import numpy as np
    from scipy.stats import spearmanr
    data = {l: load_cell(l) for l in cells()}
    ids = sorted(set.intersection(*(set(v) for v in data.values())))
    if len(ids) != ROWS:
        raise RuntimeError(f'common ids {len(ids)}')
    metrics = READINGS + ['PQ_lvl30', 'PQ_raw', 'meva']
    arr = {l: {k: np.array([data[l][i][k] for i in ids]) for k in metrics} for l in cells()}

    def x(prefix, s, c, k):
        return arr[f'{prefix.format(seed=s)}_{c}'][k]

    contrast_defs = {
        'C0_control_cfg3neg_minus_cfg0': lambda s, k: x(CTRL, s, 'mc_mf25_cfg3_neg', k) - x(CTRL, s, 'mc_mf25_cfg0', k),
        'H1_081_E1_lqneg_did': lambda s, k: (x(ARM, s, 'mc_mf25_cfg3_lqneg', k) - x(ARM, s, 'mc_mf25_cfg0', k))
        - (x(CTRL, s, 'mc_mf25_cfg3_lqneg', k) - x(CTRL, s, 'mc_mf25_cfg0', k)),
        'H2_081_E3_hqpos_cfg0_did': lambda s, k: (x(ARM, s, 'hqpos_mc_mf25_cfg0', k) - x(ARM, s, 'mc_mf25_cfg0', k))
        - (x(CTRL, s, 'hqpos_mc_mf25_cfg0', k) - x(CTRL, s, 'mc_mf25_cfg0', k)),
        'H3_084_n100_cfg0_minus_control_cfg0': lambda s, k: x(N100, s, 'mc_mf25_cfg0', k) - x(CTRL, s, 'mc_mf25_cfg0', k),
        'D1_081_hqpos_lqneg_minus_fid8': lambda s, k: x(ARM, s, 'hqpos_mc_mf25_cfg3_lqneg', k) - x(ARM, s, 'mc_mf25_cfg3_neg', k),
        'D1b_081_E2_lqneg_minus_fid8': lambda s, k: x(ARM, s, 'mc_mf25_cfg3_lqneg', k) - x(ARM, s, 'mc_mf25_cfg3_neg', k),
        'D2_n100_cfg3neg_minus_control_cfg3neg': lambda s, k: x(N100, s, 'mc_mf25_cfg3_neg', k) - x(CTRL, s, 'mc_mf25_cfg3_neg', k),
    }
    for c in QL_CELLS:
        contrast_defs[f'D2_arm_minus_control_{c}'] = (lambda c: lambda s, k: x(ARM, s, c, k) - x(CTRL, s, c, k))(c)

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, ROWS, size=(NBOOT, ROWS))
    contrasts = {}
    for name, f in contrast_defs.items():
        contrasts[name] = {}
        for k in metrics:
            per_clip = [f(s, k) for s in SEEDS]
            per_seed = [float(np.nanmean(v)) for v in per_clip]
            clip_mean = np.mean(per_clip, axis=0)
            rec = {'mean': statistics.fmean(per_seed), 'sd': statistics.stdev(per_seed), 'per_seed': per_seed}
            if not np.isnan(clip_mean).any():
                bm = clip_mean[boot_idx].mean(axis=1)
                rec['clip_boot_ci95'] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
            contrasts[name][k] = rec

    s115 = json.loads(TTM115.read_text())['systems']
    pq5 = [s115[sname]['blocks']['lvl30']['PQ'] for sname in TTM5]
    s_pq = max(pq5) - min(pq5)

    judgments = {}
    for v in VARIANTS:
        key = f'{v}_MI_lvl30'
        m = M_MUQ[v]
        j = {'reading': key, 'm': m, 'S': S_MUQ[v], 'S_PQ_lvl30': s_pq}
        c0 = contrasts['C0_control_cfg3neg_minus_cfg0'][key]['per_seed']
        j['C0'] = {'per_seed': c0, 'passed': all(t > 0 for t in c0)}
        for h in ('H1_081_E1_lqneg_did', 'H2_081_E3_hqpos_cfg0_did', 'H3_084_n100_cfg0_minus_control_cfg0'):
            c = contrasts[h]
            d_mi, d_pq = c[key]['mean'], c['PQ_lvl30']['mean']
            ratio = (d_mi / S_MUQ[v]) / (d_pq / s_pq) if d_pq else None
            j[h] = {'verdict': judge(c[key]['per_seed'], m), 'mean': d_mi, 'per_seed': c[key]['per_seed'],
                    'dPQ_lvl30': d_pq, 'R': ratio}
        n_follow = sum(j[h]['verdict'] == 'follows' for h in j if h.startswith('H'))
        n_not = sum(j[h]['verdict'] in ('does_not_follow', 'opposite') for h in j if h.startswith('H'))
        if not j['C0']['passed']:
            overall = 'harness_check_failed'
        elif n_follow >= 2:
            overall = 'holds'
        elif n_not == 3:
            overall = 'does_not_hold'
        else:
            overall = 'mixed'
        j['overall'] = overall
        judgments[v] = j

    cell_stats = {}
    for l in cells():
        a = arr[l]
        st = {k: float(np.nanmean(a[k])) for k in metrics}
        st['spearman_A1MIlvl30_vs_PQlvl30'] = float(spearmanr(a[PRIMARY], a['PQ_lvl30']).statistic)
        st['spearman_A3aMIlvl30_vs_PQlvl30'] = float(spearmanr(a['A3a_MI_lvl30'], a['PQ_lvl30']).statistic)
        if not np.isnan(a['meva']).any():
            st['spearman_A1MIlvl30_vs_meva'] = float(spearmanr(a[PRIMARY], a['meva']).statistic)
        cell_stats[l] = st

    # placement on the 11-03 Part B common prompts (descriptive only; quarter models vs 114 full)
    sel = {Path(f).stem for f in json.loads((PARTB / 'part_b_selection.json').read_text())['files']}
    idx = [n for n, i in enumerate(ids) if i in sel]
    pb = {}
    with open(PARTB / 'per_clip.tsv', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['set'] == 'ttm' and Path(r['item']).stem in sel:
                pb.setdefault(f"{r['sys']}_{r['fix']}", {'A1_MI': [], 'A3a_MI': []})
                pb[f"{r['sys']}_{r['fix']}"]['A1_MI'].append(float(r['A1_MI']))
                pb[f"{r['sys']}_{r['fix']}"]['A3a_MI'].append(float(r['A3a_MI']))
    placement = {
        'n_prompts': len(idx),
        'part_b_systems': {k: {m: statistics.fmean(v[m]) for m in v} | {'n': len(v['A1_MI'])} for k, v in pb.items()},
        'cells_seed_mean_lvl30': {}}
    for role, (prefix, names) in B.ROLES.items():
        for c in names:
            ls = [f'{prefix.format(seed=s)}_{c}' for s in SEEDS]
            placement['cells_seed_mean_lvl30'][f'{role}_{c}'] = {
                m: statistics.fmean(float(arr[l][f'{m}_lvl30'][idx].mean()) for l in ls) for m in ('A1_MI', 'A3a_MI')}

    return {'document_kind': 'muq_081_084_rescore_181_summary', 'written_at': time.strftime('%FT%T%z'),
            'design': 'docs/experiments/muq_081_084_rescore_181_20261105.md',
            'primary_reading': PRIMARY, 'deviations': DEVIATIONS,
            'scorer': 'MuQ-Eval A1 / A3a (newmetrics_muqeval, center 10 s, 24 kHz, no loudness norm), batch 1',
            'notes': 'PQ_raw / meva are the stored values of the original cells (112); PQ_lvl30 is AES PQ '
                     '(eval_metrics.score_aes, batch 32) on the lvl30 audio regenerated here, for all 36 cells '
                     '(4 control cells never had a stored lvl30 copy; the others are cross-checked in gates); lvl30 here = per-clip scalar gain to -30 LUFS, peak cap 0.999',
            'n_clips': ROWS, 'seeds': SEEDS, 'judgments': judgments, 'contrasts': contrasts,
            'cells': cell_stats, 'placement_part_b_1000': placement,
            'gates': {l: json.loads((cdir(l) / 'done.json').read_text()) for l in cells()}}


# ── modes ────────────────────────────────────────────────
def preflight():
    errs = []
    if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    for label in cells():
        try:
            a = eval_args(label)
            eval_cmd(a, Path('/dev/null'))
            if not Path(a['model_path']).is_file():
                errs.append(f'{label}: checkpoint missing {a["model_path"]}')
            if not Path(a['tsv']).is_file():
                errs.append(f'{label}: gen tsv missing {a["tsv"]}')
            if len(stored_clips(label)) != ROWS:
                errs.append(f'{label}: stored per_clip.tsv not {ROWS} rows')
            gs = lvl30_stored(label) / 'gain_status.tsv'
            if gs.is_file() and len(read_clips(gs)) != ROWS:
                errs.append(f'{label}: stored gain_status.tsv not {ROWS} rows')
            if not gs.is_file():
                log(f'PREFLIGHT note {label}: no stored lvl30 (gain cross-check skipped)')
        except Exception as e:  # noqa: BLE001
            errs.append(f'{label}: {e}')
    for p in (TTM115, PARTB / 'per_clip.tsv', PARTB / 'part_b_selection.json'):
        if not p.is_file():
            errs.append(f'missing {p}')
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if not errs:
        log(f'PREFLIGHT OK ({len(cells())} cells)')
    return 1 if errs else 0


def validate():
    pending = [l for l in cells() if not cell_done(l)]
    if pending:
        log(f'VALIDATE FAIL: {len(pending)} cells not done: {pending[:3]}')
        return 1
    if not SUMMARY.is_file():
        log('VALIDATE FAIL: summary missing')
        return 1
    s = json.loads(SUMMARY.read_text())
    if s.get('document_kind') != 'muq_081_084_rescore_181_summary' or 'A1' not in s.get('judgments', {}):
        log('VALIDATE FAIL: summary malformed')
        return 1
    log(f'VALIDATE OK  A1 overall={s["judgments"]["A1"]["overall"]}')
    return 0


def write_summary():
    tmp = SUMMARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(summarize(), indent=1))
    tmp.replace(SUMMARY)
    log(f'wrote {SUMMARY}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--score-cell')
    ap.add_argument('--only', help='run just this cell (smoke)')
    a = ap.parse_args()
    os.chdir(REPO)
    if a.score_cell:
        return score_cell(a.score_cell)
    if a.preflight:
        return preflight()
    if a.validate_only:
        return validate()
    if a.summary_only:
        write_summary()
        return validate()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [l for l in cells() if not cell_done(l)]
    if a.only:
        todo = [a.only]
    log(f'{len(cells()) - len(todo)}/{len(cells())} cells already done; {len(todo)} to run')
    for label in todo:
        if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
            raise SystemExit('disk below 20 GiB')
        run_cell(label)
        done = sum(cell_done(l) for l in cells())
        PROGRESS.write_text(json.dumps({'done': done, 'total': len(cells()), 'last': label,
                                        'written_at': time.strftime('%FT%T%z')}))
    if a.only:
        return 0
    write_summary()
    return validate()


if __name__ == '__main__':
    sys.exit(main())
