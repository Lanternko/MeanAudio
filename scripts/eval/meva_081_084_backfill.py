#!/usr/bin/env python3
"""MEva backfill for 081 (quality-label prefix) and 084 N100 (NegMF): regenerate, score, delete.

081 and 084 deleted their audio after AES scoring. MEva (pinned f03 shadow evaluator,
095 model lock) needs the waveforms, so every main-comparison cell is regenerated with
its original eval.py flags (read back from gen.log "Eval args"; only --output changes),
checked against its stored per_clip.tsv (peak / LUFS), MEva-scored, and deleted again.

The control is slot0clean_nmv2pair (081/084's control). 097's MEva control is a
different checkpoint (slot0nmv2_nmv2pair), so it is regenerated here too.

Cells (36): for each of 3 seeds
  081 arm  qlabel  x {cfg0, cfg3_neg, cfg3_lqneg, hqpos cfg0, hqpos cfg3_lqneg}
  control nmv2pair x the same five
  084 N100 negmfn100 x {cfg0, cfg3_neg}
The first cell (control s14159265 cfg0) runs first and doubles as the regeneration
check: the identity gate fails the job if it does not reproduce.

Layout per cell:
  <EVAL_ROOT>/<cell>_meva/<cell>_meva/{metrics.json,per_clip.tsv}   level only (identity)
  <EVAL_ROOT>/<cell>_meva/<cell>_meva/{meva.json,meva_per_clip.tsv} MEva
  <EVAL_ROOT>/<cell>_meva/identity.json

Modes: (default) run all pending cells; --preflight; --validate-only.
Resumable at cell level; a partial cell is wiped and regenerated from scratch.
"""
import argparse
import ast
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HOME = Path('/home/kojiek')
REPO = HOME / 'MeanAudio'
PY = HOME / 'venvs/dac/bin/python'
MEVA_PY = REPO / 'runtime/meva_20261003/venv/bin/python'
MEVA_SCORER = REPO / 'scripts/eval/meva_score_dir.py'
EVAL_ROOT = HOME / 'eval_output_nvme'
MC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
OUT = EVAL_ROOT / 'meva_081_084_backfill'
SUMMARY = OUT / 'summary.json'
SUFFIX = '_meva'
ROWS = 5521
MIN_FREE = 20 * 1024**3
RUNTIME_CHECK_N = 50

SEEDS = ['14159265', '16180339', '27182818']
ARM = 'phase8_qwen_caption2p0_slot0clean_qlabel_noq_quarter_s{seed}'
CTRL = 'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{seed}'
N100 = 'phase8_qwen_caption2p0_slot0clean_negmfn100_noq_quarter_s{seed}'
QL_CELLS = ['mc_mf25_cfg0', 'mc_mf25_cfg3_neg', 'mc_mf25_cfg3_lqneg',
            'hqpos_mc_mf25_cfg0', 'hqpos_mc_mf25_cfg3_lqneg']
N100_CELLS = ['mc_mf25_cfg0', 'mc_mf25_cfg3_neg']
ROLES = {'arm081': (ARM, QL_CELLS), 'control': (CTRL, QL_CELLS), 'n100': (N100, N100_CELLS)}

TOL_PEAK, TOL_LUFS, MIN_MATCH = 1e-4, 0.01, 0.999


def cells():
    first = f'{CTRL.format(seed=SEEDS[0])}_mc_mf25_cfg0'
    out = [first]
    for s in SEEDS:
        for prefix, names in ROLES.values():
            out += [f'{prefix.format(seed=s)}_{c}' for c in names]
    return list(dict.fromkeys(out))


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def cdir(label):
    return EVAL_ROOT / f'{label}{SUFFIX}'


def meva_json(label):
    p = cdir(label) / f'{label}{SUFFIX}' / 'meva.json'
    return json.loads(p.read_text()) if p.is_file() else None


def cell_done(label):
    ident, m = cdir(label) / 'identity.json', meva_json(label)
    return ident.is_file() and m is not None and json.loads(ident.read_text()).get('passed') is True \
        and m.get('n') == ROWS and math.isfinite(m.get('meva_mean', math.nan))


def eval_args(label):
    gl = EVAL_ROOT / label / 'gen.log'
    for line in gl.read_text(errors='replace').splitlines():
        if 'Eval args:' in line:
            raw = line.split('Eval args:', 1)[1]
            raw = raw[:raw.rindex('}') + 1].replace('PosixPath(', '(')
            return ast.literal_eval(raw)
    raise RuntimeError(f'no Eval args line in {gl}')


def eval_cmd(a, out_audio):
    if a['variant'] != 'meanaudio_s' or not a['use_meanflow'] or a['audio_path'] or a['prompt_suffix'] \
            or a['use_rope'] or a['debug'] or not a['no_text_attention_mask'] or not a['full_precision']:
        raise RuntimeError(f'unexpected eval args: {a}')
    cmd = [str(PY), 'eval.py', '--variant', a['variant'], '--model_path', a['model_path'],
           '--output', str(out_audio), '--tsv', a['tsv'], '--use_meanflow',
           '--num_steps', str(a['num_steps']), '--cfg_strength', str(a['cfg_strength']),
           '--no_text_attention_mask', '--encoder_name', a['encoder_name'],
           '--text_c_dim', str(a['text_c_dim']), '--seed', str(a['seed']), '--full_precision',
           '--duration', str(a['duration'])]
    if a.get('negative_prompt'):
        cmd += ['--negative_prompt', a['negative_prompt']]
    cmd += ['--no_q'] if a['no_q'] else ['--quality_level', str(a['quality_level'])]
    return cmd


def read_clips(path):
    with open(path, newline='') as f:
        return {r['id']: r for r in csv.DictReader(f, delimiter='\t')}


def stored_clips(label):
    return read_clips(EVAL_ROOT / label / label / 'per_clip.tsv')


def identity(label):
    old = stored_clips(label)
    new = read_clips(cdir(label) / f'{label}{SUFFIX}' / 'per_clip.tsv')
    common = sorted(set(old) & set(new))
    match, worst_peak, worst_lufs = 0, 0.0, 0.0
    for k in common:
        dp = abs(float(old[k]['peak']) - float(new[k]['peak']))
        lo, ln = old[k]['lufs'], new[k]['lufs']
        both_none = lo in ('', 'None') and ln in ('', 'None')
        dl = 0.0 if both_none else (abs(float(lo) - float(ln)) if lo not in ('', 'None') and ln not in ('', 'None') else math.inf)
        worst_peak, worst_lufs = max(worst_peak, dp), max(worst_lufs, dl)
        match += dp <= TOL_PEAK and dl <= TOL_LUFS
    frac = match / ROWS
    return {'label': label, 'n_old': len(old), 'n_new': len(new), 'n_common': len(common),
            'n_match': match, 'match_frac': frac, 'max_abs_dpeak': worst_peak,
            'max_abs_dlufs': worst_lufs if math.isfinite(worst_lufs) else 'inf',
            'tol_peak': TOL_PEAK, 'tol_lufs': TOL_LUFS, 'min_match': MIN_MATCH,
            'passed': len(common) == ROWS and frac >= MIN_MATCH}


def run_cell(label):
    d = cdir(label)
    if d.exists():
        shutil.rmtree(d)
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
    with open(d / 'metrics.log', 'w') as g:
        subprocess.run([str(PY), 'scripts/eval/eval_metrics.py', '--gen_dir', str(audio), '--tsv', str(MC_TSV),
                        '--exp_name', f'{label}{SUFFIX}', '--out_dir', str(d), '--skip_clap', '--skip_aes'],
                       cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    ident = identity(label)
    (d / 'identity.json').write_text(json.dumps(ident, indent=1))
    if not ident['passed']:
        raise RuntimeError(f'{label}: regenerated audio does not match the original: {ident}')
    log(f'MEVA {label}')
    with open(d / 'meva.log', 'w') as g:
        subprocess.run([str(MEVA_PY), str(MEVA_SCORER), '--audio_dir', str(audio),
                        '--out_dir', str(d / f'{label}{SUFFIX}')],
                       cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    m = meva_json(label)
    if not m or m['n'] != ROWS:
        raise RuntimeError(f'{label}: MEva incomplete ({m})')
    shutil.rmtree(audio)
    log(f'DONE {label}  MEva {m["meva_mean"]:.4f}  identity {ident["n_match"]}/{ROWS}')


def meva_clips(label):
    return {k: float(v['meva_raw']) for k, v in read_clips(cdir(label) / f'{label}{SUFFIX}' / 'meva_per_clip.tsv').items()}


def ranks(xs):
    order = sorted(range(len(xs)), key=xs.__getitem__)
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return statistics.correlation(ra, rb)


def cell_stats(label):
    mv, st = meva_clips(label), stored_clips(label)
    ids = sorted(set(mv) & set(st))
    pq = [float(st[k]['PQ']) for k in ids]
    m = [mv[k] for k in ids]
    return {'n': len(ids), 'meva_mean': statistics.fmean(m), 'aes_pq_mean': statistics.fmean(pq),
            'spearman_meva_pq': spearman(m, pq)}


def seed_stats(xs):
    if len(xs) < 2:
        return {'mean': xs[0] if xs else None}
    m, sd = statistics.fmean(xs), statistics.stdev(xs)
    return {'mean': m, 'sd': sd, 'per_seed': xs}


def summarize():
    stats = {l: cell_stats(l) for l in cells()}

    def g(prefix, s, c, key):
        return stats[f'{prefix.format(seed=s)}_{c}'][key]

    contrasts = {}
    for metric, key in (('meva', 'meva_mean'), ('aes_pq_raw', 'aes_pq_mean')):
        e1 = [(g(ARM, s, 'mc_mf25_cfg3_lqneg', key) - g(ARM, s, 'mc_mf25_cfg0', key))
              - (g(CTRL, s, 'mc_mf25_cfg3_lqneg', key) - g(CTRL, s, 'mc_mf25_cfg0', key)) for s in SEEDS]
        e2 = [g(ARM, s, 'mc_mf25_cfg3_lqneg', key) - g(ARM, s, 'mc_mf25_cfg3_neg', key) for s in SEEDS]
        e3 = [(g(ARM, s, 'hqpos_mc_mf25_cfg0', key) - g(ARM, s, 'mc_mf25_cfg0', key))
              - (g(CTRL, s, 'hqpos_mc_mf25_cfg0', key) - g(CTRL, s, 'mc_mf25_cfg0', key)) for s in SEEDS]
        hqlq = [g(ARM, s, 'hqpos_mc_mf25_cfg3_lqneg', key) - g(ARM, s, 'mc_mf25_cfg3_neg', key) for s in SEEDS]
        arm_ctrl = {c: seed_stats([g(ARM, s, c, key) - g(CTRL, s, c, key) for s in SEEDS]) for c in QL_CELLS}
        n100_e1 = [g(N100, s, 'mc_mf25_cfg0', key) - g(CTRL, s, 'mc_mf25_cfg0', key) for s in SEEDS]
        n100_neg = [g(N100, s, 'mc_mf25_cfg3_neg', key) - g(CTRL, s, 'mc_mf25_cfg3_neg', key) for s in SEEDS]
        ctrl_inf = [g(CTRL, s, 'mc_mf25_cfg3_neg', key) - g(CTRL, s, 'mc_mf25_cfg0', key) for s in SEEDS]
        contrasts[metric] = {
            '081_E1_lqneg_did': seed_stats(e1), '081_E2_lqneg_minus_fid8': seed_stats(e2),
            '081_E3_hqpos_cfg0_did': seed_stats(e3), '081_hqpos_lqneg_minus_fid8': seed_stats(hqlq),
            '081_arm_minus_control_by_cell': arm_ctrl,
            '084_n100_cfg0_minus_control_cfg0': seed_stats(n100_e1),
            '084_n100_cfg3neg_minus_control_cfg3neg': seed_stats(n100_neg),
            'control_cfg3neg_minus_cfg0': seed_stats(ctrl_inf)}
    return {'document_kind': 'meva_081_084_backfill_summary', 'written_at': time.strftime('%FT%T%z'),
            'evaluator': 'MEva f03 pooled/small (095 model lock), shadow only; AES PQ here is the raw '
                         '(not lvl30) value of the original cell, MEva is not loudness-aligned',
            'cells': stats, 'contrasts': contrasts}


def preflight():
    errs = []
    if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    if not MEVA_PY.is_file():
        errs.append(f'MEva venv missing: {MEVA_PY}')
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
        except Exception as e:  # noqa: BLE001
            errs.append(f'{label}: {e}')
    if not errs:
        rc = subprocess.run([str(MEVA_PY), str(MEVA_SCORER), '--runtime_check', str(RUNTIME_CHECK_N)],
                            cwd=REPO).returncode
        if rc:
            errs.append(f'MEva runtime check failed rc={rc}')
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    return 1 if errs else 0


def validate():
    pending = [l for l in cells() if not cell_done(l)]
    if pending:
        log(f'VALIDATE FAIL: {len(pending)} cells not done: {pending[:3]}')
        return 1
    if not SUMMARY.is_file():
        log('VALIDATE FAIL: summary missing')
        return 1
    log('VALIDATE OK')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    os.chdir(REPO)
    if a.preflight:
        return preflight()
    if a.validate_only:
        return validate()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [l for l in cells() if not cell_done(l)]
    log(f'{len(cells()) - len(todo)}/{len(cells())} cells already done; {len(todo)} to run')
    for i, label in enumerate(todo):
        if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
            raise SystemExit('disk below 20 GiB')
        run_cell(label)
        (OUT / 'progress.json').write_text(json.dumps({'done': len(cells()) - len(todo) + i + 1,
                                                       'total': len(cells()), 'last': label}))
    tmp = SUMMARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(summarize(), indent=1))
    tmp.replace(SUMMARY)
    log(f'wrote {SUMMARY}')
    return validate()


if __name__ == '__main__':
    sys.exit(main())
