#!/usr/bin/env python3
"""081 FAD backfill: regenerate the deleted audio of every 081 arm cell, score FAD, delete it again.

081 deleted its audio after scoring, so FAD was never computed. The control is the
nmv2pair run whose five cells already have FAD (084 cfg0/cfg3_neg, 104 the rest), so
only the 15 arm (qlabel) cells are regenerated. eval.py seeds one RNG for the whole run,
so a full 5521-row regeneration with the original flags reproduces the original audio;
every cell is checked against its stored per_clip.tsv (peak / LUFS) before its FAD is
accepted. The control cfg0 reproduction cell from 104 is reused (already done, skipped).

Generation flags are read back from the first "Eval args" line of each cell's gen.log;
only --output changes. FAD = eval_metrics.py --fad (VGGish, seeded 2048 subset,
/mnt/HDD/kojiek/musiccaps_reference), the same call as negmf_084_action.sh fad_cell.

Layout per cell (what negmf_084_analysis.py / the 081 analysis read):
  <EVAL_ROOT>/<cell>_fad/<cell>_fad/{metrics.json,per_clip.tsv}   FAD + level
  <EVAL_ROOT>/<cell>_fad/identity.json                           regeneration check
The reproduction cell writes to <cell>_fadrepro/ instead, so the existing FAD is untouched.

Modes: (default) run all pending cells; --preflight; --validate-only.
Resumable: a cell counts as done only when identity.json passed and FAD is finite;
a partial cell is wiped and regenerated from scratch (a top-up is not deterministic).
"""
import argparse
import ast
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path('/home/kojiek')
REPO = HOME / 'MeanAudio'
PY = HOME / 'venvs/dac/bin/python'
EVAL_ROOT = HOME / 'eval_output_nvme'
MC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
FAD_REF = Path('/mnt/HDD/kojiek/musiccaps_reference')
OUT = EVAL_ROOT / 'quality_label_081_fad_backfill'
SUMMARY = OUT / 'summary.json'
ROWS = 5521
MIN_FREE = 20 * 1024**3

SEEDS = ['14159265', '16180339', '27182818']
ARM = 'phase8_qwen_caption2p0_slot0clean_qlabel_noq_quarter_s{seed}'
CTRL = 'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{seed}'
ARM_CELLS = ['mc_mf25_cfg0', 'mc_mf25_cfg3_neg', 'mc_mf25_cfg3_lqneg',
             'hqpos_mc_mf25_cfg0', 'hqpos_mc_mf25_cfg3_lqneg']
CTRL_NEW = []   # control FAD for all five cells already exists (084 + 104)
CTRL_OLD = ARM_CELLS
REPRO = CTRL.format(seed=SEEDS[0]) + '_mc_mf25_cfg0'

# identity gate: same audio => same peak/LUFS up to flac float round-trip
TOL_PEAK, TOL_LUFS, MIN_MATCH = 1e-4, 0.01, 0.999


def cells():
    """(label, out_suffix) in run order: reproduction check first, then seed by seed."""
    out = [(REPRO, '_fadrepro')]
    for s in SEEDS:
        out += [(f'{ARM.format(seed=s)}_{c}', '_fad') for c in ARM_CELLS]
        out += [(f'{CTRL.format(seed=s)}_{c}', '_fad') for c in CTRL_NEW]
    return out


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def fad_metrics(label, suffix):
    p = EVAL_ROOT / f'{label}{suffix}' / f'{label}{suffix}' / 'metrics.json'
    return json.loads(p.read_text()) if p.is_file() else None


def fad_of(m):
    """eval_metrics.py writes the score under metrics.fad, not at the top level."""
    return None if not m else (m.get('metrics') or {}).get('fad')


def cell_done(label, suffix):
    ident = EVAL_ROOT / f'{label}{suffix}' / 'identity.json'
    m = fad_metrics(label, suffix)
    if not ident.is_file() or not m:
        return False
    fad = fad_of(m)
    return json.loads(ident.read_text()).get('passed') is True and \
        isinstance(fad, (int, float)) and math.isfinite(fad) and fad > 0


def eval_args(label):
    """The original eval.py arguments, from the 'Eval args: {...}' line of gen.log."""
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


def identity(label, suffix):
    old = read_clips(next((EVAL_ROOT / label).glob(f'{label}/per_clip.tsv')))
    new = read_clips(EVAL_ROOT / f'{label}{suffix}' / f'{label}{suffix}' / 'per_clip.tsv')
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


def run_cell(label, suffix):
    d = EVAL_ROOT / f'{label}{suffix}'
    if d.exists():
        shutil.rmtree(d)   # partial cell: regenerate from scratch
    audio = d / 'audio'
    audio.mkdir(parents=True)
    args = eval_args(label)
    cmd = eval_cmd(args, audio)
    (d / 'eval_cmd.json').write_text(json.dumps({'source_args': args, 'cmd': cmd}, indent=1, default=str))
    log(f'GEN {label}{suffix}  cfg={args["cfg_strength"]} neg={args.get("negative_prompt")!r} tsv={Path(args["tsv"]).name}')
    with open(d / 'gen.log', 'w') as g:
        subprocess.run(cmd, cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    n = len(list(audio.glob('*.flac')))
    if n != ROWS:
        raise RuntimeError(f'{label}: {n} flac, expected {ROWS}')
    log(f'FAD {label}{suffix}')
    with open(d / 'metrics.log', 'w') as g:
        subprocess.run([str(PY), 'scripts/eval/eval_metrics.py', '--gen_dir', str(audio), '--tsv', str(MC_TSV),
                        '--exp_name', f'{label}{suffix}', '--out_dir', str(d), '--skip_clap', '--skip_aes',
                        '--fad', '--ref_dir', str(FAD_REF), '--fad_num_samples', '2048'],
                       cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    m = fad_metrics(label, suffix)
    fad = fad_of(m)
    if not isinstance(fad, (int, float)) or not math.isfinite(fad) or fad <= 0:
        raise RuntimeError(f'{label}: FAD invalid ({fad})')
    ident = identity(label, suffix)
    (d / 'identity.json').write_text(json.dumps(ident, indent=1))
    if not ident['passed']:
        raise RuntimeError(f'{label}: regenerated audio does not match the original: {ident}')
    shutil.rmtree(audio)
    log(f'DONE {label}{suffix}  FAD {fad:.4f}  identity {ident["n_match"]}/{ROWS}')


def summarize():
    old_fad = fad_of(fad_metrics(REPRO, '_fad'))
    new_fad = fad_of(fad_metrics(REPRO, '_fadrepro'))
    repro = {'label': REPRO, 'fad_original': old_fad, 'fad_regenerated': new_fad,
             'abs_diff': None if old_fad is None or new_fad is None else abs(old_fad - new_fad)}
    table = {}
    for s in SEEDS:
        for role, prefix, names in (('arm', ARM, ARM_CELLS), ('control', CTRL, ARM_CELLS)):
            for c in names:
                m = fad_metrics(f'{prefix.format(seed=s)}_{c}', '_fad')
                table.setdefault(c, {}).setdefault(role, {})[s] = fad_of(m)
    for c, roles in table.items():
        a, k = roles.get('arm', {}), roles.get('control', {})
        diffs = [a[s] - k[s] for s in SEEDS if a.get(s) is not None and k.get(s) is not None]
        roles['arm_minus_control'] = diffs
        roles['arm_minus_control_mean'] = sum(diffs) / len(diffs) if len(diffs) == len(SEEDS) else None
    return {'document_kind': 'quality_label_081_fad_backfill_summary', 'written_at': time.strftime('%FT%T%z'),
            'fad_protocol': 'eval_metrics.py --fad, VGGish, random.seed(42) 2048-row subset, ref '
                            + str(FAD_REF), 'reproduction_check': repro, 'fad_by_cell': table}


def preflight():
    errs = []
    if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    if not FAD_REF.is_dir():
        errs.append(f'FAD ref dir missing: {FAD_REF}')
    for label, suffix in cells():
        try:
            a = eval_args(label)
            eval_cmd(a, Path('/dev/null'))
            if not Path(a['model_path']).is_file():
                errs.append(f'{label}: checkpoint missing {a["model_path"]}')
            if not Path(a['tsv']).is_file():
                errs.append(f'{label}: gen tsv missing {a["tsv"]}')
            if not list((EVAL_ROOT / label).glob(f'{label}/per_clip.tsv')):
                errs.append(f'{label}: stored per_clip.tsv missing')
        except Exception as e:  # noqa: BLE001
            errs.append(f'{label}: {e}')
    for s in SEEDS:
        for c in CTRL_OLD:
            if fad_of(fad_metrics(f'{CTRL.format(seed=s)}_{c}', '_fad')) is None:
                errs.append(f'control FAD missing for s{s} {c}')
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    return 1 if errs else 0


def validate():
    pending = [l + s for l, s in cells() if not cell_done(l, s)]
    if pending:
        log(f'VALIDATE FAIL: {len(pending)} cells not done: {pending[:3]}')
        return 1
    if not SUMMARY.is_file():
        log('VALIDATE FAIL: summary missing')
        return 1
    rep = json.loads(SUMMARY.read_text())['reproduction_check']
    if rep['abs_diff'] is None or rep['abs_diff'] > 1e-3:
        log(f'VALIDATE FAIL: FAD reproduction check {rep}')
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
    todo = [(l, s) for l, s in cells() if not cell_done(l, s)]
    log(f'{len(cells()) - len(todo)}/{len(cells())} cells already done; {len(todo)} to run')
    for i, (label, suffix) in enumerate(todo):
        if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
            raise SystemExit('disk below 20 GiB')
        run_cell(label, suffix)
        (OUT / 'progress.json').write_text(json.dumps({'done': len(cells()) - len(todo) + i + 1,
                                                       'total': len(cells()), 'last': label + suffix}))
        if label == REPRO:
            s = summarize()['reproduction_check']
            log(f'REPRO check: original {s["fad_original"]} regenerated {s["fad_regenerated"]}')
            if s['abs_diff'] is None or s['abs_diff'] > 1e-3:
                raise SystemExit(f'FAD reproduction check failed: {s}')
    tmp = SUMMARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(summarize(), indent=1))
    tmp.replace(SUMMARY)
    log(f'wrote {SUMMARY}')
    return validate()


if __name__ == '__main__':
    sys.exit(main())
