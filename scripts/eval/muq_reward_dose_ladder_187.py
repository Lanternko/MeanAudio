#!/usr/bin/env python
"""187: small-dose ladder of the MuQ A1 reward fine-tune (arm R of 183).

Design: docs/experiments/muq_reward_dose_ladder_187_20261110.md

  1. train   3 seeds x 160 steps at lam 0.1, checkpoint every 10 steps (ladder)
             3 seeds x 160 steps at lam 0   (Z160: same recipe without the reward term)
  2. select  primary dose from the Jamendo val history only, frozen before any MusicCaps cell
  3. score   C / primary dose / Z160 / fixed rungs on the 183 MusicCaps Part B harness (cfg0)
  4. summary pre-registered readings E / E_clamp / S / stealth

Training is muq_reward_ft_183.py and scoring is muq_train_exploit_183.py, both unmodified (sha-bound by
185 / 186): this file only chooses step counts and maps arm labels to checkpoints.
"""
import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/home/kojiek/MeanAudio')
sys.path[:0] = [str(REPO / 'scripts/eval')]          # python -I: the script dir is not on sys.path
import muq_train_exploit_183 as H  # noqa: E402

TORCHRUN = '/home/kojiek/venvs/dac/bin/torchrun'
FT = REPO / 'scripts/training/muq_reward_ft_183.py'
R183 = H.R_ROOT
TRAIN = Path('/home/kojiek/exps_nvme/muq_reward_dose_187')
OUT = H.EVAL_ROOT / 'muq_reward_dose_187'
DESIGN = 'docs/experiments/muq_reward_dose_ladder_187_20261110.md'
KIND = 'muq_reward_dose_ladder_187_summary'
SEEDS = H.SEEDS
PRIMARY = H.PRIMARY
LAM = 0.1
STEPS, VAL_EVERY, VAL_N, LOG_EVERY = 160, 10, 500, 10
LADDER_STEPS = list(range(VAL_EVERY, STEPS + 1, VAL_EVERY))
RUNGS = (10, 20, 40, 80, 160)
VAL_TARGET = 0.5                       # primary dose: smallest eligible step with val dA1 >= this
MAX_LUFS_DRIFT, MAX_SILENT = 3.0, 0.05  # eligibility, same as the 183 chain
FAD_TOL = 0.30                         # FAD tie band (123: training-seed floor 0.08-0.32)
TIE = 0.10                             # tie band = 0.10 x scale
NEED_FREE_MIB = 16 * 1024
MIN_FREE_TRAIN = 30 * 1024**3
BOOT_SEED = 187
log = H.log

H.P = H.Paths(OUT)
_orig_spec = H.cell_spec


# ── arms / checkpoints ───────────────────────────────────
def arm_name(step):
    return f'D{step:03d}'


def run_dir(kind, seed):
    return TRAIN / kind / f's{seed}'


def ckpt(arm, seed):
    if arm == 'Z160':
        return run_dir('zero', seed) / f'net_step{STEPS:06d}.pth'
    if len(arm) == 4 and arm[0] == 'D' and arm[1:].isdigit() and int(arm[1:]) in LADDER_STEPS:
        return run_dir('ladder', seed) / f'net_step{int(arm[1:]):06d}.pth'
    raise RuntimeError(f'unknown arm {arm}')


def cell_spec(lab):
    arm, proto, seed = H.parse(lab)
    if proto != 'cfg0':
        raise RuntimeError(f'{lab}: 187 scores cfg0 only')
    a = _orig_spec(H.label('C', proto, seed))
    if arm != 'C':
        a['model_path'] = str(ckpt(arm, seed))
    return a


H.cell_spec = cell_spec                # run_cell() resolves cell_spec through the module globals


def sel_path():
    return H.P.root / 'dose_selection.json'


def selection():
    return json.loads(sel_path().read_text()) if sel_path().is_file() else None


def arms():
    """Run order: control (harness gate), primary dose, Z160, then the remaining fixed rungs."""
    sel = selection()
    out = ['C']
    if sel:
        out.append(arm_name(sel['primary']['step']))
    out.append('Z160')
    return out + [arm_name(s) for s in RUNGS if arm_name(s) not in out]


def all_labels():
    return [H.label(a, 'cfg0', s) for a in arms() for s in SEEDS]


# ── training ─────────────────────────────────────────────
def free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def br():
    return int(json.loads((R183 / 'gate.json').read_text())['br'])


def want_steps(kind):
    return LADDER_STEPS if kind == 'ladder' else [STEPS]


def history(kind, seed):
    p = run_dir(kind, seed) / 'val_history.jsonl'
    return {int(r['step']): r for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip())}


def train_done(kind, seed):
    d = run_dir(kind, seed)
    if not (d / 'DONE').is_file():
        return False
    if not all((d / f'net_step{s:06d}.pth').is_file() for s in want_steps(kind)):
        return False
    try:
        h = history(kind, seed)
    except (OSError, ValueError):
        return False
    return all(s in h for s in [0] + want_steps(kind))


RUNS = [('ladder', s) for s in SEEDS] + [('zero', s) for s in SEEDS]


def progress(phase, last):
    H.P.root.mkdir(parents=True, exist_ok=True)
    labs = all_labels()
    H.P.progress.write_text(json.dumps({
        'phase': phase, 'train_done': sum(train_done(k, s) for k, s in RUNS), 'train_total': len(RUNS),
        'cells_done': sum(H.done(l) for l in labs), 'cells_total': len(labs), 'last': last,
        'written_at': time.strftime('%FT%T%z')}))


def train_one(kind, seed):
    d = run_dir(kind, seed)
    if train_done(kind, seed):
        return 0
    if d.exists():
        shutil.rmtree(d)                # partial run: the FT script has no mid-run resume
    f = free_mib()
    if f < NEED_FREE_MIB:
        log(f'TRAIN WAIT GPU free {f} MiB < {NEED_FREE_MIB} MiB')
        return 75
    if shutil.disk_usage(TRAIN.parent).free < MIN_FREE_TRAIN:
        raise SystemExit('NVMe below 30 GiB')
    d.mkdir(parents=True)
    lam, val_every = (LAM, VAL_EVERY) if kind == 'ladder' else (0.0, STEPS)
    cmd = [TORCHRUN, '--standalone', '--nproc_per_node=1', str(FT), '--mode', 'train', '--out_dir', str(d),
           '--seed', seed, '--lam', str(lam), '--steps', str(STEPS), '--val_every', str(val_every),
           '--val_n', str(VAL_N), '--log_every', str(LOG_EVERY), '--br', str(br()), '--save_each_val',
           '--exp_id', f'muq_reward_dose_187_{kind}_s{seed}']
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('MKL_NUM_THREADS', None)
    log(f'TRAIN {kind} s{seed}  lam={lam} steps={STEPS} val_every={val_every}')
    with open(d.parent / f's{seed}.log', 'a') as g:
        g.write(f'\n==== {time.strftime("%F %T")} {" ".join(cmd)}\n')
        g.flush()
        rc = subprocess.run(cmd, stdout=g, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
    if rc:
        log(f'TRAIN FAIL {kind} s{seed} rc={rc} (see {d.parent / f"s{seed}.log"})')
        return 1
    if not train_done(kind, seed):
        log(f'TRAIN FAIL {kind} s{seed}: DONE / checkpoints / val history incomplete')
        return 1
    h = history(kind, seed)
    log(f'TRAIN DONE {kind} s{seed}  val {PRIMARY} step0 {h[0][PRIMARY]:.3f} -> step{STEPS} {h[STEPS][PRIMARY]:.3f}')
    return 0


def train_all():
    for kind, seed in RUNS:
        rc = train_one(kind, seed)
        progress('train', f'{kind}_s{seed}')
        if rc:
            return rc
    return 0


# ── dose selection (Jamendo val only, frozen before MusicCaps) ──
def val_table():
    hist = {s: history('ladder', s) for s in SEEDS}
    rows = []
    for st in LADDER_STEPS:
        d1 = {s: hist[s][st][PRIMARY] - hist[s][0][PRIMARY] for s in SEEDS}
        d3 = {s: hist[s][st]['A3a_MI_lvl30'] - hist[s][0]['A3a_MI_lvl30'] for s in SEEDS}
        drift = max(abs(hist[s][st]['lufs_median'] - hist[s][0]['lufs_median']) for s in SEEDS)
        silent = max(hist[s][st]['silent_frac'] for s in SEEDS)
        rows.append({'step': st, 'val_dA1': statistics.fmean(d1.values()), 'val_dA1_seeds': d1,
                     'val_dA3a': statistics.fmean(d3.values()), 'val_dA3a_seeds': d3,
                     'val_A1': statistics.fmean(hist[s][st][PRIMARY] for s in SEEDS),
                     'lufs_drift_max': drift, 'silent_frac_max': silent,
                     'eligible': drift <= MAX_LUFS_DRIFT and silent <= MAX_SILENT})
    return rows, {s: hist[s][0][PRIMARY] for s in SEEDS}


def select_dose():
    sel = selection()
    if sel:
        return sel
    scored = sorted(p.stem for p in H.P.per_cell.glob('*.json') if not p.name.startswith('C_')) \
        if H.P.per_cell.is_dir() else []
    if scored:
        raise SystemExit(f'dose not frozen but MusicCaps cells already scored: {scored[:3]}')
    rows, step0 = val_table()
    elig = [r for r in rows if r['eligible']]
    hit = [r for r in elig if r['val_dA1'] >= VAL_TARGET]
    if hit:
        prim, rule = min(hit, key=lambda r: r['step']), 'smallest eligible step with val dA1 >= target'
    elif elig:
        prim, rule = max(elig, key=lambda r: r['val_dA1']), 'target not reached: eligible step with max val dA1'
    else:
        prim, rule = max(rows, key=lambda r: r['val_dA1']), 'no eligible step: max val dA1 (ineligible)'
    sel = {'document_kind': 'muq_reward_dose_ladder_187_selection', 'written_at': time.strftime('%FT%T%z'),
           'design': DESIGN, 'source': 'Jamendo track-disjoint val (first 500), 1-NFE, lvl30; no MusicCaps',
           'lam': LAM, 'steps': STEPS, 'val_n': VAL_N, 'val_target': VAL_TARGET,
           'eligibility': f'|dLUFS median| <= {MAX_LUFS_DRIFT} LU and silent <= {MAX_SILENT:.0%}, every seed',
           'rule': rule, 'primary': {'step': prim['step'], 'arm': arm_name(prim['step']),
                                     'val_dA1': prim['val_dA1'], 'eligible': prim['eligible']},
           'val_step0_A1': step0, 'table': rows}
    H.P.root.mkdir(parents=True, exist_ok=True)
    tmp = sel_path().with_suffix('.tmp')
    tmp.write_text(json.dumps(sel, indent=1, default=float))
    tmp.replace(sel_path())
    log(f'DOSE primary step {prim["step"]} (val dA1 {prim["val_dA1"]:+.3f}, eligible {prim["eligible"]}) [{rule}]')
    return sel


# ── summary ──────────────────────────────────────────────
def clamp_stats(arm, ids):
    """A1 clamped to the label range [1, 5] before differencing (arm - control, paired per prompt)."""
    def cl(v):
        return min(5.0, max(1.0, v))
    seeds = {}
    for s in SEEDS:
        x, c = H.load_cache(H.label(arm, 'cfg0', s)), H.load_cache(H.label('C', 'cfg0', s))
        xv, cv = [x[i][PRIMARY] for i in ids], [c[i][PRIMARY] for i in ids]
        srt = sorted(xv)
        seeds[s] = {'d_clamped': statistics.fmean(cl(a) - cl(b) for a, b in zip(xv, cv)),
                    'frac_gt5_arm': sum(v > 5 for v in xv) / len(xv),
                    'frac_gt5_control': sum(v > 5 for v in cv) / len(cv),
                    'frac_lt1_arm': sum(v < 1 for v in xv) / len(xv),
                    'p95_arm': srt[int(0.95 * (len(srt) - 1))], 'max_arm': srt[-1],
                    'frac_prompts_up': sum(a > b for a, b in zip(xv, cv)) / len(xv)}
    out = {'seeds': seeds}
    for k in ('d_clamped', 'frac_gt5_arm', 'frac_gt5_control', 'frac_lt1_arm', 'p95_arm', 'max_arm',
              'frac_prompts_up'):
        out[k] = statistics.fmean(seeds[s][k] for s in SEEDS)
    out['pass'] = bool(out['d_clamped'] >= H.M_A1 and all(seeds[s]['d_clamped'] > 0 for s in SEEDS))
    return out


def read_dose(arm, ids, rng, val_eligible):
    cmp = H.compare(arm, 'cfg0', ids, rng)
    m, T, S = cmp['mean'], cmp['T'], H.S
    e = bool(H.same_sign_pos(cmp, PRIMARY, H.M_A1))
    s187 = {'T_A3a_lt_0.5': T['A3a_MI_lvl30'] is not None and T['A3a_MI_lvl30'] < 0.5,
            'T_PQ_lt_0.5': T['PQ_lvl30'] is not None and T['PQ_lvl30'] < 0.5,
            'dCLAP_lt_tie': m['CLAP'] < TIE * S['CLAP'],
            'dFAD_gt_-tol': m['fad_d'] > -FAD_TOL}
    st = {'abs_dPQ_le_tie': abs(m['PQ_lvl30']) <= TIE * S['PQ_lvl30'],
          'abs_dCE_le_tie': abs(m['CE_lvl30']) <= TIE * S['CE_lvl30'],
          'abs_dCLAP_le_tie': abs(m['CLAP']) <= TIE * S['CLAP'],
          'abs_dFAD_le_tol': abs(m['fad_d']) <= FAD_TOL,
          'val_eligible': val_eligible}
    cl = clamp_stats(arm, ids)
    r = {'E': {'pass': e, 'dA1': m[PRIMARY], 'seeds': {s: cmp['seeds'][s][PRIMARY] for s in SEEDS},
               'ci95': cmp['ci95'][PRIMARY], 'dA1_over_S': m[PRIMARY] / S[PRIMARY]},
         'E_clamp': cl,
         'S': {'pass': all(s187.values()), 'parts': s187, 'T': T},
         's_rule_183': {'pass': all(H.s_rule(cmp).values()), 'parts': H.s_rule(cmp)},
         'stealth': {'pass': all(st.values()), 'parts': st,
                     'd': {k: m[k] for k in ('PQ_lvl30', 'CE_lvl30', 'CLAP', 'A3a_MI_lvl30', 'fad_d', 'lufs_d')},
                     'silent_arm': sum(cmp['seeds'][s]['silent_arm'] for s in SEEDS),
                     'silent_control': sum(cmp['seeds'][s]['silent_control'] for s in SEEDS)},
         'compare': cmp}
    r['verdict'] = ('no_effect' if not e else 'general_shift' if not r['S']['pass']
                    else 'in_range_exploit' if cl['pass'] and r['stealth']['pass']
                    else 'exploit_with_side_effects')
    return r


def spearman(xs, ys):
    import numpy as np
    from scipy.stats import rankdata
    if len(xs) < 3:
        return None
    rx, ry = rankdata(xs), rankdata(ys)
    c = np.corrcoef(rx, ry)[0, 1]
    return float(c) if math.isfinite(c) else None


def summarize():
    import numpy as np
    rng = np.random.default_rng(BOOT_SEED)
    H.load_scales()
    ids = H.ids_now()
    sel = selection()
    vt = {r['step']: r for r in sel['table']}
    out = {'document_kind': KIND, 'written_at': time.strftime('%FT%T%z'), 'design': DESIGN, 'n_prompts': len(ids),
           'S': H.S, 'm_A1': H.M_A1, 'tie': TIE, 'fad_tol': FAD_TOL, 'lam': LAM, 'steps': STEPS,
           'harness': json.loads(H.P.harness.read_text()) if H.P.harness.is_file() else None,
           'selection': {k: sel[k] for k in ('primary', 'rule', 'val_target', 'eligibility')}}
    cells = {}
    for lab in all_labels():
        rows = H.load_cache(lab)
        j = json.loads((H.P.per_cell / f'{lab}.json').read_text())
        cells[lab] = {k: H.finite_mean([r[k] for r in rows.values()]) for k in H.KEYS + list(H.BANDS) + ['lufs']}
        cells[lab].update(fad=j['fad'], silent_n=sum(int(r['silent']) for r in rows.values()))
    out['cells'] = cells
    doses = {}
    for a in arms():
        if a == 'C':
            continue
        step = STEPS if a == 'Z160' else int(a[1:])
        doses[a] = read_dose(a, ids, rng, True if a == 'Z160' else bool(vt[step]['eligible']))
        doses[a]['step'] = step
        if a != 'Z160':
            mc = doses[a]['E']['dA1']
            doses[a]['val'] = {'dA1': vt[step]['val_dA1'], 'dA3a': vt[step]['val_dA3a'],
                               'val_over_mc': vt[step]['val_dA1'] / mc if mc else None}
    out['doses'] = doses
    prim = sel['primary']['arm']
    out['primary'] = {'arm': prim, 'step': sel['primary']['step'], 'verdict': doses[prim]['verdict']}
    out['verdict'] = doses[prim]['verdict']
    # descriptive only
    d_arms = sorted((a for a in doses if a != 'Z160'), key=lambda a: doses[a]['step'])
    keys = H.KEYS + ['fad_d', 'lufs_d']
    net = {k: doses['D160']['compare']['mean'][k] - doses['Z160']['compare']['mean'][k] for k in keys}
    passing = [a for a in d_arms if doses[a]['E']['pass']]
    out['descriptive'] = {
        'curve': [{'arm': a, 'step': doses[a]['step'], 'dA1': doses[a]['E']['dA1'],
                   'dA1_clamped': doses[a]['E_clamp']['d_clamped'], 'frac_gt5': doses[a]['E_clamp']['frac_gt5_arm'],
                   **doses[a]['stealth']['d'], 'E': doses[a]['E']['pass'], 'S': doses[a]['S']['pass'],
                   'stealth': doses[a]['stealth']['pass'], 'verdict': doses[a]['verdict']} for a in d_arms],
        'spearman_step_dA1': spearman([doses[a]['step'] for a in d_arms], [doses[a]['E']['dA1'] for a in d_arms]),
        'D160_minus_Z160': net,
        'smallest_mc_dose_passing_E': {'arm': passing[0] if passing else None,
                                       'note': 'chosen by looking at MusicCaps: descriptive, never the primary'}}
    return out


def write_summary():
    s = summarize()
    tmp = H.P.summary.with_suffix('.tmp')
    tmp.write_text(json.dumps(s, indent=1, default=float))
    tmp.replace(H.P.summary)
    p = s['doses'][s['primary']['arm']]
    log(f'wrote {H.P.summary}  primary {s["primary"]["arm"]} verdict={s["verdict"]}  dA1 {p["E"]["dA1"]:+.4f} '
        f'clamped {p["E_clamp"]["d_clamped"]:+.4f}  dPQ {p["stealth"]["d"]["PQ_lvl30"]:+.4f} '
        f'dFAD {p["stealth"]["d"]["fad_d"]:+.3f}')
    return s


# ── modes ────────────────────────────────────────────────
def control_ckpt(seed):
    e = f'{H.CONTROL.format(seed=seed)}_stage2_50000'
    return REPO / 'exps' / e / f'{e}_ema_final.pth'


def preflight():
    errs = []
    if shutil.disk_usage(H.EVAL_ROOT).free < MIN_FREE_TRAIN:
        errs.append('less than 30 GiB free on NVMe')
    for p in (H.PARTB / 'part_b_selection.json', H.TTM115, H.MC_TSV, Path(H.FAD_REF), FT, Path(TORCHRUN),
              R183 / 'gate.json', R183 / 'selection.json'):
        if not Path(p).exists():
            errs.append(f'missing {p}')
    training_done = all(train_done(k, s) for k, s in RUNS)
    try:
        if not json.loads((R183 / 'gate.json').read_text())['pass']:
            errs.append('183 gate.json did not pass')
        lam = json.loads((R183 / 'selection.json').read_text())['best']['lam']
        if lam != LAM:
            errs.append(f'183 selection lam {lam} != {LAM} (design fixes lam {LAM})')
        for s in SEEDS:
            if not control_ckpt(s).is_file():
                errs.append(f'control ema_final missing {control_ckpt(s)}')
        ids = H.write_subset(None)
        if len(ids) != 1000:
            errs.append(f'{len(ids)} Part B prompts, expected 1000')
        for lab in all_labels():
            a = cell_spec(lab)
            H.eval_cmd(a, Path('/dev/null'))
            if (H.parse(lab)[0] == 'C' or training_done) and not Path(a['model_path']).is_file():
                errs.append(f'{lab}: checkpoint missing {a["model_path"]}')
        for s in SEEDS:
            p181 = H.R181 / f'{H.CONTROL.format(seed=s)}_mc_mf25_cfg0.tsv'
            if not p181.is_file() or not set(ids) <= set(H.read_clips(p181)):
                errs.append(f'181 per_cell missing/incomplete: {p181}')
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if errs:
        return 1
    log(f'PREFLIGHT OK ({len(all_labels())} cells, arms {arms()}, training done {training_done})')
    if not training_done:
        f = free_mib()
        if f < NEED_FREE_MIB:
            log(f'PREFLIGHT WAIT GPU free {f} MiB < {NEED_FREE_MIB} MiB')
            return 75
    return 0


def validate():
    if not H.P.harness.is_file() or not json.loads(H.P.harness.read_text())['passed']:
        log('VALIDATE FAIL: harness gate missing or failed (design: stop)')
        return 1
    if selection() is None:
        log('VALIDATE FAIL: dose_selection.json missing')
        return 1
    pending = [l for l in all_labels() if not H.done(l)]
    if pending:
        log(f'VALIDATE FAIL: {len(pending)} cells not done: {pending[:3]}')
        return 1
    if not H.P.summary.is_file():
        log('VALIDATE FAIL: summary missing')
        return 1
    s = json.loads(H.P.summary.read_text())
    if s.get('document_kind') != KIND or 'verdict' not in s or s.get('primary', {}).get('arm') not in s.get('doses', {}):
        log('VALIDATE FAIL: summary malformed')
        return 1
    log(f'VALIDATE OK  primary {s["primary"]["arm"]} verdict={s["verdict"]}')
    return 0


def main():
    global TRAIN
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--root', help=argparse.SUPPRESS)
    ap.add_argument('--train-root', help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.chdir(REPO)
    if a.root:
        H.P = H.Paths(Path(a.root))
    if a.train_root:
        TRAIN = Path(a.train_root)
    if a.preflight:
        return preflight()
    if a.validate_only:
        return validate()
    H.P.root.mkdir(parents=True, exist_ok=True)
    if a.summary_only:
        select_dose()
        write_summary()
        return validate()
    rc = preflight()
    if rc:
        return rc
    rc = train_all()
    if rc:
        return rc
    select_dose()
    for lab in all_labels():
        if not H.done(lab):
            if shutil.disk_usage(H.EVAL_ROOT).free < H.MIN_FREE:
                raise SystemExit('disk below 20 GiB')
            H.run_cell(lab)
        progress('score', lab)
        if lab == H.label('C', 'cfg0', SEEDS[-1]) and not H.harness():
            return 1
    write_summary()
    return validate()


if __name__ == '__main__':
    sys.exit(main())
