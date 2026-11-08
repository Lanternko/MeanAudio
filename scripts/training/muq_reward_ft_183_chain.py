#!/usr/bin/env python
"""183 arm R chain: gate -> lambda pilot -> frozen selection -> 3 seeds.

Design: docs/experiments/muq_train_exploit_183_20261108.md (arm R). MuQ is only ever
called on Jamendo training / track-disjoint Jamendo val prompts; nothing here reads
MusicCaps. Every stage writes a marker under ROOT and is skipped when the marker exists,
so the chain resumes after a pause at stage granularity. A run dir without DONE is wiped
and restarted (muq_reward_ft_183.py has no mid-run resume).

  ROOT/gate/gate_R0_br{4,2}.json   -> ROOT/gate.json {pass, br} or R_infeasible.json
  ROOT/pilot/lam{lam}/             -> 3000 steps each, --save_each_val, seed 14159265
  ROOT/selection.json              -> best val lvl30 A1_MI point (frozen) or R_no_gain.json
  ROOT/seeds/s{seed}/              -> selected lam, selected steps; net_step{steps}.pth
                                      (all 3 seeds retrained, pilot seed included: design §3.3)
  ROOT/R_DONE.json                 -> final marker (also written for infeasible / no gain)

Modes: (default) run chain; --preflight (rc 75 while VRAM is short); --validate-only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path('/home/kojiek/MeanAudio')
PY = '/home/kojiek/venvs/dac/bin/python'
TORCHRUN = '/home/kojiek/venvs/dac/bin/torchrun'
FT = REPO / 'scripts/training/muq_reward_ft_183.py'
ROOT = Path('/home/kojiek/exps_nvme/muq_reward_ft_183')
SEEDS = (14159265, 16180339, 27182818)
PILOT_SEED = 14159265
LAMS = (0.1, 0.3, 1.0)
PILOT_STEPS = 3000
VAL_EVERY = 500
MAX_LUFS_DRIFT = 3.0
MAX_SILENT = 0.05
NEED_FREE_MIB = 24 * 1024
CONTROL = 'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{seed}'


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, p)


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def free_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def ft(mode, out_dir, extra, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    env.pop('MKL_NUM_THREADS', None)
    cmd = [TORCHRUN, '--standalone', '--nproc_per_node=1', str(FT), '--mode', mode,
           '--out_dir', str(out_dir), *extra]
    print('[RUN]', ' '.join(cmd), flush=True)
    with open(log, 'a') as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode


def stage_gate():
    if (ROOT / 'gate.json').is_file() or (ROOT / 'R_infeasible.json').is_file():
        return
    tried = []
    for br in (4, 2):
        rep_p = ROOT / 'gate' / f'gate_R0_br{br}.json'
        if not rep_p.is_file():
            rc = ft('gate', ROOT / 'gate', ['--seed', str(PILOT_SEED), '--br', str(br)],
                    ROOT / 'gate' / f'gate_br{br}.log')
            if rc and not rep_p.is_file():
                tried.append({'br': br, 'rc': rc, 'report': None})
                continue
        rep = json.loads(rep_p.read_text())
        tried.append({'br': br, 'gates': rep['gates'], 'pass': rep['pass']})
        if rep['pass']:
            wjson(ROOT / 'gate.json', {'pass': True, 'br': br, 'tried': tried, 'time': now()})
            return
    wjson(ROOT / 'R_infeasible.json', {'reason': 'R0 gate failed at B_R 4 and 2', 'tried': tried,
                                       'time': now()})


def br():
    return json.loads((ROOT / 'gate.json').read_text())['br']


def train_run(out_dir, seed, lam, steps, save_each_val):
    if (out_dir / 'DONE').is_file():
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)  # partial run: no mid-run resume, restart cleanly
    extra = ['--seed', str(seed), '--lam', str(lam), '--steps', str(steps), '--val_every', str(VAL_EVERY),
             '--br', str(br()), '--exp_id', f'muq_reward_ft_183_{out_dir.name}']
    if save_each_val:
        extra.append('--save_each_val')
    rc = ft('train', out_dir, extra, out_dir.parent / f'{out_dir.name}.log')
    if rc or not (out_dir / 'DONE').is_file():
        raise SystemExit(f'[FAIL] train {out_dir} rc={rc}')


def history(run):
    return [json.loads(l) for l in (run / 'val_history.jsonl').read_text().splitlines() if l.strip()]


def stage_pilot():
    for lam in LAMS:
        train_run(ROOT / 'pilot' / f'lam{lam}', PILOT_SEED, lam, PILOT_STEPS, save_each_val=True)


def stage_select():
    """Frozen rule (design doc arm R): maximise Jamendo-val 1-NFE A1_MI_lvl30 over
    (lambda, step), excluding points whose LUFS median drifts > 3 LU from step 0 or whose
    silent fraction > 5%. Step 0 winning -> R_no_gain (no seeds are trained)."""
    if (ROOT / 'selection.json').is_file() or (ROOT / 'R_no_gain.json').is_file():
        return
    cands, base = [], None
    for lam in LAMS:
        h = history(ROOT / 'pilot' / f'lam{lam}')
        h0 = next(v for v in h if v['step'] == 0)
        base = base or h0
        for v in h:
            ok = abs(v['lufs_median'] - h0['lufs_median']) <= MAX_LUFS_DRIFT and v['silent_frac'] <= MAX_SILENT
            cands.append({'lam': lam, 'step': v['step'], 'A1_MI_lvl30': v['A1_MI_lvl30'],
                          'A3a_MI_lvl30': v['A3a_MI_lvl30'], 'lufs_median': v['lufs_median'],
                          'silent_frac': v['silent_frac'], 'eligible': ok})
    elig = [c for c in cands if c['eligible']]
    best = max(elig, key=lambda c: c['A1_MI_lvl30'])
    rec = {'rule': 'max val A1_MI_lvl30; |dLUFS|<=3 LU, silent<=5%', 'step0': base,
           'best': best, 'candidates': cands, 'time': now()}
    if best['step'] == 0:
        wjson(ROOT / 'R_no_gain.json', rec)
    else:
        wjson(ROOT / 'selection.json', rec)


def stage_seeds():
    if not (ROOT / 'selection.json').is_file():
        return
    sel = json.loads((ROOT / 'selection.json').read_text())['best']
    for seed in SEEDS:  # pilot seed retrained too, so selection and main result never share a run
        train_run(ROOT / 'seeds' / f's{seed}', seed, sel['lam'], sel['step'], save_each_val=False)
        ck = ckpt_path(seed, sel)
        assert ck.is_file(), f'[FAIL] missing {ck}'


def ckpt_path(seed, sel):
    return ROOT / 'seeds' / f's{seed}' / f"net_step{sel['step']:06d}.pth"


def final():
    status = ('infeasible' if (ROOT / 'R_infeasible.json').is_file() else
              'no_gain' if (ROOT / 'R_no_gain.json').is_file() else 'trained')
    out = {'status': status, 'time': now()}
    if status == 'trained':
        sel = json.loads((ROOT / 'selection.json').read_text())['best']
        out['selection'] = sel
        out['checkpoints'] = {str(s): str(ckpt_path(s, sel)) for s in SEEDS}
    wjson(ROOT / 'R_DONE.json', out)
    print('[DONE]', json.dumps(out), flush=True)


def preflight():
    errs = []
    for s in SEEDS:
        p = REPO / 'exps' / f'{CONTROL.format(seed=s)}_stage2_50000' / f'{CONTROL.format(seed=s)}_stage2_50000_ema_final.pth'
        if not p.is_file():
            errs.append(f'missing control {p}')
    if shutil.disk_usage('/home/kojiek').free < 30 * 2**30:
        errs.append('NVMe free < 30 GiB')
    if errs:
        print('[PREFLIGHT FAIL]', errs, flush=True)
        return 1
    if (ROOT / 'R_DONE.json').is_file():
        return 0
    f = free_mib()
    if f < NEED_FREE_MIB:
        print(f'[PREFLIGHT WAIT] GPU free {f} MiB < {NEED_FREE_MIB} MiB', flush=True)
        return 75
    return 0


def validate():
    p = ROOT / 'R_DONE.json'
    if not p.is_file():
        print('[INVALID] no R_DONE.json')
        return 1
    d = json.loads(p.read_text())
    if d['status'] == 'trained':
        missing = [c for c in d['checkpoints'].values() if not Path(c).is_file()]
        if missing:
            print('[INVALID] missing', missing)
            return 1
    print('[VALID]', d['status'])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    a = ap.parse_args()
    if a.preflight:
        return preflight()
    if a.validate_only:
        return validate()
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'R_DONE.json').is_file():
        return validate()
    stage_gate()
    if (ROOT / 'gate.json').is_file():
        stage_pilot()
        stage_select()
        stage_seeds()
    final()
    return validate()


if __name__ == '__main__':
    sys.exit(main())
