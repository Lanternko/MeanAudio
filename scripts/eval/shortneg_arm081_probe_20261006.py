#!/usr/bin/env python
"""106: the 086 short-negative wording set on the 081 arm (qlabel quarter s14159265).

086 found the ~0 gain of "Low quality recording." on the 081 control (A = nmv2pair
quarter s14159265) is mostly wording: on A the same string is the weakest negative,
below irrelevant text, while "low quality, noisy" gives +0.59. The 081 arm was trained
with that exact string as a caption prefix on the bottom-20% PQ rows, so its +1.06
diff-in-diff for "Low quality recording." mixes a training effect with a wording
handicap of the control. This runs the same 7 cells on the arm (C) so every wording
gets a clean arm-minus-control diff-in-diff.

  C = qlabel quarter s14159265 (081 arm, same seed as A)
  cells: cfg0, none, lqrec, lqnoisy, lq, irrel, fid8   (negatives as in 086)
  plus A cfg0 regenerated as a reproduction anchor vs 086's A__cfg0.json
  (recorded, not blocking: |dPQ| <= 0.005; AES batch composition drifts ~1e-3).

A's per-clip reads come from 086's cells JSON (same subset, noise, scorer, pinned shas).
Every cell: MusicCaps subset1024, MeanFlow 25, seed 42, fp32, NoMask, --no_q; scored
with eval_metrics.py (CLAP batch 1) and after -30 LUFS matching. Audio deleted after.

Usage: python shortneg_arm081_probe_20261006.py [--preflight | --validate-only]
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/kojiek/MeanAudio')
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/shortneg_arm081_probe_20261006')
A086 = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/shortneg_2x2_probe_20260928/cells')
CELLS = OUT / 'cells'
SUMMARY = OUT / 'summary.json'
PY = '/home/kojiek/venvs/dac/bin/python'
OLD = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation')
SUBSET = OLD / 'musiccaps_subset1024.tsv'
ROWS = 1024

CKPT = {
    'A': ROOT / 'exps/phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265_stage2_50000'
               '/phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265_stage2_50000_ema_final.pth',
    'C': ROOT / 'exps/phase8_qwen_caption2p0_slot0clean_qlabel_noq_quarter_s14159265_stage2_50000'
               '/phase8_qwen_caption2p0_slot0clean_qlabel_noq_quarter_s14159265_stage2_50000_ema_final.pth',
}
NEG = {
    'cfg0': None,
    'none': None,
    'lqrec': 'Low quality recording.',
    'lqnoisy': 'low quality, noisy',
    'lq': 'low quality',
    'irrel': 'a photograph of a cat, a spreadsheet, printed text',
    'fid8': 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi',
}
KEYS = ['cfg0', 'lqrec', 'lqnoisy', 'lq', 'none', 'irrel', 'fid8']
GRID = [('A', 'cfg0')] + [('C', k) for k in KEYS]   # reproduction anchor first
ANCHOR_TOL = 0.005
METRICS = ('PQ', 'CE', 'CU', 'PC', 'clap')
BOOT_N, BOOT_SEED = 10000, 20261006
HARD_STOP_FREE = 10_000_000_000
IMMUTABLE = {
    'eval.py': 'ba66c66b2ca3b7db0a698338932f6ee474208c1302c4592724f1955f3ccb2339',
    'meanaudio/model/networks.py': '5970fd615640c3d5a2b38aa025c3f3f26dee3412f5b8de810414c16d732cbe69',
    'scripts/eval/eval_metrics.py': '47406ee5bf30c837733a00be306e813d8c27301a2a8ea30de9f8f28b1dfec67d',
    'scripts/eval/level_match_rescore.py': '133ac816e3de21e732effd5a8f75a067029868aa589fa3f4d0c64ede4ca88714',
    str(SUBSET): '2e852db02fcc4d1f4d176f757b7c483a8f988ec9010b77d5e461d64166fee2e9',
}
# mean_flow.py is not pinned: the running training pipelines patch its loss()
# (S1/S2 jvp arguments); eval's MeanFlow sampling does not go through loss().


def name(ck, key):
    return f'{ck}__{key}_repro' if ck == 'A' else f'{ck}__{key}'


def cell_dir(ck, key):
    return CELLS / name(ck, key)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ── generation / scoring ─────────────────────────────────
def generate(ck, key):
    d = cell_dir(ck, key)
    shutil.rmtree(d, ignore_errors=True)          # never top up: eval.py seeds one RNG per run
    shutil.rmtree(d.parent / f'{d.name}_lvl30', ignore_errors=True)
    (d / 'audio').mkdir(parents=True)
    cmd = [PY, 'eval.py', '--variant', 'meanaudio_s', '--model_path', str(CKPT[ck]),
           '--output', str(d / 'audio'), '--tsv', str(SUBSET), '--use_meanflow',
           '--num_steps', '25', '--cfg_strength', '0.0' if key == 'cfg0' else '3.0',
           '--no_text_attention_mask', '--encoder_name', 't5_clap', '--text_c_dim', '512',
           '--seed', '42', '--full_precision', '--no_q']
    if NEG[key]:
        cmd += ['--negative_prompt', NEG[key]]
    with open(d / 'gen.log', 'w') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    n = len(list((d / 'audio').glob('*.flac')))
    if n != ROWS:
        raise SystemExit(f'[FAIL] {name(ck, key)}: {n} clips, expected {ROWS}')
    return cmd


def score(ck, key):
    d = cell_dir(ck, key)
    un = d / 'metrics' / 'per_clip.tsv'
    if not un.exists():
        subprocess.run([PY, str(ROOT / 'scripts/eval/eval_metrics.py'), '--gen_dir', str(d / 'audio'),
                        '--tsv', str(SUBSET), '--exp_name', 'metrics', '--out_dir', str(d)],
                       cwd=ROOT, check=True)
    lvl = d.parent / f'{d.name}_lvl30'
    if not list(lvl.glob('*/per_clip.tsv')):
        subprocess.run([PY, str(ROOT / 'scripts/eval/level_match_rescore.py'), '--cell_dir', str(d),
                        '--tsv', str(SUBSET)], cwd=ROOT, check=True)
    return un, next(lvl.glob('*/per_clip.tsv'))


def per_clip(path):
    with open(path, encoding='utf-8', newline='') as f:
        return {r['id']: r for r in csv.DictReader(f, delimiter='\t')}


def fnum(x):
    return float(x) if x != '' else float('nan')


def run_cell(ck, key):
    path = CELLS / f'{name(ck, key)}.json'
    if path.exists():
        return json.loads(path.read_text())
    free = shutil.disk_usage(OUT).free
    if free < HARD_STOP_FREE:
        raise SystemExit(f'[FAIL] disk hard stop: {free / 1e9:.1f} GB free')
    cmd = generate(ck, key)
    un_p, lv_p = score(ck, key)
    un, lv = per_clip(un_p), per_clip(lv_p)
    if len(un) != ROWS or set(un) != set(lv):
        raise SystemExit(f'[FAIL] {name(ck, key)}: per_clip rows {len(un)} / {len(lv)}')
    rec = {'name': name(ck, key), 'checkpoint': ck, 'checkpoint_path': str(CKPT[ck]),
           'cell': key, 'negative_prompt': NEG[key], 'cfg': 0.0 if key == 'cfg0' else 3.0,
           'command': cmd,
           'mean': {m: float(np.mean([float(un[i][m]) for i in un])) for m in METRICS},
           'mean_lvl30': {m: float(np.mean([float(lv[i][m]) for i in lv])) for m in METRICS},
           'lufs': float(np.nanmean([fnum(un[i]['lufs']) for i in un])),
           'crest': float(np.nanmean([fnum(un[i]['crest']) for i in un])),
           'silent_n': int(sum(int(un[i]['silent']) for i in un)),
           'per_clip': {i: {m: float(un[i][m]) for m in METRICS} | {f'{m}_lvl30': float(lv[i][m]) for m in METRICS}
                        for i in sorted(un)}}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec, indent=1) + '\n')
    tmp.replace(path)
    d = cell_dir(ck, key)
    shutil.rmtree(d / 'audio', ignore_errors=True)
    shutil.rmtree(d.parent / f'{d.name}_lvl30' / 'audio', ignore_errors=True)
    print(f'[cell] {rec["name"]}: PQ {rec["mean"]["PQ"]:.4f} (lvl30 {rec["mean_lvl30"]["PQ"]:.4f}) '
          f'CLAP {rec["mean"]["clap"]:.4f} LUFS {rec["lufs"]:.1f} silent {rec["silent_n"]}', flush=True)
    return rec


# ── analysis ─────────────────────────────────────────────
def boot(x, rng):
    x = np.asarray(x, dtype=np.float64)
    b = rng.choice(x, (BOOT_N, len(x))).mean(1)
    return {'mean': float(x.mean()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def load_a086():
    return {k: json.loads((A086 / f'A__{k}.json').read_text()) for k in KEYS}


def analyse(recs, a086):
    rng = np.random.default_rng(BOOT_SEED)
    ids = sorted(a086['cfg0']['per_clip'])
    pc = {('A', k): a086[k]['per_clip'] for k in KEYS} | {('C', k): recs[name('C', k)]['per_clip'] for k in KEYS}

    def gain(ck, key, m):
        return [pc[(ck, key)][i][m] - pc[(ck, 'cfg0')][i][m] for i in ids]

    ms = ('PQ', 'PQ_lvl30', 'clap', 'CE', 'PC')
    gains = {f'{ck}__{k}': {m: boot(gain(ck, k, m), rng) for m in ms}
             for ck in ('A', 'C') for k in KEYS if k != 'cfg0'}
    did = {k: {m: boot([c - a for c, a in zip(gain('C', k, m), gain('A', k, m))], rng) for m in ms}
           for k in KEYS if k != 'cfg0'}
    # cfg0 itself: arm minus control with no negative (081's "no-prefix CFG0" read)
    cfg0_diff = {m: boot([pc[('C', 'cfg0')][i][m] - pc[('A', 'cfg0')][i][m] for i in ids], rng) for m in ms}
    wording = {}
    for ck in ('A', 'C'):
        for k in ('lqnoisy', 'lq', 'irrel', 'fid8', 'none'):
            for m in ('PQ', 'PQ_lvl30'):
                wording[f'{ck}__{k}_minus_lqrec__{m}'] = boot(
                    [pc[(ck, k)][i][m] - pc[(ck, 'lqrec')][i][m] for i in ids], rng)
    # label specificity: DiD(lqrec) - DiD(k) = C(lqrec - k) - A(lqrec - k)
    specificity = {f'lqrec_minus_{k}': {m: boot([(pc[('C', 'lqrec')][i][m] - pc[('C', k)][i][m])
                                                 - (pc[('A', 'lqrec')][i][m] - pc[('A', k)][i][m]) for i in ids], rng)
                                        for m in ('PQ', 'PQ_lvl30', 'clap')}
                   for k in ('lqnoisy', 'lq', 'irrel', 'fid8')}
    rep = recs[name('A', 'cfg0')]
    new, old = rep['mean']['PQ'], a086['cfg0']['mean']['PQ']
    r = float(np.corrcoef([a086['cfg0']['per_clip'][i]['PQ'] for i in ids],
                          [rep['per_clip'][i]['PQ'] for i in ids])[0, 1])
    anchor = {'old_PQ': old, 'new_PQ': new, 'diff': new - old, 'per_clip_r': r,
              'max_abs_clip_diff': float(max(abs(rep['per_clip'][i]['PQ'] - a086['cfg0']['per_clip'][i]['PQ'])
                                             for i in ids)),
              'pass': abs(new - old) <= ANCHOR_TOL}
    return {'gains_vs_own_cfg0': gains, 'diff_in_diff_C_minus_A': did, 'cfg0_C_minus_A': cfg0_diff,
            'wording_vs_lqrec': wording, 'label_specificity': specificity, 'anchor_086': anchor}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(exist_ok=True)
    a086 = load_a086()
    recs = {}
    for ck, key in GRID:
        recs[name(ck, key)] = run_cell(ck, key)
        if ck == 'A':
            print(f"[anchor] A cfg0 repro dPQ {recs[name(ck, key)]['mean']['PQ'] - a086['cfg0']['mean']['PQ']:+.4f}",
                  flush=True)
    summary = {'experiment_id': 'shortneg-arm081-probe-20261006', 'rows': ROWS, 'subset': str(SUBSET),
               'checkpoints': {k: str(v) for k, v in CKPT.items()}, 'negatives': NEG,
               'control_cells_from': str(A086),
               'cells': {k: {x: v for x, v in r.items() if x != 'per_clip'} for k, r in recs.items()},
               **analyse(recs, a086)}
    tmp = SUMMARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2) + '\n')
    tmp.replace(SUMMARY)
    for k, g in summary['diff_in_diff_C_minus_A'].items():
        print(f'[did] {k}: dPQ {g["PQ"]["mean"]:+.3f} lvl30 {g["PQ_lvl30"]["mean"]:+.3f} '
              f'{g["PQ_lvl30"]["ci95"]} dCLAP {g["clap"]["mean"]:+.4f}', flush=True)
    print(f'wrote {SUMMARY}', flush=True)


# ── preflight / postflight ───────────────────────────────
def preflight():
    errs = []
    for rel, want in IMMUTABLE.items():
        p = Path(rel) if rel.startswith('/') else ROOT / rel
        if sha(p) != want:
            errs.append(f'{rel} changed since the contract was pinned')
    for ck, p in CKPT.items():
        if not p.exists():
            errs.append(f'missing checkpoint {ck}: {p}')
    for k in KEYS:
        if not (A086 / f'A__{k}.json').exists():
            errs.append(f'missing 086 control cell A__{k}')
    with open(SUBSET, encoding='utf-8', newline='') as f:
        n = sum(1 for _ in csv.DictReader(f, delimiter='\t'))
    if n != ROWS:
        errs.append(f'subset has {n} rows')
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUT).free < 2 * HARD_STOP_FREE:
        errs.append(f'less than {2 * HARD_STOP_FREE / 1e9:.0f} GB free at start')
    if errs:
        print('\n'.join(f'[preflight] {e}' for e in errs))
        return 1
    print('[preflight] ok')
    return 0


def validate_only():
    if not SUMMARY.exists():
        print('[validate] no summary')
        return 1
    s = json.loads(SUMMARY.read_text())
    errs = [f'{name(ck, key)} missing' for ck, key in GRID if name(ck, key) not in s['cells']]
    for k, g in s.get('gains_vs_own_cfg0', {}).items():
        if not np.isfinite(g['PQ_lvl30']['mean']):
            errs.append(f'{k}: non-finite gain')
    if len(s.get('diff_in_diff_C_minus_A', {})) != len(KEYS) - 1:
        errs.append('gain table incomplete')
    if errs:
        print('\n'.join(f'[validate] {e}' for e in errs))
        return 1
    print('[validate] ok')
    return 0


if __name__ == '__main__':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    if '--preflight' in sys.argv:
        raise SystemExit(preflight())
    if '--validate-only' in sys.argv:
        raise SystemExit(validate_only())
    run()
