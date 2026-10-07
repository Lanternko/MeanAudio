#!/usr/bin/env python
"""107: 106's short-negative wording set on the other two 081 training seeds.

106 ran the 086 wording set on the 081 arm (C) for training seed s14159265 only and
read A (control) from 086. Both pre-registered readings held (label-specific +0.44,
whole-slot amplification +0.54/+0.44) and the clean E1 was +0.43 lvl30, but the CIs
there only cover clip sampling; 081's seed-to-seed spread is far wider. This runs the
same 7 cells on A and C for s16180339 and s27182818 (28 cells) and pools the three
seeds (s14159265 from 086 A cells + 106 C cells).

Every cell: MusicCaps subset1024, MeanFlow 25, seed 42, fp32, NoMask, --no_q; scored
with eval_metrics.py (CLAP batch 1) and after -30 LUFS matching. Audio deleted after.

Usage: python shortneg_arm081_seeds_20261007.py [--preflight | --validate-only]
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
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/shortneg_arm081_seeds_20261007')
A086 = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/shortneg_2x2_probe_20260928/cells')
C106 = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/shortneg_arm081_probe_20261006/cells')
CELLS = OUT / 'cells'
SUMMARY = OUT / 'summary.json'
PY = '/home/kojiek/venvs/dac/bin/python'
OLD = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation')
SUBSET = OLD / 'musiccaps_subset1024.tsv'
ROWS = 1024

SEED0 = 's14159265'                      # read from 086 (A) and 106 (C)
NEW_SEEDS = ('s16180339', 's27182818')   # generated here
SEEDS = (SEED0,) + NEW_SEEDS
ARM = {'A': 'nmv2pair', 'C': 'qlabel'}


def ckpt(ck, seed):
    exp = f'phase8_qwen_caption2p0_slot0clean_{ARM[ck]}_noq_quarter_{seed}_stage2_50000'
    return ROOT / 'exps' / exp / f'{exp}_ema_final.pth'


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
GRID = [(s, ck, k) for s in NEW_SEEDS for ck in ('A', 'C') for k in KEYS]
METRICS = ('PQ', 'CE', 'CU', 'PC', 'clap')
THRESH = 0.19
BOOT_N, BOOT_SEED = 10000, 20261007
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


def name(seed, ck, key):
    return f'{seed}__{ck}__{key}'


def cell_dir(seed, ck, key):
    return CELLS / name(seed, ck, key)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ── generation / scoring ─────────────────────────────────
def generate(seed, ck, key):
    d = cell_dir(seed, ck, key)
    shutil.rmtree(d, ignore_errors=True)          # never top up: eval.py seeds one RNG per run
    shutil.rmtree(d.parent / f'{d.name}_lvl30', ignore_errors=True)
    (d / 'audio').mkdir(parents=True)
    cmd = [PY, 'eval.py', '--variant', 'meanaudio_s', '--model_path', str(ckpt(ck, seed)),
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
        raise SystemExit(f'[FAIL] {name(seed, ck, key)}: {n} clips, expected {ROWS}')
    return cmd


def score(seed, ck, key):
    d = cell_dir(seed, ck, key)
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


def run_cell(seed, ck, key):
    path = CELLS / f'{name(seed, ck, key)}.json'
    if path.exists():
        return json.loads(path.read_text())
    free = shutil.disk_usage(OUT).free
    if free < HARD_STOP_FREE:
        raise SystemExit(f'[FAIL] disk hard stop: {free / 1e9:.1f} GB free')
    cmd = generate(seed, ck, key)
    un_p, lv_p = score(seed, ck, key)
    un, lv = per_clip(un_p), per_clip(lv_p)
    if len(un) != ROWS or set(un) != set(lv):
        raise SystemExit(f'[FAIL] {name(seed, ck, key)}: per_clip rows {len(un)} / {len(lv)}')
    rec = {'name': name(seed, ck, key), 'training_seed': seed, 'checkpoint': ck,
           'checkpoint_path': str(ckpt(ck, seed)),
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
    d = cell_dir(seed, ck, key)
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


def load_seed0():
    out = {}
    for k in KEYS:
        out[('A', k)] = json.loads((A086 / f'A__{k}.json').read_text())['per_clip']
        out[('C', k)] = json.loads((C106 / f'C__{k}.json').read_text())['per_clip']
    return out


MS = ('PQ', 'PQ_lvl30', 'clap', 'CE', 'PC')


def contrasts(pc, ids):
    """Per-clip contrast vectors for one training seed; pc[(ck, key)] -> per_clip."""
    def gain(ck, k, m):
        return np.array([pc[(ck, k)][i][m] - pc[(ck, 'cfg0')][i][m] for i in ids])

    v = {}
    for k in KEYS[1:]:
        for m in MS:
            v[f'gain__A__{k}__{m}'] = gain('A', k, m)
            v[f'gain__C__{k}__{m}'] = gain('C', k, m)
            v[f'did__{k}__{m}'] = gain('C', k, m) - gain('A', k, m)
    for m in MS:
        v[f'cfg0_C_minus_A__{m}'] = np.array([pc[('C', 'cfg0')][i][m] - pc[('A', 'cfg0')][i][m] for i in ids])
        # 106's clean E1: arm label-string gain minus control's best short wording gain
        v[f'clean_e1__{m}'] = gain('C', 'lqrec', m) - gain('A', 'lqnoisy', m)
        for k in ('lqnoisy', 'lq', 'irrel', 'fid8'):
            v[f'spec__lqrec_minus_{k}__{m}'] = v[f'did__lqrec__{m}'] - v[f'did__{k}__{m}']
        v[f'C_lqnoisy_minus_lqrec__{m}'] = (pc_get(pc, 'C', 'lqnoisy', ids, m)
                                            - pc_get(pc, 'C', 'lqrec', ids, m))
    return v


def pc_get(pc, ck, k, ids, m):
    return np.array([pc[(ck, k)][i][m] for i in ids])


def analyse(recs):
    rng = np.random.default_rng(BOOT_SEED)
    seed0 = load_seed0()
    ids = sorted(seed0[('A', 'cfg0')])
    per_seed_pc = {SEED0: seed0}
    for s in NEW_SEEDS:
        per_seed_pc[s] = {(ck, k): recs[name(s, ck, k)]['per_clip'] for ck in ('A', 'C') for k in KEYS}
        if sorted(per_seed_pc[s][('A', 'cfg0')]) != ids:
            raise SystemExit(f'[FAIL] {s}: clip ids differ from seed0')
    vec = {s: contrasts(per_seed_pc[s], ids) for s in SEEDS}
    per_seed = {s: {k: boot(v, rng) for k, v in vec[s].items()} for s in SEEDS}
    pooled = {}
    for k in vec[SEED0]:
        means = [float(vec[s][k].mean()) for s in SEEDS]
        # clip bootstrap of the 3-seed mean (clips paired across seeds; seed noise shown by spread)
        pooled[k] = boot(np.mean([vec[s][k] for s in SEEDS], axis=0), rng) | {
            'per_seed': dict(zip(SEEDS, means)),
            'seed_sd': float(np.std(means, ddof=1)),
            'seed_min': min(means), 'seed_max': max(means),
            'n_pos': int(sum(x > 0 for x in means))}

    def all_same_sign_pos(k):
        return pooled[k]['n_pos'] == len(SEEDS)

    m = 'PQ_lvl30'
    spec_k = f'spec__lqrec_minus_lqnoisy__{m}'
    verdict = {
        'label_specific': {
            'rule': f'3-seed mean lqrec-lqnoisy DiD >= {THRESH} and > 0 in all 3 seeds',
            'mean': pooled[spec_k]['mean'], 'n_pos': pooled[spec_k]['n_pos'],
            'pass': pooled[spec_k]['mean'] >= THRESH and all_same_sign_pos(spec_k)},
        'whole_slot_amplification': {
            'rule': f'3-seed mean DiD(lqnoisy) and DiD(fid8) >= {THRESH}, each > 0 in all 3 seeds',
            'lqnoisy': pooled[f'did__lqnoisy__{m}']['mean'], 'fid8': pooled[f'did__fid8__{m}']['mean'],
            'pass': all(pooled[f'did__{k}__{m}']['mean'] >= THRESH and all_same_sign_pos(f'did__{k}__{m}')
                        for k in ('lqnoisy', 'fid8'))},
        'clean_e1': {
            'mean': pooled[f'clean_e1__{m}']['mean'], 'per_seed': pooled[f'clean_e1__{m}']['per_seed'],
            'raw_did_lqrec': pooled[f'did__lqrec__{m}']['mean'],
            'fraction_of_raw': pooled[f'clean_e1__{m}']['mean'] / pooled[f'did__lqrec__{m}']['mean'],
            'clap': pooled['clean_e1__clap']['mean'], 'clap_per_seed': pooled['clean_e1__clap']['per_seed']},
        'exploratory_lq_failure': {
            'rule': 'DiD(lq) < 0 in all 3 seeds (106 single seed: -0.21)',
            'per_seed': pooled[f'did__lq__{m}']['per_seed'],
            'holds': pooled[f'did__lq__{m}']['n_pos'] == 0},
    }
    return {'per_seed': per_seed, 'pooled_3seed': pooled, 'verdict': verdict}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(exist_ok=True)
    recs = {name(s, ck, k): run_cell(s, ck, k) for s, ck, k in GRID}
    summary = {'experiment_id': 'shortneg-arm081-seeds-20261007', 'rows': ROWS, 'subset': str(SUBSET),
               'seeds': SEEDS, 'seed0_cells_from': {'A': str(A086), 'C': str(C106)},
               'checkpoints': {f'{s}__{ck}': str(ckpt(ck, s)) for s in SEEDS for ck in ('A', 'C')},
               'negatives': NEG,
               'cells': {k: {x: v for x, v in r.items() if x != 'per_clip'} for k, r in recs.items()},
               **analyse(recs)}
    tmp = SUMMARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2) + '\n')
    tmp.replace(SUMMARY)
    for k, v in summary['verdict'].items():
        print(f'[verdict] {k}: {json.dumps(v)}', flush=True)
    print(f'wrote {SUMMARY}', flush=True)


# ── preflight / postflight ───────────────────────────────
def preflight():
    errs = []
    for rel, want in IMMUTABLE.items():
        p = Path(rel) if rel.startswith('/') else ROOT / rel
        if sha(p) != want:
            errs.append(f'{rel} changed since the contract was pinned')
    for s in SEEDS:
        for ck in ('A', 'C'):
            if not ckpt(ck, s).exists():
                errs.append(f'missing checkpoint {s} {ck}: {ckpt(ck, s)}')
    for k in KEYS:
        if not (A086 / f'A__{k}.json').exists():
            errs.append(f'missing 086 control cell A__{k}')
        if not (C106 / f'C__{k}.json').exists():
            errs.append(f'missing 106 arm cell C__{k}')
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
    errs = [f'{name(*g)} missing' for g in GRID if name(*g) not in s['cells']]
    for k, g in s.get('pooled_3seed', {}).items():
        if not np.isfinite(g['mean']):
            errs.append(f'{k}: non-finite pooled mean')
    if set(s.get('per_seed', {})) != set(SEEDS):
        errs.append('per-seed table incomplete')
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
