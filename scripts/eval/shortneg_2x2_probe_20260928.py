#!/usr/bin/env python
"""086: why does a short "low quality" negative prompt do nothing on the nmv2pair control?

081's control (slot0clean_nmv2pair quarter) got lvl30 dPQ -0.007 from the negative
prompt "Low quality recording." at CFG 3, while the 09-03 ablation on c2p0_slot0
(multisent full 200k) got +0.971 raw PQ from "low quality, noisy" and +0.357 from
irrelevant text on a 1024-row MusicCaps subset. Two things differ at once:
checkpoint and wording. This crosses them on the same subset / noise / scorer.

  A = nmv2pair quarter s14159265 (081 control, 075/080 base)
  B = c2p0_slot0 (the 09-03 checkpoint)

  cell       A  B   negative prompt (CFG 3 unless cfg0)
  cfg0       x  x   -- (CFG 0, conditional branch only)
  none       x      stored null (no --negative_prompt)
  lqrec      x  x   "Low quality recording."            (081 wording)
  lqnoisy    x  x   "low quality, noisy"                (09-03 fidelity_short)
  lq         x  x   "low quality"                       (isolates "noisy")
  irrel      x      "a photograph of a cat, a spreadsheet, printed text"
  fid8       x      fidelity8

Every cell: MusicCaps subset1024 (seed 20260830, the 09-03 subset), MeanFlow 25,
seed 42, fp32, NoMask, --no_q; scored with eval_metrics.py (CLAP batch 1) and after
-30 LUFS matching (level_match_rescore.py). Audio is deleted once both reads exist.

Anchor to 09-03 (recorded, not blocking): B cfg0 and B lqnoisy raw PQ means must be
within 0.01 of 6.5822 / 7.5534 (09-03 used AES batch 32, which drifts ~1e-3). A miss
means the old numbers are not reproducible with today's code; the within-probe cross
stays valid because every cell here is generated and scored the same way.

Usage: python shortneg_2x2_probe_20260928.py [--preflight | --validate-only]
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
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/shortneg_2x2_probe_20260928')
CELLS = OUT / 'cells'
SUMMARY = OUT / 'summary.json'
PY = '/home/kojiek/venvs/dac/bin/python'
OLD = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/negprompt_ablation')
SUBSET = OLD / 'musiccaps_subset1024.tsv'
ROWS = 1024

CKPT = {
    'A': ROOT / 'exps/phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265_stage2_50000'
               '/phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265_stage2_50000_ema_final.pth',
    'B': ROOT / 'exps/phase8_qwen_caption10s_multisent_noq_full_stage2_200000'
               '/phase8_qwen_caption10s_multisent_noq_full_stage2_200000_ema_final.pth',
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
GRID = [('B', 'cfg0'), ('B', 'lqnoisy'),            # 09-03 anchor first
        ('A', 'cfg0'), ('A', 'lqrec'), ('A', 'lqnoisy'), ('A', 'lq'),
        ('B', 'lqrec'), ('B', 'lq'),
        ('A', 'none'), ('A', 'irrel'), ('A', 'fid8')]
ANCHOR = {('B', 'cfg0'): ('c2p0_slot0__cfg0__none', 6.58222681004554),
          ('B', 'lqnoisy'): ('c2p0_slot0__cfg3.0__fidelity_short', 7.553362264763564)}
ANCHOR_TOL = 0.01
METRICS = ('PQ', 'CE', 'CU', 'PC', 'clap')
BOOT_N, BOOT_SEED = 10000, 20260928
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
    return f'{ck}__{key}'


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


def analyse(recs):
    rng = np.random.default_rng(BOOT_SEED)
    ids = sorted(recs[name('A', 'cfg0')]['per_clip'])
    pc = {k: r['per_clip'] for k, r in recs.items()}

    def gain(ck, key, m):
        return [pc[name(ck, key)][i][m] - pc[name(ck, 'cfg0')][i][m] for i in ids]

    gains = {}
    for ck, key in GRID:
        if key == 'cfg0':
            continue
        gains[name(ck, key)] = {m: boot(gain(ck, key, m), rng)
                                for m in ('PQ', 'PQ_lvl30', 'clap', 'CE', 'PC')}
    inter = {}
    for key in ('lqnoisy', 'lq'):
        for m in ('PQ', 'PQ_lvl30'):
            # wording effect on B minus wording effect on A (vs lqrec), all vs own cfg0
            x = [(pc[name('B', key)][i][m] - pc[name('B', 'lqrec')][i][m])
                 - (pc[name('A', key)][i][m] - pc[name('A', 'lqrec')][i][m]) for i in ids]
            inter[f'{key}_vs_lqrec__B_minus_A__{m}'] = boot(x, rng)
    for key in ('lqrec', 'lqnoisy', 'lq'):
        for m in ('PQ', 'PQ_lvl30'):
            inter[f'{key}__gain_B_minus_A__{m}'] = boot(
                [a - b for a, b in zip(gain('B', key, m), gain('A', key, m))], rng)
    anchor = {}
    for (ck, key), (old_label, old_pq) in ANCHOR.items():
        new = recs[name(ck, key)]['mean']['PQ']
        old = json.loads((OLD / f'{old_label}.json').read_text())['per_clip']
        r = float(np.corrcoef([old[i]['PQ'] for i in ids], [pc[name(ck, key)][i]['PQ'] for i in ids])[0, 1])
        anchor[name(ck, key)] = {'old_label': old_label, 'old_PQ': old_pq, 'new_PQ': new,
                                 'diff': new - old_pq, 'per_clip_r': r,
                                 'pass': abs(new - old_pq) <= ANCHOR_TOL}
    return {'gains_vs_own_cfg0': gains, 'contrasts': inter, 'anchor_0903': anchor}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(exist_ok=True)
    recs = {}
    for ck, key in GRID:
        recs[name(ck, key)] = run_cell(ck, key)
        if (ck, key) == ('B', 'lqnoisy'):
            print(f'[anchor] {analyse_anchor_only(recs)}', flush=True)
    summary = {'experiment_id': 'shortneg-2x2-probe-20260928', 'rows': ROWS, 'subset': str(SUBSET),
               'checkpoints': {k: str(v) for k, v in CKPT.items()}, 'negatives': NEG,
               'cells': {k: {x: v for x, v in r.items() if x != 'per_clip'} for k, r in recs.items()},
               **analyse(recs)}
    tmp = SUMMARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2) + '\n')
    tmp.replace(SUMMARY)
    for k, g in summary['gains_vs_own_cfg0'].items():
        print(f'[gain] {k}: dPQ {g["PQ"]["mean"]:+.3f} lvl30 {g["PQ_lvl30"]["mean"]:+.3f} '
              f'{g["PQ_lvl30"]["ci95"]} dCLAP {g["clap"]["mean"]:+.4f}', flush=True)
    print(f'wrote {SUMMARY}', flush=True)


def analyse_anchor_only(recs):
    out = {}
    for (ck, key), (_, old_pq) in ANCHOR.items():
        if name(ck, key) in recs:
            out[name(ck, key)] = round(recs[name(ck, key)]['mean']['PQ'] - old_pq, 4)
    return out


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
    for old_label, _ in ANCHOR.values():
        if not (OLD / f'{old_label}.json').exists():
            errs.append(f'missing 09-03 anchor {old_label}')
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
    if len(s.get('gains_vs_own_cfg0', {})) != len(GRID) - 2:
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
