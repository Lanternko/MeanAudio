#!/usr/bin/env python
"""107: irrelevant-text CFG3 negative on the nmv2pair control, 3 seeds, full MusicCaps.

084 Stage C split the NegMF N100 gain into a polarity part (~62%) and a part that the
reversed text ("high quality recording, clean, ...") also gets (~38%). On the same
control checkpoints the reversed text as an inference-time negative gives G_rev +0.26
lvl30 PQ (3 seeds). 086 measured irrelevant text ("a photograph of a cat, ...") at
+0.52 on s14159265, but only on subset1024 and one seed. If arbitrary non-null text
gets as much as the reversed text, the 38% is a "any non-null text" layer, not a
fidelity-vocabulary layer. This fills the missing cell: irrel CFG3 on all 3 control
seeds, full 5521, same protocol, and reads it against the existing neg / revneg /
lqneg cells of the same checkpoints.

  per seed: mc_mf25_negvariant_eval.sh <ctrl> --no_q --neg IRREL --tag irrel
            -> FAD (2048, musiccaps_reference) -> -30 LUFS rescore -> drop audio
  then:     G_k = cell_k - cfg0 per clip for k in neg, revneg, lqneg, irrel;
            primary D = G_rev - G_irr on PQ lvl30, pooled over seeds (clip bootstrap).

Usage: python irrelneg_nmv2pair_probe_20261007.py [--preflight | --validate-only]
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
EV = Path('/home/kojiek/eval_output_nvme')
LVL_STOCK = EV / 'd2_075_lvl30'
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/irrelneg_nmv2pair_probe_20261007')
SUMMARY = OUT / 'summary.json'
PY = '/home/kojiek/venvs/dac/bin/python'
MC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
FAD_REF = '/mnt/HDD/kojiek/musiccaps_reference'
ROWS = 5521
SEEDS = (14159265, 16180339, 27182818)
CTRL = 'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s{}'
IRREL = 'a photograph of a cat, a spreadsheet, printed text'
TAG = 'irrel'
EXISTING = {'neg': 'cfg3_neg', 'revneg': 'cfg3_revneg', 'lqneg': 'cfg3_lqneg'}
NEGTEXT = {
    'neg': 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi',
    'revneg': 'high quality recording, clean, professional, pristine, hi-fi',
    'lqneg': 'Low quality recording.',
    'irrel': IRREL,
}
METRICS = ('PQ', 'CE', 'CU', 'PC', 'clap')
BOOT_N, BOOT_SEED = 10000, 20261007
MARGIN = 0.10
HARD_STOP_FREE = 10_000_000_000
IMMUTABLE = {
    'eval.py': 'ba66c66b2ca3b7db0a698338932f6ee474208c1302c4592724f1955f3ccb2339',
    'meanaudio/model/networks.py': '5970fd615640c3d5a2b38aa025c3f3f26dee3412f5b8de810414c16d732cbe69',
    'scripts/eval/eval_metrics.py': '47406ee5bf30c837733a00be306e813d8c27301a2a8ea30de9f8f28b1dfec67d',
    'scripts/eval/level_match_rescore.py': '133ac816e3de21e732effd5a8f75a067029868aa589fa3f4d0c64ede4ca88714',
    'scripts/eval/mc_mf25_negvariant_eval.sh': '04d6b684c5e60f331356be0b5055f977525dcdfe7fcab00bdfd478eee8becc63',
    str(MC_TSV): 'de567b13c39b6e7f7b3666f257817322ea119bcdece82fb5e8700b4a7470e51f',
}


def ctrl(seed):
    return CTRL.format(seed)


def ema(seed):
    e = f'{ctrl(seed)}_stage2_50000'
    return ROOT / 'exps' / e / f'{e}_ema_final.pth'


def label(seed, cell):
    return f'{ctrl(seed)}_mc_mf25_{cell}'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with open(path, encoding='utf-8', newline='') as f:
        return {r['id']: r for r in csv.DictReader(f, delimiter='\t')}


def raw_path(lab):
    return EV / lab / lab / 'per_clip.tsv'


def lvl_path(lab):
    for root in (EV, LVL_STOCK):
        p = root / f'{lab}_lvl30' / f'{lab}_lvl30' / 'per_clip.tsv'
        if p.exists():
            return p
    return None


def fad_value(lab):
    p = EV / f'{lab}_fad' / f'{lab}_fad' / 'metrics.json'
    return json.loads(p.read_text())['metrics']['fad'] if p.exists() else None


# ── generation ───────────────────────────────────────────
def run_seed(seed):
    lab = label(seed, f'cfg3_{TAG}')
    d = EV / lab
    if lvl_path(lab) and raw_path(lab).exists() and fad_value(lab) is not None:
        print(f'[skip] {lab} complete', flush=True)
        return
    free = shutil.disk_usage(EV).free
    if free < HARD_STOP_FREE:
        raise SystemExit(f'[FAIL] disk hard stop: {free / 1e9:.1f} GB free')
    # The wrapper skips on an existing report and regenerates a report-less dir whole.
    subprocess.run(['bash', str(ROOT / 'scripts/eval/mc_mf25_negvariant_eval.sh'), ctrl(seed), str(ema(seed)),
                    '--no_q', '--neg', IRREL, '--tag', TAG], cwd=ROOT, check=True)
    if not (d / f'{lab}_REPORT.json').exists():
        raise SystemExit(f'[FAIL] {lab}: no report')
    if fad_value(lab) is None:
        if not (d / 'audio').is_dir():
            raise SystemExit(f'[FAIL] {lab}: no audio left for FAD')
        subprocess.run([PY, str(ROOT / 'scripts/eval/eval_metrics.py'), '--gen_dir', str(d / 'audio'),
                        '--tsv', str(MC_TSV), '--exp_name', f'{lab}_fad', '--out_dir', str(EV / f'{lab}_fad'),
                        '--skip_clap', '--skip_aes', '--skip_level', '--fad', '--ref_dir', FAD_REF,
                        '--fad_num_samples', '2048'], cwd=ROOT, check=True)
    if lvl_path(lab) is None:
        if not (d / 'audio').is_dir():
            raise SystemExit(f'[FAIL] {lab}: no audio left for the lvl30 rescore')
        subprocess.run([PY, str(ROOT / 'scripts/eval/level_match_rescore.py'), '--cell_dir', str(d),
                        '--tsv', str(MC_TSV)], cwd=ROOT, check=True)
    if lvl_path(lab) is None or fad_value(lab) is None:
        raise SystemExit(f'[FAIL] {lab}: lvl30 or FAD missing after scoring')
    shutil.rmtree(d / 'audio', ignore_errors=True)
    shutil.rmtree(EV / f'{lab}_lvl30' / 'audio', ignore_errors=True)


# ── analysis ─────────────────────────────────────────────
def num(x):
    return float(x) if x not in ('', None) else float('nan')


def boot(x, rng):
    x = np.asarray(x, dtype=np.float64)
    b = rng.choice(x, (BOOT_N, len(x))).mean(1)
    return {'mean': float(x.mean()), 'ci95': [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def load(seed, cell):
    lab = label(seed, cell)
    raw, lv = read(raw_path(lab)), read(lvl_path(lab))
    return {i: {m: num(raw[i][m]) for m in METRICS} | {'PQ_lvl30': num(lv[i]['PQ']),
                                                         'silent': int(raw[i]['silent']),
                                                         'lufs': num(raw[i]['lufs']),
                                                         'crest': num(raw[i]['crest'])}
            for i in raw}


def analyse():
    rng = np.random.default_rng(BOOT_SEED)
    keys = ['neg', 'revneg', 'lqneg', TAG]
    cell = {k: EXISTING.get(k, f'cfg3_{TAG}') for k in keys}
    data = {s: {'cfg0': load(s, 'cfg0')} | {k: load(s, cell[k]) for k in keys} for s in SEEDS}
    ids = sorted(set.intersection(*(set(data[s][c]) for s in SEEDS for c in data[s])))
    if len(ids) != ROWS:
        raise SystemExit(f'[FAIL] only {len(ids)} common clips')
    ms = ('PQ_lvl30', 'PQ', 'CE', 'CU', 'PC', 'clap')

    def g(s, k, m, i):
        return data[s][k][i][m] - data[s]['cfg0'][i][m]

    cells = {f's{s}__{k}': {'label': label(s, cell.get(k, k)), 'negative': NEGTEXT.get(k),
                            'mean': {m: float(np.nanmean([data[s][k][i][m] for i in ids])) for m in ms},
                            'silent_n': int(sum(data[s][k][i]['silent'] for i in ids)),
                            'lufs': float(np.nanmean([data[s][k][i]['lufs'] for i in ids])),
                            'crest': float(np.nanmean([data[s][k][i]['crest'] for i in ids])),
                            'fad': fad_value(label(s, cell.get(k, k)))}
             for s in SEEDS for k in ['cfg0'] + keys}
    gains = {k: {m: {'per_seed': {str(s): float(np.nanmean([g(s, k, m, i) for i in ids])) for s in SEEDS},
                     'pooled': boot([np.mean([g(s, k, m, i) for s in SEEDS]) for i in ids], rng)}
                 for m in ms} for k in keys}

    def contrast(a, b, m, excl_silent=False):
        per_seed, pooled = {}, []
        for s in SEEDS:
            keep = [i for i in ids if not (excl_silent and (data[s][a][i]['silent'] or data[s][b][i]['silent']))]
            per_seed[str(s)] = float(np.mean([data[s][a][i][m] - data[s][b][i][m] for i in keep]))
        for i in ids:
            v = [data[s][a][i][m] - data[s][b][i][m] for s in SEEDS
                 if not (excl_silent and (data[s][a][i]['silent'] or data[s][b][i]['silent']))]
            if v:
                pooled.append(np.mean(v))
        return {'per_seed': per_seed, 'pooled': boot(pooled, rng)}

    contrasts = {f'{a}_minus_{TAG}': {m: contrast(a, TAG, m) for m in ms} for a in ('revneg', 'lqneg', 'neg')}
    contrasts_nosilent = {f'{a}_minus_{TAG}': contrast(a, TAG, 'PQ_lvl30', True) for a in ('revneg', 'lqneg', 'neg')}
    ratios = {k: {str(s): gains[k]['PQ_lvl30']['per_seed'][str(s)] / gains['neg']['PQ_lvl30']['per_seed'][str(s)]
                  for s in SEEDS} | {'pooled': gains[k]['PQ_lvl30']['pooled']['mean'] / gains['neg']['PQ_lvl30']['pooled']['mean']}
              for k in keys}
    d = contrasts[f'revneg_minus_{TAG}']['PQ_lvl30']['pooled']
    if d['mean'] >= MARGIN and d['ci95'][0] > 0:
        verdict = 'reversed_above_irrelevant'
    elif d['mean'] <= -MARGIN and d['ci95'][1] < 0:
        verdict = 'reversed_below_irrelevant'
    else:
        verdict = 'reversed_equivalent_to_irrelevant'
    return {'clips': len(ids), 'cells': cells, 'gains_vs_cfg0': gains, 'contrasts': contrasts,
            'contrasts_PQ_lvl30_excluding_silent': contrasts_nosilent, 'ratio_to_G_neg_PQ_lvl30': ratios,
            'primary': {'contrast': 'G_rev - G_irr, PQ lvl30, pooled 3 seeds', 'margin': MARGIN, **d,
                        'verdict': verdict}}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        run_seed(s)
    summary = {'experiment_id': 'irrelneg-nmv2pair-probe-20261007', 'seeds': SEEDS, 'tsv': str(MC_TSV),
               'negatives': NEGTEXT, **analyse()}
    tmp = SUMMARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2) + '\n')
    tmp.replace(SUMMARY)
    for k, g in summary['gains_vs_cfg0'].items():
        print(f'[G] {k}: lvl30 {g["PQ_lvl30"]["pooled"]["mean"]:+.3f} {g["PQ_lvl30"]["pooled"]["ci95"]} '
              f'per seed {g["PQ_lvl30"]["per_seed"]}', flush=True)
    p = summary['primary']
    print(f'[primary] G_rev - G_irr {p["mean"]:+.3f} {p["ci95"]} -> {p["verdict"]}', flush=True)
    print(f'wrote {SUMMARY}', flush=True)


# ── preflight / postflight ───────────────────────────────
def preflight():
    errs = []
    for rel, want in IMMUTABLE.items():
        p = Path(rel) if rel.startswith('/') else ROOT / rel
        if sha(p) != want:
            errs.append(f'{rel} changed since the contract was pinned')
    for s in SEEDS:
        if not ema(s).exists():
            errs.append(f'missing checkpoint {ema(s)}')
        for cell in ['cfg0'] + list(EXISTING.values()):
            lab = label(s, cell)
            if not raw_path(lab).exists() or lvl_path(lab) is None:
                errs.append(f'missing existing cell {lab} (raw or lvl30)')
    if not Path(FAD_REF).is_dir():
        errs.append(f'missing FAD reference {FAD_REF}')
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(EV).free < 2 * HARD_STOP_FREE:
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
    errs = []
    if s.get('clips') != ROWS:
        errs.append(f'clips {s.get("clips")} != {ROWS}')
    for seed in SEEDS:
        c = s['cells'].get(f's{seed}__{TAG}')
        if not c or c.get('fad') is None:
            errs.append(f's{seed} irrel cell or FAD missing')
    for k, g in s.get('gains_vs_cfg0', {}).items():
        if not np.isfinite(g['PQ_lvl30']['pooled']['mean']):
            errs.append(f'{k}: non-finite gain')
    if 'verdict' not in s.get('primary', {}):
        errs.append('no primary verdict')
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
