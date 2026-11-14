#!/usr/bin/env python3
"""182: how high can MuQ-Eval A1 go with existing PQ-oriented levers + best-of-N?

Design (frozen before generation): docs/experiments/muq_stack_bestofn_182_20261108.md.

Prompts: the 1,000 Part B common prompts (muq_attack_fix_20261103/part_b_selection.json).
Part A: 5 cells x 3 training seeds (081 arm quarter with HQ prefix + LQ / fid8 / LQ+fid8 at
cfg 3 / 4.5; N100 full recipe cfg3+fid8), inference seed 42. Part B: the A cell with the
highest 3-seed mean A1_MI_lvl30 (no lowpass), training seed 14159265, inference seeds
42 (reused from A) + 7 more; per-prompt best-of-k by A1_MI_lvl30, k in 1/2/4/8.

Per cell: generate (eval.py, subset TSV) -> level + lvl30 copies (CPU pool, main process)
-> subprocess --score-cell: MuQ A1/A3a x MI/TA x raw/lvl30/lp6k(lvl30), AES PQ/CE lvl30,
CLAP raw vs ORIGINAL caption (batch 1), FAD raw vs musiccaps_reference -> per_cell tsv/json.
Audio kept for training seed 14159265 (Part A candidates for B) and all B cells until the
summary is written; everything else deleted after scoring.

Modes: (default) run all, then summary; --preflight; --validate-only; --summary-only;
--score-cell LABEL (internal); --only LABEL (smoke); --limit N (smoke: first N prompts,
separate output root).
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
from meva_081_084_backfill import PY, REPO, EVAL_ROOT, MC_TSV, eval_args, eval_cmd, read_clips  # noqa: E402

OUT = EVAL_ROOT / 'muq_stack_bestofn_182'
MIN_FREE = 20 * 1024**3
TARGET, PEAK_CAP, LP_FC = -30.0, 0.999, 6000.0
SEEDS = ['14159265', '16180339', '27182818']
B_TRAIN_SEED = '14159265'
INF_SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
KS = [1, 2, 4, 8]
HQ_TSV = Path('/home/kojiek/exps_nvme/quality_label_081/musiccaps_test_hqprefix.tsv')
FAD_REF = '/mnt/HDD/kojiek/musiccaps_reference'
PARTB = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/muq_attack_fix_20261103')
TTM115 = EVAL_ROOT / 'ttm_metrics_115_20261009/summary.json'
R181 = EVAL_ROOT / 'muq_081_084_rescore_181/per_cell'
FID8 = 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
LQ = 'Low quality recording.'
ARM = 'phase8_qwen_caption2p0_slot0clean_qlabel_noq_quarter_s{seed}'
N100F = 'phase8_qwen_caption2p0_slot0clean_negmfn100_noq_full_s{seed}'
SRC_HQ = ARM + '_hqpos_mc_mf25_cfg3_lqneg'          # eval-arg source (gen.log) for the 081 arm cells
SRC_N100 = N100F + '_mc_mf25_cfg3_neg'
# design §1 table: name -> (source label, tsv kind, negative prompt, cfg)
A_CELLS = {
    'A_ref': (SRC_HQ, 'hq', LQ, 3.0),
    'A_fid': (SRC_HQ, 'hq', FID8, 3.0),
    'A_comb': (SRC_HQ, 'hq', LQ + ' ' + FID8, 3.0),
    'A_comb45': (SRC_HQ, 'hq', LQ + ' ' + FID8, 4.5),
    'A_n100': (SRC_N100, 'mc', FID8, 3.0),
}
VARIANTS, HEADS = ('A1', 'A3a'), ('MI', 'TA')
CONDS = ('raw', 'lvl30', 'lp6k')
MUQ_COLS = [f'{v}_{h}_{c}' for c in CONDS for v in VARIANTS for h in HEADS]
COLS = MUQ_COLS + ['PQ_lvl30', 'CE_lvl30', 'CLAP', 'lufs', 'silent']
PRIMARY = 'A1_MI_lvl30'
# pre-registered (design §3)
S = {'A1_MI_lvl30': 3.541 - 2.990, 'A3a_MI_lvl30': 3.716 - 3.093, 'PQ_lvl30': 0.914}
M = {'A1': 0.055, 'A3a': 0.062}
MUSICGEN_A1, REAL_A1 = 3.541, 3.832
HARNESS_MEAN, HARNESS_SEED = 0.05, 0.10
TTM5 = ('musicgen_medium', 'meanaudio_cfg3_neg', 'musicldm_fid8', 'musicldm', 'meanaudio_cfg0')
NBOOT, BOOT_SEED = 2000, 182


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


class Paths:
    def __init__(self, root):
        self.root = root
        self.cells = root / 'cells'
        self.per_cell = root / 'per_cell'
        self.tsv = root / 'tsv'
        self.summary = root / 'summary.json'
        self.progress = root / 'progress.json'
        self.bsel = root / 'b_selection.json'


P = Paths(OUT)


# ── prompts ──────────────────────────────────────────────
def part_b_ids(limit=None):
    sel = json.loads((PARTB / 'part_b_selection.json').read_text())
    ids = {Path(f).stem for f in sel['files']}
    with open(MC_TSV, newline='', encoding='utf-8') as f:
        order = [r['id'] for r in csv.DictReader(f, delimiter='\t') if r['id'] in ids]
    if len(order) != len(ids):
        raise SystemExit(f'Part B ids not all in MusicCaps TSV: {len(order)}/{len(ids)}')
    return order[:limit] if limit else order


def write_subsets(limit=None):
    ids = part_b_ids(limit)
    keep = set(ids)
    P.tsv.mkdir(parents=True, exist_ok=True)
    for kind, src in (('mc', MC_TSV), ('hq', HQ_TSV)):
        with open(src, newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f, delimiter='\t')
            fields, rows = rd.fieldnames, [r for r in rd if r['id'] in keep]
        if [r['id'] for r in rows] != ids:
            raise SystemExit(f'{src}: subset order/ids mismatch')
        with open(P.tsv / f'{kind}.tsv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    return ids


def ids_now():
    with open(P.tsv / 'mc.tsv', newline='', encoding='utf-8') as f:
        return [r['id'] for r in csv.DictReader(f, delimiter='\t')]


# ── cells ────────────────────────────────────────────────
def a_label(name, seed):
    return f'{name}_s{seed}'


def b_label(inf_seed):
    return f'B_i{inf_seed}'


def a_labels():
    return [a_label(n, s) for n in A_CELLS for s in SEEDS]


def cell_spec(label):
    """(eval args, tsv kind) for a cell label; B cells need b_selection.json."""
    if label.startswith('B_i'):
        sel = json.loads(P.bsel.read_text())['cell']
        a, kind = cell_spec(a_label(sel, B_TRAIN_SEED))
        a = dict(a, seed=int(label[3:]))
        return a, kind
    name, seed = label.rsplit('_s', 1)
    src, kind, neg, cfg = A_CELLS[name]
    a = dict(eval_args(src.format(seed=seed)))
    if not a['no_q'] or a['seed'] != 42 or a['num_steps'] != 25 or a['duration'] != 9.975:
        raise RuntimeError(f'{label}: unexpected source args {a}')
    a.update(negative_prompt=neg, cfg_strength=cfg, tsv=str(P.tsv / f'{kind}.tsv'), seed=42)
    return a, kind


def cdir(label):
    return P.cells / label


def done(label):
    j = P.per_cell / f'{label}.json'
    t = P.per_cell / f'{label}.tsv'
    return j.is_file() and t.is_file() and json.loads(j.read_text()).get('passed') is True


def keep_audio(label):
    return label.startswith('B_i') or label.endswith(f'_s{B_TRAIN_SEED}')


def run_cell(label):
    d = cdir(label)
    if d.exists():
        shutil.rmtree(d)
    for suf in ('.tsv', '.json'):
        (P.per_cell / f'{label}{suf}').unlink(missing_ok=True)
    ids = ids_now()
    if label == b_label(42):
        # inference seed 42 is the A cell itself: reuse its audio and scores (design §1)
        sel = json.loads(P.bsel.read_text())['cell']
        src = a_label(sel, B_TRAIN_SEED)
        d.mkdir(parents=True)
        (d / 'audio').symlink_to(cdir(src) / 'audio')
        shutil.copy(P.per_cell / f'{src}.tsv', P.per_cell / f'{label}.tsv')
        j = json.loads((P.per_cell / f'{src}.json').read_text())
        j['reused_from'] = src
        (P.per_cell / f'{label}.json').write_text(json.dumps(j, indent=1))
        log(f'DONE {label} (reused {src})')
        return
    audio = d / 'audio'
    audio.mkdir(parents=True)
    a, kind = cell_spec(label)
    cmd = eval_cmd(a, audio)
    (d / 'eval_cmd.json').write_text(json.dumps({'args': a, 'tsv_kind': kind, 'cmd': cmd}, indent=1, default=str))
    log(f'GEN {label}  cfg={a["cfg_strength"]} seed={a["seed"]} neg={a["negative_prompt"]!r} tsv={kind}')
    t0 = time.time()
    with open(d / 'gen.log', 'w') as g:
        subprocess.run(cmd, cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    got = sorted(p.stem for p in audio.glob('*.flac'))
    if got != sorted(ids):
        raise RuntimeError(f'{label}: {len(got)} flac, expected {len(ids)}')
    gen_s = time.time() - t0
    log(f'LEVEL {label}')
    from eval_metrics import _level_one
    from level_match_rescore import _gain_one
    (d / 'lvl30').mkdir()
    with Pool(16) as pool:
        lv = pool.map(_level_one, [str(audio / f'{i}.flac') for i in ids], chunksize=32)
        gains = pool.map(_gain_one, [(str(audio / f'{i}.flac'), str(d / 'lvl30' / f'{i}.flac'), TARGET, PEAK_CAP)
                                     for i in ids], chunksize=32)
    with open(d / 'level.tsv', 'w') as f:
        f.write('id\tlufs\tsilent\tpeak\tgain_status\tgain_db\n')
        for i, l, g in zip(ids, lv, gains):
            f.write(f'{i}\t{l["lufs"]}\t{l["silent"]}\t{l["peak"]!r}\t{g[1]}\t{g[3]!r}\n')
    log(f'SCORE {label}')
    with open(d / 'score.log', 'w') as g:
        subprocess.run([str(PY), '-I', str(Path(__file__).resolve()), '--root', str(P.root), '--score-cell', label],
                       cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    rows = read_clips(P.per_cell / f'{label}.tsv')
    if sorted(rows) != sorted(ids):
        raise RuntimeError(f'{label}: scored {len(rows)} rows')
    shutil.rmtree(d / 'lvl30')
    if not keep_audio(label):
        shutil.rmtree(audio)
    j = json.loads((P.per_cell / f'{label}.json').read_text())
    j.update(gen_seconds=gen_s, passed=True)
    (P.per_cell / f'{label}.json').write_text(json.dumps(j, indent=1))
    log(f'DONE {label}  {PRIMARY} {statistics.fmean(float(r[PRIMARY]) for r in rows.values()):.4f}  '
        f'FAD {j["fad"]:.3f}')


def score_cell(label):
    """Subprocess entry: all readings of one cell."""
    sys.path[:0] = [str(REPO), str(REPO / 'scripts/eval')]
    import soundfile as sf
    from muq_gen_steer_179_score import muq_hf_cache, standard_hf_cache
    from muq_lowpass_probe_20261028 import lp
    from newmetrics_muqeval import score_muqeval
    from eval_metrics import load_rows, score_aes, score_clap, score_fad
    d = cdir(label)
    ids = ids_now()
    rec = {i: {} for i in ids}
    with muq_hf_cache():
        for v in VARIANTS:
            t = time.time()
            for i in ids:
                for cond, sub in (('raw', 'audio'), ('lvl30', 'lvl30')):
                    s = score_muqeval(str(d / sub / f'{i}.flac'), None, variant=v)
                    for h in HEADS:
                        rec[i][f'{v}_{h}_{cond}'] = s[h]
                x, sr = sf.read(str(d / 'lvl30' / f'{i}.flac'), dtype='float32')
                s = score_muqeval(lp(x, sr, LP_FC).astype('float32'), sr, variant=v)
                for h in HEADS:
                    rec[i][f'{v}_{h}_lp6k'] = s[h]
            log(f'{label} {v} {len(ids) * 3 / (time.time() - t):.1f} clip/s')
    with standard_hf_cache():
        paths = [str(d / 'lvl30' / f'{i}.flac') for i in ids]
        aes, failed = score_aes(paths, progress=False)
        if failed:
            raise SystemExit(f'{label}: AES failed on {len(failed)} clips')
        for i, p in zip(ids, paths):
            rec[i]['PQ_lvl30'], rec[i]['CE_lvl30'] = aes[p]['PQ'], aes[p]['CE']
        rows = load_rows(P.tsv / 'mc.tsv')          # ORIGINAL captions, also for hq cells
        clap = score_clap(rows, d / 'audio', progress=False)
        if len(clap) != len(ids):
            raise SystemExit(f'{label}: CLAP scored {len(clap)}')
        for i in ids:
            rec[i]['CLAP'] = clap[i]
        fad, pairs = score_fad(rows, d / 'audio', FAD_REF, num_samples=0)
    lv = read_clips(d / 'level.tsv')
    for i in ids:
        rec[i]['lufs'] = float(lv[i]['lufs']) if lv[i]['lufs'] not in ('None', '') else math.nan
        rec[i]['silent'] = int(lv[i]['silent'])
    P.per_cell.mkdir(parents=True, exist_ok=True)
    tmp = P.per_cell / f'{label}.tsv.tmp'
    with open(tmp, 'w') as f:
        f.write('id\t' + '\t'.join(COLS) + '\n')
        for i in ids:
            vals = [rec[i][k] for k in COLS]
            if not all(math.isfinite(x) for x in vals[:len(MUQ_COLS) + 3]):
                raise SystemExit(f'{label}: non-finite score on {i}')
            f.write(i + '\t' + '\t'.join(repr(x) for x in vals) + '\n')
    tmp.replace(P.per_cell / f'{label}.tsv')
    if pairs != len(ids):
        raise SystemExit(f'{label}: FAD pairs {pairs} != {len(ids)}')
    (P.per_cell / f'{label}.json').write_text(json.dumps({'fad': fad, 'fad_pairs': pairs, 'passed': False}, indent=1))
    return 0


# ── selection / best-of-k ───────────────────────────────
def load(label):
    return {i: {k: float(v) for k, v in r.items() if k != 'id'} for i, r in read_clips(P.per_cell / f'{label}.tsv').items()}


def cell_mean(label, key=PRIMARY):
    return statistics.fmean(r[key] for r in load(label).values())


def select_b():
    means = {n: statistics.fmean(cell_mean(a_label(n, s)) for s in SEEDS) for n in A_CELLS}
    best = max(means, key=means.get)
    P.bsel.write_text(json.dumps({'rule': 'max 3-seed mean A1_MI_lvl30 over A cells (no lowpass)',
                                  'means': means, 'cell': best}, indent=1))
    log(f'B SELECT {best}  {means}')
    for n in A_CELLS:   # Part A candidates no longer needed except the selected one
        if n != best:
            shutil.rmtree(cdir(a_label(n, B_TRAIN_SEED)) / 'audio', ignore_errors=True)
    return best


def boot_ci(vals, rng):
    import numpy as np
    v = np.asarray(vals, dtype=float)
    idx = rng.integers(0, len(v), size=(NBOOT, len(v)))
    bm = v[idx].mean(axis=1)
    return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def bestofk(ids, data, k, key):
    """per prompt: index (into INF_SEEDS[:k]) of the max `key`."""
    return {i: max(range(k), key=lambda j: data[j][i][key]) for i in ids}


def fad_of_pick(ids, pick, name):
    sys.path[:0] = [str(REPO / 'scripts/eval')]
    from eval_metrics import score_fad
    from muq_gen_steer_179_score import standard_hf_cache
    d = P.root / 'pick' / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i in ids:
        (d / f'{i}.flac').symlink_to((cdir(b_label(INF_SEEDS[pick[i]])) / 'audio' / f'{i}.flac').resolve())
    rows = [(i, '') for i in ids]
    with standard_hf_cache():
        fad, pairs = score_fad(rows, d, FAD_REF, num_samples=0)
    shutil.rmtree(d)
    return fad, pairs


def summarize():
    import numpy as np
    rng = np.random.default_rng(BOOT_SEED)
    ids = ids_now()
    s115 = json.loads(TTM115.read_text())['systems']
    S_ = dict(S)
    S_['CE_lvl30'] = (lambda v: max(v) - min(v))([s115[n]['blocks']['lvl30']['CE'] for n in TTM5])
    S_['CLAP'] = (lambda v: max(v) - min(v))([s115[n]['blocks']['raw']['clap'] for n in TTM5])
    out = {'document_kind': 'muq_stack_bestofn_182_summary', 'written_at': time.strftime('%FT%T%z'),
           'design': 'docs/experiments/muq_stack_bestofn_182_20261108.md', 'n_prompts': len(ids),
           'S': S_, 'm': M, 'refs': {'musicgen_medium_A1': MUSICGEN_A1, 'real_A1': REAL_A1}}
    # Part A cells
    cells = {}
    keys = ['A1_MI_lvl30', 'A1_MI_lp6k', 'A1_MI_raw', 'A3a_MI_lvl30', 'A3a_MI_lp6k', 'PQ_lvl30', 'CE_lvl30', 'CLAP']
    for n in A_CELLS:
        per = {s: load(a_label(n, s)) for s in SEEDS}
        c = {'seeds': {}}
        for s in SEEDS:
            c['seeds'][s] = {k: statistics.fmean(r[k] for r in per[s].values()) for k in keys}
            j = json.loads((P.per_cell / f'{a_label(n, s)}.json').read_text())
            c['seeds'][s].update(fad=j['fad'], silent_n=sum(int(r['silent']) for r in per[s].values()),
                                 lufs_mean=statistics.fmean(r['lufs'] for r in per[s].values()
                                                            if math.isfinite(r['lufs'])))
        c['mean'] = {k: statistics.fmean(c['seeds'][s][k] for s in SEEDS) for k in keys + ['fad', 'silent_n']}
        clip_mean = [statistics.fmean(per[s][i][PRIMARY] for s in SEEDS) for i in ids]
        c['A1_lvl30_prompt_boot_ci95'] = boot_ci(clip_mean, rng)
        cells[n] = c
    out['cells'] = cells
    # harness gate vs 181 on the same prompts
    hg = {}
    for s in SEEDS:
        p181 = R181 / f'{SRC_HQ.format(seed=s)}.tsv'
        old = read_clips(p181)
        m181 = statistics.fmean(float(old[i][PRIMARY]) for i in ids)
        hg[s] = {'r181': m181, 'r182': cells['A_ref']['seeds'][s][PRIMARY],
                 'd': cells['A_ref']['seeds'][s][PRIMARY] - m181}
    dm = statistics.fmean(v['d'] for v in hg.values())
    out['harness'] = {'seeds': hg, 'mean_d': dm,
                      'passed': abs(dm) <= HARNESS_MEAN and all(abs(v['d']) <= HARNESS_SEED for v in hg.values())}
    # Q1
    def q1(key):
        best = max(A_CELLS, key=lambda n: cells[n]['mean'][key])
        mean = cells[best]['mean'][key]
        seeds = [cells[best]['seeds'][s][key] for s in SEEDS]
        d = mean - MUSICGEN_A1
        if d >= M['A1'] and all(x > MUSICGEN_A1 for x in seeds):
            v = 'beats'
        elif abs(d) < M['A1']:
            v = 'ties'
        elif d <= -M['A1']:
            v = 'loses'
        else:
            v = 'undetermined'
        return {'cell': best, 'mean': mean, 'seeds': seeds, 'd_vs_musicgen': d, 'd_vs_real': mean - REAL_A1,
                'verdict': v}
    out['Q1'] = {'no_lowpass': q1('A1_MI_lvl30'), 'lp6k': q1('A1_MI_lp6k')}
    # Part B
    bsel = json.loads(P.bsel.read_text())
    data = [load(b_label(x)) for x in INF_SEEDS]
    bk = {}
    base = {key: [data[0][i][key] for i in ids] for key in keys}
    for sel_key, tag in (('A1_MI_lvl30', 'byA1'), ('A3a_MI_lvl30', 'byA3a')):
        for k in KS:
            if tag == 'byA3a' and k != 8:
                continue
            pick = bestofk(ids, data, k, sel_key)
            r = {}
            for key in keys:
                v = [data[pick[i]][i][key] for i in ids]
                dv = [a - b for a, b in zip(v, base[key])]
                r[key] = {'mean': statistics.fmean(v), 'd_vs_bo1': statistics.fmean(dv),
                          'd_ci95': boot_ci(dv, rng) if k > 1 else [0.0, 0.0]}
            r['silent_n'] = sum(int(data[pick[i]][i]['silent']) for i in ids)
            r['seed_pick_counts'] = [sum(1 for i in ids if pick[i] == j) for j in range(k)]
            if k == 1:
                r['fad'] = json.loads((P.per_cell / f'{b_label(42)}.json').read_text())['fad']
            else:
                r['fad'], _ = fad_of_pick(ids, pick, f'{tag}_k{k}')
            bk[f'{tag}_k{k}'] = r
    # descriptive control (not in the verdict): a random seed per prompt among the 8, to show the
    # FAD / metric spread that any re-pick of seeds produces without selection
    import random
    rr = random.Random(BOOT_SEED)
    pick = {i: rr.randrange(8) for i in ids}
    r = {}
    for key in keys:
        v = [data[pick[i]][i][key] for i in ids]
        dv = [a - b for a, b in zip(v, base[key])]
        r[key] = {'mean': statistics.fmean(v), 'd_vs_bo1': statistics.fmean(dv), 'd_ci95': boot_ci(dv, rng)}
    r['silent_n'] = sum(int(data[pick[i]][i]['silent']) for i in ids)
    r['fad'], _ = fad_of_pick(ids, pick, 'rand_k8')
    bk['rand_k8'] = r
    b8 = bk['byA1_k8']
    dA1 = b8['A1_MI_lvl30']['d_vs_bo1']
    T = {}
    for key in ('A3a_MI_lvl30', 'PQ_lvl30', 'CE_lvl30', 'CLAP'):
        T[key] = (b8[key]['d_vs_bo1'] / S_[key]) / (dA1 / S_['A1_MI_lvl30']) if dA1 else None
    fad_worse = b8['fad'] > bk['byA1_k1']['fad']
    if (T['A3a_MI_lvl30'] is not None and T['A3a_MI_lvl30'] < 0.5) or \
            (T['PQ_lvl30'] < 0.25 and T['CLAP'] < 0.25) or fad_worse:
        q3 = 'hacking_signature'
    elif T['A3a_MI_lvl30'] >= 0.8 and T['PQ_lvl30'] >= 0.5 and T['CLAP'] >= 0.5:
        q3 = 'general_improvement'
    else:
        q3 = 'partial_transfer'
    # Q2 placement of bo-8 (prompt bootstrap of the bo-8 mean)
    v8 = [data[bestofk(ids, data, 8, PRIMARY)[i]][i][PRIMARY] for i in ids]
    ci8 = boot_ci(v8, rng)
    m8 = statistics.fmean(v8)
    if m8 - MUSICGEN_A1 >= M['A1'] and ci8[0] > MUSICGEN_A1:
        q2 = 'beats'
    elif abs(m8 - MUSICGEN_A1) < M['A1']:
        q2 = 'ties'
    elif m8 - MUSICGEN_A1 <= -M['A1']:
        q2 = 'loses'
    else:
        q2 = 'undetermined'
    out['B'] = {'selection': bsel, 'train_seed': B_TRAIN_SEED, 'inf_seeds': INF_SEEDS, 'bestofk': bk}
    out['Q2'] = {'bo8_A1': m8, 'bo8_A1_ci95': ci8, 'd_vs_musicgen': m8 - MUSICGEN_A1, 'd_vs_real': m8 - REAL_A1,
                 'verdict_vs_musicgen': q2,
                 'bo8_lp6k_A1_selected_by_A1': b8['A1_MI_lp6k']['mean']}
    # pre-registered rule (design §3): T_A3a < 0.5, OR (PQ and CLAP both < 0.25), OR FAD worse -> hacking_signature
    out['Q3'] = {'T': T, 'fad_bo1': bk['byA1_k1']['fad'], 'fad_bo8': b8['fad'], 'fad_worse': fad_worse,
                 'verdict': q3}
    return out


def write_summary():
    s = summarize()
    tmp = P.summary.with_suffix('.tmp')
    tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(P.summary)
    log(f'wrote {P.summary}  harness={s["harness"]["passed"]} Q1={s["Q1"]["no_lowpass"]["verdict"]} '
        f'Q2={s["Q2"]["verdict_vs_musicgen"]} Q3={s["Q3"]["verdict"]}')
    return s


def cleanup_audio():
    for label in a_labels() + [b_label(x) for x in INF_SEEDS]:
        a = cdir(label) / 'audio'
        if a.is_symlink():
            a.unlink()
        elif a.exists():
            shutil.rmtree(a)


# ── modes ────────────────────────────────────────────────
def preflight():
    errs = []
    if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
        errs.append('less than 20 GiB free on NVMe')
    for p in (PARTB / 'part_b_selection.json', TTM115, HQ_TSV, MC_TSV, Path(FAD_REF)):
        if not p.exists():
            errs.append(f'missing {p}')
    try:
        ids = write_subsets(LIMIT)
        if not LIMIT and len(ids) != 1000:
            errs.append(f'{len(ids)} Part B prompts, expected 1000')
        for label in a_labels():
            a, _ = cell_spec(label)
            eval_cmd(a, Path('/dev/null'))
            if not Path(a['model_path']).is_file():
                errs.append(f'{label}: checkpoint missing {a["model_path"]}')
        for s in SEEDS:
            p181 = R181 / f'{SRC_HQ.format(seed=s)}.tsv'
            if not p181.is_file() or not set(ids) <= set(read_clips(p181)):
                errs.append(f'181 per_cell missing/incomplete: {p181}')
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    for e in errs:
        log(f'PREFLIGHT FAIL {e}')
    if not errs:
        log(f'PREFLIGHT OK ({len(a_labels())} A cells + {len(INF_SEEDS)} B, {len(ids)} prompts)')
    return 1 if errs else 0


def validate():
    labels = a_labels() + [b_label(x) for x in INF_SEEDS]
    pending = [l for l in labels if not done(l)]
    if pending:
        log(f'VALIDATE FAIL: {len(pending)} cells not done: {pending[:3]}')
        return 1
    if not P.summary.is_file():
        log('VALIDATE FAIL: summary missing')
        return 1
    s = json.loads(P.summary.read_text())
    if s.get('document_kind') != 'muq_stack_bestofn_182_summary' or 'Q3' not in s:
        log('VALIDATE FAIL: summary malformed')
        return 1
    log(f'VALIDATE OK  harness={s["harness"]["passed"]} Q1={s["Q1"]["no_lowpass"]["verdict"]} '
        f'Q3={s["Q3"]["verdict"]}')
    return 0


def progress(last):
    labels = a_labels() + [b_label(x) for x in INF_SEEDS]
    P.progress.write_text(json.dumps({'done': sum(done(l) for l in labels), 'total': len(labels), 'last': last,
                                      'written_at': time.strftime('%FT%T%z')}))


LIMIT = None


def main():
    global P, LIMIT
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--score-cell')
    ap.add_argument('--only', help='run just this A cell (smoke)')
    ap.add_argument('--limit', type=int, help='smoke: first N Part B prompts, output under <OUT>_smoke')
    ap.add_argument('--root', help=argparse.SUPPRESS)
    a = ap.parse_args()
    os.chdir(REPO)
    if a.limit:
        LIMIT = a.limit
        P = Paths(Path(str(OUT) + '_smoke'))
    if a.root:
        P = Paths(Path(a.root))
    if a.score_cell:
        return score_cell(a.score_cell)
    if a.preflight:
        return preflight()
    if a.validate_only:
        return validate()
    P.root.mkdir(parents=True, exist_ok=True)
    if a.summary_only:
        write_summary()
        return validate()
    if preflight():
        return 75
    if a.only:
        run_cell(a.only)
        return 0
    for label in a_labels():
        if not done(label):
            if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
                raise SystemExit('disk below 20 GiB')
            run_cell(label)
        progress(label)
    if not P.bsel.is_file():
        select_b()
    for x in INF_SEEDS:
        label = b_label(x)
        if not done(label):
            if shutil.disk_usage(EVAL_ROOT).free < MIN_FREE:
                raise SystemExit('disk below 20 GiB')
            run_cell(label)
        progress(label)
    write_summary()
    rc = validate()
    if rc == 0:
        cleanup_audio()
    return rc


if __name__ == '__main__':
    sys.exit(main())
