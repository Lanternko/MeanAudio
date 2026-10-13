#!/usr/bin/env python3
"""AES + MEva + VGGish FAD + MIR on the real-piece MIDI control stimuli (p2 123).

Stimuli come from midi_realpiece_stimuli.py: 150 MAESTRO excerpts x 2 SoundFonts x 57
cases (clean, exaggerated degradations, should-not-change controls, ambiguous edits,
below-noise floors), 30 s, already at lvl30 (-30 LUFS scalar gain, peak cap 0.999,
mono 16 kHz, FLAC PCM_24).

Two views are scored, each split by SoundFont (4 units of 8,550 clips):

  full30   the rendered 30 s clips
  crop10   seconds 0.5-10.5 of each clip, re-leveled to -30 LUFS, so the readings sit
           on the same duration as the 10 s text-to-music systems (MEva is
           duration-dependent)

Readings: every case minus the clean render of the same excerpt and SoundFont
(cluster bootstrap over the 150 excerpts, 10,000 draws); the green / red scale
(clean = 1, a floor stimulus = 0) with the lvl30 scores of the p2 115 systems placed
on it; FAD of every case against MusicCaps real and against the same-bank clean set;
and the FAD checks asked for in the 2026-10-14 meeting (real split-half, CFG0 against
CFG3+neg with MeanAudio as the reference, seed against seed).

  (no flag)        run, then write <OUT>/summary.json
  --preflight      inputs, runtimes and free space
  --validate-only  every unit and the summary are complete
  --analyze-only   rebuild the summary from the stored scores
  --limit N        first N excerpts only, into <OUT>/smoke
"""
import argparse
import collections
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# MKL_NUM_THREADS set in the parent makes the forked FAD loader pool hang on its first task (first p2 123 attempt)
os.environ.pop('MKL_NUM_THREADS', None)

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path('/home/kojiek/MeanAudio')
EV = Path('/home/kojiek/eval_output_nvme')
MIDI = EV / 'midi_realpiece_123'
OUT = EV / 'midi_realpiece_control_123'
R115 = EV / 'ttm_metrics_115_20261009'
FAD_EMB = EV / 'ttm_fad_lvl30_20261013/emb'
MEVA_PY = str(REPO / 'runtime/meva_20261003/venv/bin/python')
SR, TARGET, PEAK_CAP, N_BOOT, N_FAD_BOOT, SEED = 16000, -30.0, 0.999, 10000, 100, 20261014
LEAD_SEC, CROP_SEC, MIN_FREE = 0.5, 10.0, 12 * 2 ** 30
MIN_SPAN = 0.25  # green minus red below this: the scale is not usable for that metric
BANKS, VIEWS = ('generaluser', 'fluidr3'), ('full30', 'crop10')
METRICS = ('PQ', 'CE', 'CU', 'PC', 'meva')
MIR = ('pulse_clarity', 'ibi_cv', 'key_cnn_conf', 'chroma_entropy')
SYSTEMS = ('musiccaps_real', 'musicgen_medium', 'musicldm', 'musicldm_fid8', 'meanaudio_cfg0', 'meanaudio_cfg3_neg')
MA_SEEDS = (14159265, 27182818, 16180339)
ROOT = OUT


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def items(limit=None):
    rows = json.loads((MIDI / 'items.json').read_text())
    if limit:
        keep = set(sorted({r['excerpt'] for r in rows})[:limit])
        rows = [r for r in rows if r['excerpt'] in keep]
    return rows


def unit_rows(rows, bank):
    return sorted((r for r in rows if r['bank'] == bank), key=lambda r: r['key'])


def audio_path(r, view):
    return ROOT / view / r['bank'] / 'audio' / Path(r['path']).name  # full30: a symlink to the render


def crop_one(job):
    import pyloudnorm
    src, dst = job
    x, sr = sf.read(src, dtype='float64')
    if sr != SR or x.ndim != 1:
        raise RuntimeError(f'expected mono {SR} Hz: {src}')
    x = x[int(LEAD_SEC * SR):int((LEAD_SEC + CROP_SEC) * SR)]
    lufs, peak = float(pyloudnorm.Meter(SR).integrated_loudness(x)), float(np.max(np.abs(x)))
    if not np.isfinite(lufs) or peak <= 0:  # a floor stimulus may be digitally silent in this window: keep it as is
        sf.write(dst, x, SR, format='FLAC', subtype='PCM_24')
        return 2
    g = 10 ** ((TARGET - lufs) / 20)
    capped = peak * g > PEAK_CAP
    sf.write(dst, x * (PEAK_CAP / peak if capped else g), SR, format='FLAC', subtype='PCM_24')
    return int(capped)


def unit_done(d, stems):
    try:
        aes = json.loads((d / 'aes.json').read_text())
        meva = {r['id']: float(r['meva_raw']) for r in csv.DictReader(open(d / 'meva' / 'meva_per_clip.tsv'), delimiter='\t')}
        return (set(aes) == set(meva) == set(stems) and all(math.isfinite(v) for a in aes.values() for v in a.values())
                and all(map(math.isfinite, meva.values())) and (d / 'fad_emb.npz').is_file())
    except (OSError, ValueError, KeyError):
        return False


def run_mir(d, paths):
    from mir_metrics import score_mir
    per = score_mir([str(p) for p in paths], madmom_workers=8, chroma_workers=4)
    (d / 'mir.json').write_text(json.dumps({Path(p).stem: v for p, v in per.items()}))
    log(f'  MIR done {d.relative_to(ROOT)} n={len(per)}')


def run_unit(view, bank, rows, embedder):
    from eval_metrics import score_aes
    import ttm_fad_lvl30 as F
    d = ROOT / view / bank
    stems = [Path(r['path']).stem for r in rows]
    if unit_done(d, stems):
        return
    d.mkdir(parents=True, exist_ok=True)
    paths = [audio_path(r, view) for r in rows]
    (d / 'audio').mkdir(exist_ok=True)
    if view == 'full30':
        for r, p in zip(rows, paths):
            if not p.is_symlink():
                p.symlink_to(r['path'])
    else:
        with Pool(16) as pool:
            flags = pool.map(crop_one, [(r['path'], str(p)) for r, p in zip(rows, paths)], chunksize=32)
        (d / 'crop.json').write_text(json.dumps({'n': len(flags), 'capped': flags.count(1), 'unmeasurable': flags.count(2)}))
    per, failed = score_aes(paths)
    if failed or len(per) != len(paths):
        raise SystemExit(f'{view}/{bank}: AES incomplete {len(per)}/{len(paths)} failed={len(failed)}')
    (d / 'aes.json').write_text(json.dumps({Path(p).stem: v for p, v in per.items()}))
    log(f'  AES done {view}/{bank}')
    shutil.rmtree(d / 'meva', ignore_errors=True)
    (d / 'meva').mkdir()
    with open(d / 'meva.log', 'w') as g:
        subprocess.run([MEVA_PY, str(REPO / 'scripts/eval/meva_score_dir.py'), '--audio_dir', str(paths[0].parent),
                        '--out_dir', str(d / 'meva')], cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    log(f'  MEva done {view}/{bank}')
    F.OUT = d / 'fad_tmp'
    emb = embedder.run('emb', [(Path(p).stem, str(p), False) for p in paths])
    np.savez(d / 'fad_emb.npz', **emb)
    shutil.rmtree(d / 'fad_tmp', ignore_errors=True)
    if not unit_done(d, stems):
        raise SystemExit(f'{view}/{bank}: scores incomplete')
    log(f'unit done {view}/{bank} n={len(stems)}')


def run(rows, limit=None):
    import ttm_fad_lvl30 as F
    embedder = F.Embedder()  # forks its loader pool: must happen while this process is single-threaded
    procs = []
    for bank in BANKS:  # CPU-only, alongside the GPU scoring; own process so no thread is alive at any fork here
        d = ROOT / 'full30' / bank
        d.mkdir(parents=True, exist_ok=True)
        if not (d / 'mir.json').is_file():
            cmd = [sys.executable, __file__, '--mir-bank', bank] + (['--limit', str(limit)] if limit else [])
            procs.append((bank, subprocess.Popen(cmd, stdout=open(d / 'mir.log', 'w'), stderr=subprocess.STDOUT)))
    for view in VIEWS:
        for bank in BANKS:
            run_unit(view, bank, unit_rows(rows, bank), embedder)
    embedder.pool.close()
    for bank, pr in procs:
        if pr.wait() != 0:
            raise SystemExit(f'MIR failed for {bank}: see full30/{bank}/mir.log')
    for bank in BANKS:
        if not (ROOT / 'full30' / bank / 'mir.json').is_file():
            raise SystemExit(f'MIR missing for {bank}')


# ── analysis ──
def ci(by_cluster, rng):
    m = np.array([np.mean(v) for v in by_cluster.values()])
    boot = m[rng.integers(0, len(m), (N_BOOT, len(m)))].mean(axis=1)
    return [float(m.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def load_scores(view, rows):
    sc = {}
    for bank in BANKS:
        d = ROOT / view / bank
        aes = json.loads((d / 'aes.json').read_text())
        meva = {r['id']: float(r['meva_raw']) for r in csv.DictReader(open(d / 'meva' / 'meva_per_clip.tsv'), delimiter='\t')}
        mir = json.loads((d / 'mir.json').read_text()) if view == 'full30' else {}
        for r in unit_rows(rows, bank):
            stem = Path(r['path']).stem
            sc[r['key']] = {**aes[stem], 'meva': meva[stem], **{m: mir[stem][m] for m in MIR if stem in mir and m in mir[stem]}}
    return sc


def system_scores():
    out = {}
    for s in SYSTEMS:
        d = R115 / s / 'lvl30'
        per = {r['id']: {m: float(r[m]) for m in METRICS[:4]} for r in csv.DictReader(open(d / 'scores' / f'{s}_lvl30' / 'per_clip.tsv'), delimiter='\t')}
        for r in csv.DictReader(open(d / 'meva' / 'meva_per_clip.tsv'), delimiter='\t'):
            if r['id'] in per:
                per[r['id']]['meva'] = float(r['meva_raw'])
        out[s] = {m: float(np.mean([v[m] for v in per.values() if m in v])) for m in METRICS}
    return out


def fad_block(rows, rng):
    from ttm_fad_lvl30 import cat, frechet, stats
    z = np.load(FAD_EMB / 'ref_lvl30_full.npz')
    real = {k: z[k] for k in z.files}
    ref_real = stats(np.concatenate(list(real.values())))
    res = {'reference': 'MusicCaps real lvl30, 5,131 clips, VGGish frames', 'views': {}}
    for view in VIEWS:
        V = {}
        emb, clean = {}, {}
        for bank in BANKS:
            z = np.load(ROOT / view / bank / 'fad_emb.npz')
            ur = unit_rows(rows, bank)
            for r in ur:
                emb[r['key']] = z[Path(r['path']).stem]
            clean[bank] = [r['key'] for r in ur if r['case'] == 'clean']
        for bank in BANKS:
            ref_clean = stats(cat(emb, clean[bank]))
            halves = []
            for _ in range(20):
                p = list(rng.permutation(clean[bank]))
                halves.append(frechet(stats(cat(emb, p[:len(p) // 2])), stats(cat(emb, p[len(p) // 2:]))))
            B = {'clean_vs_real': frechet(ref_clean, ref_real), 'clean_n': len(clean[bank]),
                 'clean_split_half': {'mean': float(np.mean(halves)), 'min': float(min(halves)), 'max': float(max(halves))},
                 'cases': {}}
            by_case = collections.defaultdict(list)
            for r in unit_rows(rows, bank):
                by_case[f'{r["class"]}__{r["case"]}'].append(r['key'])
            for case, keys in sorted(by_case.items()):
                st = stats(cat(emb, keys))
                e = {'n': len(keys), 'vs_real': frechet(st, ref_real)}
                if case != 'base__clean':
                    e['vs_clean'] = frechet(st, ref_clean)
                    boots = []
                    for _ in range(N_FAD_BOOT):  # excerpt-paired: the same resampled excerpts on both sides
                        idx = rng.integers(0, len(keys), len(keys))
                        boots.append(frechet(stats(cat(emb, [keys[i] for i in idx])), ref_real)
                                     - frechet(stats(cat(emb, [clean[bank][i] for i in idx])), ref_real))
                    e['d_vs_real_minus_clean'] = [e['vs_real'] - B['clean_vs_real'], *map(float, np.percentile(boots, [2.5, 97.5]))]
                B['cases'][case] = e
            V[bank] = B
        V['bank_vs_bank_clean'] = frechet(stats(cat(emb, clean[BANKS[0]])), stats(cat(emb, clean[BANKS[1]])))
        res['views'][view] = V
    # meeting checks, on the cached lvl30 embeddings of p2 115/118 (1,924 prompts per system)
    chk = {}
    ids = sorted(real)
    halves = []
    for _ in range(20):
        p = list(rng.permutation(ids))
        halves.append(frechet(stats(cat(real, p[:len(p) // 2])), stats(cat(real, p[len(p) // 2:]))))
    chk['real_split_half_2565'] = {'mean': float(np.mean(halves)), 'min': float(min(halves)), 'max': float(max(halves))}
    S = {}
    for name in [f'meanaudio_s{s}_{c}' for s in MA_SEEDS for c in ('cfg0', 'cfg3_neg')] + ['musicgen_medium', 'musicldm', 'musicldm_fid8']:
        z = np.load(FAD_EMB / f'{name}_lvl30.npz')
        S[name] = {k: z[k] for k in z.files}
    sub = sorted(set.intersection(*[set(v) for v in S.values()]) & set(real))
    st = {n: stats(cat(e, sub)) for n, e in S.items()}
    st['real_same_prompts'] = stats(cat(real, sub))
    chk['n_prompts'] = len(sub)
    halves = []
    for _ in range(20):
        p = list(rng.permutation(sub))
        halves.append(frechet(stats(cat(real, p[:len(p) // 2])), stats(cat(real, p[len(p) // 2:]))))
    chk['real_split_half_same_prompts'] = {'mean': float(np.mean(halves)), 'min': float(min(halves)), 'max': float(max(halves))}
    chk['vs_real_full'] = {n: frechet(v, ref_real) for n, v in st.items()}
    chk['vs_real_same_prompts'] = {n: frechet(v, st['real_same_prompts']) for n, v in st.items() if n != 'real_same_prompts'}
    chk['cfg0_vs_cfg3neg_same_seed'] = {str(s): frechet(st[f'meanaudio_s{s}_cfg0'], st[f'meanaudio_s{s}_cfg3_neg']) for s in MA_SEEDS}
    chk['seed_vs_seed_same_cell'] = {f'{c}|{a}|{b}': frechet(st[f'meanaudio_s{a}_{c}'], st[f'meanaudio_s{b}_{c}'])
                                     for c in ('cfg0', 'cfg3_neg') for i, a in enumerate(MA_SEEDS) for b in MA_SEEDS[i + 1:]}
    chk['cfg0_vs_cfg3neg_cross_seed'] = {f'{a}|{b}': frechet(st[f'meanaudio_s{a}_cfg0'], st[f'meanaudio_s{b}_cfg3_neg'])
                                         for a in MA_SEEDS for b in MA_SEEDS if a != b}
    for n in ('musicgen_medium', 'musicldm'):
        chk[f'{n}_vs_meanaudio'] = {c: frechet(st[n], st[f'meanaudio_s{MA_SEEDS[0]}_{c}']) for c in ('cfg0', 'cfg3_neg')}
    res['meeting_checks'] = chk
    return res


def analyze(rows):
    rng = np.random.default_rng(SEED)
    res = {'protocol': {'level': 'lvl30', 'n_boot': N_BOOT, 'n_fad_boot': N_FAD_BOOT, 'seed': SEED, 'n_clips': len(rows),
                        'n_excerpts': len({r['excerpt'] for r in rows}), 'bootstrap_unit': 'excerpt',
                        'crop10': f'seconds {LEAD_SEC}-{LEAD_SEC + CROP_SEC} of the 30 s render, re-leveled to {TARGET} LUFS'},
           'views': {}}
    sysm = system_scores()
    for view in VIEWS:
        sc = load_scores(view, rows)
        mets = METRICS + (MIR if view == 'full30' else ())
        V = {'absolute': {}, 'paired': {}, 'classes': {}}
        tree = lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        ab, pr = tree(), tree()
        for r in rows:
            for name in (r['case'], f'{r["case"]}|{r["bank"]}'):
                for m in mets:
                    if m in sc[r['key']]:
                        ab[name][m][r['excerpt']].append(sc[r['key']][m])
            if r['base']:
                for name in (f'{r["class"]}__{r["case"]}', f'{r["class"]}__{r["case"]}|{r["bank"]}'):
                    for m in mets:
                        if m in sc[r['key']] and m in sc[r['base']]:
                            pr[name][m][r['excerpt']].append(sc[r['key']][m] - sc[r['base']][m])
        for name, g in sorted(ab.items()):
            V['absolute'][name] = {m: ci(g[m], rng) for m in mets if g[m]}
        for name, g in sorted(pr.items()):
            V['paired'][name] = {m: ci(g[m], rng) for m in mets if g[m]}
        pooled = {n: v for n, v in V['paired'].items() if '|' not in n}
        for cls in ('degrade', 'control', 'ambiguous', 'floor'):
            cases = {n: v for n, v in pooled.items() if n.startswith(cls + '__')}
            if cases:
                V['classes'][cls] = {m: {'n_cases': len(cases), 'ci_below_zero': sum(v[m][2] < 0 for v in cases.values() if m in v),
                                         'ci_above_zero': sum(v[m][1] > 0 for v in cases.values() if m in v),
                                         'ci_spans_zero': sum(v[m][1] <= 0 <= v[m][2] for v in cases.values() if m in v),
                                         'mean_delta': float(np.mean([v[m][0] for v in cases.values() if m in v])),
                                         'max_abs_delta': float(max(abs(v[m][0]) for v in cases.values() if m in v))}
                                     for m in mets if any(m in v for v in cases.values())}
        # green / red scale: clean = 1, floor case = 0; the 10 s systems are only placed on the 10 s view
        green = {m: V['absolute']['clean'][m][0] for m in METRICS}
        reds = {n.split('__', 1)[1]: {m: V['absolute'][n.split('__', 1)[1]][m][0] for m in METRICS} for n in pooled if n.startswith('floor__')}
        worst = {m: float(np.mean(sorted(V['absolute'][n.split('__', 1)[1]][m][0] for n in pooled if n.startswith('degrade__'))[:5])) for m in METRICS}
        V['scale'] = {'green_clean': green, 'red_floor': reds, 'mean_of_5_worst_degrade_cases': worst}
        if view == 'crop10':
            V['scale']['systems'] = sysm
            V['scale']['position'] = {red: {s: {m: (sysm[s][m] - reds[red][m]) / (green[m] - reds[red][m]) if green[m] - reds[red][m] >= MIN_SPAN else None
                                                for m in METRICS} for s in SYSTEMS} for red in reds}
        res['views'][view] = V
    res['fad'] = fad_block(rows, rng)
    res['manipulation_checks'] = json.loads((MIDI / 'checks.json').read_text())
    (ROOT / 'summary.json').write_text(json.dumps(res, indent=1))
    log(f'ANALYZE -> {ROOT / "summary.json"}')


def preflight():
    rows = items()
    keys = {r['key'] for r in rows}
    missing = [r['path'] for r in rows if not Path(r['path']).is_file()]
    bad_base = [r['key'] for r in rows if r['base'] and r['base'] not in keys]
    OUT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUT).free
    need = ([FAD_EMB / 'ref_lvl30_full.npz'] + [R115 / s / 'lvl30/meva/meva_per_clip.tsv' for s in SYSTEMS]
            + [Path(MEVA_PY), Path('/home/kojiek/venvs/madmom/bin/python'), Path('/home/kojiek/venvs/mir/bin/python')])
    absent = [str(p) for p in need if not p.exists()]
    ok = len(rows) == len(keys) == 17100 and not missing and not bad_base and not absent and free >= MIN_FREE
    print(json.dumps({'n': len(rows), 'missing': len(missing), 'bad_base': len(bad_base), 'absent': absent, 'free_gb': free / 2 ** 30, 'ok': ok}))
    return 0 if ok else 1


def validate(rows):
    for view in VIEWS:
        for bank in BANKS:
            if not unit_done(ROOT / view / bank, [Path(r['path']).stem for r in unit_rows(rows, bank)]):
                print(json.dumps({'ok': False, 'incomplete': f'{view}/{bank}'}))
                return 1
    s = json.loads((ROOT / 'summary.json').read_text())
    ok = s['protocol']['n_clips'] == len(rows) and all(len(s['views'][v]['paired']) >= 56 for v in VIEWS) and 'meeting_checks' in s['fad']
    print(json.dumps({'n': len(rows), 'paired': len(s['views']['full30']['paired']), 'ok': ok}))
    return 0 if ok else 1


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--mir-bank', choices=BANKS)
    a = ap.parse_args()
    if a.preflight:
        return preflight()
    if a.limit:
        ROOT = OUT / 'smoke'
    rows = items(a.limit)
    if a.mir_bank:
        return run_mir(ROOT / 'full30' / a.mir_bank, [r['path'] for r in unit_rows(rows, a.mir_bank)])
    if a.validate_only:
        return validate(rows)
    if not a.analyze_only:
        run(rows, a.limit)
    analyze(rows)
    return validate(rows)


if __name__ == '__main__':
    sys.exit(main())
