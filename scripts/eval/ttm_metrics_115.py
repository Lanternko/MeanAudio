#!/usr/bin/env python3
"""p2 115 action: loudness blocks + metrics for the TTM comparison table.

Contract: docs/experiments/ttm_metrics_115_20261009_contract.json (GPU_QUEUE_CONTRACT).
Design: docs/experiments/ttm_quality_comparison_plan_20261007.md §7.7 (loudness rules), §4.

Systems: the 113 external baselines (required), the MusicCaps real reference (anchor,
5,131 of 5,521 ids exist), and the two p2 114 MeanAudio cells (included only if both
REPORTs exist when this job first runs; the choice is frozen in selection.json).

Blocks (block-major order, lvl30 first because it is the primary block):
  lvl30  scalar gain to -30 LUFS (BS.1770, pyloudnorm)
  raw    the system's own output, untouched
  lvl23  scalar gain to -23 LUFS (sensitivity)
Gain rule: near-silent clips (raw RMS < -45 dBFS, or integrated loudness < -70 LUFS or not
finite) are NOT gained; they stay in the block as raw audio and count as failures. If the
gain would push the peak above 0.999, the gain is reduced so the peak is 0.999 (same rule
as level_match_rescore.py, no limiter); every such clip is logged as peak_capped.
Gained audio is FLAC PCM_24 (deviation from the plan's float32 WAV: every scorer reads
{id}.flac; the 24-bit floor is ~-144 dBFS, 100 dB below where PCM_16 starts to move PQ).

Per block: eval_metrics.py (CLAP batch 1, AES, level), MEva. FAD (explicit --ref_dir) and
MIR on raw only: the FAD reference set is raw real audio and VGGish features depend on
level, so a lvl30 FAD would partly measure the level offset; MIR is level-free.

  (no flag)        run every unfinished step (each step is skipped once its output validates)
  --preflight      inputs, bindings, free disk, MEva runtime check
  --validate-only  every step present and complete; writes the registered summary.json
"""
import csv
import hashlib
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

import numpy as np
import soundfile as sf

REPO = Path('/home/kojiek/MeanAudio')
SILENT_DBFS = -45.0
SILENT_LUFS = -70.0
PEAK_CAP = 0.999
UNEQUAL_RATE = 0.01
BLOCKS = {'lvl30': -30.0, 'raw': None, 'lvl23': -23.0}
AES = ('CE', 'CU', 'PC', 'PQ')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def contract():
    return json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())


def log(msg):
    print(f'[{time.strftime("%F %T")}] {msg}', flush=True)


def tsv_ids(tsv):
    with open(tsv, newline='', encoding='utf-8') as f:
        return [r['id'] for r in csv.DictReader(f, delimiter='\t')]


def root(c):
    return Path(c['storage']['path'])


# ── system selection ─────────────────────────────────────
def selection(c, freeze=False):
    """Systems in this run. MeanAudio cells are decided once, at the first run."""
    path = root(c) / 'selection.json'
    if path.is_file():
        return json.loads(path.read_text())
    systems = []
    for s in c['external_systems']:
        systems.append({'name': s['name'], 'kind': 'external', 'audio': s['audio'],
                        'sampling_rate': s['sampling_rate'], 'fad': True})
    ma = c['meanaudio_cells']
    reports = [Path(x['report']) for x in ma['cells']]
    included = all(r.is_file() for r in reports)
    if included:
        for x in ma['cells']:
            systems.append({'name': x['name'], 'kind': 'meanaudio', 'audio': x['audio'],
                            'sampling_rate': 16000, 'fad': True,
                            'report_sha256': sha256(x['report'])})
    ref = c['reference']
    systems.append({'name': ref['name'], 'kind': 'reference', 'audio': str(root(c) / ref['name'] / 'raw_audio'),
                    'source': ref['dir'], 'sampling_rate': 16000, 'fad': False})
    sel = {'written_at': time.strftime('%FT%T%z'), 'meanaudio_included': included,
           'meanaudio_reports': [str(r) for r in reports], 'systems': systems}
    if freeze:
        root(c).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sel, indent=1))
    return sel


def expected_ids(c, s):
    ids = tsv_ids(c['fixed_protocol']['tsv'])
    if s['kind'] != 'reference':
        return ids
    return [i for i in ids if (Path(s['source']) / f'{i}.wav').is_file()]


def link_reference(c, s):
    """eval_metrics/MEva read {id}.flac; libsndfile reads the wav by content."""
    d = Path(s['audio'])
    ids = expected_ids(c, s)
    if d.is_dir() and len(list(d.glob('*.flac'))) == len(ids):
        return
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for i in ids:
        (d / f'{i}.flac').symlink_to(Path(s['source']) / f'{i}.wav')


# ── loudness blocks ──────────────────────────────────────
def _gain_one(args):
    import pyloudnorm
    src, dst, target = args
    wav, sr = sf.read(src, dtype='float64', always_2d=True)
    rms_db = 20 * math.log10(math.sqrt(float(np.mean(wav ** 2))) + 1e-12)
    lufs = float(pyloudnorm.Meter(sr).integrated_loudness(wav))
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if rms_db < SILENT_DBFS or not math.isfinite(lufs) or lufs < SILENT_LUFS or peak == 0.0:
        status, gain_db = 'silent_raw', 0.0
    else:
        gain_db, status = target - lufs, 'ok'
        if peak * 10 ** (gain_db / 20) > PEAK_CAP:
            gain_db, status = 20 * math.log10(PEAK_CAP / peak), 'peak_capped'
    out = np.clip(wav * 10 ** (gain_db / 20), -1.0, 1.0)
    sf.write(dst, out, sr, format='FLAC', subtype='PCM_24')
    return (Path(src).stem, status, round(rms_db, 4), lufs if math.isfinite(lufs) else '',
            round(peak, 6), round(gain_db, 6))


def gain_summary_path(c, s, block):
    return root(c) / s['name'] / block / 'gain_summary.json'


def gain_done(c, s, block):
    p = gain_summary_path(c, s, block)
    if not p.is_file():
        return False
    g = json.loads(p.read_text())
    return g['n'] == len(expected_ids(c, s)) and len(list((p.parent / 'audio').glob('*.flac'))) == g['n']


def make_block(c, s, block):
    target = BLOCKS[block]
    d = root(c) / s['name'] / block
    tmp = d.with_name(block + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)
    (tmp / 'audio').mkdir(parents=True)
    ids = expected_ids(c, s)
    jobs = [(str(Path(s['audio']) / f'{i}.flac'), str(tmp / 'audio' / f'{i}.flac'), target) for i in ids]
    with Pool(c['fixed_protocol']['gain_workers']) as pool:
        res = pool.map(_gain_one, jobs, chunksize=32)
    counts = {}
    with open(tmp / 'gain_status.tsv', 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['id', 'status', 'src_rms_dbfs', 'src_lufs', 'src_peak', 'gain_db'])
        for r in res:
            w.writerow(r)
            counts[r[1]] = counts.get(r[1], 0) + 1
    gained = len(res) - counts.get('silent_raw', 0)
    capped = counts.get('peak_capped', 0)
    (tmp / 'gain_summary.json').write_text(json.dumps({
        'n': len(res), 'target_lufs': target, 'peak_cap': PEAK_CAP, 'status': counts,
        'peak_capped_rate': capped / gained if gained else None,
        'processing_unequal': bool(gained and capped / gained > UNEQUAL_RATE),
        'format': 'FLAC PCM_24', 'source': s['audio']}, indent=1))
    tmp.rename(d)
    log(f'GAIN {s["name"]} {block}: {counts}')


def block_audio(c, s, block):
    return Path(s['audio']) if block == 'raw' else root(c) / s['name'] / block / 'audio'


# ── scorers ──────────────────────────────────────────────
def metrics_json(c, s, block):
    return root(c) / s['name'] / block / 'scores' / f'{s["name"]}_{block}' / 'metrics.json'


def metrics_done(c, s, block):
    p = metrics_json(c, s, block)
    if not p.is_file():
        return False
    m = json.loads(p.read_text())
    want_fad = block == 'raw' and s['fad']
    return (m['n_present'] == len(expected_ids(c, s)) and m['gen_dir'] == str(block_audio(c, s, block).resolve())
            and m['metrics'].get('aes_n') == m['n_present'] and m['metrics'].get('clap_n') == m['n_present']
            and (not want_fad or m['metrics'].get('fad', -1) >= 0))


def run_metrics(c, s, block):
    p = c['fixed_protocol']
    out = root(c) / s['name'] / block / 'scores'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cmd = [p['python'], str(REPO / 'scripts/eval/eval_metrics.py'), '--gen_dir', str(block_audio(c, s, block)),
           '--tsv', p['tsv'], '--exp_name', f'{s["name"]}_{block}', '--out_dir', str(out)]
    if s['kind'] == 'reference':
        cmd.append('--allow_missing')
    if block == 'raw' and s['fad']:
        cmd += ['--fad', '--ref_dir', p['fad_ref_dir'], '--fad_num_samples', str(p['fad_num_samples']),
                '--seed', str(p['fad_seed'])]
    log(f'METRICS {s["name"]} {block}')
    with open(out.parent / 'metrics.log', 'w') as g:
        subprocess.run(cmd, cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    if not metrics_done(c, s, block):
        raise SystemExit(f'{s["name"]} {block}: metrics incomplete')


def meva_json(c, s, block):
    return root(c) / s['name'] / block / 'meva' / 'meva.json'


def meva_done(c, s, block):
    p = meva_json(c, s, block)
    return p.is_file() and json.loads(p.read_text())['n'] == len(expected_ids(c, s))


def run_meva(c, s, block):
    p = c['fixed_protocol']
    out = meva_json(c, s, block).parent
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    log(f'MEVA {s["name"]} {block}')
    with open(out.parent / 'meva.log', 'w') as g:
        subprocess.run([p['meva_python'], str(REPO / 'scripts/eval/meva_score_dir.py'),
                        '--audio_dir', str(block_audio(c, s, block)), '--out_dir', str(out)],
                       cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    if not meva_done(c, s, block):
        raise SystemExit(f'{s["name"]} {block}: MEva incomplete')


def mir_json(c, s):
    return root(c) / s['name'] / 'raw' / 'mir' / f'{s["name"]}_raw' / 'mir_metrics.json'


def run_mir(c, s):
    p = c['fixed_protocol']
    out = mir_json(c, s).parent.parent
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    cmd = [p['python'], str(REPO / 'scripts/eval/mir_metrics.py'), '--gen_dir', str(block_audio(c, s, 'raw')),
           '--tsv', p['tsv'], '--exp_name', f'{s["name"]}_raw', '--out_dir', str(out)]
    if s['kind'] == 'reference':
        cmd.append('--allow_missing')
    log(f'MIR {s["name"]}')
    with open(out.parent / 'mir.log', 'w') as g:
        subprocess.run(cmd, cwd=REPO, stdout=g, stderr=subprocess.STDOUT, check=True)
    if not mir_json(c, s).is_file():
        raise SystemExit(f'{s["name"]}: MIR incomplete')


# ── modes ────────────────────────────────────────────────
def preflight(c):
    p, b = c['fixed_protocol'], c['bindings']
    if sha256(p['tsv']) != p['tsv_sha256']:
        sys.exit('TSV sha mismatch')
    for k in ('eval_metrics', 'meva_scorer', 'mir_metrics'):
        if sha256(b[k]) != b[f'{k}_sha256']:
            sys.exit(f'{k} sha mismatch')
    for s in c['external_systems']:
        man = Path(s['manifest'])
        if sha256(man) != s['manifest_sha256']:
            sys.exit(f'{s["name"]}: 113 manifest changed')
        if len(list(Path(s['audio']).glob('*.flac'))) != p['n_prompts']:
            sys.exit(f'{s["name"]}: audio count != {p["n_prompts"]}')
    if not Path(p['fad_ref_dir']).is_dir() or not Path(c['reference']['dir']).is_dir():
        sys.exit('reference dir missing')
    for py in (p['python'], p['meva_python'], *p['mir_pythons']):
        if not os.access(py, os.X_OK):
            sys.exit(f'missing interpreter {py}')
    root(c).mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root(c)).free < c['storage']['min_free_bytes']:
        sys.exit('not enough free disk')
    if not (root(c) / 'meva_runtime_check.json').is_file():
        r = subprocess.run([p['meva_python'], str(REPO / 'scripts/eval/meva_score_dir.py'),
                            '--runtime_check', str(p['meva_runtime_check_n'])],
                           cwd=REPO, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'MEva runtime check failed: {r.stdout[-500:]} {r.stderr[-500:]}')
        (root(c) / 'meva_runtime_check.json').write_text(r.stdout.strip().splitlines()[-1])
    sel = selection(c)
    print(f'preflight ok; systems: {[s["name"] for s in sel["systems"]]}')


def run(c):
    sel = selection(c, freeze=True)
    for s in sel['systems']:
        if s['kind'] == 'reference':
            link_reference(c, s)
    for block in BLOCKS:
        for s in sel['systems']:
            if BLOCKS[block] is not None and not gain_done(c, s, block):
                make_block(c, s, block)
            if not metrics_done(c, s, block):
                run_metrics(c, s, block)
            if not meva_done(c, s, block):
                run_meva(c, s, block)
            if block == 'raw' and not mir_json(c, s).is_file():
                run_mir(c, s)
    log('all steps done')


def per_clip(path, key='id'):
    with open(path, newline='', encoding='utf-8') as f:
        return {r[key]: r for r in csv.DictReader(f, delimiter='\t')}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def fmean(xs):
    xs = [x for x in xs if math.isfinite(x)]
    return statistics.fmean(xs) if xs else None


def validate(c):
    sel = selection(c)
    bad = []
    for s in sel['systems']:
        for block in BLOCKS:
            if BLOCKS[block] is not None and not gain_done(c, s, block):
                bad.append(f'{s["name"]} {block}: gain')
            if not metrics_done(c, s, block):
                bad.append(f'{s["name"]} {block}: metrics')
            if not meva_done(c, s, block):
                bad.append(f'{s["name"]} {block}: meva')
        if not mir_json(c, s).is_file():
            bad.append(f'{s["name"]}: mir')
    if bad:
        sys.exit('; '.join(bad))

    gen = [s for s in sel['systems'] if s['kind'] != 'reference']
    silent = {s['name']: {i for i, r in per_clip(root(c) / s['name'] / 'lvl30' / 'gain_status.tsv').items()
                          if r['status'] == 'silent_raw'} for s in sel['systems']}
    union = set().union(*(silent[s['name']] for s in gen))
    out = {'document_kind': 'ttm_metrics_115_summary_v1', 'written_at': time.strftime('%FT%T%z'),
           'tsv': c['fixed_protocol']['tsv'], 'meanaudio_included': sel['meanaudio_included'],
           'silent_rule': f'raw RMS < {SILENT_DBFS} dBFS or LUFS < {SILENT_LUFS} / not finite: not gained, kept, counted',
           'peak_rule': f'gain reduced so peak = {PEAK_CAP}; processing_unequal if lvl30 rate > {UNEQUAL_RATE}',
           'silent_union_generated_n': len(union), 'markers': c['markers'], 'systems': {}}
    for s in sel['systems']:
        rec = {'kind': s['kind'], 'sampling_rate': s['sampling_rate'], 'audio': s['audio'],
               'n': len(expected_ids(c, s)), 'silent_n': len(silent[s['name']]), 'blocks': {}}
        rec['silent_rate'] = rec['silent_n'] / rec['n']
        for block in BLOCKS:
            m = json.loads(metrics_json(c, s, block).read_text())['metrics']
            pc = per_clip(metrics_json(c, s, block).parent / 'per_clip.tsv')
            mv = {k: float(v['meva_raw']) for k, v in per_clip(meva_json(c, s, block).parent / 'meva_per_clip.tsv').items()}
            b = {'clap': m['clap_score'], **{k: m[f'aes_{k}'] for k in AES},
                 'meva': json.loads(meva_json(c, s, block).read_text())['meva_mean'],
                 'lufs_mean': m['level_lufs_mean'], 'crest_mean': m['level_crest_mean'],
                 'clipped_n': m['level_clipped_n'], 'silent_rms_n': m['level_silent_n']}
            if 'fad' in m:
                b['fad'], b['fad_pairs'] = m['fad'], m['fad_pairs']
            if BLOCKS[block] is not None:
                g = json.loads(gain_summary_path(c, s, block).read_text())
                b['gain_status'], b['peak_capped_rate'] = g['status'], g['peak_capped_rate']
                b['processing_unequal'] = g['processing_unequal']
            keep = [i for i in pc if i not in union]
            b['excl_silent_union'] = {
                'n': len(keep), 'clap': fmean(num(pc[i]['clap']) for i in keep),
                **{k: fmean(num(pc[i][k]) for i in keep) for k in AES},
                'meva': fmean(mv[i] for i in keep if i in mv)}
            rec['blocks'][block] = b
        mir = json.loads(mir_json(c, s).read_text())
        rec['mir_raw'] = mir.get('metrics', mir)
        out['systems'][s['name']] = rec
    report = Path(c['reports'][0]['path'])
    report.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    c = contract()
    if '--preflight' in sys.argv:
        preflight(c)
    elif '--validate-only' in sys.argv:
        validate(c)
    else:
        run(c)
