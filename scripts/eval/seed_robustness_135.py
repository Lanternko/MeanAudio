#!/usr/bin/env python3
"""p2 135/136/137 action: table B inference-seed robustness (generation + lvl30 scoring).

Spec (pre-registered, frozen): docs/experiments/seed_robustness_table_b_135_20261021.md
Contract: GPU_QUEUE_CONTRACT (job = "135" ours, "136" MusicGen-medium, "137" MusicLDM).

One *share* = one system at one inference seed over the 5,521 MusicCaps prompts, stored at
  /mnt/seagate/meanaudio/seed_robustness_135/<system>_<seed>/
    audio/            raw output (same file format as the seed-42 originals)
    share_manifest.json   n, missing, silent, NaN, timing, audio sha, driver/ckpt sha
    gen.log, lvl30/   (lvl30 = the 115 block: gain, eval_metrics CLAP batch 1 + AES, MEva)
A share with a complete manifest is not regenerated; a partial one is deleted and regenerated
whole (eval.py seeds one RNG per run and skips existing clips without drawing noise, so
topping up gives different audio than a fresh run).

Generation is the original protocol with the seed as the only variable:
  ours      eval.py with exactly the flags of scripts/eval/mc_mf25_eval.sh (--no_q, 25 MeanFlow
            steps, NoMask, fp32, CFG 0 / CFG 3.0 + fidelity8), only --seed changes.
  musicgen  scripts/eval/ttm_external_generate_seeded.py (copy of ttm_external_generate.py,
            --base_seed), batch 16 as in 113; PCM_16 FLAC like 113 so that seed 42 is byte-equal.
  musicldm  same generator, batch 8 as in 113.
Scoring is not re-implemented: every step is a function of scripts/eval/ttm_metrics_115.py.

Reproduction gate (spec §3.1, §3.2), the first step of every job and also runnable alone:
  --gate {ours,musicgen,musicldm,score,all} [--n 256] [--root DIR]
The gate regenerates the first N prompts at seed 42 and compares them with the existing
audio (max abs sample difference <= 1e-4, or per-clip lvl30 PQ |diff| median <= 0.01 and
system mean diff <= 0.005); `score` rescored an existing seed-42 cell and compares lvl30 CLAP/PQ
with the 118 summary (<= 0.002). A failed gate makes the job exit non-zero before any new seed.

  (no flag)        gate, then every unfinished share of the contract's job
  --preflight      inputs, shas, interpreters, free disk
  --validate-only  every share complete and scored; writes the registered summary
"""
import argparse
import csv
import hashlib
import json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ttm_metrics_115 as p115  # noqa: E402

REPO = Path('/home/kojiek/MeanAudio')
ROOT = Path('/mnt/seagate/meanaudio/seed_robustness_135')
TSV = '/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv'
TSV_SHA = 'de567b13c39b6e7f7b3666f257817322ea119bcdece82fb5e8700b4a7470e51f'
N_PROMPTS = 5521
NEG = 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
PY_DAC = '/home/kojiek/venvs/dac/bin/python'
PY_TTM = '/home/kojiek/venvs/ttm/bin/python'
PY_MEVA = '/home/kojiek/MeanAudio/runtime/meva_20261003/venv/bin/python'
FAD_REF = '/mnt/HDD/kojiek/musiccaps_reference'
GENERATOR = HERE / 'ttm_external_generate_seeded.py'
EVAL_NVME = Path('/home/kojiek/eval_output_nvme')
OURS_PREFIX = 'phase8_qwen_caption2p0_slot0clean_nmv2matched_noq_full'
TRAIN_SEEDS = {'s14159265': 114, 's27182818': 116, 's16180339': 117}
CFGS = ('cfg0', 'cfg3_neg')
OURS_SEEDS = (101, 202, 303, 404)
MUSICGEN_SEEDS = (101, 202, 303, 404)
MUSICLDM_SEEDS = (101, 202)
EXTERNAL = {  # batch size = 113's
    'musicgen_medium': {'batch_size': 16, 'sr': 32000, 'ref113': 'musicgen_medium'},
    'musicldm': {'batch_size': 8, 'sr': 16000, 'ref113': 'musicldm'},
}
EXT113 = EVAL_NVME / 'ttm_external_20261008'
M115 = EVAL_NVME / 'ttm_metrics_115_20261009'
M118 = EVAL_NVME / 'ttm_metrics_seeds_118_20261011'
SUMMARY_118 = M118 / 'summary.json'
GATE_N = 256
SILENT_DBFS = p115.SILENT_DBFS
MIN_SECONDS = 9.9


def sha256_file(p):
    return p115.sha256(p)


def log(msg):
    p115.log(msg)


# ── share definitions ────────────────────────────────────
def ckpt_path(train):
    stem = f'{OURS_PREFIX}_{train}_stage2_200000'
    return Path(f'/home/kojiek/exps_nvme/{stem}/{stem}_ema_final.pth')


def ours_cell(train, cfg):
    return EVAL_NVME / f'{OURS_PREFIX}_{train}_mc_mf25_{cfg}'


def share(system, seed, **kw):
    return {'name': f'{system}_{seed}', 'system': system, 'seed': seed, **kw}


def ours_share(train, cfg, seed):
    return share(f'ours_{train}_{cfg}', seed, kind='ours', train=train, cfg=cfg)


def ext_share(system, seed):
    return share(system, seed, kind='external')


def job_shares(job):
    if job == '135':
        return [ours_share(t, cfg, s) for s in OURS_SEEDS for t in TRAIN_SEEDS for cfg in CFGS]
    if job == '136':
        return [ext_share('musicgen_medium', s) for s in MUSICGEN_SEEDS]
    if job == '137':
        return [ext_share('musicldm', s) for s in MUSICLDM_SEEDS]
    raise SystemExit(f'unknown job {job}')


def existing_audio(sh):
    """Seed-42 audio this share's generator must reproduce."""
    if sh['kind'] == 'ours':
        return ours_cell(sh['train'], sh['cfg']) / 'audio'
    return EXT113 / EXTERNAL[sh['system']]['ref113'] / 'audio'


def existing_lvl30_dir(sh):
    """The 115 / 118 lvl30 block of the seed-42 originals."""
    if sh['kind'] == 'external':
        n = EXTERNAL[sh['system']]['ref113']
        return M115 / n / 'lvl30', n
    if sh['train'] == 's14159265':
        n = f'meanaudio_{sh["cfg"]}'
        return M115 / n / 'lvl30', n
    n = f'meanaudio_{sh["train"]}_{sh["cfg"]}'
    return M118 / n / 'lvl30', n


def existing_per_clip(sh):
    d, n = existing_lvl30_dir(sh)
    return d / 'scores' / f'{n}_lvl30' / 'per_clip.tsv'


def sampling_rate(sh):
    return 16000 if sh['kind'] == 'ours' else EXTERNAL[sh['system']]['sr']


# ── TSV / contract-like dict for the 115 functions ───────
def write_subset_tsv(dst, n):
    """First n csv records (not lines: some captions contain newlines inside quotes)."""
    with open(TSV, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f, delimiter='\t')
        rows = [r for _, r in zip(range(n), rd)]
        fields = rd.fieldnames
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    return dst


def c115(root, tsv):
    """The slice of a 115-style contract that the imported step functions read."""
    return {'storage': {'path': str(root)},
            'fixed_protocol': {'tsv': str(tsv), 'python': PY_DAC, 'meva_python': PY_MEVA,
                               'gain_workers': 12, 'fad_ref_dir': FAD_REF,
                               'fad_num_samples': 2048, 'fad_seed': 42}}


def sysrec(sh, root, audio=None):
    return {'name': sh['name'], 'kind': 'meanaudio' if sh['kind'] == 'ours' else 'external',
            'audio': str(audio or Path(root) / sh['name'] / 'audio'),
            'sampling_rate': sampling_rate(sh), 'fad': False}


# ── generation ───────────────────────────────────────────
def _probe(args):
    path, sr = args
    p = Path(path)
    if not p.is_file():
        return (p.stem, False, True, None, 0, '')
    x, fsr = sf.read(str(p), dtype='float32')
    finite = bool(np.isfinite(x).all()) and x.ndim == 1 and fsr == sr
    rms = 20 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12) if finite else None
    return (p.stem, True, finite, rms, len(x), sha256_file(p))


def inspect_audio(audio_dir, ids, sr):
    with Pool(12) as pool:
        res = pool.map(_probe, [(str(Path(audio_dir) / f'{i}.flac'), sr) for i in ids], chunksize=32)
    present = [r for r in res if r[1]]
    listing = hashlib.sha256()
    for r in sorted(present):
        listing.update(f'{r[0]}\t{r[5]}\n'.encode())
    return {
        'n_expected': len(ids), 'n_present': len(present), 'n_missing': len(ids) - len(present),
        'n_nonfinite_or_badformat': sum(not r[2] for r in present),
        'n_silent_rms_lt_-45dbfs': sum(1 for r in present if r[3] is not None and r[3] < SILENT_DBFS),
        'n_short': sum(1 for r in present if r[4] < MIN_SECONDS * sr),
        'audio_listing_sha256': listing.hexdigest()}


def manifest_path(sh, root):
    return Path(root) / sh['name'] / 'share_manifest.json'


def generation_complete(sh, root, ids, tsv_sha):
    mp = manifest_path(sh, root)
    if not mp.is_file():
        return False
    m = json.loads(mp.read_text())
    audio = Path(root) / sh['name'] / 'audio'
    return (m.get('status') == 'complete' and m.get('tsv_sha256') == tsv_sha and m.get('seed') == sh['seed']
            and m.get('n_expected') == len(ids) and m.get('n_present') == len(ids)
            and len(list(audio.glob('*.flac'))) == len(ids))


def gen_command(sh, d, tsv, n_rows):
    if sh['kind'] == 'ours':
        cfg3 = sh['cfg'] == 'cfg3_neg'
        cmd = [PY_DAC, 'eval.py', '--variant', 'meanaudio_s', '--model_path', str(ckpt_path(sh['train'])),
               '--output', str(d / 'audio'), '--tsv', str(tsv), '--use_meanflow',
               '--num_steps', '25', '--cfg_strength', '3.0' if cfg3 else '0']
        if cfg3:
            cmd += ['--negative_prompt', NEG]
        return cmd + ['--no_text_attention_mask', '--encoder_name', 't5_clap', '--text_c_dim', '512',
                      '--seed', str(sh['seed']), '--full_precision', '--no_q']
    return [PY_TTM, str(GENERATOR), '--system', sh['system'], '--tsv', str(tsv), '--out_dir', str(d),
            '--batch_size', str(EXTERNAL[sh['system']]['batch_size']), '--base_seed', str(sh['seed'])]


def git_rev():
    return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def generate_share(sh, root, tsv, tsv_sha, ids):
    d = Path(root) / sh['name']
    if generation_complete(sh, root, ids, tsv_sha):
        log(f'GEN skip (complete manifest) {sh["name"]}')
        return json.loads(manifest_path(sh, root).read_text())
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
           'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    for attempt in (1, 2):
        shutil.rmtree(d / 'audio', ignore_errors=True)
        for stale in ('manifest.json', 'share_manifest.json'):
            (d / stale).unlink(missing_ok=True)
        d.mkdir(parents=True, exist_ok=True)
        cmd = gen_command(sh, d, tsv, len(ids))
        log(f'GEN {sh["name"]} attempt {attempt}: {" ".join(cmd[:4])} ...')
        t0, t0s = time.time(), time.strftime('%FT%T%z')
        with open(d / 'gen.log', 'w') as g:
            rc = subprocess.run(cmd, cwd=REPO, stdout=g, stderr=subprocess.STDOUT, env=env).returncode
        seconds = time.time() - t0
        if rc:
            raise SystemExit(f'{sh["name"]}: generator exited {rc} (see {d / "gen.log"})')
        info = inspect_audio(d / 'audio', ids, sampling_rate(sh))
        log(f'GEN {sh["name"]}: {info["n_present"]}/{info["n_expected"]} present, '
            f'{info["n_silent_rms_lt_-45dbfs"]} silent, {info["n_nonfinite_or_badformat"]} bad, {seconds:.0f}s')
        if info['n_missing'] == 0 and info['n_nonfinite_or_badformat'] == 0 and info['n_short'] == 0:
            break
        log(f'GEN {sh["name"]}: incomplete, regenerating whole share' if attempt == 1 else 'GEN incomplete again')
    else:
        raise SystemExit(f'{sh["name"]}: still incomplete after a second full run: {info}')
    manifest = {
        'status': 'complete', 'system': sh['system'], 'seed': sh['seed'], 'kind': sh['kind'],
        'tsv': str(tsv), 'tsv_sha256': tsv_sha, 'sampling_rate': sampling_rate(sh),
        'command': cmd, 'started_at': t0s, 'finished_at': time.strftime('%FT%T%z'),
        'generation_seconds': round(seconds, 1), 'seconds_per_clip': round(seconds / len(ids), 4),
        'driver_sha256': sha256_file(__file__), 'git_rev': git_rev(), **info}
    if sh['kind'] == 'ours':
        manifest.update(train=sh['train'], cfg=sh['cfg'], checkpoint=str(ckpt_path(sh['train'])))
    else:
        manifest.update(generator_sha256=sha256_file(GENERATOR),
                        generator_manifest=json.loads((d / 'manifest.json').read_text()))
        manifest['generator_manifest'].pop('batches', None)
    manifest_path(sh, root).write_text(json.dumps(manifest, indent=1))
    return manifest


# ── scoring (115 functions, lvl30 only) ──────────────────
def scored(c, s):
    return p115.gain_done(c, s, 'lvl30') and p115.metrics_done(c, s, 'lvl30') and p115.meva_done(c, s, 'lvl30')


def score_share(c, s):
    if not p115.gain_done(c, s, 'lvl30'):
        p115.make_block(c, s, 'lvl30')
    if not p115.metrics_done(c, s, 'lvl30'):
        p115.run_metrics(c, s, 'lvl30')
    if not p115.meva_done(c, s, 'lvl30'):
        p115.run_meva(c, s, 'lvl30')


def read_tsv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return {r['id']: r for r in csv.DictReader(f, delimiter='\t')}


# ── reproduction gate ────────────────────────────────────
def gate_report_path(root, kind):
    return Path(root) / f'gate_{kind}.json'


def gate_passed(root, kind, n):
    p = gate_report_path(root, kind)
    if not p.is_file():
        return False
    g = json.loads(p.read_text())
    return g.get('passed') is True and g.get('n') == n and g.get('driver_sha256') == sha256_file(__file__)


def compare_audio(new_dir, old_dir, ids):
    worst, exact, shape_bad = 0.0, 0, 0
    for i in ids:
        a, sa = sf.read(str(Path(new_dir) / f'{i}.flac'), dtype='float64')
        b, sb = sf.read(str(Path(old_dir) / f'{i}.flac'), dtype='float64')
        if a.shape != b.shape or sa != sb:
            shape_bad += 1
            continue
        dmax = float(np.max(np.abs(a - b))) if a.size else 0.0
        worst = max(worst, dmax)
        exact += dmax == 0.0
    return {'max_abs_diff': worst, 'n_bit_identical': exact, 'n_shape_or_sr_mismatch': shape_bad}


def gate_system(sh, root, tsv, tsv_sha, ids):
    gm = generate_share(sh, root, tsv, tsv_sha, ids)
    new_dir = Path(root) / sh['name'] / 'audio'
    res = {'system': sh['system'], 'n': len(ids), 'seconds_per_clip': gm['seconds_per_clip'],
           'generation_seconds': gm['generation_seconds'], **compare_audio(new_dir, existing_audio(sh), ids)}
    audio_ok = res['n_shape_or_sr_mismatch'] == 0 and res['max_abs_diff'] <= 1e-4
    c = c115(root, tsv)
    score_share(c, sysrec(sh, root))
    new_pc = read_tsv(Path(root) / sh['name'] / 'lvl30' / 'scores' / f'{sh["name"]}_lvl30' / 'per_clip.tsv')
    old_pc = read_tsv(existing_per_clip(sh))
    d = [float(new_pc[i]['PQ']) - float(old_pc[i]['PQ']) for i in ids]
    dc = [float(new_pc[i]['clap']) - float(old_pc[i]['clap']) for i in ids]
    res.update(pq_abs_diff_median=statistics.median(abs(x) for x in d), pq_mean_diff=statistics.fmean(d),
               clap_mean_diff=statistics.fmean(dc))
    pq_ok = res['pq_abs_diff_median'] <= 0.01 and abs(res['pq_mean_diff']) <= 0.005
    res.update(audio_criterion_pass=audio_ok, pq_criterion_pass=pq_ok, passed=bool(audio_ok or pq_ok))
    return res


def score_gate(root):
    """Spec §3.2: rescore an existing seed-42 cell with this scoring path; 118 summary +-0.002."""
    sh = ours_share('s27182818', 'cfg3_neg', 42)
    c = c115(Path(root) / 'score_gate', TSV)
    s = {'name': 'scoregate_s27182818_cfg3_neg', 'kind': 'meanaudio', 'audio': str(existing_audio(sh)),
         'sampling_rate': 16000, 'fad': False}
    score_share(c, s)
    m = json.loads(p115.metrics_json(c, s, 'lvl30').read_text())['metrics']
    mv = json.loads(p115.meva_json(c, s, 'lvl30').read_text())['meva_mean']
    ref = json.loads(SUMMARY_118.read_text())['systems']['meanaudio_s27182818_cfg3_neg']['blocks']['lvl30']
    new = {'clap': m['clap_score'], 'PQ': m['aes_PQ'], 'meva': mv}
    diff = {k: new[k] - ref[k] for k in new}
    return {'cell': s['audio'], 'new': new, 'ref_118': {k: ref[k] for k in new}, 'diff': diff,
            'tolerance': 0.002, 'passed': abs(diff['clap']) <= 0.002 and abs(diff['PQ']) <= 0.002}


def run_gate(kind, root=ROOT / 'gate', n=GATE_N, force=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not force and gate_passed(root, kind, n):
        log(f'GATE {kind}: already passed for this driver sha')
        return True
    report = {'kind': kind, 'n': n, 'driver_sha256': sha256_file(__file__), 'git_rev': git_rev(),
              'started_at': time.strftime('%FT%T%z'), 'systems': []}
    if kind == 'score':
        report['score_gate'] = score_gate(root)
        report['passed'] = report['score_gate']['passed']
    else:
        tsv = write_subset_tsv(root / f'tsv{n}.tsv', n)
        tsv_sha = sha256_file(tsv)
        ids = list(read_tsv(tsv))
        subjects = {'ours': [ours_share('s14159265', cfg, 42) for cfg in CFGS],
                    'musicgen': [ext_share('musicgen_medium', 42)],
                    'musicldm': [ext_share('musicldm', 42)]}[kind]
        for sh in subjects:
            r = gate_system(sh, root, tsv, tsv_sha, ids)
            report['systems'].append(r)
            log(f'GATE {kind} {sh["name"]}: max|d|={r["max_abs_diff"]:.2e} bit-identical {r["n_bit_identical"]}/{n} '
                f'PQ|d| med {r["pq_abs_diff_median"]:.4f} mean {r["pq_mean_diff"]:+.4f} '
                f'{r["seconds_per_clip"]:.3f}s/clip -> {"PASS" if r["passed"] else "FAIL"}')
        report['passed'] = all(r['passed'] for r in report['systems'])
    report['finished_at'] = time.strftime('%FT%T%z')
    gate_report_path(root, kind).write_text(json.dumps(report, indent=1))
    log(f'GATE {kind}: {"PASS" if report["passed"] else "FAIL"} -> {gate_report_path(root, kind)}')
    return report['passed']


# ── job modes ────────────────────────────────────────────
def contract():
    return json.loads(Path(os.environ['GPU_QUEUE_CONTRACT']).read_text())


def gates_for(job):
    return {'135': ['ours', 'score'], '136': ['musicgen'], '137': ['musicldm']}[job]


def preflight(c):
    job, p, b = c['job'], c['fixed_protocol'], c['bindings']
    if sha256_file(TSV) != TSV_SHA:
        sys.exit('TSV sha mismatch')
    for k in ('generator', 'ttm_metrics_115', 'eval_metrics', 'meva_scorer'):
        if sha256_file(b[k]) != b[f'{k}_sha256']:
            sys.exit(f'{k} sha mismatch')
    if sha256_file(__file__) != b['action_sha256']:
        sys.exit('action sha mismatch')
    for py in (PY_DAC, PY_TTM, PY_MEVA):
        if not os.access(py, os.X_OK):
            sys.exit(f'missing interpreter {py}')
    if not Path(FAD_REF).is_dir():
        sys.exit('FAD reference dir missing')
    for sh in job_shares(job):
        if sh['seed'] == 42:
            sys.exit('seed 42 is the existing run; refusing')
    if job == '135':
        for train in TRAIN_SEEDS:
            rep = next(ours_cell(train, 'cfg0').glob('*_REPORT.json'))
            want = json.loads(rep.read_text())['checkpoint_sha256']
            if sha256_file(ckpt_path(train)) != want:
                sys.exit(f'{train}: checkpoint sha differs from the seed-42 REPORT')
            for cfg in CFGS:
                if len(list((ours_cell(train, cfg) / 'audio').glob('*.flac'))) != N_PROMPTS:
                    sys.exit(f'{train} {cfg}: seed-42 audio count != {N_PROMPTS}')
    else:
        for n in {sh['system'] for sh in job_shares(job)}:
            if len(list(existing_audio(ext_share(n, 42)).glob('*.flac'))) != N_PROMPTS:
                sys.exit(f'{n}: 113 audio count != {N_PROMPTS}')
    ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < p['min_free_bytes']:
        sys.exit('not enough free disk')
    chk = ROOT / 'meva_runtime_check.json'
    if not chk.is_file():
        r = subprocess.run([PY_MEVA, str(REPO / 'scripts/eval/meva_score_dir.py'),
                            '--runtime_check', '50'], cwd=REPO, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'MEva runtime check failed: {r.stdout[-500:]} {r.stderr[-500:]}')
        chk.write_text(r.stdout.strip().splitlines()[-1])
    print(f'preflight ok; job {job}: {len(job_shares(job))} shares')


def run(c):
    job = c['job']
    for kind in gates_for(job):
        if not run_gate(kind):
            sys.exit(f'reproduction gate "{kind}" failed: no new seed is generated '
                     f'(see {gate_report_path(ROOT / "gate", kind)})')
    ids = list(read_tsv(TSV))
    cc = c115(ROOT, TSV)
    for sh in job_shares(job):
        generate_share(sh, ROOT, TSV, TSV_SHA, ids)
        score_share(cc, sysrec(sh, ROOT))
    log('all steps done')


def lvl30_means(c, s):
    m = json.loads(p115.metrics_json(c, s, 'lvl30').read_text())['metrics']
    return {'clap': m['clap_score'], **{k: m[f'aes_{k}'] for k in p115.AES},
            'meva': json.loads(p115.meva_json(c, s, 'lvl30').read_text())['meva_mean'],
            'lufs_mean': m['level_lufs_mean'], 'silent_rms_n': m['level_silent_n']}


def validate(c):
    job = c['job']
    ids = list(read_tsv(TSV))
    cc = c115(ROOT, TSV)
    bad, out = [], {}
    for sh in job_shares(job):
        s = sysrec(sh, ROOT)
        if not generation_complete(sh, ROOT, ids, TSV_SHA):
            bad.append(f'{sh["name"]}: generation')
            continue
        m = json.loads(manifest_path(sh, ROOT).read_text())
        if m['n_missing'] or m['n_nonfinite_or_badformat'] or m['n_short']:
            bad.append(f'{sh["name"]}: manifest reports missing/bad/short')
        if not scored(cc, s):
            bad.append(f'{sh["name"]}: scoring')
            continue
        out[sh['name']] = {'system': sh['system'], 'seed': sh['seed'], 'n': m['n_present'],
                           'silent_n': m['n_silent_rms_lt_-45dbfs'], 'nonfinite_n': m['n_nonfinite_or_badformat'],
                           'generation_seconds': m['generation_seconds'], 'seconds_per_clip': m['seconds_per_clip'],
                           'audio_listing_sha256': m['audio_listing_sha256'],
                           'manifest_sha256': sha256_file(manifest_path(sh, ROOT)), 'lvl30': lvl30_means(cc, s)}
    for kind in gates_for(job):
        if not gate_passed(ROOT / 'gate', kind, GATE_N):
            bad.append(f'gate {kind}')
    if bad:
        sys.exit('; '.join(bad))
    gates = {k: json.loads(gate_report_path(ROOT / 'gate', k).read_text()) for k in gates_for(job)}
    summary = {'document_kind': f'seed_robustness_135_job{job}_summary_v1', 'written_at': time.strftime('%FT%T%z'),
               'spec': 'docs/experiments/seed_robustness_table_b_135_20261021.md', 'tsv': TSV, 'n_prompts': len(ids),
               'root': str(ROOT), 'gates': gates, 'shares': out,
               'total_generation_seconds': round(sum(v['generation_seconds'] for v in out.values()), 1)}
    Path(c['reports'][0]['path']).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'gates'}, indent=1)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--gate', choices=['ours', 'musicgen', 'musicldm', 'score', 'all'])
    ap.add_argument('--n', type=int, default=GATE_N)
    ap.add_argument('--root', default=str(ROOT / 'gate'))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.gate:
        kinds = ['ours', 'musicgen', 'musicldm', 'score'] if a.gate == 'all' else [a.gate]
        ok = [run_gate(k, a.root, a.n, a.force) for k in kinds]
        return 0 if all(ok) else 1
    c = contract()
    if a.preflight:
        preflight(c)
    elif a.validate_only:
        validate(c)
    else:
        run(c)
    return 0


if __name__ == '__main__':
    sys.exit(main())
