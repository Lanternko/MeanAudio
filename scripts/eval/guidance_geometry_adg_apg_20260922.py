#!/usr/bin/env python3
"""073: guidance geometry (ADG norm-preserving / APG orthogonal projection) in place
of naive CFG extrapolation.

Design: docs/experiments/guidance_geometry_adg_apg_20260922.md
Contract: docs/experiments/guidance_geometry_adg_apg_20260922_contract.json

Vanilla guidance is   v = A + (c-1)*(A - B)   with A the conditional branch, B the
negative (or stored-null) branch. It rotates the prediction toward the prompt and
lengthens it at the same time; the length is what saturates the waveform. The two
interventions here are inference-only:

    ADG   v <- v * ((1-g) + g*||A||/||v||)      norm-preserving, g in [0,1]
    APG   v <- A + (c-1)*(d_perp + e*d_par)     orthogonal projection, e in [0,1]

g=0 and e=1 are numerically vanilla but NOT bit-identical (different float order),
so the replication gate runs the unpatched code path instead: cell G0 regenerates
the canonical c2p0_slot0 CFG3+fidelity8 audio and compares per-clip sha256 against
the 051 manifest with zero tolerance.

networks.py and eval.py are never edited: eval.py runs verbatim through runpy in
this process, with MeanAudio.ode_wrapper monkeypatched around it.

Usage:
    guidance_geometry_adg_apg_20260922.py                # run (resumable)
    guidance_geometry_adg_apg_20260922.py --validate-only # postflight
    guidance_geometry_adg_apg_20260922.py --smoke N       # N rows, pilot cells only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import runpy
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'eval'))

OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/guidance_geometry_adg_apg_20260922')
AUDIO_ROOT = OUT / '_audio'
CELLS_DIR = OUT / 'cells'
SUMMARY = OUT / 'summary.json'

EXP_ID = 'phase8_qwen_caption10s_multisent_noq_full_stage2_200000'
CKPT = ROOT / 'exps' / EXP_ID / f'{EXP_ID}_ema_final.pth'

MC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
SUBSET_TSV = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/'
                  'negprompt_ablation/musiccaps_subset1024.tsv')
BASELINE_MANIFEST = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/'
                         'loudness_aes_cfg3_20260911/audio_manifest.json')

FIDELITY8 = ('low quality recording, noisy, amateur, distorted, muffled, '
             'poor fidelity, hiss, lo-fi')
FAMILIES = {'N0': None, 'N8': FIDELITY8}

AES_BATCH = 32
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260922
HARD_STOP_FREE_BYTES = 20_000_000_000

# early-kill gate (docs section "Early-kill gate")
GATE_PQ = 0.28           # 2x the inference-seed PQ noise floor (0.142)
GATE_CREST = 0.5
GATE_CLAP_FLOOR = -0.005
LOUDNESS_GATE_LU = 0.5   # beyond this, the matched-loudness read is mandatory
SILENT_RATIO_FLAG = 2.0

GEOMETRIES = {
    'vanilla':        {'kind': 'vanilla'},
    'adg_g0.5':       {'kind': 'adg', 'gamma': 0.5, 'ref': 'A', 'axis': 'global'},
    'adg_g1.0':       {'kind': 'adg', 'gamma': 1.0, 'ref': 'A', 'axis': 'global'},
    'adg_g1.0_refB':  {'kind': 'adg', 'gamma': 1.0, 'ref': 'B', 'axis': 'global'},
    'adg_g1.0_frame': {'kind': 'adg', 'gamma': 1.0, 'ref': 'A', 'axis': 'frame'},
    'apg_e0.5':       {'kind': 'apg', 'eta': 0.5, 'axis': 'global'},
    'apg_e0.25':      {'kind': 'apg', 'eta': 0.25, 'axis': 'global'},
    'apg_e0.0':       {'kind': 'apg', 'eta': 0.0, 'axis': 'global'},
}
PILOT_GEOMETRIES = list(GEOMETRIES)


# ── cells ────────────────────────────────────────────────
def cell(name, *, family, cfg, geometry, tsv, stage):
    return {'name': name, 'family': family, 'cfg': float(cfg), 'geometry': geometry,
            'tsv': str(tsv), 'stage': stage}


def gate_cells():
    return [cell('G0__N8__cfg3.0__vanilla', family='N8', cfg=3.0,
                 geometry='vanilla', tsv=MC_TSV, stage='gate')]


def pilot_cells():
    cells = []
    for fam in FAMILIES:
        cells.append(cell(f'A__{fam}__cfg3.0__vanilla', family=fam, cfg=3.0,
                          geometry='vanilla', tsv=SUBSET_TSV, stage='pilot'))
        for geom in PILOT_GEOMETRIES:
            cells.append(cell(f'A__{fam}__cfg4.5__{geom}', family=fam, cfg=4.5,
                              geometry=geom, tsv=SUBSET_TSV, stage='pilot'))
    return cells


def full_cells(selected):
    """selected: {family: {'adg': geom_name, 'apg': geom_name}}"""
    cells = []
    for fam in FAMILIES:
        for cfg in (3.0, 4.5):
            if not (fam == 'N8' and cfg == 3.0):   # that one is the gate cell G0
                cells.append(cell(f'B__{fam}__cfg{cfg}__vanilla', family=fam, cfg=cfg,
                                  geometry='vanilla', tsv=MC_TSV, stage='full'))
            for kind in ('adg', 'apg'):
                geom = selected[fam][kind]
                cells.append(cell(f'B__{fam}__cfg{cfg}__{geom}', family=fam, cfg=cfg,
                                  geometry=geom, tsv=MC_TSV, stage='full'))
    return cells


def vanilla_partner(c):
    """the in-run vanilla cell every geometry cell is paired against"""
    if c['stage'] == 'pilot':
        return f"A__{c['family']}__cfg{c['cfg']}__vanilla"
    if c['family'] == 'N8' and c['cfg'] == 3.0:
        return 'G0__N8__cfg3.0__vanilla'
    return f"B__{c['family']}__cfg{c['cfg']}__vanilla"


# ── geometry patch ───────────────────────────────────────
def _norm(x, axis):
    dims = tuple(range(1, x.ndim)) if axis == 'global' else (-1,)
    return x.pow(2).sum(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)


def make_ode_wrapper(spec):
    import torch

    def ode_wrapper(self, t, r, latent, conditions, empty_conditions, cfg_strength, q=None):
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        r = r * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if q is None:
            q = torch.full((len(latent),), 10, dtype=torch.long, device=latent.device)
        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, r, conditions, q)
        a = self.predict_flow(latent, t, r, conditions, q)
        b = self.predict_flow(latent, t, r, empty_conditions, q)
        delta = a - b
        if spec['kind'] == 'apg':
            axis = spec['axis']
            a_hat = a / _norm(a, axis)
            dims = tuple(range(1, a.ndim)) if axis == 'global' else (-1,)
            par = (delta * a_hat).sum(dim=dims, keepdim=True) * a_hat
            perp = delta - par
            return a + (cfg_strength - 1.0) * (perp + spec['eta'] * par)
        v = a + (cfg_strength - 1.0) * delta
        if spec['kind'] == 'adg':
            axis = spec['axis']
            ref = a if spec['ref'] == 'A' else b
            scale = _norm(ref, axis) / _norm(v, axis)
            gamma = spec['gamma']
            return v * ((1.0 - gamma) + gamma * scale)
        raise SystemExit(f'[FAIL] unknown geometry kind {spec["kind"]}')

    return ode_wrapper


@contextmanager
def geometry(name):
    from meanaudio.model.networks import MeanAudio
    spec = GEOMETRIES[name]
    if spec['kind'] == 'vanilla':          # unpatched: the bit-exact reference path
        yield
        return
    original = MeanAudio.ode_wrapper
    MeanAudio.ode_wrapper = make_ode_wrapper(spec)
    try:
        yield
    finally:
        MeanAudio.ode_wrapper = original


# ── generation ───────────────────────────────────────────
def audio_dir(c):
    return AUDIO_ROOT / c['name']


def generate(c, limit=None):
    """Run eval.py verbatim (runpy) under the geometry patch."""
    out = audio_dir(c)
    tsv = Path(c['tsv'])
    if limit:                                   # smoke only; never in a contract run
        rows = tsv.read_text(encoding='utf-8').splitlines(True)
        tsv = OUT / f'_smoke_{limit}.tsv'
        tsv.write_text(''.join(rows[:limit + 1]), encoding='utf-8')
    if out.exists():
        # eval.py seeds one RNG for the whole run and skips clips that already
        # exist without drawing noise, so a partial dir cannot be topped up.
        shutil.rmtree(out)
    out.mkdir(parents=True)
    argv = ['eval.py', '--variant', 'meanaudio_s', '--model_path', str(CKPT),
            '--output', str(out), '--tsv', str(tsv), '--use_meanflow',
            '--num_steps', '25', '--cfg_strength', str(c['cfg']),
            '--no_text_attention_mask', '--encoder_name', 't5_clap',
            '--text_c_dim', '512', '--seed', '42', '--full_precision', '--no_q']
    neg = FAMILIES[c['family']]
    if neg:
        argv += ['--negative_prompt', neg]
    saved_argv, saved_cwd = sys.argv, os.getcwd()
    os.chdir(ROOT)
    sys.argv = argv
    # eval.py installs a log handler on import; runpy re-executes it once per cell,
    # so without this every line would be printed once per cell run so far.
    import logging
    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    root_logger.handlers = []
    try:
        with geometry(c['geometry']):
            runpy.run_path(str(ROOT / 'eval.py'), run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise SystemExit(f'[FAIL] eval.py exited {exc.code} for {c["name"]}')
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)
        logging.getLogger().handlers = saved_handlers
    return out, tsv


# ── scoring ──────────────────────────────────────────────
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def score_dir(gen_dir, rows, clap_model):
    import eval_metrics as em
    paths = [em.audio_path(gen_dir, cid) for cid, _ in rows]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f'[FAIL] {len(missing)} missing clips in {gen_dir} '
                         f'(first: {missing[0]})')
    clap = em.score_clap(rows, gen_dir, model=clap_model)
    aes, failed = em.score_aes(paths, batch_size=AES_BATCH)
    if failed:
        raise SystemExit(f'[FAIL] AES failed on {len(failed)} clips in {gen_dir}')
    level = em.score_level(paths)
    per = {}
    for cid, _ in rows:
        p = str(em.audio_path(gen_dir, cid))
        per[cid] = {'clap': clap[cid], **aes[p], **level[p]}
    return per


def aggregate(per):
    keys = ('clap', 'CE', 'CU', 'PC', 'PQ', 'lufs', 'rms_dbfs', 'crest')
    agg = {'n': len(per)}
    for k in keys:
        vals = [v[k] for v in per.values() if v.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None
    agg['crest_min'] = min((v['crest'] for v in per.values() if v.get('crest')), default=None)
    agg['silent_n'] = sum(v['silent'] for v in per.values())
    return agg


def score_cell(c, clap_model, limit=None):
    import eval_metrics as em
    path = CELLS_DIR / f'{c["name"]}.json'
    if path.exists():
        return json.loads(path.read_text())
    free = shutil.disk_usage(OUT).free
    if free < HARD_STOP_FREE_BYTES:
        raise SystemExit(f'[FAIL] {free} bytes free, below the registered hard stop')
    t0 = time.time()
    gen_dir, tsv = generate(c, limit=limit)
    rows = em.load_rows(tsv)
    per = score_dir(gen_dir, rows, clap_model)
    record = {'document_kind': 'guidance_geometry_cell_v1', **c,
              'tsv_used': str(tsv), 'n_rows': len(rows),
              'geometry_spec': GEOMETRIES[c['geometry']],
              'negative_prompt': FAMILIES[c['family']],
              'aggregates': aggregate(per), 'per_clip': per,
              'audio_sha256': {cid: sha256(em.audio_path(gen_dir, cid)) for cid, _ in rows},
              'wall_seconds': round(time.time() - t0, 1)}
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    tmp.replace(path)
    return record


# ── replication gate ─────────────────────────────────────
def replication_gate(gate_record):
    manifest = json.loads(BASELINE_MANIFEST.read_text())['audio_sha256']
    ours = gate_record['audio_sha256']
    shared = sorted(set(manifest) & set(ours))
    mismatched = [cid for cid in shared if manifest[cid] != ours[cid]]
    return {'reference': str(BASELINE_MANIFEST),
            'n_reference': len(manifest), 'n_ours': len(ours), 'n_compared': len(shared),
            'n_mismatch': len(mismatched), 'first_mismatches': mismatched[:10],
            'tolerated_mismatches': 0,
            'passed': len(shared) == len(manifest) == len(ours) and not mismatched}


def equivalence_check(n_clips=32):
    """gamma=0 and eta=1 must reproduce vanilla on the latent to <=1e-5 relative.
    Float ordering differs, so this is numeric, not bit-exact, by design."""
    import torch
    from meanaudio.model.networks import MeanAudio
    torch.manual_seed(20260922)
    rng = np.random.default_rng(20260922)
    worst = {}
    for name, spec in (('adg_gamma0', {'kind': 'adg', 'gamma': 0.0, 'ref': 'A', 'axis': 'global'}),
                       ('apg_eta1', {'kind': 'apg', 'eta': 1.0, 'axis': 'global'})):
        patched = make_ode_wrapper(spec)
        worst[name] = 0.0
        for _ in range(n_clips):
            a = torch.from_numpy(rng.standard_normal((1, 312, 20), dtype=np.float32))
            b = torch.from_numpy(rng.standard_normal((1, 312, 20), dtype=np.float32))
            c = float(rng.uniform(1.5, 5.0))

            class Stub:
                def predict_flow(self, latent, t, r, conditions, q):
                    return a if conditions == 'cond' else b
            v_ref = c * a + (1 - c) * b
            v_new = patched(Stub(), torch.tensor(0.5), torch.tensor(0.0),
                            torch.zeros(1, 312, 20), 'cond', 'uncond', c,
                            q=torch.zeros(1, dtype=torch.long))
            rel = (v_new - v_ref).abs().max().item() / max(v_ref.abs().max().item(), 1e-12)
            worst[name] = max(worst[name], rel)
    return {'tolerance': 1e-5, 'worst_relative_error': worst,
            'passed': all(v <= 1e-5 for v in worst.values()),
            'note': 'gamma=0 / eta=1 are algebraically vanilla but evaluated in a '
                    'different float order, so bit-exactness is not expected and not '
                    'required here; the bit-exact gate is G0 against the 051 manifest.'}


# ── loudness-matched read ────────────────────────────────
def loudness_matched(geo_record, van_record, clap_model):
    """Rescale every geometry clip to its paired vanilla clip's LUFS and rescore.
    ADG changes output amplitude by construction, and 063/065 showed PQ/CU rise as
    level falls while CLAP rises as it rises, so an unmatched delta cannot be read."""
    import eval_metrics as em
    import soundfile as sf
    src = audio_dir(geo_record)
    dst = AUDIO_ROOT / f'{geo_record["name"]}__lufsmatched'
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    ids = [cid for cid in geo_record['per_clip'] if cid in van_record['per_clip']]
    excluded = []
    for cid in ids:
        target = van_record['per_clip'][cid]['lufs']
        current = geo_record['per_clip'][cid]['lufs']
        if target is None or current is None:
            excluded.append([cid, 'non-finite loudness'])
            continue
        wav, sr = sf.read(src / f'{cid}.flac', dtype='float32', always_2d=True)
        gain = 10 ** ((target - current) / 20.0)
        out = wav * gain
        if float(np.max(np.abs(out))) > 0.999:
            excluded.append([cid, 'matching gain would clip'])
            continue
        sf.write(dst / f'{cid}.flac', out, sr)
    kept = [cid for cid in ids if (dst / f'{cid}.flac').exists()]
    captions = dict(em.load_rows(Path(geo_record['tsv_used'])))
    per = score_dir(dst, [(cid, captions[cid]) for cid in kept], clap_model)
    return {'n_matched': len(kept), 'n_excluded': len(excluded),
            'excluded': excluded[:20],
            'aggregates': aggregate(per),
            'paired': paired_deltas(per, van_record['per_clip'], kept),
            'audio_dir': str(dst)}


# ── analysis ─────────────────────────────────────────────
def paired_deltas(per_a, per_b, ids=None):
    ids = ids if ids is not None else sorted(set(per_a) & set(per_b))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {'n_paired': len(ids)}
    for key in ('clap', 'CE', 'CU', 'PC', 'PQ', 'crest', 'lufs'):
        d = np.array([per_a[i][key] - per_b[i][key] for i in ids
                      if per_a[i].get(key) is not None and per_b[i].get(key) is not None],
                     dtype=np.float64)
        if d.size == 0:
            out[key] = None
            continue
        boot = np.empty(BOOTSTRAP_N, dtype=np.float64)
        for start in range(0, BOOTSTRAP_N, 500):   # chunked: the full index matrix
            stop = min(start + 500, BOOTSTRAP_N)   # would be hundreds of MB per key
            idx = rng.integers(0, d.size, size=(stop - start, d.size))
            boot[start:stop] = d[idx].mean(axis=1)
        out[key] = {'mean_delta': float(d.mean()),
                    'ci95': [float(np.percentile(boot, 2.5)),
                             float(np.percentile(boot, 97.5))],
                    'frac_improved': float((d > 0).mean()), 'n': int(d.size)}
    return out


def contrast(geo, van, clap_model):
    ids = sorted(set(geo['per_clip']) & set(van['per_clip']))
    row = {'cell': geo['name'], 'vanilla': van['name'], 'family': geo['family'],
           'cfg': geo['cfg'], 'geometry': geo['geometry'],
           'paired': paired_deltas(geo['per_clip'], van['per_clip'], ids),
           'geometry_aggregates': geo['aggregates'],
           'vanilla_aggregates': van['aggregates'],
           'crest_min_delta': (None if geo['aggregates']['crest_min'] is None
                               else geo['aggregates']['crest_min'] - van['aggregates']['crest_min']),
           'silent_n': [geo['aggregates']['silent_n'], van['aggregates']['silent_n']]}
    dl = abs(row['paired']['lufs']['mean_delta'])
    row['loudness_gate'] = {'delta_lufs_mean': row['paired']['lufs']['mean_delta'],
                            'threshold_lu': LOUDNESS_GATE_LU,
                            'matched_read_required': dl > LOUDNESS_GATE_LU}
    row['silence_flag'] = (van['aggregates']['silent_n'] > 0 and
                           geo['aggregates']['silent_n'] >
                           SILENT_RATIO_FLAG * van['aggregates']['silent_n'])
    if row['loudness_gate']['matched_read_required'] and clap_model is not None:
        row['loudness_matched'] = loudness_matched(geo, van, clap_model)
    return row


def passes_early_kill(rows):
    for r in rows:
        if r['geometry'] == 'vanilla':
            continue
        p = r['paired']
        if not (p.get('clap') and p.get('PQ') and p.get('crest')):
            continue
        if p['clap']['mean_delta'] < GATE_CLAP_FLOOR:
            continue
        crest_ok = (p['crest']['mean_delta'] >= GATE_CREST and
                    (r['crest_min_delta'] or 0) >= 0)
        if p['PQ']['mean_delta'] >= GATE_PQ or crest_ok:
            return True, r['cell']
    return False, None


def select_for_full(rows):
    """max paired dPQ subject to dCLAP >= -0.005; ties broken by dcrest."""
    sel = {}
    for fam in FAMILIES:
        sel[fam] = {}
        for kind in ('adg', 'apg'):
            pool = [r for r in rows
                    if r['family'] == fam and r['geometry'].startswith(kind)
                    and r['paired']['clap']['mean_delta'] >= GATE_CLAP_FLOOR]
            if not pool:
                pool = [r for r in rows if r['family'] == fam and r['geometry'].startswith(kind)]
            best = max(pool, key=lambda r: (r['paired']['PQ']['mean_delta'],
                                            r['paired']['crest']['mean_delta']))
            sel[fam][kind] = best['geometry']
    return sel


# ── driver ───────────────────────────────────────────────
def write_summary(payload):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    tmp.replace(SUMMARY)


def run(smoke=None):
    import eval_metrics as em
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    clap_model = em.load_clap()

    equivalence = equivalence_check()
    if not equivalence['passed'] and not smoke:
        write_summary({'document_kind': 'guidance_geometry_summary_v1',
                       'status': 'failed_equivalence_check', 'equivalence': equivalence})
        raise SystemExit('[FAIL] gamma=0 / eta=1 do not reproduce vanilla numerically')

    records = {}
    gate = None
    if not smoke:
        for c in gate_cells():
            records[c['name']] = score_cell(c, clap_model)
        gate = replication_gate(records['G0__N8__cfg3.0__vanilla'])
        if not gate['passed']:
            write_summary({'document_kind': 'guidance_geometry_summary_v1',
                           'status': 'failed_replication_gate',
                           'equivalence': equivalence, 'replication_gate': gate})
            raise SystemExit('[FAIL] vanilla path did not reproduce the 051 baseline audio')

    for c in pilot_cells():
        records[c['name']] = score_cell(c, clap_model, limit=smoke)

    pilot_rows = [contrast(records[c['name']], records[vanilla_partner(c)], clap_model)
                  for c in pilot_cells() if c['geometry'] != 'vanilla']
    launched, trigger = passes_early_kill(pilot_rows)

    payload = {'document_kind': 'guidance_geometry_summary_v1',
               'experiment_id': 'guidance-geometry-adg-apg-20260922',
               'checkpoint': str(CKPT), 'checkpoint_sha256': sha256(CKPT),
               'smoke_rows': smoke,
               'equivalence': equivalence, 'replication_gate': gate,
               'geometries': GEOMETRIES,
               'early_kill': {'thresholds': {'pq': GATE_PQ, 'crest': GATE_CREST,
                                             'clap_floor': GATE_CLAP_FLOOR},
                              'stage_b_launched': bool(launched and not smoke),
                              'triggering_cell': trigger},
               'pilot': pilot_rows}
    write_summary(payload)

    if not launched or smoke:
        payload['status'] = 'completed_null_at_pilot' if not launched else 'completed_smoke'
        write_summary(payload)
        return

    selected = select_for_full(pilot_rows)
    payload['selected_for_full'] = selected
    write_summary(payload)
    done = {}
    for c in full_cells(selected):
        records[c['name']] = score_cell(c, clap_model)
        for x in full_cells(selected):
            if (x['geometry'] == 'vanilla' or x['name'] in done
                    or x['name'] not in records or vanilla_partner(x) not in records):
                continue
            done[x['name']] = contrast(records[x['name']],
                                       records[vanilla_partner(x)], clap_model)
        payload['full'] = [done[k] for k in sorted(done)]
        write_summary(payload)

    payload['fad'] = fad_block(records, selected)
    payload['status'] = 'completed'
    write_summary(payload)


def fad_block(records, selected):
    import eval_metrics as em
    rows = em.load_rows(MC_TSV)
    out = {}
    for c in full_cells(selected) + gate_cells():
        if c['name'] not in records:
            continue
        value, pairs = em.score_fad(rows, audio_dir(c), em.DEFAULT_REF_DIR,
                                    num_samples=2048, seed=42)
        out[c['name']] = {'fad': value, 'pairs': pairs}
    return out


def validate_only():
    if not SUMMARY.exists():
        raise SystemExit('[FAIL] summary.json missing')
    s = json.loads(SUMMARY.read_text())
    if s.get('status') not in ('completed', 'completed_null_at_pilot'):
        raise SystemExit(f'[FAIL] summary status {s.get("status")!r}')
    if not s['equivalence']['passed']:
        raise SystemExit('[FAIL] equivalence check not passed')
    if not (s.get('replication_gate') or {}).get('passed'):
        raise SystemExit('[FAIL] replication gate not passed')
    for row in s.get('pilot', []) + s.get('full', []):
        if row['loudness_gate']['matched_read_required'] and 'loudness_matched' not in row:
            raise SystemExit(f'[FAIL] {row["cell"]} tripped the loudness gate '
                             'without a matched-loudness read')
    print(json.dumps({'status': s['status'],
                      'stage_b_launched': s['early_kill']['stage_b_launched'],
                      'pilot_cells': len(s.get('pilot', [])),
                      'full_cells': len(s.get('full', []))}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--smoke', type=int, default=None,
                    help='rows per cell; pilot cells only, skips the replication gate')
    args = ap.parse_args()
    if args.validate_only:
        validate_only()
        return
    run(smoke=args.smoke)


if __name__ == '__main__':
    main()
