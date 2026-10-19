#!/usr/bin/env python3
"""126d: SongEval (arXiv 2505.10793, ASLP-lab/SongEval) on the 126/126b/126c stimulus sets.

Design: docs/experiments/metric_win_tie_126d_songeval_20261020.md
Judging rule: same as 126 (d = score(case) - score(clean base); tie if |d| <= m, m = 0.10 S;
pass = 80 %).  S_i = range of 5 TTM-system clean means for each SongEval dim i.

Modes (run sequentially):
  score-126c   SongEval on 36,366 pre-built 126c stimuli
  score-126b   SongEval on 87,653 pre-built 126b stimuli (with identity gate check)
  score-126a   SongEval on 126a clips (regenerate from R115 source + transform)
  score-partb  Part B: detune control (B1) + karaoke LUFS comparison (B2)
  analyze      Compute S, judge win/tie/lose, write summary.json

Default (no mode argument): run all modes in order.

SongEval setup (pinned):
  repo commit: 848fb2ff3a2a9d64dcb20d46f07238c28abd7add
  ckpt sha256: 0cb1eb8be02b96e9949ca3a2b66997c5ac7d9da3567240df4dbfc2048eb98efd
  MuQ model:   OpenMuQ/MuQ-large-msd-iter  rev 0562a57814f6f8bbd9fdea0a25921a2fce1a841a
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.pop('MKL_NUM_THREADS', None)

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

REPO = Path('/home/kojiek/MeanAudio')
sys.path.insert(0, str(REPO / 'scripts/eval'))
sys.path.insert(0, str(REPO / 'scripts/eval/songeval_src'))
import metric_win_tie_126a as A  # noqa: E402
import metric_win_tie_126b as B  # noqa: E402

EV = Path('/home/kojiek/eval_output_nvme')
OUT = EV / 'metric_win_tie_126d_20261020'
OUT_126C = EV / 'metric_win_tie_126c_20261019'
OUT_126B = EV / 'metric_win_tie_126b_20261016'
OUT_126A = EV / 'metric_win_tie_126a_20261016'
R115 = EV / 'ttm_metrics_115_20261009'
SONGEVAL_SRC = REPO / 'scripts/eval/songeval_src'
CKPT = SONGEVAL_SRC / 'ckpt/model.safetensors'
CKPT_SHA = '0cb1eb8be02b96e9949ca3a2b66997c5ac7d9da3567240df4dbfc2048eb98efd'
MUQ_REPO = 'OpenMuQ/MuQ-large-msd-iter'
MUQ_REVISION = '0562a57814f6f8bbd9fdea0a25921a2fce1a841a'

REPORT = REPO / 'docs/experiments/results/metric_win_tie_126d_summary.json'
SE_DIMS = ('se_coh', 'se_mem', 'se_nat', 'se_cla', 'se_mus')
SE_SR = 24000
BATCH = 16
MIN_FREE = 5 * 2 ** 30  # 5 GB hard floor
M_FRAC, PASS = 0.10, 0.80
SYSTEMS = A.SYSTEMS  # {'real': 'musiccaps_real', ...}
KARAOKE_DIR = EV / 'ttm_midi_karaoke_meva_20261012'
log = A.log


# ── SongEval model ──
def load_songeval(device='cuda'):
    from muq import MuQ
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    # Verify ckpt integrity
    h = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    if h != CKPT_SHA:
        raise SystemExit(f'CKPT sha mismatch: got {h}')

    config = OmegaConf.load(SONGEVAL_SRC / 'config.yaml')
    gen = instantiate(config.generator)
    state = load_file(str(CKPT), device='cpu')
    gen.load_state_dict(state, strict=False)
    gen = gen.to(device).eval()

    muq = MuQ.from_pretrained(MUQ_REPO, revision=MUQ_REVISION)
    muq = muq.to(device).eval()
    return muq, gen


def score_songeval_batch(wavs_24k, muq, gen, device):
    """Score a list of 24kHz float32 numpy arrays. Returns (N, 5) float32 array."""
    import torch
    max_len = max(len(w) for w in wavs_24k)
    padded = np.zeros((len(wavs_24k), max_len), dtype=np.float32)
    for i, w in enumerate(wavs_24k):
        padded[i, :len(w)] = w
    x = torch.from_numpy(padded).to(device)
    with torch.no_grad():
        feat = muq(x, output_hidden_states=True).last_hidden_state
        scores = gen(feat).cpu().numpy()
    return scores.astype(np.float32)


def load_24k(path):
    x, sr = sf.read(str(path), dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SE_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SE_SR)
    return x


def score_files(paths, muq, gen, device, progress=True):
    """Score a list of audio file paths. Returns dict path -> (5,) array."""
    results = {}
    for i in range(0, len(paths), BATCH):
        batch_paths = paths[i:i + BATCH]
        wavs = [load_24k(p) for p in batch_paths]
        scores = score_songeval_batch(wavs, muq, gen, device)
        for p, s in zip(batch_paths, scores):
            results[str(p)] = s
        if progress and (i // BATCH) % 50 == 0:
            log(f'  scored {min(i + BATCH, len(paths))}/{len(paths)}')
    return results


def write_tsv(path, rows, dims=SE_DIMS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['key', *dims])
        for key, scores in rows.items():
            w.writerow([key, *[f'{x:.6f}' for x in scores]])


def read_tsv(path, dims=SE_DIMS):
    """Returns dict key -> np.array(5)"""
    result = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            result[row['key']] = np.array([float(row[d]) for d in dims])
    return result


# ── score 126c ──
def score_126c(muq, gen, device):
    out = OUT / '126c_songeval.tsv'
    if out.is_file():
        log('126c songeval already done, skipping')
        return
    log('=== scoring 126c ===')
    stimuli = OUT_126C / 'stimuli'
    items = json.loads((stimuli / 'items.json').read_text())
    # Build path→key mapping
    pk = {r['path']: r['key'] for r in items}
    paths = sorted(set(r['path'] for r in items))
    paths = [p for p in paths if Path(p).is_file()]
    log(f'  {len(paths)} clips')
    scored = score_files(paths, muq, gen, device)
    # Re-key by item key
    rows = {}
    for r in items:
        if r['path'] in scored:
            rows[r['key']] = scored[r['path']]
    write_tsv(out, rows)
    log(f'  wrote {out}')


# ── score 126b ──
def score_126b(muq, gen, device):
    out = OUT / '126b_songeval.tsv'
    if out.is_file():
        log('126b songeval already done, skipping')
        return
    log('=== scoring 126b ===')
    stimuli = OUT_126B / 'stimuli'
    items = json.loads((stimuli / 'items.json').read_text())
    pk = {r['path']: r['key'] for r in items}
    paths = sorted(set(r['path'] for r in items))
    paths = [p for p in paths if Path(p).is_file()]
    log(f'  {len(paths)} clips')
    scored = score_files(paths, muq, gen, device)
    rows = {}
    for r in items:
        if r['path'] in scored:
            rows[r['key']] = scored[r['path']]
    write_tsv(out, rows)
    log(f'  wrote {out}')
    _identity_gate_126b(rows, items)


def _identity_gate_126b(se_rows, items):
    """Spot-check ~200 126b clips against stored AES PQ scores (batch composition drift gate)."""
    # Read stored AES scores from 126b units
    aes_stored = {}
    for d in (OUT_126B / 'units').iterdir():
        tsv = d / 'scores.tsv'
        if tsv.is_file():
            for row in csv.DictReader(open(tsv), delimiter='\t'):
                aes_stored[row['key']] = float(row['PQ'])

    # Re-score ~200 clips with AES to verify identity
    from eval_metrics import score_aes
    sample_items = [r for r in items if r['key'] in aes_stored][:200]
    if not sample_items:
        log('  WARNING: no 126b stored AES scores found for identity gate')
        return

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Symlink audio for AES batch
        paths_batch = []
        for r in sample_items:
            dst = td / f'{r["key"]}.flac'
            dst.symlink_to(r['path'])
            paths_batch.append(dst)
        aes_new, _ = score_aes(paths_batch, progress=False)
        aes_new = {Path(p).stem: v['PQ'] for p, v in aes_new.items()}

    diffs = []
    for key in aes_new:
        if key in aes_stored:
            diffs.append(abs(aes_new[key] - aes_stored[key]))
    if not diffs:
        log('  WARNING: no matched keys for identity gate')
        return
    max_diff = max(diffs)
    threshold = 1e-3
    log(f'  identity gate: max AES PQ diff = {max_diff:.2e} (threshold {threshold:.1e}), n={len(diffs)}')
    if max_diff > threshold:
        log(f'  WARNING: identity gate exceeded threshold {max_diff:.2e} > {threshold:.1e} -- investigate before reporting')
    else:
        log(f'  identity gate PASSED')


# ── score 126a ──
def _126a_unit_done(d):
    return (d / 'done.json').is_file() and (d / 'songeval.tsv').is_file()


def score_126a(muq, gen, device):
    log('=== scoring 126a ===')
    # Read the list of IDs used in 126a (from clean/real unit)
    clean_real = OUT_126A / 'units/clean/real/scores.tsv'
    ids = [row['id'] for row in csv.DictReader(open(clean_real), delimiter='\t')]
    log(f'  N={len(ids)} clips per (variant, system)')

    # Iterate variant-major (same order as 126a units())
    todo = [(v['name'], s) for v in A.variants() for s in A.SYSTEMS]
    done = sum(1 for vname, sys in todo
               if _126a_unit_done(OUT / '126a' / vname / sys))
    log(f'  {done}/{len(todo)} units already done')

    for k, (vname, system) in enumerate(todo):
        udir = OUT / '126a' / vname / system
        if _126a_unit_done(udir):
            continue
        if shutil.disk_usage(OUT).free < MIN_FREE:
            raise SystemExit('disk below minimum free')
        log(f'[{k + 1}/{len(todo)}] {vname}/{system}')
        _score_126a_unit(vname, system, ids, muq, gen, device, udir)


def _score_126a_unit(vname, system, ids, muq, gen, device, udir):
    src_dir = R115 / SYSTEMS[system] / 'lvl30/audio'
    var = A.VARIANTS[vname]
    t0 = time.time()

    wavs_24k = []
    valid_ids = []
    for id_ in ids:
        src = src_dir / f'{id_}.flac'
        if not src.is_file():
            continue
        x, sr = sf.read(str(src), dtype='float64')
        assert x.ndim == 1, src
        y, _ = A.transform(x, var, A.rng_for(system, vname, id_), int(sr))
        y_f32 = y.astype(np.float32)
        # Resample to 24kHz for SongEval
        if sr != SE_SR:
            y_24k = librosa.resample(y_f32, orig_sr=int(sr), target_sr=SE_SR)
        else:
            y_24k = y_f32
        wavs_24k.append(y_24k)
        valid_ids.append(id_)

    # Batch score
    rows = {}
    for i in range(0, len(wavs_24k), BATCH):
        batch_wavs = wavs_24k[i:i + BATCH]
        batch_ids = valid_ids[i:i + BATCH]
        scores = score_songeval_batch(batch_wavs, muq, gen, device)
        for id_, s in zip(batch_ids, scores):
            rows[id_] = s

    udir.mkdir(parents=True, exist_ok=True)
    write_tsv(udir / 'songeval.tsv', rows)
    (udir / 'done.json').write_text(json.dumps({'n': len(rows), 'seconds': time.time() - t0,
                                                'vname': vname, 'system': system}))
    log(f'  done {vname}/{system} n={len(rows)} {time.time() - t0:.0f}s')


# ── Part B ──
def score_partb(muq, gen, device):
    log('=== Part B ===')
    _partb1_detune_control(muq, gen, device)
    _partb2_karaoke_lufs(muq, gen, device)


def _partb1_detune_control(muq, gen, device):
    out = OUT / 'partb1_detune_control.tsv'
    if out.is_file():
        log('  Part B1 already done')
        return
    log('  Part B1: detune control (+X-X vs +2X-X)')
    from eval_metrics import score_aes

    cache_dir = OUT_126C / 'stimuli/cache'
    windows_json = OUT_126C / 'stimuli/windows.json'
    if not windows_json.is_file():
        log('  WARNING: 126c windows.json not found, skip B1')
        return
    windows = json.loads(windows_json.read_text())
    # Filter windows that have vocal track
    win_ids = [w['win'] for w in windows if (cache_dir / f'{w["win"]}.npz').is_file()][:100]
    log(f'  using {len(win_ids)} windows')

    cases = {
        'pv0': (50, -50),      # +50 then -50 = vocal_pv0 (processing reference)
        'pv_2xminus': (100, -50),   # +100 then -50 = net +50c
        'pv_2x_cancel': (100, -100),  # +100 then -100 = net 0c (more processing)
    }

    rows_all = []
    for win_id in win_ids:
        npz = np.load(cache_dir / f'{win_id}.npz')
        if 'vocals' not in npz:
            continue
        voc = npz['vocals'].astype(np.float64)
        mix = npz['mixture'].astype(np.float64)
        acc = mix - voc  # accompaniment
        sr = 16000

        def rb(x, cents):
            """rubberband via ffmpeg"""
            with tempfile.TemporaryDirectory() as td:
                src_p = Path(td) / 'a.wav'
                dst_p = Path(td) / 'b.wav'
                sf.write(str(src_p), x, sr, subtype='FLOAT')
                ratio = 2 ** (cents / 1200)
                subprocess.run(
                    ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                     '-i', str(src_p), '-af', f'rubberband=pitch={ratio:.8f}:pitchq=quality',
                     '-c:a', 'pcm_f32le', str(dst_p)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                y, _ = sf.read(str(dst_p), dtype='float64')
            return np.pad(y, (0, max(len(x) - len(y), 0)))[:len(x)]

        def lufs_gain(x, target_lufs=-30.0):
            import pyloudnorm
            meter = pyloudnorm.Meter(sr)
            L = meter.integrated_loudness(x)
            if not math.isfinite(L):
                return x
            return x * 10 ** ((target_lufs - L) / 20)

        # Compute processed vocals for each case
        case_wavs = {}
        case_wavs['clean'] = lufs_gain(mix)
        for case_name, (c1, c2) in cases.items():
            voc_proc = rb(rb(voc, c1), c2)
            mix_proc = acc + voc_proc
            case_wavs[case_name] = lufs_gain(mix_proc)

        for case_name, wav in case_wavs.items():
            wav_24k = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SE_SR)
            rows_all.append({'win': win_id, 'case': case_name, 'wav_24k': wav_24k, 'wav': wav})

    # Score SongEval in batches
    wavs = [r['wav_24k'] for r in rows_all]
    se_all = []
    for i in range(0, len(wavs), BATCH):
        se_all.extend(list(score_songeval_batch(wavs[i:i + BATCH], muq, gen, device)))

    # Score AES
    import tempfile as tmpmod
    with tmpmod.TemporaryDirectory() as td:
        td = Path(td)
        paths = []
        for j, r in enumerate(rows_all):
            p = td / f'{j}.flac'
            sf.write(str(p), r['wav'], 16000, format='FLAC', subtype='PCM_24')
            paths.append(p)
        aes_res, _ = score_aes(paths, progress=False)
        aes_by_idx = {Path(p).stem: v for p, v in aes_res.items()}

    # Write result
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['win', 'case', 'PQ', 'CE', 'CU', 'PC', *SE_DIMS])
        for j, r in enumerate(rows_all):
            aes = aes_by_idx.get(str(j), {})
            se = se_all[j]
            w.writerow([r['win'], r['case'],
                        f'{aes.get("PQ", float("nan")):.6f}',
                        f'{aes.get("CE", float("nan")):.6f}',
                        f'{aes.get("CU", float("nan")):.6f}',
                        f'{aes.get("PC", float("nan")):.6f}',
                        *[f'{x:.6f}' for x in se]])
    log(f'  Part B1 done: {len(rows_all)} clips -> {out}')


def _partb2_karaoke_lufs(muq, gen, device):
    out = OUT / 'partb2_karaoke_lufs.tsv'
    if out.is_file():
        log('  Part B2 already done')
        return
    log('  Part B2: karaoke LUFS comparison')
    if not (KARAOKE_DIR / 'manifest.tsv').is_file():
        log('  WARNING: karaoke manifest not found, skip B2')
        return

    from eval_metrics import score_aes

    manifest = list(csv.DictReader(open(KARAOKE_DIR / 'manifest.tsv'), delimiter='\t'))
    # Filter for karaoke entries with original/demucs_drums_off/demucs_vocals_off cases
    target_ops = {'original', 'demucs_drums_off', 'demucs_vocals_off'}
    kar_rows = [r for r in manifest
                if r['key'].startswith('kar__') and any(op in r['key'] for op in target_ops)]
    log(f'  {len(kar_rows)} karaoke clips for B2')

    def gain_inv(path, gain_db):
        x, sr = sf.read(str(path), dtype='float32')
        if x.ndim > 1:
            x = x.mean(axis=1)
        scale = 10 ** (-float(gain_db) / 20)
        return (x * scale).astype(np.float32), sr

    rows_all = []
    for r in kar_rows:
        p = KARAOKE_DIR / 'audio' / f'{r["key"]}.flac'
        if not p.is_file():
            continue
        # Condition 1: normalized (-30 LUFS, as stored)
        x_norm, sr = sf.read(str(p), dtype='float32')
        if x_norm.ndim > 1:
            x_norm = x_norm.mean(axis=1)
        rows_all.append({'key': r['key'], 'lufs_cond': 'norm30', 'wav': x_norm, 'sr': sr,
                         'gain_db': float(r['gain_db'])})
        # Condition 2: unnormalized (original Codex LUFS)
        x_unnorm, _ = gain_inv(p, r['gain_db'])
        rows_all.append({'key': r['key'], 'lufs_cond': 'unnorm', 'wav': x_unnorm, 'sr': sr,
                         'gain_db': float(r['gain_db'])})

    # SongEval scoring
    wavs_24k = []
    for r in rows_all:
        sr = r['sr']
        x = r['wav']
        if sr != SE_SR:
            x = librosa.resample(x, orig_sr=sr, target_sr=SE_SR)
        wavs_24k.append(x)

    se_all = []
    for i in range(0, len(wavs_24k), BATCH):
        se_all.extend(list(score_songeval_batch(wavs_24k[i:i + BATCH], muq, gen, device)))

    # AES scoring
    import tempfile as tmpmod
    with tmpmod.TemporaryDirectory() as td:
        td = Path(td)
        paths = []
        for j, r in enumerate(rows_all):
            p = td / f'{j}.flac'
            sf.write(str(p), r['wav'], r['sr'], format='FLAC', subtype='PCM_24')
            paths.append(p)
        aes_res, _ = score_aes(paths, progress=False)
        aes_by_idx = {Path(p).stem: v for p, v in aes_res.items()}

    # Write
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['key', 'lufs_cond', 'gain_db', 'PQ', 'CE', 'CU', 'PC', *SE_DIMS])
        for j, r in enumerate(rows_all):
            aes = aes_by_idx.get(str(j), {})
            se = se_all[j]
            w.writerow([r['key'], r['lufs_cond'], f'{r["gain_db"]:.3f}',
                        f'{aes.get("PQ", float("nan")):.6f}',
                        f'{aes.get("CE", float("nan")):.6f}',
                        f'{aes.get("CU", float("nan")):.6f}',
                        f'{aes.get("PC", float("nan")):.6f}',
                        *[f'{x:.6f}' for x in se]])
    log(f'  Part B2 done: {len(rows_all)} clips -> {out}')


# ── analysis ──
def analyze():
    log('=== analyze ===')
    # Load 126a summary to get S for AES/MEva (for reference)
    a126_report = json.loads((REPO / 'docs/experiments/results/metric_win_tie_126a_summary.json').read_text())
    S_aes = a126_report.get('S', {})

    # Compute S_songeval from 126a clean clips
    log('  computing SongEval S values from 126a clean...')
    sys_means = {}
    for sys_name in A.SYSTEMS:
        tsv = OUT / '126a/clean' / sys_name / 'songeval.tsv'
        if not tsv.is_file():
            log(f'  WARNING: missing clean SongEval for {sys_name}')
            continue
        rows = read_tsv(tsv)
        sys_means[sys_name] = np.array(list(rows.values())).mean(0)  # (5,)

    if len(sys_means) == 5:
        mat = np.stack(list(sys_means.values()))  # (5, 5)
        S_se = {d: float(mat[:, i].max() - mat[:, i].min()) for i, d in enumerate(SE_DIMS)}
        log(f'  SongEval S = {S_se}')
    else:
        S_se = {d: 1.0 for d in SE_DIMS}
        log(f'  WARNING: using S=1 (missing systems)')
    (OUT / 's_values.json').write_text(json.dumps({'S_songeval': S_se, 'S_aes': S_aes}))

    # Judge 126c
    results_126c = _judge_126c(S_se)
    # Judge 126b
    results_126b = _judge_126b(S_se)
    # Judge 126a
    results_126a = _judge_126a(S_se)
    # Part B summary
    results_partb = _analyze_partb()

    summary = {
        'S_songeval': S_se,
        'S_aes': S_aes,
        'system_means_songeval': {k: v.tolist() for k, v in sys_means.items()},
        '126c': results_126c,
        '126b': results_126b,
        '126a': results_126a,
        'partb': results_partb,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    REPORT.write_text(json.dumps(summary, indent=2))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    log(f'  wrote {REPORT}')
    return summary


def _judge_with_se(d_arr, S_se, kind):
    """Given array of (N,5) deltas, return {dim: {'mean_d': float, 'pass_pct': float}} using win/tie rule."""
    results = {}
    for i, dim in enumerate(SE_DIMS):
        S = S_se.get(dim, 1.0)
        m = M_FRAC * S
        d = d_arr[:, i]
        mean_d = float(np.mean(d))
        if kind == 'tie':
            pass_pct = float(np.mean(np.abs(d) <= m))
        elif kind == 'lose':
            pass_pct = float(np.mean(d < -m))
        else:
            pass_pct = float(np.mean(np.abs(d) <= m))
        results[dim] = {'mean_d': mean_d, 'pass_pct': pass_pct, 'passed': pass_pct >= PASS}
    return results


def _judge_126c(S_se):
    se_tsv = OUT / '126c_songeval.tsv'
    if not se_tsv.is_file():
        return {}
    se = read_tsv(se_tsv)  # key -> (5,)
    items = json.loads((OUT_126C / 'stimuli/items.json').read_text())
    # Group by case, reference is mix (case_name matches the base)
    by_case = {}
    for r in items:
        by_case.setdefault(r.get('case', r.get('kind', 'unknown')), []).append(r)

    # Build (case, kind) -> list of (d_5,)
    from_items = {}
    # Find clean base per window
    clean_keys = {}
    for r in items:
        if r.get('case') == 'ref' or r.get('kind') == 'ref':
            clean_keys[r.get('win', r.get('source', ''))] = r['key']

    results = {}
    by_case_kind = {}
    for r in items:
        win = r.get('win', r.get('source', ''))
        case = r.get('case', r.get('name', ''))
        kind = r.get('kind', '')
        base_key = clean_keys.get(win)
        if base_key and case != 'ref' and r['key'] in se and base_key in se:
            d = se[r['key']] - se[base_key]
            by_case_kind.setdefault((case, kind), []).append(d)

    for (case, kind), ds in by_case_kind.items():
        d_arr = np.stack(ds)
        results[case] = {'kind': kind, **_judge_with_se(d_arr, S_se, kind)}

    return results


def _judge_126b(S_se):
    se_tsv = OUT / '126b_songeval.tsv'
    if not se_tsv.is_file():
        return {}
    se = read_tsv(se_tsv)
    items = json.loads((OUT_126B / 'stimuli/items.json').read_text())

    # Use same logic as 126b.py: d = score(case) - score(clean base of same source+excerpt)
    clean_keys = {}
    for r in items:
        if r.get('case') == 'clean' or r.get('kind') == 'clean':
            clean_keys[(r['source'], r.get('excerpt', r.get('block', '')))] = r['key']

    results = {}
    by_case_kind = {}
    for r in items:
        src = r['source']
        exc = r.get('excerpt', r.get('block', ''))
        case = r.get('case', r.get('name', ''))
        kind = r.get('kind', '')
        base_key = clean_keys.get((src, exc))
        if base_key and case not in ('clean',) and r['key'] in se and base_key in se:
            d = se[r['key']] - se[base_key]
            by_case_kind.setdefault((case, kind), []).append(d)

    for (case, kind), ds in by_case_kind.items():
        d_arr = np.stack(ds)
        results[case] = {'kind': kind, **_judge_with_se(d_arr, S_se, kind)}

    return results


def _judge_126a(S_se):
    # For each variant, read SongEval scores and compute d vs clean
    clean_scores = {}
    for sys in A.SYSTEMS:
        tsv = OUT / '126a/clean' / sys / 'songeval.tsv'
        if tsv.is_file():
            clean_scores[sys] = read_tsv(tsv)

    results = {}
    for vname, var in A.VARIANTS.items():
        if vname == 'clean':
            continue
        kind = var['kind']
        ds = []
        for sys in A.SYSTEMS:
            tsv = OUT / '126a' / vname / sys / 'songeval.tsv'
            if not tsv.is_file() or sys not in clean_scores:
                continue
            se_case = read_tsv(tsv)
            se_clean = clean_scores[sys]
            for id_, case_scores in se_case.items():
                if id_ in se_clean:
                    ds.append(case_scores - se_clean[id_])
        if ds:
            d_arr = np.stack(ds)
            results[vname] = {'kind': kind, **_judge_with_se(d_arr, S_se, kind)}

    return results


def _analyze_partb():
    out = {}
    # Part B1: detune control
    b1 = OUT / 'partb1_detune_control.tsv'
    if b1.is_file():
        rows = list(csv.DictReader(open(b1), delimiter='\t'))
        for case_name in ('pv0', 'pv_2xminus', 'pv_2x_cancel'):
            case_rows = [r for r in rows if r['case'] == case_name]
            clean_rows = [r for r in rows if r['case'] == 'clean']
            if case_rows and clean_rows:
                clean_pq = {r['win']: float(r['PQ']) for r in clean_rows}
                ds = [float(r['PQ']) - clean_pq.get(r['win'], float('nan'))
                      for r in case_rows if r['win'] in clean_pq]
                ds = [d for d in ds if math.isfinite(d)]
                out[f'b1_{case_name}_PQ_mean'] = float(np.mean(ds)) if ds else None

    # Part B2: karaoke LUFS
    b2 = OUT / 'partb2_karaoke_lufs.tsv'
    if b2.is_file():
        rows = list(csv.DictReader(open(b2), delimiter='\t'))
        for lufs_cond in ('norm30', 'unnorm'):
            cond_rows = {r['key']: r for r in rows if r['lufs_cond'] == lufs_cond}
            for op in ('demucs_drums_off', 'demucs_vocals_off'):
                orig_keys = [k for k in cond_rows if k.endswith('__original')]
                remove_keys = [k for k in cond_rows if f'__{op}' in k]
                ds = []
                for rk in remove_keys:
                    song = rk.split(f'__{op}')[0]
                    ok = song + '__original'
                    if ok in cond_rows:
                        ds.append(float(cond_rows[rk]['PQ']) - float(cond_rows[ok]['PQ']))
                if ds:
                    out[f'b2_{lufs_cond}_{op}_PQ_mean'] = float(np.mean(ds))
    return out


# ── main / preflight ──
def preflight():
    errors = []
    if not CKPT.is_file():
        errors.append(f'SongEval ckpt not found: {CKPT}')
    else:
        h = hashlib.sha256(CKPT.read_bytes()).hexdigest()
        if h != CKPT_SHA:
            errors.append(f'CKPT sha mismatch: got {h}, want {CKPT_SHA}')
    if shutil.disk_usage(str(EV)).free < MIN_FREE:
        errors.append(f'disk too full: {shutil.disk_usage(str(EV)).free / 2**30:.1f} GB free')
    if not (OUT_126C / 'stimuli/items.json').is_file():
        errors.append(f'126c stimuli not found')
    if not (OUT_126B / 'stimuli/items.json').is_file():
        errors.append(f'126b stimuli not found')
    if errors:
        for e in errors:
            print(f'PREFLIGHT ERROR: {e}', flush=True)
        return 75  # EX_TEMPFAIL: queue retries
    OUT.mkdir(parents=True, exist_ok=True)
    print('PREFLIGHT OK', flush=True)
    return 0


def validate():
    if not REPORT.is_file():
        print(f'VALIDATE FAIL: report not found: {REPORT}', flush=True)
        return 1
    report = json.loads(REPORT.read_text())
    if not report.get('generated_at'):
        print('VALIDATE FAIL: missing generated_at', flush=True)
        return 1
    print(f'VALIDATE OK: {REPORT}', flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--mode', choices=['score-126c', 'score-126b', 'score-126a',
                                       'score-partb', 'analyze', 'all'],
                    default='all')
    ap.add_argument('--mir-worker', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.preflight:
        sys.exit(preflight())
    if args.validate_only:
        sys.exit(validate())

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device={device}')
    OUT.mkdir(parents=True, exist_ok=True)

    if args.mode in ('score-126c', 'all'):
        muq, gen = load_songeval(device)
        score_126c(muq, gen, device)
    if args.mode in ('score-126b', 'all'):
        if args.mode == 'all':
            pass  # already loaded
        else:
            muq, gen = load_songeval(device)
        score_126b(muq, gen, device)
    if args.mode in ('score-126a', 'all'):
        if args.mode not in ('score-126c', 'score-126b'):
            muq, gen = load_songeval(device)
        score_126a(muq, gen, device)
    if args.mode in ('score-partb', 'all'):
        if args.mode not in ('score-126c', 'score-126b', 'score-126a'):
            muq, gen = load_songeval(device)
        score_partb(muq, gen, device)
    if args.mode in ('analyze', 'all'):
        analyze()


if __name__ == '__main__':
    main()
