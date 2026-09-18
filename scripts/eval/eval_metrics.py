#!/usr/bin/env python
"""Canonical metrics for a directory of generated audio (<id>.flac per TSV row).

Replaces ~/research/meanaudio_eval/phase4_eval.py for new work. That file stays
byte-identical because historical contracts/manifests pin it; its numbers and
this script's agree to the last bit on the same audio (verified 2026-09-18 on
slot0nm quarter CFG3+neg: CLAP 0.2314, AES CE/CU/PC/PQ 6.8336/7.5070/4.7384/7.4360).

Metrics
  CLAP   laion_clap HTSAT-base, music_speech_audioset ckpt, ONE file per forward
         (batch 1). laion_clap pads differently above batch 8, so batched CLAP
         sits +0.004..+0.025 higher and can flip rankings (062 rescore). Batch 1
         is the project standard; there is deliberately no batch-size flag.
  AES    Audiobox Aesthetics CE/CU/PC/PQ (batch-invariant, batch 32).
  level  integrated LUFS, RMS dBFS, peak, crest, silent (RMS < -45 dBFS).
         AES and CLAP both move with playback level (051/063), and sparse-prompt
         arms can collapse to silence without moving mean CLAP (slot0nm), so every
         eval records level alongside the scores.
  FAD    optional (--fad), VGGish at 16 kHz on a seeded subset, as phase4_eval.

Differences from phase4_eval.py
  - --tsv is required (phase4_eval silently defaulted to the 90k Jamendo TSV).
  - Missing or unscorable clips fail the run instead of shrinking the mean;
    --allow_missing downgrades that to a recorded count.
  - Writes per_clip.tsv and metrics.json (provenance: TSV sha256, git rev,
    scorer versions, n scored per metric) next to the usual metrics.txt.
  - FAD scratch dirs are per-run temp dirs, not a shared /tmp path.

Usage
  python scripts/eval/eval_metrics.py --gen_dir <run>/audio --tsv <tsv> --exp_name <name>

Library use (sweep drivers should import these rather than re-implement CLAP):
  from eval_metrics import load_rows, load_clap, score_clap, score_aes, score_level
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLAP_CKPT = ROOT / 'weights' / 'music_speech_audioset_epoch_15_esc_89.98.pt'
DEFAULT_OUT_DIR = ROOT / 'eval_output' / 'metrics'
DEFAULT_REF_DIR = '/mnt/HDD/kojiek/music_semantic_fidelity/original_audio'
SILENT_DBFS = -45.0
CLAP_METHOD = 'laion_clap HTSAT-base, batch 1 (= phase4_eval.compute_clap_score)'
AES_KEYS = ('CE', 'CU', 'PC', 'PQ')


# ── inputs ───────────────────────────────────────────────
def load_rows(tsv, limit=None):
    """[(id, caption), ...] in TSV order; same csv parsing as eval.py/phase4_eval."""
    rows = []
    with open(tsv, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            rows.append((row['id'], row['caption']))
            if limit and len(rows) >= limit:
                break
    ids = [i for i, _ in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f'[FAIL] duplicate ids in {tsv}')
    return rows


def audio_path(gen_dir, clip_id):
    return Path(gen_dir) / f'{clip_id}.flac'


# ── CLAP ─────────────────────────────────────────────────
def load_clap(ckpt=DEFAULT_CLAP_CKPT):
    import torch
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(str(ckpt))
    model.eval()
    return model.to('cuda' if torch.cuda.is_available() else 'cpu')


def score_clap(rows, gen_dir, model=None, progress=True):
    """{id: cosine(audio, caption)} with one file per forward. Missing files are
    left out; unreadable files raise."""
    import torch
    from tqdm import tqdm
    owned = model is None
    if owned:
        model = load_clap()
    per = {}
    with torch.no_grad():
        for clip_id, caption in tqdm(rows, desc='CLAP', disable=not progress):
            path = audio_path(gen_dir, clip_id)
            if not path.exists():
                continue
            ae = model.get_audio_embedding_from_filelist([str(path)], use_tensor=True)
            te = model.get_text_embedding([caption], use_tensor=True)
            per[clip_id] = float(torch.nn.functional.cosine_similarity(ae, te, dim=-1).item())
    if owned:
        del model
        torch.cuda.empty_cache()
    return per


# ── Audiobox Aesthetics ──────────────────────────────────
def _read_wav_sf(meta):
    # torchcodec/FFmpeg are not available in this env; read with soundfile.
    import soundfile as sf
    import torch
    wav, sr = sf.read(meta['path'], dtype='float32', always_2d=True)
    wav = torch.from_numpy(wav.T)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    return wav, sr


def score_aes(paths, batch_size=32, progress=True):
    """{path: {CE, CU, PC, PQ}}. A failing batch is retried file by file so one
    bad clip costs one clip, and the failures are returned instead of dropped."""
    import torch
    from tqdm import tqdm
    import audiobox_aesthetics.infer as aes_infer
    aes_infer.read_wav = _read_wav_sf
    from audiobox_aesthetics.infer import AesPredictor

    predictor = AesPredictor(checkpoint_pth=None, batch_size=batch_size)
    per, failed = {}, {}
    paths = [str(p) for p in paths]
    for i in tqdm(range(0, len(paths), batch_size), desc='AES', disable=not progress):
        chunk = paths[i:i + batch_size]
        try:
            results = list(zip(chunk, predictor.forward([{'path': p} for p in chunk])))
        except Exception:
            results = []
            for p in chunk:
                try:
                    results.append((p, predictor.forward([{'path': p}])[0]))
                except Exception as e:
                    failed[p] = repr(e)
        for p, r in results:
            per[p] = {k: float(r[k]) for k in AES_KEYS}
    del predictor
    torch.cuda.empty_cache()
    return per, failed


# ── level ────────────────────────────────────────────────
def _level_one(path):
    import soundfile as sf
    import pyloudnorm
    wav, sr = sf.read(path, dtype='float32', always_2d=True)
    mono = wav.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms_dbfs = 20 * math.log10(rms + 1e-9)
    lufs = float(pyloudnorm.Meter(sr).integrated_loudness(wav.astype(np.float64)))
    return {
        'lufs': lufs if math.isfinite(lufs) else None,
        'rms_dbfs': rms_dbfs,
        'peak': peak,
        'crest': peak / rms if rms > 0 else None,
        'silent': int(rms_dbfs < SILENT_DBFS),
    }


def score_level(paths, workers=16):
    paths = [str(p) for p in paths]
    with Pool(workers) as pool:
        return dict(zip(paths, pool.map(_level_one, paths, chunksize=64)))


# ── FAD ──────────────────────────────────────────────────
def score_fad(rows, gen_dir, ref_dir, num_samples=2048, seed=42):
    import librosa
    import soundfile as sf
    from frechet_audio_distance import FrechetAudioDistance
    sys.path.insert(0, str(Path.home() / 'research' / 'meanaudio_eval'))
    from phase4_eval import get_ref_audio_path

    if num_samples and num_samples < len(rows):
        random.seed(seed)
        sampled = random.sample(rows, num_samples)
    else:
        sampled = rows
    tmp = Path(tempfile.mkdtemp(prefix='eval_metrics_fad_'))
    ref_tmp, gen_tmp = tmp / 'ref', tmp / 'gen'
    ref_tmp.mkdir()
    gen_tmp.mkdir()
    pairs = 0
    try:
        for clip_id, _ in sampled:
            ref = get_ref_audio_path(clip_id, ref_dir)
            gen = audio_path(gen_dir, clip_id)
            if not ref or not os.path.exists(ref) or not gen.exists():
                continue
            for src, dst in ((ref, ref_tmp), (gen, gen_tmp)):
                audio, _ = librosa.load(str(src), sr=16000, mono=True)
                sf.write(dst / f'{clip_id}.wav', audio, 16000)
            pairs += 1
        fad = FrechetAudioDistance(use_pca=False, use_activation=False, verbose=False)
        return float(fad.score(str(ref_tmp), str(gen_tmp))), pairs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── provenance ───────────────────────────────────────────
def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _git_rev():
    try:
        rev = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', 'HEAD', '--', __file__]).returncode != 0
        return rev + ('+dirty' if dirty else '')
    except Exception:
        return None


def _versions():
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for pkg in ('torch', 'laion_clap', 'audiobox_aesthetics', 'pyloudnorm'):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return (float(np.mean(values)) if values else None), len(values)


def _write_atomic(path, text):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


# ── main ─────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(description='CLAP (batch 1) + Audiobox Aesthetics + level (+ FAD) for generated audio')
    p.add_argument('--gen_dir', required=True, help='directory of <id>.flac')
    p.add_argument('--tsv', required=True, help='TSV with id, caption (the captions CLAP is scored against)')
    p.add_argument('--exp_name', default='', help='output subdir name (default: gen_dir name, or its parent if it is "audio")')
    p.add_argument('--out_dir', default=str(DEFAULT_OUT_DIR))
    p.add_argument('--clap_ckpt', default=str(DEFAULT_CLAP_CKPT))
    p.add_argument('--aes_batch_size', type=int, default=32)
    p.add_argument('--skip_clap', action='store_true')
    p.add_argument('--skip_aes', action='store_true')
    p.add_argument('--skip_level', action='store_true')
    p.add_argument('--fad', action='store_true', help='also compute FAD (off by default)')
    p.add_argument('--fad_num_samples', '--num_samples', dest='fad_num_samples', type=int, default=2048,
                   help='FAD subset size only; CLAP/AES/level always use every row')
    p.add_argument('--ref_dir', default=DEFAULT_REF_DIR, help='reference audio root for FAD')
    p.add_argument('--seed', type=int, default=42, help='FAD subset seed')
    p.add_argument('--limit', type=int, default=None, help='score only the first N TSV rows (smoke test)')
    p.add_argument('--allow_missing', action='store_true',
                   help='record missing/unscorable clips instead of failing')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    gen_dir = Path(args.gen_dir)
    exp_name = args.exp_name or (gen_dir.parent.name if gen_dir.name == 'audio' else gen_dir.name)
    out = Path(args.out_dir) / exp_name
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.tsv, limit=args.limit)
    ids = [i for i, _ in rows]
    present = [i for i in ids if audio_path(gen_dir, i).exists()]
    present_set = set(present)
    missing = [i for i in ids if i not in present_set]
    print(f'[eval_metrics] {exp_name}: {len(present)}/{len(rows)} clips in {gen_dir}')
    if missing and not args.allow_missing:
        raise SystemExit(f'[FAIL] {len(missing)} TSV rows have no audio (first: {missing[:3]}); '
                         f'pass --allow_missing to score the rest')
    if not present:
        raise SystemExit('[FAIL] no audio to score')

    per = {i: {} for i in present}
    failures = {}
    summary = {}

    # level first: its worker pool must fork before CUDA is initialised
    if not args.skip_level:
        paths = {str(audio_path(gen_dir, i)): i for i in present}
        for p, r in score_level(list(paths)).items():
            per[paths[p]].update(r)
        lvl = [per[i] for i in present]
        summary['level_lufs_mean'], summary['level_lufs_n'] = _mean(r['lufs'] for r in lvl)
        summary['level_rms_dbfs_mean'], _ = _mean(r['rms_dbfs'] for r in lvl)
        summary['level_crest_mean'], _ = _mean(r['crest'] for r in lvl)
        summary['level_clipped_n'] = sum(r['peak'] >= 0.999 for r in lvl)
        summary['level_silent_n'] = sum(r['silent'] for r in lvl)
        lufs = summary['level_lufs_mean']
        print(f'  level: LUFS {"n/a" if lufs is None else f"{lufs:.2f}"}  silent(<{SILENT_DBFS:g} dBFS) '
              f'{summary["level_silent_n"]}  clipped {summary["level_clipped_n"]}')

    if not args.skip_clap:
        clap = score_clap([r for r in rows if r[0] in per], gen_dir, model=load_clap(args.clap_ckpt))
        for i, s in clap.items():
            per[i]['clap'] = s
        summary['clap_score'], n = _mean(clap.values())
        summary['clap_n'] = n
        print(f'  clap_score: {summary["clap_score"]:.4f}  (n={n})')

    if not args.skip_aes:
        paths = {str(audio_path(gen_dir, i)): i for i in present}
        aes, failed = score_aes(list(paths), batch_size=args.aes_batch_size)
        for p, r in aes.items():
            per[paths[p]].update(r)
        failures['aes'] = {paths[p]: e for p, e in failed.items()}
        for k in AES_KEYS:
            summary[f'aes_{k}'], n = _mean(r[k] for r in aes.values())
        summary['aes_n'] = n
        print('  ' + '  '.join(f'aes_{k}: {summary[f"aes_{k}"]:.4f}' for k in AES_KEYS) + f'  (n={n})')

    if args.fad:
        summary['fad'], summary['fad_pairs'] = score_fad(rows, gen_dir, args.ref_dir,
                                                         args.fad_num_samples, args.seed)
        print(f'  fad: {summary["fad"]:.4f}  (pairs={summary["fad_pairs"]})')

    bad = [k for k, v in summary.items() if isinstance(v, float) and not math.isfinite(v)]
    if bad:
        raise SystemExit(f'[FAIL] non-finite metrics: {bad}')
    unscored = sorted(failures.get('aes', {}))
    if unscored and not args.allow_missing:
        raise SystemExit(f'[FAIL] {len(unscored)} clips failed AES (first: {unscored[:3]})')

    # per_clip.tsv
    cols = ['id', 'clap', *AES_KEYS, 'lufs', 'rms_dbfs', 'peak', 'crest', 'silent']
    lines = ['\t'.join(cols)]
    for i in present:
        lines.append('\t'.join([i] + ['' if per[i].get(c) is None else repr(per[i][c]) for c in cols[1:]]))
    _write_atomic(out / 'per_clip.tsv', '\n'.join(lines) + '\n')

    meta = {
        'document_kind': 'eval_metrics_v1',
        'exp_name': exp_name,
        'gen_dir': str(gen_dir.resolve()),
        'tsv': str(Path(args.tsv).resolve()),
        'tsv_sha256': _sha256(args.tsv),
        'n_rows': len(rows),
        'n_present': len(present),
        'missing_ids': missing,
        'failures': failures,
        'metrics': summary,
        'clap_method': CLAP_METHOD,
        'clap_ckpt': str(args.clap_ckpt),
        'silent_threshold_dbfs': SILENT_DBFS,
        'fad_num_samples': args.fad_num_samples if args.fad else None,
        'limit': args.limit,
        'script': str(Path(__file__).resolve()),
        'git_rev': _git_rev(),
        'versions': _versions(),
        'written_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    }
    _write_atomic(out / 'metrics.json', json.dumps(meta, indent=1, ensure_ascii=False) + '\n')

    # metrics.txt keeps phase4_eval's layout so existing parsers keep working
    txt = [f'Experiment: {exp_name}', f'Test TSV: {args.tsv}', f'Generated audio: {args.gen_dir}',
           f'Test clips: {len(present)}', '─' * 40]
    order = sorted(summary, key=lambda k: (k.startswith('level_'), k.endswith('_n') or k == 'fad_pairs'))
    txt += [f'{k}: {summary[k]:.4f}' if isinstance(summary[k], float) else f'{k}: {summary[k]}'
            for k in order if summary[k] is not None]
    _write_atomic(out / 'metrics.txt', '\n'.join(txt) + '\n')
    print(f'[eval_metrics] wrote {out}/metrics.txt, metrics.json, per_clip.tsv')


if __name__ == '__main__':
    main()
