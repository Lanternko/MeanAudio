#!/usr/bin/env python
"""FAD on raw audio, with a cached reference-embedding table.

Mirrors scripts/eval/eval_metrics.py:score_fad bit for bit on the preprocessing
path (librosa.load 16 kHz mono -> PCM_16 wav -> sf.read float32 -> VGGish), so
the full-5521 cfg3 cell must reproduce 5.0836 (G0/FAD parity, tolerance 1e-3).

The reference side is the only part shared by every trial on a split, so its
VGGish embeddings are cached under a key derived from the sorted id list that
actually produced a valid pair. Gen-side embeddings are always recomputed.

    fad_cache.py --gen-dir DIR --tsv TSV --split NAME --out OUT.json
                 [--ref-dir /mnt/HDD/kojiek/musiccaps_reference]
                 [--cache-dir <scratch>/fad_cache] [--num-samples 2048] [--seed 42]
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.pop('MKL_NUM_THREADS', None)      # fork/thread pools hang with it (123 incident)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                  # noqa: E402

REF_DIR = common.REF_DIR
CACHE_VERSION = 'fad-ref-vggish-v1'


def ref_cache_key(ids):
    return hashlib.sha256((CACHE_VERSION + '\0' + '\n'.join(sorted(ids))).encode()).hexdigest()


def sampled_pairs(rows, num_samples, seed):
    """eval_metrics.score_fad's sample, including its exact skip rule."""
    if num_samples and num_samples < len(rows):
        random.seed(seed)
        return random.sample(rows, num_samples)
    return list(rows)


def get_ref_audio_path(clip_id, ref_dir):
    sys.path.insert(0, str(common.ROOT / 'research' / 'eval'))
    from phase4_eval import get_ref_audio_path as impl
    return impl(clip_id, str(ref_dir))


def write_wav16(src, dst):
    import librosa
    import soundfile as sf
    audio, _ = librosa.load(str(src), sr=16000, mono=True)
    sf.write(str(dst), audio, 16000)          # PCM_16, same as score_fad


def score(gen_dir, rows, ref_dir=REF_DIR, cache_dir=None, num_samples=2048, seed=42):
    import numpy as np
    from frechet_audio_distance import FrechetAudioDistance

    gen_dir = Path(gen_dir)
    ref_dir = Path(ref_dir)
    pairs = []
    for clip_id, _ in sampled_pairs(rows, num_samples, seed):
        ref = get_ref_audio_path(clip_id, ref_dir)
        gen = gen_dir / f'{clip_id}.flac'
        if not ref or not os.path.exists(ref) or not gen.exists():
            continue
        pairs.append((clip_id, ref, gen))
    if not pairs:
        raise SystemExit(f'[FAIL] no valid FAD pairs in {gen_dir}')

    ids = [p[0] for p in pairs]
    key = ref_cache_key(ids)
    cache_path = Path(cache_dir) / f'{key}.npy' if cache_dir else None
    tmp = Path(tempfile.mkdtemp(prefix='fad_cache_'))
    ref_tmp, gen_tmp = tmp / 'ref', tmp / 'gen'
    ref_tmp.mkdir()
    gen_tmp.mkdir()
    cached = bool(cache_path and cache_path.exists())
    try:
        if not cached:
            for clip_id, ref, _ in pairs:
                write_wav16(ref, ref_tmp / f'{clip_id}.wav')
        for clip_id, _, gen in pairs:
            write_wav16(gen, gen_tmp / f'{clip_id}.wav')
        fad = FrechetAudioDistance(use_pca=False, use_activation=False, verbose=False)
        value = float(fad.score(str(ref_tmp), str(gen_tmp), background_embds_path=str(cache_path)
                                if cache_path else None))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if cache_path and not cached:
        common.atomic_json(cache_path.with_suffix('.ids.json'),
                           {'key': key, 'n': len(ids), 'ids_sha256': hashlib.sha256(
                               '\n'.join(sorted(ids)).encode()).hexdigest(), 'ids': sorted(ids)})
    if not np.isfinite(value) or value < 0:
        raise SystemExit(f'[FAIL] FAD returned {value} for {gen_dir}')
    return {'value': value, 'pairs': len(pairs), 'sampled': len(sampled_pairs(rows, num_samples, seed)),
            'ref_cache': str(cache_path) if cache_path else None, 'ref_cache_hit': cached,
            'ref_ids_sha256': hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--gen-dir', required=True)
    p.add_argument('--tsv', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ref-dir', default=str(REF_DIR))
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--num-samples', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(argv)
    rows = common.rows_for_tsv(a.tsv)
    t0 = time.time()
    rec = score(a.gen_dir, rows, ref_dir=a.ref_dir, cache_dir=a.cache_dir,
                num_samples=a.num_samples, seed=a.seed)
    rec.update({'split': a.split, 'gen_dir': str(a.gen_dir), 'seconds': round(time.time() - t0, 1),
                'written_at': common.now_iso()})
    common.atomic_json(a.out, rec)
    print(json.dumps(rec, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
