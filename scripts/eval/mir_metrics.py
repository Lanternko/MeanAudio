#!/usr/bin/env python
"""MIR readings for a directory of generated audio, next to eval_metrics.py.

Four features that added validity over AES for human OVL on MusicEval
(docs/experiments/results/mir_incremental_validity_musiceval_20260930_results.md):
  pulse_clarity   madmom RNNBeat activation autocorrelation max, 40-220 BPM lags  (higher = clearer pulse)
  ibi_cv          std/mean of DBN inter-beat intervals, >= 4 beats                (lower = steadier beat)
  key_cnn_conf    max of madmom's 24-way CNN key posterior                        (higher = clearer key)
  chroma_entropy  mean normalised chroma_cqt entropy over non-silent frames       (lower = clearer tonality)
Definitions are identical to research/eval/mir_features_{madmom,essentia}.py.
They are readings next to AES, not headline metrics: on MusicEval the beat
features separated systems but not clips within a system, and they have not been
validated on differences between our own arms.

madmom needs py3.9 (~/venvs/madmom) and chroma uses librosa 1.0 (~/venvs/mir),
so the driver (any python, normally ~/venvs/dac) runs this same file as a worker
in those two envs and merges the results.

Usage
  # one finished mc_mf25 cell (reads ids and label from its *_REPORT.json)
  python scripts/eval/mir_metrics.py --cell ~/eval_output_nvme/<label>
  # any audio dir
  python scripts/eval/mir_metrics.py --gen_dir <run>/audio --tsv <tsv> --exp_name <name> --out_dir <dir>

Output: <out_dir>/<exp_name>/{mir_metrics.json, mir_metrics.txt, mir_per_clip.tsv}
Clips where a feature is undefined (silence, < 4 beats) count as NaN and are
reported per feature (*_n); missing audio fails unless --allow_missing.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')
MADMOM_PY = os.path.expanduser('~/venvs/madmom/bin/python')
MIR_PY = os.path.expanduser('~/venvs/mir/bin/python')
FEATS = ['pulse_clarity', 'ibi_cv', 'key_cnn_conf', 'chroma_entropy']
MADMOM_COLS = ['pulse_clarity', 'ibi_cv', 'key_cnn_conf', 'n_beats']
CHROMA_COLS = ['chroma_entropy']
SR44 = 44100
FPS = 100
LAG_MIN, LAG_MAX = int(round(60 * FPS / 220)), int(round(60 * FPS / 40))
HOP = 512
P = {}


# ── workers (run inside ~/venvs/madmom or ~/venvs/mir) ──
def _init_madmom():
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.key import CNNKeyRecognitionProcessor
    P['beat'] = RNNBeatProcessor()
    P['dbn'] = DBNBeatTrackingProcessor(fps=FPS)
    P['key'] = CNNKeyRecognitionProcessor()


def _madmom_one(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    from madmom.audio.signal import Signal
    out = {c: np.nan for c in MADMOM_COLS}
    try:
        wav, sr = sf.read(path, dtype='float32', always_2d=True)
        wav = wav.mean(1)
        if sr != SR44:
            g = np.gcd(sr, SR44)
            wav = resample_poly(wav, SR44 // g, sr // g).astype(np.float32)
        sig = Signal(wav, sample_rate=SR44, num_channels=1)
        act = P['beat'](sig)
        beats = P['dbn'](act)
        out['n_beats'] = len(beats)
        a = act - act.mean()
        n = len(a)
        if n > LAG_MAX + 1 and np.any(a):
            spec = np.fft.rfft(a, 2 * n)
            ac = np.fft.irfft(spec * np.conj(spec))[:n]
            out['pulse_clarity'] = float((ac[LAG_MIN:LAG_MAX + 1] / ac[0]).max())
        if len(beats) >= 4:
            ibi = np.diff(beats)
            out['ibi_cv'] = float(ibi.std() / ibi.mean())
        out['key_cnn_conf'] = float(np.max(P['key'](sig)))
    except Exception as e:
        print('[WARN] %s: %r' % (path, e), file=sys.stderr)
    return path, out


def _chroma_one(path):
    import soundfile as sf
    import librosa
    out = {c: np.nan for c in CHROMA_COLS}
    try:
        wav, sr = sf.read(path, dtype='float32', always_2d=True)
        wav = wav.mean(1)
        ch = librosa.feature.chroma_cqt(y=wav, sr=sr, hop_length=HOP)
        rms = librosa.feature.rms(y=wav, hop_length=HOP)[0][:ch.shape[1]]
        ok = rms > 1e-3
        if ok.sum() >= 10:
            c = ch[:, ok] / (ch[:, ok].sum(0, keepdims=True) + 1e-12)
            out['chroma_entropy'] = float((-(c * np.log(c + 1e-12)).sum(0) / np.log(12)).mean())
    except Exception as e:
        print('[WARN] %s: %r' % (path, e), file=sys.stderr)
    return path, out


def worker(kind, paths_file, out_file, workers):
    paths = [l.rstrip('\n') for l in open(paths_file) if l.strip()]
    fn, init, cols = ((_madmom_one, _init_madmom, MADMOM_COLS) if kind == 'madmom'
                      else (_chroma_one, None, CHROMA_COLS))
    with open(out_file, 'w') as f, Pool(workers, initializer=init) as pool:
        f.write('path\t' + '\t'.join(cols) + '\n')
        for i, (p, o) in enumerate(pool.imap_unordered(fn, paths, chunksize=4)):
            f.write(p + '\t' + '\t'.join('%.6g' % o[c] for c in cols) + '\n')
            if i % 500 == 0:
                print('[mir %s] %d/%d' % (kind, i, len(paths)), flush=True)


# ── driver ─────────────────────────────────────────────
def _versions():
    v = {}
    for name, py, mod in (('madmom', MADMOM_PY, 'madmom'), ('librosa', MIR_PY, 'librosa')):
        try:
            v[name] = subprocess.run([py, '-W', 'ignore', '-c', 'import %s;print(%s.__version__)' % (mod, mod)],
                                     capture_output=True, text=True, check=True).stdout.strip()
        except Exception as e:
            v[name] = 'unknown (%r)' % e
    return v


def score_mir(paths, madmom_workers=12, chroma_workers=4):
    """{path: {feature: float}} for the four readings (+ n_beats)."""
    tmp = Path(tempfile.mkdtemp(prefix='mir_metrics_'))
    (tmp / 'paths.txt').write_text('\n'.join(paths) + '\n')
    me = str(Path(__file__).resolve())
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    procs = [subprocess.Popen([py, me, '_worker', kind, str(tmp / 'paths.txt'), str(tmp / f'{kind}.tsv'), str(w)],
                              env=env)
             for kind, py, w in (('madmom', MADMOM_PY, madmom_workers), ('chroma', MIR_PY, chroma_workers))]
    rcs = [p.wait() for p in procs]
    if any(rcs):
        raise SystemExit(f'[FAIL] MIR workers exited {rcs} (scratch: {tmp})')
    per = {p: {} for p in paths}
    for kind in ('madmom', 'chroma'):
        for r in csv.DictReader(open(tmp / f'{kind}.tsv'), delimiter='\t'):
            per[r.pop('path')].update({k: float(v) for k, v in r.items()})
    short = [p for p, r in per.items() if len(r) != len(MADMOM_COLS) + len(CHROMA_COLS)]
    if short:
        raise SystemExit(f'[FAIL] {len(short)} clips missing from worker output (scratch: {tmp})')
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    return per


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_metrics import load_rows, audio_path, _sha256, _git_rev, _write_atomic
    import datetime

    ap = argparse.ArgumentParser(description='MIR readings (pulse clarity, beat steadiness, key/chroma clarity)')
    ap.add_argument('--cell', help='finished mc_mf25 cell dir (has <label>_REPORT.json and audio/)')
    ap.add_argument('--gen_dir')
    ap.add_argument('--tsv')
    ap.add_argument('--exp_name', default='')
    ap.add_argument('--out_dir')
    ap.add_argument('--madmom_workers', type=int, default=12)
    ap.add_argument('--chroma_workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=None, help='score only the first N TSV rows (smoke test)')
    ap.add_argument('--allow_missing', action='store_true')
    ap.add_argument('--force', action='store_true', help='rescore even if mir_metrics.json exists')
    args = ap.parse_args(argv)

    if args.cell:
        cell = Path(args.cell)
        reports = sorted(cell.glob('*_REPORT.json'))
        if len(reports) != 1:
            raise SystemExit(f'[FAIL] expected one *_REPORT.json in {cell}, found {len(reports)}')
        rep = json.loads(reports[0].read_text())
        gen_dir, tsv, exp_name, out_dir = cell / 'audio', rep['score_tsv'], rep['label'], cell
    else:
        if not (args.gen_dir and args.tsv and args.out_dir):
            raise SystemExit('[FAIL] pass --cell, or all of --gen_dir --tsv --out_dir')
        gen_dir, tsv, out_dir = Path(args.gen_dir), args.tsv, Path(args.out_dir)
        exp_name = args.exp_name or (gen_dir.parent.name if gen_dir.name == 'audio' else gen_dir.name)
    out = Path(out_dir) / exp_name
    if (out / 'mir_metrics.json').exists() and not args.force:
        print(f'[mir_metrics] SKIP exists: {out}/mir_metrics.json')
        return
    out.mkdir(parents=True, exist_ok=True)

    ids = [i for i, _ in load_rows(tsv, limit=args.limit)]
    present = [i for i in ids if audio_path(gen_dir, i).exists()]
    missing = sorted(set(ids) - set(present))
    print(f'[mir_metrics] {exp_name}: {len(present)}/{len(ids)} clips in {gen_dir}', flush=True)
    if missing and not args.allow_missing:
        raise SystemExit(f'[FAIL] {len(missing)} TSV rows have no audio (first: {missing[:3]})')
    if not present:
        raise SystemExit('[FAIL] no audio to score')

    paths = {str(audio_path(gen_dir, i)): i for i in present}
    per = {paths[p]: r for p, r in score_mir(list(paths), args.madmom_workers, args.chroma_workers).items()}

    summary = {}
    for k in FEATS:
        v = np.array([per[i][k] for i in present])
        ok = np.isfinite(v)
        summary[f'mir_{k}'] = float(v[ok].mean()) if ok.any() else None
        summary[f'mir_{k}_n'] = int(ok.sum())
    summary['mir_lt4_beats_n'] = int(sum(per[i]['n_beats'] < 4 for i in present))
    print('  ' + '  '.join(f'{k}: {summary["mir_" + k]:.4f} (n={summary["mir_" + k + "_n"]})'
                           for k in FEATS if summary['mir_' + k] is not None), flush=True)

    cols = FEATS + ['n_beats']
    lines = ['id\t' + '\t'.join(cols)]
    for i in present:
        lines.append(i + '\t' + '\t'.join('' if not np.isfinite(per[i][c]) else repr(per[i][c]) for c in cols))
    _write_atomic(out / 'mir_per_clip.tsv', '\n'.join(lines) + '\n')

    meta = {
        'document_kind': 'mir_metrics_v1',
        'exp_name': exp_name,
        'gen_dir': str(Path(gen_dir).resolve()),
        'tsv': str(Path(tsv).resolve()),
        'tsv_sha256': _sha256(tsv),
        'n_rows': len(ids), 'n_present': len(present), 'missing_ids': missing,
        'metrics': summary,
        'definitions': 'docs/experiments/mir_incremental_validity_musiceval_20260930.md',
        'limit': args.limit,
        'script': str(Path(__file__).resolve()),
        'script_sha256': _sha256(__file__),
        'git_rev': _git_rev(),
        'versions': _versions(),
        'written_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    }
    _write_atomic(out / 'mir_metrics.json', json.dumps(meta, indent=1, ensure_ascii=False) + '\n')
    txt = [f'Experiment: {exp_name}', f'Test TSV: {tsv}', f'Generated audio: {gen_dir}',
           f'Test clips: {len(present)}', '─' * 40]
    txt += [f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' for k, v in summary.items() if v is not None]
    _write_atomic(out / 'mir_metrics.txt', '\n'.join(txt) + '\n')
    print(f'[mir_metrics] wrote {out}/mir_metrics.txt, mir_metrics.json, mir_per_clip.tsv')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_worker':
        worker(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        main()
