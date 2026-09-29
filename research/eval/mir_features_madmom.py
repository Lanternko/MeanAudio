"""madmom rhythm/key features for the MIR incremental-validity study.

Runs in ~/venvs/madmom (py3.9). Input: manifest TSV (set, key, path). Output TSV
keyed by (set, key). Audio is resampled to 44.1 kHz first because every madmom
model here was trained at 44.1 kHz and frames by sample count.

Features (see docs/experiments/mir_incremental_validity_musiceval_20260930.md):
  beat_act       mean RNN beat activation at DBN beats
  pulse_clarity  max normalised autocorrelation of the beat activation, 40-220 BPM lags
  ibi_cv         std/mean of inter-beat intervals (>= 4 beats)
  tempo_drift    |log(median IBI 2nd half / 1st half)| (>= 3 IBIs per half)
  downbeat_act   mean RNN downbeat activation at DBN downbeats (3/4 or 4/4)
  key_cnn_conf   max of the 24-way CNN key posterior

Usage:
    OMP_NUM_THREADS=1 ~/venvs/madmom/bin/python research/eval/mir_features_madmom.py \
        --manifest M.tsv --out F.tsv --workers 16
"""
import argparse
import csv
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

warnings.filterwarnings('ignore')
FPS = 100
SR = 44100
LAG_MIN, LAG_MAX = int(round(60 * FPS / 220)), int(round(60 * FPS / 40))
COLS = ['beat_act', 'pulse_clarity', 'ibi_cv', 'tempo_drift', 'downbeat_act', 'key_cnn_conf', 'n_beats']
P = {}


def init():
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    from madmom.features.key import CNNKeyRecognitionProcessor
    P['beat'] = RNNBeatProcessor()
    P['dbn'] = DBNBeatTrackingProcessor(fps=FPS)
    P['down'] = RNNDownBeatProcessor()
    P['ddbn'] = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=FPS)
    P['key'] = CNNKeyRecognitionProcessor()


def load(path):
    from madmom.audio.signal import Signal
    wav, sr = sf.read(path, dtype='float32', always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    return Signal(wav, sample_rate=SR, num_channels=1)


def at(act, times):
    idx = np.clip(np.round(np.asarray(times) * FPS).astype(int), 0, len(act) - 1)
    return float(act[idx].mean()) if len(idx) else np.nan


def feats(row):
    out = {c: np.nan for c in COLS}
    try:
        sig = load(row['path'])
        act = P['beat'](sig)
        beats = P['dbn'](act)
        out['n_beats'] = len(beats)
        out['beat_act'] = at(act, beats)
        a = act - act.mean()
        n = len(a)
        if n > LAG_MAX + 1 and np.any(a):
            spec = np.fft.rfft(a, 2 * n)
            ac = np.fft.irfft(spec * np.conj(spec))[:n]
            out['pulse_clarity'] = float((ac[LAG_MIN:LAG_MAX + 1] / ac[0]).max())
        if len(beats) >= 4:
            ibi = np.diff(beats)
            out['ibi_cv'] = float(ibi.std() / ibi.mean())
            h = len(ibi) // 2
            if h >= 3:
                out['tempo_drift'] = float(abs(np.log(np.median(ibi[h:]) / np.median(ibi[:h]))))
        dact = P['down'](sig)
        try:
            db = P['ddbn'](dact)
            downs = db[db[:, 1] == 1, 0] if len(db) else []
            out['downbeat_act'] = at(dact[:, 1], downs)
        except Exception:
            pass
        out['key_cnn_conf'] = float(np.max(P['key'](sig)))
    except Exception as e:
        print(f"[WARN] {row['key']}: {e!r}", file=sys.stderr)
    return row['set'], row['key'], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.manifest), delimiter='\t'))
    done = set()
    if os.path.exists(args.out):
        done = {(r['set'], r['key']) for r in csv.DictReader(open(args.out), delimiter='\t')}
    todo = [r for r in rows if (r['set'], r['key']) not in done]
    print(f'{len(rows)} rows, {len(done)} done, {len(todo)} todo', flush=True)
    new = not os.path.exists(args.out)
    with open(args.out, 'a') as f, Pool(args.workers, initializer=init) as pool:
        if new:
            f.write('set\tkey\t' + '\t'.join(COLS) + '\n')
        for i, (s, k, o) in enumerate(pool.imap_unordered(feats, todo, chunksize=4)):
            f.write(f'{s}\t{k}\t' + '\t'.join(f'{o[c]:.6g}' for c in COLS) + '\n')
            if i % 200 == 0:
                f.flush()
                print(f'{i}/{len(todo)}', flush=True)


if __name__ == '__main__':
    main()
