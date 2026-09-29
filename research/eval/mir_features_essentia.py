"""essentia + librosa rhythm/tonal features for the MIR incremental-validity study.

Runs in ~/venvs/mir (essentia, librosa). Input: manifest TSV (set, key, path).

Features (see docs/experiments/mir_incremental_validity_musiceval_20260930.md):
  rhythm_conf     RhythmExtractor2013(multifeature) confidence   (44.1 kHz)
  danceability    Danceability                                     (44.1 kHz)
  key_strength    KeyExtractor(profileType='edma') strength        (44.1 kHz)
  dissonance      mean frame Dissonance over non-silent frames     (44.1 kHz)
  key_stability   share of 5 s windows (hop 2.5 s) whose Krumhansl key equals the
                  whole-clip key                                   (native sr, librosa chroma_cqt)
  chroma_entropy  mean normalised chroma entropy over non-silent frames

Usage:
    OMP_NUM_THREADS=1 ~/venvs/mir/bin/python research/eval/mir_features_essentia.py \
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
SR = 44100
COLS = ['rhythm_conf', 'danceability', 'key_strength', 'dissonance', 'key_stability', 'chroma_entropy']
HOP = 512
# Krumhansl-Kessler profiles
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PROF = np.stack([np.roll(MAJ, i) for i in range(12)] + [np.roll(MIN, i) for i in range(12)])
PROF = (PROF - PROF.mean(1, keepdims=True)) / PROF.std(1, keepdims=True)
E = {}


def init():
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    import essentia.standard as es
    E['rhythm'] = es.RhythmExtractor2013(method='multifeature')
    E['dance'] = es.Danceability(sampleRate=SR)
    E['key'] = es.KeyExtractor(profileType='edma', sampleRate=SR)
    E['win'] = es.Windowing(type='hann')
    E['spec'] = es.Spectrum()
    E['peaks'] = es.SpectralPeaks(orderBy='frequency', minFrequency=20, maxFrequency=8000,
                                  magnitudeThreshold=1e-5, sampleRate=SR)
    E['diss'] = es.Dissonance()
    E['frames'] = es.FrameGenerator


def ks_key(ch):
    v = ch.mean(1)
    if not np.any(v):
        return -1
    v = (v - v.mean()) / (v.std() + 1e-12)
    return int(np.argmax(PROF @ v))


def feats(row):
    import librosa
    out = {c: np.nan for c in COLS}
    try:
        wav, sr = sf.read(row['path'], dtype='float32', always_2d=True)
        wav = wav.mean(1)
        g = np.gcd(sr, SR)
        w44 = resample_poly(wav, SR // g, sr // g).astype(np.float32) if sr != SR else wav

        bpm, ticks, conf, _, _ = E['rhythm'](w44)
        out['rhythm_conf'] = float(conf)
        out['danceability'] = float(E['dance'](w44)[0])
        _, _, strength = E['key'](w44)
        out['key_strength'] = float(strength)
        d = []
        for fr in E['frames'](w44, frameSize=2048, hopSize=1024, startFromZero=True):
            if np.sqrt(np.mean(fr ** 2)) < 1e-3:
                continue
            f, m = E['peaks'](E['spec'](E['win'](fr)))
            if len(f) > 1:
                d.append(E['diss'](f, m))
        out['dissonance'] = float(np.mean(d)) if d else np.nan

        ch = librosa.feature.chroma_cqt(y=wav, sr=sr, hop_length=HOP)
        rms = librosa.feature.rms(y=wav, hop_length=HOP)[0][:ch.shape[1]]
        ok = rms > 1e-3
        if ok.sum() >= 10:
            c = ch[:, ok] / (ch[:, ok].sum(0, keepdims=True) + 1e-12)
            out['chroma_entropy'] = float((-(c * np.log(c + 1e-12)).sum(0) / np.log(12)).mean())
            gk = ks_key(ch[:, ok])
            win, hop = int(5 * sr / HOP), int(2.5 * sr / HOP)
            keys = []
            for s in range(0, max(ch.shape[1] - win, 0) + 1, hop):
                m = ok[s:s + win]
                if m.sum() >= win // 2:
                    keys.append(ks_key(ch[:, s:s + win][:, m]))
            if keys:
                out['key_stability'] = float(np.mean(np.array(keys) == gk))
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
