"""e00 factor decomposition: timbre x filter x drums x score, 12 tonics C4-B4 (+ pitch extension abs vs rel lowpass C3-B5).
Same lock/scorer as ce_pitch_octaves_20261118 (-30 LUFS, 16 kHz, tier 2, empty prompt). Post-hoc, not preregistered."""
import os, sys, json
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ.pop('MKL_NUM_THREADS', None)
from pathlib import Path
sys.path.insert(0, str(Path.home() / 'MeanAudio/scripts/eval'))
import multiprocessing as mp
import numpy as np
import attack_metric_matrix_197 as A
import ce_push_196 as D
C = A.C
E = sys.modules['ce_weakness_195'] if 'ce_weakness_195' in sys.modules else __import__('ce_weakness_195')
OUT = Path(__file__).resolve().parent
SR, NS = E.SR, E.NS
T = NS / SR
P0 = np.clip(np.asarray(D.e00_params(), float), 0, 1)
WTS = (P0[0:3] + 1e-3) / (P0[0:3] + 1e-3).sum()
NLEN = 0.05 + P0[6] * 0.95
AA, DD, SS, RR = 0.002 + P0[7] * 0.2, 0.01 + P0[8] * 0.5, P0[9], 0.01 + P0[10] * 0.5
STEP = 1.0 / (1.0 + P0[11] * 7.0)
TILT, LP = -12 + P0[13] * 18, 500.0 * 16.0 ** P0[14]
LK, LS, LH = P0[15], P0[16], P0[17]
DYN, CENTS = P0[18], P0[19] * 30.0
E00_MIDI = 12 * (2 + min(int(P0[5] * 4), 3) + 1) + int(P0[4] * 12) % 12   # 61
TRI = [0, 4, 7]
SCALE = [0, 2, 4, 5, 7, 9, 11, 12]


def score_notes(kind, tonic, chord=(0, 4, 7)):
    TRI = list(chord)
    """-> list of (onset, nlen, midi, velfactor) ; velfactor uses e00's _VELR sequence by note index."""
    out = []
    ons = np.arange(0.0, T - 0.05, STEP)
    rng = np.random.default_rng(1119 + tonic)
    if kind == 'e00':
        out = [(on, NLEN, tonic + TRI[k % 3]) for k, on in enumerate(ons)]
    elif kind == 'short':
        out = [(on, 0.20, tonic + TRI[k % 3]) for k, on in enumerate(ons)]
    elif kind == 'grid':          # on the 120 BPM drum grid, same overlap ratio
        ons = np.arange(0.0, T - 0.05, 0.25)
        out = [(on, NLEN * 0.25 / STEP, tonic + TRI[k % 3]) for k, on in enumerate(ons)]
    elif kind == 'block':         # whole triad every 3 steps, same note count
        for j, on in enumerate(ons[::3]):
            out += [(on, NLEN + 2 * STEP * 0.5, tonic + i) for i in TRI]
    elif kind == 'prog':          # I V vi IV, each chord 2 arpeggio cycles
        for k, on in enumerate(ons):
            r, third = A.PROG['maj'][(k // 6) % 4]
            out.append((on, NLEN, tonic + r + [0, third, 7][k % 3]))
    elif kind == 'root':
        out = [(on, NLEN, tonic) for on in ons]
    elif kind == 'rscale':
        out = [(on, NLEN, tonic + int(rng.choice(SCALE))) for on in ons]
    return [(on, nl, m, 1.0 - DYN * E._VELR[k % 256]) for k, (on, nl, m) in enumerate(out)]


def osc_buf(notes, timbre):
    buf = np.zeros(NS)
    for on, nl, m, vf in notes:
        f = 440.0 * 2 ** ((m - 69) / 12)
        i0 = int(on * SR)
        L = min(int((nl + RR) * SR), NS - i0)
        if L < 8:
            continue
        t = np.arange(L) / SR
        xp = np.maximum.accumulate(np.array([0.0, AA, AA + DD, nl, nl + RR]) + np.arange(5) * 1e-6)
        env = np.interp(t, xp, [0.0, 1.0, SS, SS, 0.0])
        if timbre == 'e00osc':
            fs = [f] + ([f * 2 ** (CENTS / 1200)] if CENTS > 0.5 else [])
            w = 0.0
            for ff in fs:
                ph = ff * t
                w = w + WTS[0] * np.sin(2 * np.pi * ph) + WTS[1] * (2 * (ph % 1) - 1) + WTS[2] * np.sign(np.sin(2 * np.pi * ph))
            w = w / len(fs)
        elif timbre == 'sine':
            w = np.sin(2 * np.pi * f * t)
        buf[i0:i0 + L] += w * env * vf
    return buf


def piano_buf(notes, bank):
    nn = [(on, on + nl, int(m), int(np.clip(round(100 * vf), 20, 127))) for on, nl, m, vf in notes]
    x = A.MS.synth(nn, bank, 0, 0)
    i0 = int(A.MS.LEAD * SR)
    return np.asarray(x[i0:i0 + NS], np.float64)


def drums(r0):
    dr = np.zeros(NS)
    tt = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 * tt + 75 * 0.03 * (1 - np.exp(-tt / 0.03)))) * np.exp(-tt / 0.05)
    snare = E._NOISE[:len(tt)] * np.exp(-tt / 0.04)
    tt2 = np.arange(int(0.05 * SR)) / SR
    hat = np.diff(E._NOISE[:len(tt2) + 1]) * np.exp(-tt2 / 0.01)
    for pat, per, off in ((kick * LK * r0 * 2.5, 0.5, 0.0), (snare * LS * r0 * 1.5, 1.0, 0.5), (hat * LH * r0 * 2.0, 0.25, 0.0)):
        for on in np.arange(off, T, per):
            i0 = int(on * SR)
            m = min(len(pat), NS - i0)
            if m > 0:
                dr[i0:i0 + m] += pat[:m]
    return dr


def filt(buf, lp):
    X_ = np.fft.rfft(buf)
    f_ = np.fft.rfftfreq(NS, 1 / SR)
    g = 10 ** (TILT * np.log2(np.maximum(f_, 50.0) / 1000.0) / 20) / np.sqrt(1 + (f_ / lp) ** 8)
    return np.fft.irfft(X_ * g, NS)


def render(k):
    timbre, fmode, dr, sc, tonic, ch = k
    if sc == 'none':
        pb = np.zeros(NS); r0 = 0.05
    else:
        notes = score_notes(sc, tonic, E._CH[ch])
        pb = (A.MS.synth([(on, on + nl, int(m), int(np.clip(round(100 * vf), 20, 127))) for on, nl, m, vf in notes], 'generaluser', GM[timbre], 0)[int(A.MS.LEAD * SR):int(A.MS.LEAD * SR) + NS].astype(np.float64)
              if timbre in GM else osc_buf(notes, timbre))
        r0 = float(np.sqrt(np.mean(pb ** 2))) or 0.05
    buf = pb + (drums(r0) if dr else 0.0)
    if fmode == 'abs':
        buf = filt(buf, LP)
    elif fmode == 'rel':
        buf = filt(buf, LP * 2 ** ((tonic - E00_MIDI) / 12))
    return np.asarray(C.lock(buf), np.float32)


GM = {'piano': 0, 'violin': 40, 'guitar': 24, 'organ': 19, 'flute': 73, 'strings': 48}
TIMBRES = ['e00osc', 'sine'] + list(GM)
SCORES = ['e00', 'short', 'grid', 'block', 'prog', 'root', 'rscale']
TON = list(range(60, 72))

if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    ref = np.asarray(E.synth_E(list(P0)), np.float32)
    mine = render(('e00osc', 'abs', 1, 'e00', E00_MIDI, 0))
    print('exact check max|diff|', float(np.max(np.abs(ref - mine))), flush=True)
    keys = [(t, f, d, s, n, 0) for t in TIMBRES for f in ('off', 'abs') for d in (0, 1) for s in SCORES for n in TON]
    keys += [(t, f, d, 'e00', n, c) for t in TIMBRES for f in ('off', 'abs') for d in (0, 1) for n in TON for c in range(1, 6)]
    keys += [('e00osc', f, 1, 'e00', n, 0) for f in ('abs', 'rel') for n in list(range(48, 60)) + list(range(72, 84))]
    keys += [('e00osc', 'rel', 1, 'e00', n, 0) for n in TON]
    keys += [('-', f, 1, 'none', 61, 0) for f in ('off', 'abs')]
    if smoke:
        keys = keys[:4] + [k for k in keys if k[0] == 'violin'][:4] + keys[-2:]
    keys = list(dict.fromkeys(keys))
    print('n clips', len(keys), flush=True)
    with mp.get_context('fork').Pool(16) as pool:
        W = pool.map(render, keys, chunksize=4)
    import torch
    from audiobox_aesthetics.infer import AesPredictor
    aes = AesPredictor(checkpoint_pth=None, batch_size=32)
    rows = []
    for s0 in range(0, len(W), 32):
        items = [{'path': torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': SR} for w in W[s0:s0 + 32]]
        rows += aes.forward(items)
        if s0 % 640 == 0:
            print('scored', len(rows), flush=True)
    res = []
    for k, r in zip(keys, rows):
        d = dict(zip(('timbre', 'filter', 'drums', 'score', 'midi', 'chord'), k))
        d.update({kk: float(r[kk]) for kk in ('CE', 'CU', 'PC', 'PQ')})
        res.append(d)
    (OUT / ('scores_smoke.json' if smoke else 'scores.json')).write_text(json.dumps(res, indent=1))
    print('done', len(res), flush=True)
