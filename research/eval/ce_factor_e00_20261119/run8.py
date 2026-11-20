"""Round 8 (post-hoc, AES only): Mary interval stretch with mean pitch held at the original's (tonic+3) and every note inside
a per-timbre playable window. c{k}: p' = mean + k*(p - mean). Out-of-window (timbre, key, k) cells are rendered but flagged
in_range=False and excluded (paired with orig) in the analysis."""
import sys, json
import numpy as np
import multiprocessing as mp
import run4
import run as M
SK = (2, 3, 4, 6)
WIN = {'piano': (36, 96), 'organ': (36, 96), 'strings': (36, 96), 'guitar': (40, 84), 'violin': (55, 96), 'flute': (60, 96),
       'sine': (36, 96), 'e00osc': (36, 96)}


def pitches8(kind, tonic):
    base = np.array([tonic + p for p, _ in run4.MEL]); mu = tonic + 3
    k = 1 if kind == 'orig' else int(kind[1:])
    return (mu + k * (base - mu)).tolist()


_prev = M.score_notes


def score_notes8(kind, tonic, chord=(0, 4, 7)):
    if not kind.startswith('mary8:'):
        return _prev(kind, tonic, chord)
    ps = pitches8(kind[6:], tonic); out, t = [], 0.0
    for i, ((_, d), p) in enumerate(zip(run4.MEL, ps)):
        out.append((t, d * run4.Q * 0.9, int(p), 1.0 - M.DYN * M.E._VELR[i % 256])); t += d * run4.Q
    return out


M.score_notes = score_notes8
KINDS = ['orig'] + [f'c{k}' for k in SK]

if __name__ == '__main__':
    keys = [(t, 'off', 0, 'mary8:' + k, n, 0) for t in M.TIMBRES for n in M.TON for k in KINDS]
    with mp.get_context('fork').Pool(12) as pool:
        W = pool.map(M.render, keys, chunksize=4)
    import torch
    from audiobox_aesthetics.infer import AesPredictor
    aes = AesPredictor(checkpoint_pth=None, batch_size=32)
    rows = []
    for s0 in range(0, len(W), 32):
        rows += aes.forward([{'path': torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': M.SR} for w in W[s0:s0 + 32]])
    res = []
    for k, r in zip(keys, rows):
        kind = k[3][6:]; ps = np.array(pitches8(kind, k[4])); lo, hi = WIN[k[0]]
        d = {'timbre': k[0], 'key': k[4], 'kind': kind, 'mean_pitch': float(ps.mean()), 'min_p': int(ps.min()), 'max_p': int(ps.max()),
             'in_range': bool(ps.min() >= lo and ps.max() <= hi)}
        d.update({kk: float(r[kk]) for kk in ('CE', 'CU', 'PC', 'PQ')})
        res.append(d)
    (M.OUT / 'scores8.json').write_text(json.dumps(res, indent=1))
    print('done', len(res), flush=True)
