"""Round 7 (post-hoc, AES only): Mary stretch with a fixed anchor instead of centring on E.
up{k}: lowest note (tonic C) fixed, intervals x k upward; dn{k}: highest note (G) fixed, intervals x k downward;
tr_up{k}/tr_dn{k}: original melody (k=1 intervals) transposed so its mean pitch matches up{k}/dn{k} (register-only control).
8 timbres x 12 keys; orig/centred stretch come from scores4.json (same renderer)."""
import sys, json
import numpy as np
import multiprocessing as mp
import run4
import run as M
SK = (2, 3, 4, 6)


def pitches7(kind, tonic):
    base = np.array([tonic + p for p, _ in run4.MEL])
    if kind == 'orig':
        return base.tolist()
    name, k = kind.rstrip('0123456789'), int(kind[len(kind.rstrip('0123456789')):])
    up = tonic + k * (base - tonic); dn = (tonic + 7) - k * ((tonic + 7) - base)
    if name == 'up':
        return up.tolist()
    if name == 'dn':
        return dn.tolist()
    if name == 'tr_up':
        return (base + int(round(up.mean() - base.mean()))).tolist()
    if name == 'tr_dn':
        return (base + int(round(dn.mean() - base.mean()))).tolist()
    raise KeyError(kind)


_prev = M.score_notes


def score_notes7(kind, tonic, chord=(0, 4, 7)):
    if not kind.startswith('mary7:'):
        return _prev(kind, tonic, chord)
    ps = pitches7(kind[6:], tonic); out, t = [], 0.0
    for i, ((_, d), p) in enumerate(zip(run4.MEL, ps)):
        out.append((t, d * run4.Q * 0.9, int(p), 1.0 - M.DYN * M.E._VELR[i % 256])); t += d * run4.Q
    return out


M.score_notes = score_notes7
KINDS = ['orig'] + [f'{n}{k}' for n in ('up', 'dn', 'tr_up', 'tr_dn') for k in SK]

if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    keys = [(t, 'off', 0, 'mary7:' + k, n, 0) for t in M.TIMBRES for n in M.TON for k in KINDS]
    if smoke:
        keys = [k for k in keys if k[0] == 'piano' and k[4] == M.TON[0]]
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
        kind = k[3][6:]; ps = np.array(pitches7(kind, k[4]))
        d = {'timbre': k[0], 'key': k[4], 'kind': kind, 'mean_pitch': float(ps.mean()), 'range': int(ps.max() - ps.min()),
             'min_p': int(ps.min()), 'max_p': int(ps.max())}
        d.update({kk: float(r[kk]) for kk in ('CE', 'CU', 'PC', 'PQ')})
        res.append(d)
    (M.OUT / ('scores7_smoke.json' if smoke else 'scores7.json')).write_text(json.dumps(res, indent=1))
    if smoke:
        for d in res: print(d['kind'], d['min_p'], d['max_p'], round(d['mean_pitch'], 1), round(d['CE'], 2))
    print('done', len(res), flush=True)
