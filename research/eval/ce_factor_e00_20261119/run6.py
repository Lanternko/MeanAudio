"""Round 6 (post-hoc, AES only): monotone sweeps that cover the full range with small steps (answers: walk never reached its nominal range).
u{W}: chromatic ascent lo->hi spread evenly over the clip (exact range W, exact mean = centre, step ~W/32, repeats when W<32);
d{W}: mirror (descent); z{W}: major-scale ascent from lo, wrapping to lo when it passes hi (small steps + one big drop per cycle).
8 timbres x 9 widths x 3 rules x 12 tonics, no filter, no drums; centre = tonic+4 as in round 3."""
import json
import numpy as np
import multiprocessing as mp
import run4  # noqa: F401  (patch chain)
import run as M
WIDTHS = (0, 2, 4, 7, 12, 19, 24, 36, 48)
MAJ = (0, 2, 4, 5, 7, 9, 11)


def seq6(kind, tonic):
    rule, w = kind[0], int(kind[1:].split('_')[0])
    c = tonic + 4; lo = c - w // 2; hi = lo + w
    ons = np.arange(0.0, M.T - 0.05, M.STEP); n = len(ons)
    if rule in 'ud':
        ps = [int(round(lo + w * k / (n - 1))) for k in range(n)]
        ps = ps if rule == 'u' else ps[::-1]
    else:
        cand = [p for p in range(lo, hi + 1) if (p - tonic) % 12 in MAJ] or [c]
        ps = [cand[k % len(cand)] for k in range(n)]
    return ons, ps


_prev = M.score_notes


def score_notes6(kind, tonic, chord=(0, 4, 7)):
    if not (kind[:1] in 'udz' and kind.endswith('_s0')):
        return _prev(kind, tonic, chord)
    ons, ps = seq6(kind, tonic)
    return [(on, M.NLEN, p, 1.0 - M.DYN * M.E._VELR[k % 256]) for k, (on, p) in enumerate(zip(ons, ps))]


M.score_notes = score_notes6

if __name__ == '__main__':
    keys = [(t, 'off', 0, f'{r}{w}_s0', n, 0) for t in M.TIMBRES for w in WIDTHS for r in 'udz' for n in M.TON]
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
        _, ps = seq6(k[3], k[4]); ps = np.array(ps)
        d = {'timbre': k[0], 'rule': k[3][0], 'width': int(k[3][1:].split('_')[0]), 'midi': k[4], 'mean_pitch': float(ps.mean()),
             'mean_abs_int': float(np.abs(np.diff(ps)).mean()), 'min_p': int(ps.min()), 'max_p': int(ps.max()), 'n_unique': int(len(set(ps.tolist())))}
        d.update({kk: float(r[kk]) for kk in ('CE', 'CU', 'PC', 'PQ')})
        res.append(d)
    (M.OUT / 'scores6.json').write_text(json.dumps(res, indent=1))
    print('done', len(res), flush=True)
