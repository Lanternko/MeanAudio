"""Round 3 (post-hoc, AES only): pitch-range ladder. Random chromatic notes in a pool of width W semitones centred on tonic+4
(mean pitch fixed by design), two step rules at the same range: 'j' = independent uniform jumps, 'w' = random walk (+-1..2 st,
reflected at the bounds) -> separates jump size from range. 8 timbres x 9 widths x 2 rules x 12 tonics x 3 seeds, no filter, no drums."""
import sys, json
import numpy as np
import multiprocessing as mp
import run2  # noqa: F401  (keeps round-2 patch chain)
import run as M
WIDTHS = (0, 2, 4, 7, 12, 19, 24, 36, 48)
SEEDS = (0, 1, 2)
_prev = M.score_notes


def seq(kind, tonic):
    rule, w, seed = kind[0], int(kind[1:].split('_')[0]), int(kind.split('_s')[1])
    c = tonic + 4; lo, hi = c - w // 2, c - w // 2 + w
    rng = np.random.default_rng(31337 + 97 * tonic + 1009 * w + 7 * seed + (0 if rule == 'j' else 50000))
    ons = np.arange(0.0, M.T - 0.05, M.STEP)
    ps, p = [], c
    for _ in ons:
        if w == 0:
            p = c
        elif rule == 'j':
            p = int(rng.integers(lo, hi + 1))
        else:
            p = p + int(rng.choice([-2, -1, 1, 2]))
            if p > hi: p = 2 * hi - p
            if p < lo: p = 2 * lo - p
            p = int(np.clip(p, lo, hi))
        ps.append(p)
    return ons, ps


def score_notes3(kind, tonic, chord=(0, 4, 7)):
    if not (kind[:1] in 'jw' and '_s' in kind):
        return _prev(kind, tonic, chord)
    ons, ps = seq(kind, tonic)
    return [(on, M.NLEN, p, 1.0 - M.DYN * M.E._VELR[k % 256]) for k, (on, p) in enumerate(zip(ons, ps))]


M.score_notes = score_notes3

if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    keys = [(t, 'off', 0, f'{r}{w}_s{s}', n, 0) for t in M.TIMBRES for w in WIDTHS for r in 'jw' for n in M.TON for s in SEEDS]
    if smoke:
        keys = keys[:4] + [k for k in keys if k[0] == 'piano' and k[3].startswith('j48')][:2]
    with mp.get_context('fork').Pool(16) as pool:
        W = pool.map(M.render, keys, chunksize=4)
    import torch
    from audiobox_aesthetics.infer import AesPredictor
    aes = AesPredictor(checkpoint_pth=None, batch_size=32)
    rows = []
    for s0 in range(0, len(W), 32):
        rows += aes.forward([{'path': torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': M.SR} for w in W[s0:s0 + 32]])
    res = []
    for k, r in zip(keys, rows):
        ons, ps = seq(k[3], k[4]); ps = np.array(ps)
        d = {'timbre': k[0], 'rule': k[3][0], 'width': int(k[3][1:].split('_')[0]), 'seed': int(k[3].split('_s')[1]), 'midi': k[4],
             'mean_pitch': float(ps.mean()), 'pitch_sd': float(ps.std()), 'mean_abs_int': float(np.abs(np.diff(ps)).mean()),
             'min_p': int(ps.min()), 'max_p': int(ps.max())}
        d.update({kk: float(r[kk]) for kk in ('CE', 'CU', 'PC', 'PQ')})
        res.append(d)
    (M.OUT / ('scores3_smoke.json' if smoke else 'scores3.json')).write_text(json.dumps(res, indent=1))
    print('done', len(res), flush=True)
