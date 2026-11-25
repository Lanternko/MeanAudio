"""Round 5 (post-hoc, AES only): register ladder. Same random-jump rule as round 3 (j{W}), widths 0/6/12, pool centre swept
C2..C6 (MIDI 36..84, step 4) -> does CE prefer a register? 8 timbres x 13 centres x 3 seeds, no filter, no drums."""
import sys, json
import numpy as np
import multiprocessing as mp
import run3
import run as M
CENTRES = list(range(36, 85, 4))
if __name__ == '__main__':
    keys = [(t, 'off', 0, f'j{w}_s{s}', c - 4, 0) for t in M.TIMBRES for c in CENTRES for w in (0, 6, 12) for s in ((0,) if w == 0 else (0, 1, 2))]
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
        ons, ps = run3.seq(k[3], k[4]); ps = np.array(ps)
        d = {'timbre': k[0], 'centre': k[4] + 4, 'width': int(k[3][1:].split('_')[0]), 'seed': int(k[3].split('_s')[1]),
             'mean_pitch': float(ps.mean()), 'mean_abs_int': float(np.abs(np.diff(ps)).mean())}
        d.update({kk: float(r[kk]) for kk in ('CE', 'CU', 'PC', 'PQ')})
        res.append(d)
    (M.OUT / 'scores5.json').write_text(json.dumps(res, indent=1))
    print('done', len(res), flush=True)
