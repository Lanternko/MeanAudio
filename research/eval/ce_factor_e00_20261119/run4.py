"""Round 4 (post-hoc, AES only): 'Mary Had a Little Lamb' (C D E G, 32 quarters fitted to 10 s), same rhythm, pitch variants:
orig / shuf (same pitches, order permuted) / rand{W} (each note an independent random pitch in a W-semitone pool centred on E) /
stretch{k} (intervals around E multiplied by k: same contour, 4 distinct pitches, wide) / map{W} (the 4 pitch classes mapped to 4 random
distinct pitches in a W pool: same symbol sequence, wide). 8 timbres x 12 keys, no filter, no drums."""
import sys, json
import numpy as np
import multiprocessing as mp
import run3  # noqa: F401
import run as M
MEL = [(4,1),(2,1),(0,1),(2,1),(4,1),(4,1),(4,2),(2,1),(2,1),(2,2),(4,1),(7,1),(7,2),
       (4,1),(2,1),(0,1),(2,1),(4,1),(4,1),(4,1),(4,1),(2,1),(2,1),(4,1),(2,1),(0,4)]
Q = M.T / 32.0
SEEDS = (0, 1, 2)
RW = (2, 4, 7, 12, 24, 48)
SK = (2, 3, 4, 6)
MW = (12, 24, 48)


def kinds():
    ks = ['orig'] + [f'shuf_s{s}' for s in SEEDS] + [f'rand{w}_s{s}' for w in RW for s in SEEDS] + [f'stretch{k}' for k in SK] + \
         [f'map{w}_s{s}' for w in MW for s in SEEDS]
    return ks


def pitches(kind, tonic):
    base = [tonic + p for p, _ in MEL]; c = tonic + 4
    name, _, s = kind.partition('_s'); s = int(s) if s else 0
    rng = np.random.default_rng(2026 + 131 * tonic + 17 * s + sum(map(ord, name)))
    if name == 'orig':
        return base
    if name == 'shuf':
        return list(rng.permutation(base))
    if name.startswith('rand'):
        w = int(name[4:]); lo = c - w // 2
        return [int(rng.integers(lo, lo + w + 1)) for _ in base]
    if name.startswith('stretch'):
        k = int(name[7:]); return [c + k * (p - c) for p in base]
    if name.startswith('map'):
        w = int(name[3:]); lo = c - w // 2
        tgt = rng.choice(np.arange(lo, lo + w + 1), 4, replace=False)
        mp_ = dict(zip((0, 2, 4, 7), tgt)); return [int(mp_[p - tonic]) for p in base]
    raise KeyError(kind)


_prev = M.score_notes


def score_notes4(kind, tonic, chord=(0, 4, 7)):
    if not kind.startswith('mary:'):
        return _prev(kind, tonic, chord)
    ps = pitches(kind[5:], tonic); out, t = [], 0.0
    for k, ((_, d), p) in enumerate(zip(MEL, ps)):
        out.append((t, d * Q * 0.9, int(p), 1.0 - M.DYN * M.E._VELR[k % 256])); t += d * Q
    return out


M.score_notes = score_notes4

if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    keys = [(t, 'off', 0, 'mary:' + k, n, 0) for t in M.TIMBRES for n in M.TON for k in kinds()]
    if smoke:
        keys = [k for k in keys if k[0] == 'piano' and k[4] == 60][:40]
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
        kind = k[3][5:]; ps = np.array(pitches(kind, k[4]))
        name = kind.partition('_s')[0]
        d = {'timbre': k[0], 'key': k[4], 'kind': kind, 'cond': name, 'seed': int(kind.partition('_s')[2] or 0),
             'n_pc': int(len(set(ps.tolist()))), 'range': int(ps.max() - ps.min()), 'mean_abs_int': float(np.abs(np.diff(ps)).mean()),
             'min_p': int(ps.min()), 'max_p': int(ps.max())}
        d.update({kk: float(r[kk]) for kk in ('CE', 'CU', 'PC', 'PQ')})
        res.append(d)
    (M.OUT / ('scores4_smoke.json' if smoke else 'scores4.json')).write_text(json.dumps(res, indent=1))
    if smoke:
        import soundfile as sf
        sp = '/tmp/claude-1005/-home-kojiek-MeanAudio/5863b2db-6efb-478f-b52d-c0d162e128a3/scratchpad/mary_smoke'
        import os; os.makedirs(sp, exist_ok=True)
        for k, w, d in zip(keys, W, res):
            sf.write(f'{sp}/{d["kind"]}_CE{d["CE"]:.2f}.wav', w, M.SR)
    print('done', len(res), flush=True)
