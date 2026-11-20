"""Round 2 (post-hoc, AES only): (A) well-known simple drum grooves vs random-placement controls, 3 kits x 3 humanize seeds;
(B) random wide-range pitch jumps (12/16/26 semitones, chromatic vs scale) x 8 timbres x filter x drums x 12 tonics."""
import os, sys, json
import run as M
import numpy as np
import multiprocessing as mp
sys.path.insert(0, '/home/kojiek/Documents/Codex/2026-10-01/files-pasted-by-the-user-negmf/work/midi-packages')
SR, NS, T = M.SR, M.NS, M.T
GMK = {'K': 36, 'S': 38, 'H': 42, 'O': 46, 'R': 51, 'C': 39, 'X': 37, 'L': 45, 'B': 56}
# (bpm, steps per bar, {voice: [steps]}) ; ghost notes as lowercase voice -> low velocity
GROOVES = {
    'rock':        (110, 16, {'K': [0, 8], 'S': [4, 12], 'H': [0, 2, 4, 6, 8, 10, 12, 14]}),
    'rock_16':     (100, 16, {'K': [0, 7, 8, 10], 'S': [4, 12], 'H': list(range(16))}),
    'four_floor':  (124, 16, {'K': [0, 4, 8, 12], 'C': [4, 12], 'O': [2, 6, 10, 14], 'H': list(range(0, 16, 1))}),
    'disco':       (118, 16, {'K': [0, 4, 8, 12], 'S': [4, 12], 'O': [2, 6, 10, 14], 'H': [0, 4, 8, 12]}),
    'boom_bap':    (90, 16, {'K': [0, 7, 10], 'S': [4, 12], 'H': [0, 2, 4, 6, 8, 10, 12, 14]}),
    'funk':        (100, 16, {'K': [0, 2, 10], 'S': [4, 12], 's': [7, 9, 15], 'H': [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15], 'O': [7]}),
    'bossa':       (130, 16, {'K': [0, 3, 4, 7, 8, 11, 12, 15], 'X': [0, 3, 6, 10, 12], 'H': [0, 2, 4, 6, 8, 10, 12, 14]}),
    'one_drop':    (75, 16, {'K': [8], 'X': [8], 'H': [0, 2, 4, 6, 8, 10, 12, 14]}),
    'shuffle':     (100, 12, {'K': [0, 6], 'S': [3, 9], 'H': [0, 2, 3, 5, 6, 8, 9, 11]}),
    'motown':      (120, 16, {'K': [0, 8, 10], 'S': [0, 4, 8, 12], 'B': [0, 2, 4, 6, 8, 10, 12, 14]}),
    'stomp_clap':  (81, 16, {'K': [0, 2, 8, 10], 'C': [4, 12]}),
    'halftime':    (140, 16, {'K': [0, 10], 'S': [8], 'H': [0, 2, 4, 6, 8, 10, 12, 14]}),
    'trap':        (140, 16, {'K': [0, 7, 11], 'S': [8], 'H': list(range(0, 16, 2)) + [13, 15]}),
    'amen':        (170, 16, {'K': [0, 2, 10, 11], 'S': [4, 12], 's': [7, 9, 15], 'R': [0, 2, 4, 6, 8, 10, 12, 14]}),
    'ride_jazz':   (140, 12, {'R': [0, 3, 5, 6, 9, 11], 'H': [3, 9], 'K': [0]}),
}
KITS = ('e00', 'generaluser', 'fluidr3')
SEEDS = (0, 1, 2)


def events(name, mode, seed):
    bpm, n, pat = GROOVES[name]
    dt = 60.0 / bpm / (n / 4)
    bar = dt * n
    rng = np.random.default_rng(seed * 1000 + sum(map(ord, name)))
    ev = []
    for v, steps in pat.items():
        vel = 45 if v.islower() else (70 if v in 'HRB' else 100)
        for b in range(int(np.ceil(T / bar))):
            ss = steps
            if mode == 'rgrid':
                ss = rng.choice(n, len(steps), replace=False)
            for s in ss:
                t = b * bar + s * dt if mode != 'rfree' else b * bar + rng.uniform(0, bar)
                t += rng.normal(0, 0.006)                      # humanize timing
                if 0 <= t < T - 0.05:
                    ev.append((t, v.upper(), int(np.clip(vel + rng.normal(0, 8), 20, 127))))
    return ev


def e00_kit(ev):
    """e00's own synthetic kick/snare/hat (same gains as e00 at r0=0.05); clap/rim -> snare, open hat/ride/bell -> long hat, tom -> kick."""
    dr = np.zeros(NS)
    tt = np.arange(int(0.15 * SR)) / SR
    kick = np.sin(2 * np.pi * (45 * tt + 75 * 0.03 * (1 - np.exp(-tt / 0.03)))) * np.exp(-tt / 0.05) * M.LK * 2.5
    snare = M.E._NOISE[:len(tt)] * np.exp(-tt / 0.04) * M.LS * 1.5
    tt2 = np.arange(int(0.05 * SR)) / SR
    hat = np.diff(M.E._NOISE[:len(tt2) + 1]) * np.exp(-tt2 / 0.01) * M.LH * 2.0
    tt3 = np.arange(int(0.3 * SR)) / SR
    ohat = np.diff(M.E._NOISE[1000:1000 + len(tt3) + 1]) * np.exp(-tt3 / 0.07) * M.LH * 1.2
    S = {'K': kick, 'L': kick, 'S': snare, 'C': snare, 'X': snare * 0.6, 'H': hat, 'O': ohat, 'R': ohat * 0.7, 'B': hat}
    for t, v, vel in ev:
        x = S[v] * vel / 100
        i0 = int(t * SR); m = min(len(x), NS - i0)
        dr[i0:i0 + m] += x[:m]
    return dr


def gm_kit(ev, bank):
    import soxr, tinysoundfont
    s = tinysoundfont.Synth(gain=-6, samplerate=44100)
    sid = s.sfload(str(M.A.MS.BANKS[bank]))
    s.program_select(9, sid, 128, 0, True)
    N = int(44100 * (T + 0.5)); x = np.zeros((N, 2), np.float32); cur = 0
    for t, v, vel in sorted(ev):
        end = int(round(t * 44100))
        if end > cur:
            x[cur:end] = np.frombuffer(s.generate_simple(end - cur), np.float32).reshape(-1, 2); cur = end
        s.noteon(9, GMK[v], vel)
    x[cur:] = np.frombuffer(s.generate_simple(N - cur), np.float32).reshape(-1, 2)
    return soxr.resample(x.mean(1).astype(np.float64), 44100, SR)[:NS]


def render_drum(k):
    name, mode, kit, seed = k
    ev = events(name, mode, seed)
    buf = e00_kit(ev) if kit == 'e00' else gm_kit(ev, kit)
    return np.asarray(M.C.lock(buf), np.float32)


JUMPS = {'rchrom12': (12, 'chrom'), 'rchrom16': (16, 'chrom'), 'rchrom26': (26, 'chrom'), 'rscale16': (16, 'scale'), 'rscale26': (26, 'scale')}
_orig = M.score_notes


def score_notes2(kind, tonic, chord=(0, 4, 7)):
    if kind not in JUMPS:
        return _orig(kind, tonic, chord)
    w, how = JUMPS[kind]
    lo = tonic + 4 - w // 2
    pool = list(range(lo, lo + w + 1))
    if how == 'scale':
        pool = [p for p in pool if (p - tonic) % 12 in (0, 2, 4, 5, 7, 9, 11)]
    rng = np.random.default_rng(1119 + tonic + 7 * w)
    ons = np.arange(0.0, T - 0.05, M.STEP)
    out = [(on, M.NLEN, int(rng.choice(pool))) for on in ons]
    return [(on, nl, m, 1.0 - M.DYN * M.E._VELR[k % 256]) for k, (on, nl, m) in enumerate(out)]


M.score_notes = score_notes2

if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    dk = [(g, m, kit, s) for g in GROOVES for m in ('groove', 'rgrid', 'rfree') for kit in KITS for s in SEEDS]
    jk = [(t, f, d, j, n, 0) for t in M.TIMBRES for f in ('off', 'abs') for d in (0, 1) for j in JUMPS for n in M.TON]
    if smoke:
        dk = dk[:3] + [k for k in dk if k[2] == 'generaluser'][:2]; jk = jk[:3]
    with mp.get_context('fork').Pool(16) as pool:
        W = pool.map(render_drum, dk, chunksize=2) + pool.map(M.render, jk, chunksize=4)
    import torch
    from audiobox_aesthetics.infer import AesPredictor
    aes = AesPredictor(checkpoint_pth=None, batch_size=32)
    rows = []
    for s0 in range(0, len(W), 32):
        rows += aes.forward([{'path': torch.from_numpy(np.ascontiguousarray(w))[None], 'sample_rate': SR} for w in W[s0:s0 + 32]])
    res = []
    for k, r in zip(dk + jk, rows):
        d = dict(zip(('groove', 'mode', 'kit', 'seed'), k)) if len(k) == 4 else dict(zip(('timbre', 'filter', 'drums', 'score', 'midi', 'chord'), k))
        d.update({kk: float(r[kk]) for kk in ('CE', 'CU', 'PC', 'PQ')})
        res.append(d)
    (M.OUT / ('scores2_smoke.json' if smoke else 'scores2.json')).write_text(json.dumps(res, indent=1))
    if smoke:
        import soundfile as sf
        for k, w in zip(dk, W):
            sf.write(f'/tmp/claude-1005/-home-kojiek-MeanAudio/5863b2db-6efb-478f-b52d-c0d162e128a3/scratchpad/smk_{"_".join(map(str, k))}.wav', w, SR)
    print('done', len(res), flush=True)
