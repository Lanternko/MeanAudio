"""Human listening test (2026-11-13): original vs post-hoc 6 kHz low-pass.

Design (pre-registered): docs/experiments/human_listen_lp6k_20261113.md
Stimuli are rebuilt with the 193 functions (189 load_one + 193 apply_cell 'base'/'lp6k'); nothing is re-implemented.
Only `verify` needs the GPU (MuQ + AES scorers); select/prep/predict/page are CPU only.

Usage: python human_listen_lp6k_20261113.py {select|prep|verify|predict|page}   (dac venv)
"""
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/human_listen_lp6k_20261113')
E193 = Path('/home/kojiek/eval_output_nvme/muq_filter_ladder_193')
SEL_SEED = 20261115
NAME_SEED = 20261116
BASES = ['real', 'ma_cfg3neg_s14159265', 'musicgen', 'musicldm']
N_PER_BASE = 12
N_CATCH = 6
N_REPEATS = 6
MAX_LU = 0.5
PAGE_TEMPLATE = HERE / 'human_listen_lp6k_20261113_page.html'


def _mods():
    import muq_overlay_matrix_189 as M
    import muq_filter_ladder_193 as L
    return M, L


_CACHE = {}


def wave(base, i, version):
    """float32 16 kHz 10 s -30 LUFS waveform exactly as 193 scored it ('base' or 'lp6k')."""
    M, L = _mods()
    k = (base, i)
    if k not in _CACHE:
        _CACHE[k] = M.load_one(base, i)
    return L.apply_cell(_CACHE[k], version)


def lufs(w):
    import numpy as np
    import pyloudnorm
    return float(pyloudnorm.Meter(16000).integrated_loudness(np.asarray(w, np.float64)[:, None]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(E193 / 'split.json'))['test_mc']
    rng = random.Random(SEL_SEED)
    pool = list(pool)
    rng.shuffle(pool)
    cur = 0
    ids, excluded = {}, []
    for b in BASES:
        ids[b] = []
        while len(ids[b]) < N_PER_BASE:
            i = pool[cur]
            cur += 1
            dl = abs(lufs(wave(b, i, 'base')) - lufs(wave(b, i, 'lp6k')))
            if dl > MAX_LU:
                excluded.append(dict(base=b, id=i, abs_lufs_diff=dl))
                continue
            ids[b].append(i)
    catch = []
    for k in range(N_CATCH):
        catch.append(dict(base=BASES[k % 4], id=pool[cur]))
        cur += 1
    calib = pool[cur]
    cur += 1
    trials = []
    for b in BASES:
        for i in ids[b]:
            a, bb = ('base', 'lp6k') if rng.random() < 0.5 else ('lp6k', 'base')
            trials.append(dict(kind='main', base=b, id=i, a_version=a, b_version=bb))
    for c in catch:
        trials.append(dict(kind='catch', base=c['base'], id=c['id'], a_version='base', b_version='base'))
    rng.shuffle(trials)
    # repeats: from main trials in the first half, each base >= 1, A/B flipped
    half = len(trials) // 2
    first = [t for t in trials[:half] if t['kind'] == 'main']
    chosen = []
    for b in BASES:
        chosen.append(rng.choice([t for t in first if t['base'] == b]))
    rest = [t for t in first if t not in chosen]
    chosen += rng.sample(rest, N_REPEATS - len(chosen))
    reps = []
    for t in chosen:
        reps.append(dict(kind='repeat', base=t['base'], id=t['id'], a_version=t['b_version'],
                         b_version=t['a_version'], _orig=trials.index(t)))
    n0 = len(trials)
    for r in reps:
        lo = max(n0 // 2, r['_orig'] + 10)
        trials.insert(rng.randint(lo, len(trials)), r)
    for k, t in enumerate(trials):
        t['trial'] = k
    for t in trials:
        if t['kind'] == 'repeat':
            del t['_orig']
            t['repeat_of'] = next(u['trial'] for u in trials
                                  if u['kind'] == 'main' and u['base'] == t['base'] and u['id'] == t['id'])
        else:
            t['repeat_of'] = None
    for t in trials:   # sanity: repeat >= 10 after original
        if t['kind'] == 'repeat':
            assert t['trial'] - t['repeat_of'] >= 10, t
    json.dump(dict(seed=SEL_SEED, ids=ids, catch=catch, calibration=dict(base='real', id=calib),
                   excluded=excluded, n_trials=len(trials), trials=trials),
              open(OUT / 'selection.json', 'w'), indent=1)
    kinds = {k: sum(t['kind'] == k for t in trials) for k in ('main', 'repeat', 'catch')}
    print(f'trials {len(trials)} {kinds}; excluded {len(excluded)}: {excluded}; calib {calib}')


def prep():
    import numpy as np
    import soundfile as sf
    sel = json.load(open(OUT / 'selection.json'))
    deliver = OUT / 'deliver'
    deliver.mkdir(exist_ok=True)
    rng = random.Random(NAME_SEED)
    key, page = {}, []

    def export(base, i, version, **meta):
        w = wave(base, i, version)
        name = '%08x.wav' % rng.getrandbits(32)
        while name in key:
            name = '%08x.wav' % rng.getrandbits(32)
        sf.write(deliver / name, w, 16000, subtype='PCM_24')
        y, sr = sf.read(deliver / name, dtype='float64', always_2d=True)
        assert sr == 16000 and y.shape[1] == 1 and len(y) == len(w)
        err = float(np.max(np.abs(y[:, 0] - w.astype(np.float64))))
        assert err <= 2 ** -22, (name, err)
        key[name] = dict(base=base, id=i, version=version, sha256=sha(deliver / name), max_abs_err=err, **meta)
        return name

    for t in sel['trials']:
        a = export(t['base'], t['id'], t['a_version'], trial=t['trial'], kind=t['kind'], side='A')
        b = export(t['base'], t['id'], t['b_version'], trial=t['trial'], kind=t['kind'], side='B')
        page.append(dict(trial=t['trial'], a=a, b=b))
    c = sel['calibration']
    calib = export(c['base'], c['id'], 'base', trial=None, kind='calibration', side=None)
    json.dump(key, open(OUT / 'key.json', 'w'), indent=1)
    json.dump(dict(calibration=calib, trials=page), open(OUT / 'page_trials.json', 'w'), indent=1)
    print(f'{len(key)} wav files -> {deliver}')


def _read_wav(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    assert sr == 16000 and x.shape[1] == 1
    return x[:, 0]


def verify():
    import numpy as np
    M, L = _mods()
    P = M.P
    L.setup(False)
    P.SC._muq()
    P.SC._aes()
    key = json.load(open(OUT / 'key.json'))
    cols = ['A1_MI', 'A3a_MI', 'PQ', 'CE']
    files = sorted(n for n, v in key.items() if v['kind'] == 'main')
    wavs = [_read_wav(OUT / 'deliver' / n) for n in files]
    rows = []
    for s in range(0, len(wavs), 16):
        rows += L.X.score(wavs[s:s + 16], None, 1, 'empty', sr=16000, clap=False)
    ref = {}
    for b in BASES:
        for v in ('base', 'lp6k'):
            ref[(b, v)] = M.tsv_read(E193 / 'per_clip' / f'T_{b}_{v}.tsv')
    diffs = {c: [] for c in cols}
    per = []
    for n, r in zip(files, rows):
        k = key[n]
        d = {c: abs(r[c] - ref[(k['base'], k['version'])][k['id']][c]) for c in cols}
        for c in cols:
            diffs[c].append(d[c])
        per.append(dict(file=n, base=k['base'], id=k['id'], version=k['version'], **d))
    summ = {c: dict(max=float(np.max(diffs[c])), mean=float(np.mean(diffs[c]))) for c in cols}
    ok = summ['A1_MI']['max'] <= 0.02
    json.dump(dict(n_files=len(files), summary=summ, passed_A1_tol_0p02=ok, per_file=per),
              open(OUT / 'verify.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))
    if not ok:
        raise SystemExit('VERIFY FAILED: A1_MI max abs diff > 0.02')
    print('verify OK')


def predict():
    import csv
    import numpy as np
    sel = json.load(open(OUT / 'selection.json'))
    assert not (OUT / 'ratings').exists()
    cols = ['A1_MI', 'A3a_MI', 'PQ', 'CE', 'PAM', 'CMI_q']

    def rd(b, v):
        with open(E193 / 'per_clip' / f'T_{b}_{v}.tsv', newline='') as f:
            return {r['id']: {c: float(r[c]) for c in cols} for r in csv.DictReader(f, delimiter='\t')}
    per_id = []
    for b in BASES:
        o, l = rd(b, 'base'), rd(b, 'lp6k')
        for i in sel['ids'][b]:
            per_id.append(dict(base=b, id=i, **{c: l[i][c] - o[i][c] for c in cols}))
    out = dict(written_at=time.strftime('%F %T'), n=len(per_id), definition='d = lp6k - base; prefers_lp6k_frac = frac(d>0)',
               metrics={}, per_base={}, per_id=per_id)
    for c in cols:
        d = np.array([r[c] for r in per_id])
        out['metrics'][c] = dict(prefers_lp6k_frac=float((d > 0).mean()), mean_d=float(d.mean()))
        for b in BASES:
            db = np.array([r[c] for r in per_id if r['base'] == b])
            out['per_base'].setdefault(b, {})[c] = dict(prefers_lp6k_frac=float((db > 0).mean()), mean_d=float(db.mean()))
    json.dump(out, open(OUT / 'predictions.json', 'w'), indent=1)
    print(' '.join(f"{c} lp6k-pref {out['metrics'][c]['prefers_lp6k_frac']:.2f} mean_d {out['metrics'][c]['mean_d']:+.3f}"
                    for c in cols))


def page():
    site = OUT / 'site'
    (site / 'audio').mkdir(parents=True, exist_ok=True)
    pt = json.load(open(OUT / 'page_trials.json'))
    html = PAGE_TEMPLATE.read_text().replace('__PAGE__', json.dumps(pt))
    (site / 'index.html').write_text(html)
    n = 0
    for p in sorted((OUT / 'deliver').glob('*.wav')):
        shutil.copy2(p, site / 'audio' / p.name)
        n += 1
    print(f'site: index.html + {n} wavs -> {site}')


if __name__ == '__main__':
    fns = dict(select=select, prep=prep, verify=verify, predict=predict, page=page)
    fns[sys.argv[1]]()
