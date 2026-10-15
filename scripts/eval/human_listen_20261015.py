"""Human listening pilot (2026-10-15, meeting item 7): MeanAudio vs MusicCaps real recordings.

Design: docs/experiments/human_listen_real_vs_ours_20261015.md

Three paired comparisons on the same 20 MusicCaps prompts (seeded sample, no filtering;
silent outputs stay in and count as failures):
  real_fid8  ours CFG3+fidelity8  vs MusicCaps real recording   (the meeting's ask)
  real_cfg0  ours CFG0            vs MusicCaps real recording
  mg         ours CFG3+fidelity8  vs MusicGen-medium            (table E primary, pilot)
Ours = slot0clean nmv2matched full s14159265 (p2 114/115).

Playback files are the 115 lvl30 block (-30 LUFS, FLAC PCM_24) decoded to WAV PCM_24, so the
samples are bit-identical to what the metrics scored (TTM plan 7.7; checked in prep). Native
bandwidth: MusicGen 32 kHz, the rest 16 kHz. Every (comparison, prompt, system) gets its own
opaque file name, so a clip reused across comparisons cannot be recognised by name.

Usage: python human_listen_20261015.py {select|prep|predict|all}   (dac venv)
"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/human_listen_20261015')
MC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
B115 = Path('/home/kojiek/eval_output_nvme/ttm_metrics_115_20261009')
SEED = 20261015
N_PROMPTS = 20
N_REPEATS = 6           # 2 per comparison, A/B flipped, second half, >= 10 trials after the original
N_BLIND = 6             # end-of-test "which one is the real recording?" (3 per real comparison)

SYSTEMS = ('meanaudio_cfg3_neg', 'meanaudio_cfg0', 'musiccaps_real', 'musicgen_medium')
# comparison -> (ours, other)
COMPARISONS = {
    'real_fid8': ('meanaudio_cfg3_neg', 'musiccaps_real'),
    'real_cfg0': ('meanaudio_cfg0', 'musiccaps_real'),
    'mg': ('meanaudio_cfg3_neg', 'musicgen_medium'),
}


def lvl30(system):
    return B115 / system / 'lvl30'


def read_tsv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return [(r['id'], r['caption']) for r in csv.DictReader(f, delimiter='\t')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gain_status(system):
    with open(lvl30(system) / 'gain_status.tsv', newline='') as f:
        return {r['id']: r['status'] for r in csv.DictReader(f, delimiter='\t')}


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    mc = read_tsv(MC_TSV)
    present = set.intersection(*[{p.stem for p in (lvl30(s) / 'audio').glob('*.flac')} for s in SYSTEMS])
    pool = [r for r in mc if r[0] in present]   # TSV order (csv records, not lines)
    rng = random.Random(SEED)
    picked = rng.sample(pool, N_PROMPTS)
    ids = {i for i, _ in picked}
    # calibration clip: a real recording outside the set with status ok
    real_ok = gain_status('musiccaps_real')
    calib = rng.choice([i for i, _ in pool if i not in ids and real_ok.get(i) == 'ok'])
    trials = [dict(comp=c, id=i) for c in COMPARISONS for i, _ in picked]
    for t in trials:
        t['ours_side'] = rng.choice('AB')
    rng.shuffle(trials)
    reps = []
    for c in COMPARISONS:
        src = rng.sample([t for t in trials[:len(trials) - 20] if t['comp'] == c], N_REPEATS // len(COMPARISONS))
        for t in src:
            reps.append(dict(comp=c, id=t['id'], ours_side='B' if t['ours_side'] == 'A' else 'A',
                             repeat_of=trials.index(t)))
    for r in reps:
        lo = max(len(trials) // 2, r['repeat_of'] + 10)
        trials.insert(rng.randint(lo, len(trials)), r)
    for k, t in enumerate(trials):
        t['trial'] = k
    for t in trials:
        if 'repeat_of' in t:
            t['repeat_of'] = next(u['trial'] for u in trials
                                  if u['comp'] == t['comp'] and u['id'] == t['id'] and 'repeat_of' not in u)
    # blind check: fixed real-vs-ours pairs, re-using the main-trial files, sides re-drawn
    blind = []
    for c in ('real_fid8', 'real_cfg0'):
        for i in rng.sample(sorted(ids), N_BLIND // 2):
            blind.append(dict(comp=c, id=i, ours_side=rng.choice('AB')))
    rng.shuffle(blind)
    json.dump(dict(seed=SEED, pool_size=len(pool), prompts=picked, calibration=calib,
                   trials=trials, blind=blind),
              open(OUT / 'selection.json', 'w'), indent=1, ensure_ascii=False)
    print(f'pool {len(pool)}, picked {len(picked)}, trials {len(trials)}, blind {len(blind)}, calib {calib}')


def prep():
    import numpy as np
    import soundfile as sf
    sel = json.load(open(OUT / 'selection.json'))
    deliver = OUT / 'deliver'
    deliver.mkdir(exist_ok=True)
    rng = random.Random(SEED + 1)
    key, files = {}, {}

    def export(system, i, tag):
        src = lvl30(system) / 'audio' / f'{i}.flac'
        info = sf.info(str(src))
        assert info.subtype == 'PCM_24', (src, info.subtype)
        x, sr = sf.read(src, dtype='int32', always_2d=True)
        name = '%08x.wav' % rng.getrandbits(32)
        assert name not in key
        sf.write(deliver / name, x, sr, subtype='PCM_24')
        y, sr2 = sf.read(deliver / name, dtype='int32', always_2d=True)
        assert sr2 == sr and np.array_equal(x, y), f'not bit-identical: {src}'
        key[name] = dict(system=system, id=i, tag=tag, src=str(src), src_sha256=sha(src),
                         sr=sr, n_samples=len(x), sha256=sha(deliver / name))
        return name

    for comp, pair in COMPARISONS.items():
        for i, _ in sel['prompts']:
            for s in pair:
                files[(comp, i, s)] = export(s, i, comp)
    calib = export('musiccaps_real', sel['calibration'], 'calibration')
    caps = dict(read_tsv(MC_TSV))

    def page_pair(t):
        ours, other = COMPARISONS[t['comp']]
        a, b = (ours, other) if t['ours_side'] == 'A' else (other, ours)
        return dict(caption=caps[t['id']], a=files[(t['comp'], t['id'], a)], b=files[(t['comp'], t['id'], b)])

    page = dict(calibration=calib,
                trials=[dict(trial=t['trial'], **page_pair(t)) for t in sel['trials']],
                blind=[dict(blind=k, **page_pair(t)) for k, t in enumerate(sel['blind'])])
    json.dump(key, open(OUT / 'key.json', 'w'), indent=1)
    json.dump(page, open(OUT / 'page_trials.json', 'w'), indent=1, ensure_ascii=False)
    print(f'{len(key)} files -> {deliver}')


def _per_clip(system):
    rows = {}
    with open(lvl30(system) / 'scores' / f'{system}_lvl30' / 'per_clip.tsv', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows[r['id']] = {k: float(r[k]) if r[k] != '' else float('nan') for k in ('clap', 'CE', 'PQ', 'lufs', 'silent')}
    with open(lvl30(system) / 'meva' / 'meva_per_clip.tsv', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            rows[r['id']]['meva'] = float(r['meva_raw'])
    return rows


def predict():
    """Metric predictions on exactly the delivered clips (lvl30 scores = delivered samples)."""
    sel = json.load(open(OUT / 'selection.json'))
    ids = [i for i, _ in sel['prompts']]
    scores = {s: _per_clip(s) for s in SYSTEMS}
    status = {s: gain_status(s) for s in SYSTEMS}
    out = {}
    for comp, (ours, other) in COMPARISONS.items():
        res = {}
        for m in ('PQ', 'CE', 'meva', 'clap'):
            d = [scores[ours][i][m] - scores[other][i][m] for i in ids]
            res[m] = dict(mean_diff_ours_minus_other=sum(d) / len(d), ours_wins=sum(x > 0 for x in d),
                          n=len(d), per_id={i: x for i, x in zip(ids, d)})
        res['silent_ours'] = [i for i in ids if status[ours][i] == 'silent_raw']
        res['silent_other'] = [i for i in ids if status[other][i] == 'silent_raw']
        out[comp] = res
        print(f'{comp:9s} ' + '  '.join(f'{m} {res[m]["mean_diff_ours_minus_other"]:+.3f} '
                                       f'({res[m]["ours_wins"]}/{len(ids)})' for m in ('PQ', 'CE', 'meva', 'clap'))
              + f'  silent ours {len(res["silent_ours"])} other {len(res["silent_other"])}')
    json.dump(out, open(OUT / 'predictions.json', 'w'), indent=1)


if __name__ == '__main__':
    step = sys.argv[1]
    for name, fn in (('select', select), ('prep', prep), ('predict', predict)):
        if step in (name, 'all'):
            fn()
