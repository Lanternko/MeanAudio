"""Human blind-listening study (2026-10-06): ground truth for AES / MEva on our own arms.

Design: docs/experiments/blind_listen_20261006.md

Three comparisons on the same 24 MusicCaps prompts (seeded sample, no filtering):
  nmv2   nmv2pair quarter s14159265  CFG0  vs CFG3+neg(fidelity8)   AES and MEva agree
  dlab   defectlab quarter s14159265 CFG0  vs CFG3+neg(fidelity8)   AES and MEva disagree
  q081   081 arm HQ prefix + neg 'Low quality recording.' vs 081 control CFG3+neg(fidelity8)
nmv2/dlab audio comes from the existing full-run cells; q081 cells were deleted and are
regenerated on the 24 prompts only (different noise from the full run; every delivered
file is re-scored, so predictions refer to exactly what listeners hear).

Each pair is loudness-matched to a common target (-23 LUFS, lowered per pair so both
peaks stay <= -1 dBFS) and exported as 16-bit WAV under opaque names (artifacts do not serve FLAC). The condition key stays
local (key.json); the published page only sees opaque names.

Usage: python blind_listen_20261006.py {select|generate|prep|score|all}   (dac venv)
       runtime/meva_20261003/venv/bin/python blind_listen_20261006.py meva
"""
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = '/home/kojiek/venvs/dac/bin/python'
OUT = Path('/home/kojiek/nvme_experiment_artifacts/meanaudio/blind_listen_20261006')
MC_TSV = Path('/mnt/HDD/kojiek/phase4_jamendo_data/musiccaps_test.tsv')
HQ_TSV = Path('/home/kojiek/exps_nvme/quality_label_081/musiccaps_test_hqprefix.tsv')
EV = Path('/home/kojiek/eval_output_nvme')
EXPS = Path('/home/kojiek/exps_nvme')
SEED = 20261006
N_PROMPTS = 24
N_REPEATS = 6           # 2 per comparison, A/B order flipped relative to first showing
TARGET_LUFS = -23.0
PEAK_CEIL_DB = -1.0
FID8 = 'low quality recording, noisy, amateur, distorted, muffled, poor fidelity, hiss, lo-fi'
LQREC = 'Low quality recording.'


def _ckpt(exp):
    return EXPS / exp / f'{exp}_ema_final.pth'


ARM081 = 'phase8_qwen_caption2p0_slot0clean_qlabel_noq_quarter_s14159265_stage2_50000'
CTL081 = 'phase8_qwen_caption2p0_slot0clean_nmv2pair_noq_quarter_s14159265_stage2_50000'
NMV2 = 'phase8_qwen_caption2p0_slot0nmv2_nmv2pair_noq_quarter_s14159265_mc_mf25'
DLAB = 'phase8_qwen_caption2p0_slot0clean_defectlab_noq_quarter_s14159265_mc_mf25'

# condition -> audio dir (existing or regenerated)
CONDS = {
    'nmv2_cfg0': EV / f'{NMV2}_cfg0/audio',
    'nmv2_cfg3neg': EV / f'{NMV2}_cfg3_neg/audio',
    'dlab_cfg0': EV / f'{DLAB}_cfg0/audio',
    'dlab_cfg3neg': EV / f'{DLAB}_cfg3_neg/audio',
    'q081_arm_hqlq': OUT / 'gen/q081_arm_hqlq',
    'q081_ctl_fid8': OUT / 'gen/q081_ctl_fid8',
}
# comparison -> (treatment, reference); "treatment" is the arm the metrics favor
COMPARISONS = {
    'nmv2': ('nmv2_cfg3neg', 'nmv2_cfg0'),
    'dlab': ('dlab_cfg3neg', 'dlab_cfg0'),
    'q081': ('q081_arm_hqlq', 'q081_ctl_fid8'),
}
GEN = {
    'q081_arm_hqlq': (ARM081, 'subset_hq.tsv', LQREC),
    'q081_ctl_fid8': (CTL081, 'subset.tsv', FID8),
}


def read_tsv(path):
    with open(path, encoding='utf-8', newline='') as f:
        return [(r['id'], r['caption']) for r in csv.DictReader(f, delimiter='\t')]


def write_tsv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['id', 'caption'])
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    mc = read_tsv(MC_TSV)
    hq = dict(read_tsv(HQ_TSV))
    present = set.intersection(*[{p.stem for p in CONDS[c].glob('*.flac')}
                                 for c in ('nmv2_cfg0', 'nmv2_cfg3neg', 'dlab_cfg0', 'dlab_cfg3neg')])
    pool = [r for r in mc if r[0] in present and r[0] in hq]   # TSV order; csv records not lines
    rng = random.Random(SEED)
    picked = rng.sample(pool, N_PROMPTS)
    write_tsv(OUT / 'subset.tsv', picked)
    write_tsv(OUT / 'subset_hq.tsv', [(i, hq[i]) for i, _ in picked])
    # trials: every (comparison, prompt) once, plus repeats; shuffled order; random A/B side
    trials = [dict(comp=c, id=i) for c in COMPARISONS for i, _ in picked]
    for t in trials:
        t['treat_side'] = rng.choice('AB')
    rng.shuffle(trials)
    reps = []
    for c in COMPARISONS:
        src = rng.sample([t for t in trials[:len(trials) - 20] if t['comp'] == c],
                         N_REPEATS // len(COMPARISONS))
        for t in src:
            reps.append(dict(comp=c, id=t['id'], treat_side='B' if t['treat_side'] == 'A' else 'A',
                             repeat_of=trials.index(t)))
    for r in reps:   # insert repeats in the second half, at least 10 trials after the original
        lo = max(len(trials) // 2, r['repeat_of'] + 10)
        trials.insert(rng.randint(lo, len(trials)), r)
    for k, t in enumerate(trials):
        t['trial'] = k
    for t in trials:
        if 'repeat_of' in t:   # re-point to the original's final index after insertion
            t['repeat_of'] = next(u['trial'] for u in trials
                                  if u['comp'] == t['comp'] and u['id'] == t['id'] and 'repeat_of' not in u)
    json.dump(dict(seed=SEED, pool_size=len(pool), prompts=picked, trials=trials),
              open(OUT / 'selection.json', 'w'), indent=1, ensure_ascii=False)
    print(f'pool {len(pool)}, picked {len(picked)}, trials {len(trials)}')


def generate():
    for cond, (exp, tsv, neg) in GEN.items():
        d = CONDS[cond]
        if d.exists() and len(list(d.glob('*.flac'))) == N_PROMPTS:
            continue
        subprocess.run(['rm', '-rf', str(d)], check=True)   # never top up: one RNG per run
        cmd = [PY, 'eval.py', '--variant', 'meanaudio_s', '--model_path', str(_ckpt(exp)),
               '--output', str(d), '--tsv', str(OUT / tsv), '--use_meanflow',
               '--num_steps', '25', '--cfg_strength', '3.0', '--negative_prompt', neg,
               '--no_text_attention_mask', '--encoder_name', 't5_clap', '--text_c_dim', '512',
               '--seed', '42', '--full_precision', '--no_q']
        with open(OUT / f'gen_{cond}.log', 'w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        n = len(list(d.glob('*.flac')))
        assert n == N_PROMPTS, f'{cond}: {n} files'


def prep():
    import numpy as np
    import pyloudnorm
    import soundfile as sf
    sel = json.load(open(OUT / 'selection.json'))
    ids = [i for i, _ in sel['prompts']]
    deliver = OUT / 'deliver'
    deliver.mkdir(exist_ok=True)
    rng = random.Random(SEED + 1)
    key, files = {}, {}
    for comp, (treat, ref) in COMPARISONS.items():
        for i in ids:
            waves = {}
            for c in (treat, ref):
                w, sr = sf.read(CONDS[c] / f'{i}.flac', dtype='float64', always_2d=True)
                waves[c] = (w.mean(axis=1), sr)
            meter = pyloudnorm.Meter(waves[treat][1])
            lufs = {c: meter.integrated_loudness(w) for c, (w, _) in waves.items()}
            # common target, lowered so neither clip peaks above the ceiling
            headroom = min(PEAK_CEIL_DB - 20 * np.log10(np.abs(w).max()) - (TARGET_LUFS - lufs[c])
                           for c, (w, _) in waves.items())
            target = TARGET_LUFS + min(0.0, headroom)
            for c, (w, sr) in waves.items():
                out = w * 10 ** ((target - lufs[c]) / 20)
                name = '%08x.wav' % rng.getrandbits(32)
                sf.write(deliver / name, out.astype(np.float32), sr, subtype='PCM_16')   # served types: wav not flac
                key[name] = dict(cond=c, comp=comp, id=i, src=str(CONDS[c] / f'{i}.flac'),
                                 src_sha256=sha(CONDS[c] / f'{i}.flac'), lufs_src=lufs[c],
                                 target_lufs=target, sha256=sha(deliver / name))
                files[(comp, i, c)] = name
    caps = dict(sel['prompts'])
    page_trials = []
    for t in sel['trials']:
        treat, ref = COMPARISONS[t['comp']]
        a, b = (treat, ref) if t['treat_side'] == 'A' else (ref, treat)
        page_trials.append(dict(trial=t['trial'], caption=caps[t['id']],
                                a=files[(t['comp'], t['id'], a)], b=files[(t['comp'], t['id'], b)]))
    json.dump(key, open(OUT / 'key.json', 'w'), indent=1)
    json.dump(page_trials, open(OUT / 'page_trials.json', 'w'), indent=1, ensure_ascii=False)
    print(f'{len(key)} files, {len(page_trials)} trials, targets '
          f'{min(v["target_lufs"] for v in key.values()):.1f}..{max(v["target_lufs"] for v in key.values()):.1f}')


def score():
    sys.path.insert(0, str(ROOT / 'scripts/eval'))
    key = json.load(open(OUT / 'key.json'))
    caps = dict(json.load(open(OUT / 'selection.json'))['prompts'])
    deliver = OUT / 'deliver'
    from eval_metrics import score_aes, load_clap
    names = sorted(key)
    aes, failed = score_aes([deliver / n for n in names], batch_size=1, progress=False)
    assert not failed, failed
    import torch
    clap = load_clap()
    with torch.no_grad():
        for n in names:
            ae = clap.get_audio_embedding_from_filelist([str(deliver / n)], use_tensor=True)
            te = clap.get_text_embedding([caps[key[n]['id']]], use_tensor=True)
            key[n]['clap'] = float(torch.nn.functional.cosine_similarity(ae, te, dim=-1).item())
    del clap
    torch.cuda.empty_cache()
    for n in names:
        key[n].update({f'aes_{k}': v for k, v in aes[str(deliver / n)].items()})
    json.dump(key, open(OUT / 'key_scored.json', 'w'), indent=1)


def meva():
    # MEva needs its pinned venv: runtime/meva_20261003/venv/bin/python
    sys.path.insert(0, str(ROOT / 'scripts/eval'))
    from meva_runtime import Evaluator
    key = json.load(open(OUT / 'key_scored.json'))
    model = Evaluator()
    for n in sorted(key):
        key[n]['meva'] = model.score(OUT / 'deliver' / n)['meva_raw']
    json.dump(key, open(OUT / 'key_scored.json', 'w'), indent=1)
    summary()


def summary():
    key = json.load(open(OUT / 'key_scored.json'))
    for comp, (treat, ref) in COMPARISONS.items():
        rows = {}
        for v in key.values():
            if v['comp'] == comp:
                rows.setdefault(v['id'], {})[v['cond']] = v
        for m in ('aes_PQ', 'aes_CE', 'meva', 'clap'):
            if 'meva' == m and 'meva' not in next(iter(key.values())):
                continue
            d = [r[treat][m] - r[ref][m] for r in rows.values()]
            print(f'{comp:5s} {m:7s} mean diff {sum(d)/len(d):+.3f}  treat wins {sum(x > 0 for x in d)}/{len(d)}')


if __name__ == '__main__':
    step = sys.argv[1]
    for name, fn in (('select', select), ('generate', generate), ('prep', prep), ('score', score),
                     ('meva', meva), ('summary', summary)):
        if step == name or (step == 'all' and name not in ('meva', 'summary')):
            fn()
